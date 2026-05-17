"""Tests for the Phase 1 scenario infrastructure.

Covers:
- ``ScenarioLoader`` discovery, validation, and cross-file integrity.
- ``ScenarioJudge`` prompt assembly, score parsing, and aggregation.
- ``scripts/check_no_keyword_rules`` violation detection and allowlisting.

All LLM calls are mocked via ``unittest.mock.AsyncMock`` on
``LLMRouter.complete``; no network is touched.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.evals.scenario_eval import (
    ScenarioJudge,
    ScenarioScore,
    _JUDGE_RUBRIC_PROMPT,
)
from app.evals.scenario_loader import ScenarioLoader
from app.intelligence.situation_report import (
    Citation,
    Claim,
    Severity,
    SituationReport,
    SuggestedAction,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_scenario(
    root: Path,
    scenario_id: str,
    *,
    split: str = "train",
    pii_passed: bool = True,
    bad_citation: bool = False,
) -> Path:
    folder = root / scenario_id
    folder.mkdir(parents=True, exist_ok=True)
    obs = [
        {"observation_id": "obs_1", "source": "x", "timestamp": None,
         "text": "evidence text here"},
        {"observation_id": "obs_2", "source": "y", "timestamp": None,
         "text": "second piece"},
    ]
    (folder / "observations.jsonl").write_text(
        "\n".join(json.dumps(o) for o in obs) + "\n", encoding="utf-8"
    )
    gold = SituationReport(
        signal_type="security_concern",
        severity=Severity.SEV_3,
        calibrated_confidence=0.7,
        summary="summary",
        citations=[
            Citation(
                post_id="missing_obs" if bad_citation else "obs_1",
                char_start=0, char_end=8,
            )
        ],
        claims=[Claim(text="c", citation_ids=[0], confidence=0.7)],
        suggested_actions=[
            SuggestedAction(text="act", rationale_claim_ids=[0], priority=2)
        ],
    )
    (folder / "gold_report.json").write_text(
        gold.model_dump_json(indent=2), encoding="utf-8"
    )
    (folder / "metadata.yaml").write_text(
        "scenario_id: {sid}\n"
        "split: {split}\n"
        "author_id: A01\n"
        "annotator_ids:\n  - A01\n  - A02\n"
        "adjudicator_id: null\n"
        "guideline_version: 1.0.0\n"
        "created_on: 2026-05-17\n"
        "adversarial_tags: []\n"
        "pii_review_passed: {pii}\n"
        "pii_reviewer_id: {reviewer}\n".format(
            sid=scenario_id,
            split=split,
            pii=str(pii_passed).lower(),
            reviewer="P01" if pii_passed else "null",
        ),
        encoding="utf-8",
    )
    return folder


def test_loader_discovers_and_validates(tmp_path):
    _write_scenario(tmp_path, "scen_001", split="train")
    _write_scenario(tmp_path, "scen_002", split="heldout")
    loader = ScenarioLoader(tmp_path)
    train = list(loader.discover(split="train"))
    heldout = list(loader.discover(split="heldout"))
    assert [c.scenario_id for c in train] == ["scen_001"]
    assert [c.scenario_id for c in heldout] == ["scen_002"]
    assert train[0].observations[0].observation_id == "obs_1"


def test_loader_skips_underscore_template(tmp_path):
    _write_scenario(tmp_path, "_template", split="train")
    _write_scenario(tmp_path, "scen_010", split="train")
    loader = ScenarioLoader(tmp_path)
    ids = [c.scenario_id for c in loader.discover()]
    assert ids == ["scen_010"]


def test_loader_filters_unsigned_pii_by_default(tmp_path):
    _write_scenario(tmp_path, "scen_020", split="train", pii_passed=False)
    loader = ScenarioLoader(tmp_path)
    assert list(loader.discover()) == []
    assert len(list(loader.discover(require_pii_signoff=False))) == 1


def test_loader_rejects_bad_citation_post_id(tmp_path):
    _write_scenario(tmp_path, "scen_030", bad_citation=True)
    loader = ScenarioLoader(tmp_path)
    with pytest.raises(ValueError, match="references unknown"):
        list(loader.discover())


def test_loader_rejects_mismatched_folder_name(tmp_path):
    folder = _write_scenario(tmp_path, "scen_040")
    meta = folder / "metadata.yaml"
    meta.write_text(
        meta.read_text(encoding="utf-8").replace(
            "scenario_id: scen_040", "scenario_id: wrong_id"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not\\s+match folder name"):
        list(ScenarioLoader(tmp_path).discover())


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def _fake_router(response_text: str):
    router = AsyncMock()
    router.complete = AsyncMock(return_value=response_text)
    return router


def test_rubric_prompt_is_a_string_not_code():
    assert isinstance(_JUDGE_RUBRIC_PROMPT, str)
    assert "Accuracy" in _JUDGE_RUBRIC_PROMPT
    assert "Groundedness" in _JUDGE_RUBRIC_PROMPT
    assert "Calibration" in _JUDGE_RUBRIC_PROMPT
    assert "Abstention" in _JUDGE_RUBRIC_PROMPT
    assert "Actionability" in _JUDGE_RUBRIC_PROMPT


def test_judge_score_one_parses_json(tmp_path):
    _write_scenario(tmp_path, "scen_100", split="heldout")
    case = next(ScenarioLoader(tmp_path).discover(split="heldout"))
    candidate = case.gold_report
    payload = {
        "accuracy": 18, "groundedness": 17, "calibration": 16,
        "abstention": 20, "actionability": 15, "total": 86,
        "rationale": "looks fine",
    }
    judge = ScenarioJudge(router=_fake_router(json.dumps(payload)))
    result = asyncio.run(judge.score_one(case, candidate))
    assert result.scenario_id == "scen_100"
    assert result.score.total == 86


def test_judge_recomputes_inconsistent_total(tmp_path):
    _write_scenario(tmp_path, "scen_101", split="heldout")
    case = next(ScenarioLoader(tmp_path).discover(split="heldout"))
    payload = {
        "accuracy": 10, "groundedness": 10, "calibration": 10,
        "abstention": 10, "actionability": 10, "total": 99,
        "rationale": "bad sum",
    }
    judge = ScenarioJudge(router=_fake_router(json.dumps(payload)))
    result = asyncio.run(judge.score_one(case, case.gold_report))
    assert result.score.total == 50


def test_judge_aggregates_floor_pass(tmp_path):
    _write_scenario(tmp_path, "s_a", split="heldout")
    _write_scenario(tmp_path, "s_b", split="heldout")
    cases = list(ScenarioLoader(tmp_path).discover(split="heldout"))
    high = json.dumps({"accuracy": 18, "groundedness": 18, "calibration": 18,
                       "abstention": 18, "actionability": 18, "total": 90,
                       "rationale": "good"})
    router = AsyncMock()
    router.complete = AsyncMock(return_value=high)
    judge = ScenarioJudge(router=router, passing_floor=85)
    report = asyncio.run(
        judge.score_many([(c, c.gold_report) for c in cases])
    )
    assert report.n_scenarios == 2
    assert report.mean_total == 90.0
    assert report.floor_pass is True


def test_judge_rejects_non_json(tmp_path):
    _write_scenario(tmp_path, "scen_102", split="heldout")
    case = next(ScenarioLoader(tmp_path).discover(split="heldout"))
    judge = ScenarioJudge(router=_fake_router("the model returned prose"))
    with pytest.raises(ValueError, match="no JSON object"):
        asyncio.run(judge.score_one(case, case.gold_report))


# ---------------------------------------------------------------------------
# CI guard
# ---------------------------------------------------------------------------

def _run_guard(target: Path):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_no_keyword_rules.py"),
         "--target", str(target)],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )


def test_guard_passes_on_clean_tree(tmp_path):
    (tmp_path / "ok.py").write_text(
        "def f():\n    return 1\n", encoding="utf-8"
    )
    proc = _run_guard(tmp_path)
    assert proc.returncode == 0
    assert "[OK]" in proc.stdout


def test_guard_fails_on_phrase_registry(tmp_path):
    (tmp_path / "bad.py").write_text(
        'KEYWORDS = ["breach", "leak"]\n', encoding="utf-8"
    )
    proc = _run_guard(tmp_path)
    assert proc.returncode == 1
    assert "phrase_or_keyword_registry" in proc.stderr


def test_guard_fails_on_detector_module_name(tmp_path):
    (tmp_path / "ok.py").write_text(
        "from app import signal_detectors\n", encoding="utf-8"
    )
    proc = _run_guard(tmp_path)
    assert proc.returncode == 1
    assert "signal_detector_module" in proc.stderr


def test_guard_fails_on_report_section_template(tmp_path):
    (tmp_path / "bad.py").write_text(
        'TEMPLATE = "## Section 1: Confirmed Findings"\n',
        encoding="utf-8",
    )
    proc = _run_guard(tmp_path)
    assert proc.returncode == 1
    assert "report_template_string" in proc.stderr


def test_guard_passes_on_live_repo():
    proc = _run_guard(REPO_ROOT / "app" / "intelligence")
    assert proc.returncode == 0, proc.stderr

