"""Tests for the ``quarantined`` metadata field and the build-set skip rule.

Covers the deliverables.md §3.4 / §4.4 contract: a quarantined scenario is a
valid ``ScenarioMetadata`` (so the Lead can flag it) but is excluded from
``train.jsonl`` / ``val.jsonl`` by ``scripts/build_training_set.py``. The
schema still forbids *unknown* fields (``extra="forbid"``).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evals.scenario_loader import ScenarioMetadata
from app.intelligence.situation_report import (
    Citation, Claim, Severity, SituationReport, SuggestedAction,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _meta(**over) -> dict:
    base = dict(
        scenario_id="scen_q", split="train", author_id="A01",
        annotator_ids=["A01", "A02"], adjudicator_id=None,
        guideline_version="1.0.0", created_on="2026-05-17",
        adversarial_tags=[], pii_review_passed=True, pii_reviewer_id="P01",
    )
    base.update(over)
    return base


def test_quarantined_defaults_false():
    m = ScenarioMetadata.model_validate(_meta())
    assert m.quarantined is False


def test_quarantined_true_is_valid():
    m = ScenarioMetadata.model_validate(_meta(quarantined=True))
    assert m.quarantined is True


def test_unknown_field_still_forbidden():
    with pytest.raises(ValidationError):
        ScenarioMetadata.model_validate(_meta(not_a_field=1))


def _write_scenario(root: Path, sid: str, *, quarantined: bool) -> None:
    folder = root / sid
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "observations.jsonl").write_text(
        json.dumps({"observation_id": "obs_1", "source": "x",
                    "timestamp": None, "text": "evidence text here"}) + "\n",
        encoding="utf-8",
    )
    gold = SituationReport(
        signal_type="bug_report", severity=Severity.SEV_3,
        calibrated_confidence=0.7, summary="summary",
        citations=[Citation(post_id="obs_1", char_start=0, char_end=8)],
        claims=[Claim(text="c", citation_ids=[0], confidence=0.7)],
        suggested_actions=[SuggestedAction(text="a", rationale_claim_ids=[0], priority=3)],
    )
    (folder / "gold_report.json").write_text(gold.model_dump_json(indent=2), encoding="utf-8")
    q = "true" if quarantined else "false"
    (folder / "metadata.yaml").write_text(
        f"scenario_id: {sid}\nsplit: train\nauthor_id: A01\n"
        "annotator_ids:\n  - A01\n  - A02\nadjudicator_id: null\n"
        "guideline_version: 1.0.0\ncreated_on: 2026-05-17\n"
        f"adversarial_tags: []\npii_review_passed: true\npii_reviewer_id: P01\n"
        f"quarantined: {q}\n",
        encoding="utf-8",
    )


def test_build_training_set_skips_quarantined(tmp_path):
    scen = tmp_path / "scenarios"
    out = tmp_path / "training"
    _write_scenario(scen, "keep_001", quarantined=False)
    _write_scenario(scen, "drop_001", quarantined=True)
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "build_training_set.py"),
         "--scenarios-root", str(scen), "--out-dir", str(out)],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    train = (out / "train.jsonl").read_text(encoding="utf-8")
    assert "internal_label/1.0.0/keep_001" in train
    assert "drop_001" not in train
