"""Unit tests for ``app.llm.training.promotion.evaluate_and_promote``.

The gate is the single source of truth that decides whether a fine-tuned
checkpoint is allowed to ship. These tests drive it with mocked
inferencers and judges so no GPU or LLM is required.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

from app.evals.scenario_eval import ScenarioJudge
from app.evals.scenario_loader import ScenarioLoader
from app.evals.synthetic_scenarios import generate_corpus
from app.llm.training.inferencer import SituationEngineInferencer
from app.llm.training.promotion import evaluate_and_promote


def _heldout_cases(tmp_path: Path):
    generate_corpus(
        tmp_path / "scenarios", n_train=0, n_val=0, n_heldout=3, seed=1,
    )
    loader = ScenarioLoader(tmp_path / "scenarios")
    return list(loader.discover(split="heldout"))


def _make_echo_inferencer(cases):
    by_text = {
        tuple(o.text for o in c.observations): c.gold_report for c in cases
    }

    def _generate(messages):
        payload = json.loads(messages[1]["content"])
        key = tuple(o["text"] for o in payload["observations"])
        return by_text[key].model_dump_json()

    return SituationEngineInferencer(_generate)


def _fake_judge(response_payload: dict) -> ScenarioJudge:
    router = AsyncMock()
    router.complete = AsyncMock(return_value=json.dumps(response_payload))
    return ScenarioJudge(router=router, passing_floor=85)


def test_promotion_succeeds_when_judge_mean_clears_floor(tmp_path):
    cases = _heldout_cases(tmp_path)
    inferencer = _make_echo_inferencer(cases)
    judge = _fake_judge({
        "accuracy": 19, "groundedness": 19, "calibration": 19,
        "abstention": 19, "actionability": 19, "total": 95,
        "rationale": "ok",
    })
    manifest = tmp_path / "promotion.json"
    decision = asyncio.run(evaluate_and_promote(
        cases=cases, inferencer=inferencer, judge=judge,
        floor=85, checkpoint_path=tmp_path, manifest_path=manifest,
    ))
    assert decision.promoted is True
    assert decision.judge_report.mean_total == 95.0
    assert decision.calibration.n == len(cases)
    assert manifest.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["promoted"] is True
    assert payload["judge"]["floor_pass"] is True


def test_promotion_rejects_when_below_floor(tmp_path):
    cases = _heldout_cases(tmp_path)
    inferencer = _make_echo_inferencer(cases)
    judge = _fake_judge({
        "accuracy": 10, "groundedness": 10, "calibration": 10,
        "abstention": 10, "actionability": 10, "total": 50,
        "rationale": "weak",
    })
    decision = asyncio.run(evaluate_and_promote(
        cases=cases, inferencer=inferencer, judge=judge,
        floor=85, manifest_path=None,
    ))
    assert decision.promoted is False
    assert decision.judge_report.mean_total == 50.0


def test_promotion_rejects_when_any_inference_fails(tmp_path):
    cases = _heldout_cases(tmp_path)

    def _broken_generate(messages):
        raise RuntimeError("boom")

    inferencer = SituationEngineInferencer(_broken_generate)
    judge = _fake_judge({
        "accuracy": 20, "groundedness": 20, "calibration": 20,
        "abstention": 20, "actionability": 20, "total": 100,
        "rationale": "perfect",
    })
    decision = asyncio.run(evaluate_and_promote(
        cases=cases, inferencer=inferencer, judge=judge,
        floor=85, manifest_path=None,
    ))
    assert decision.promoted is False
    assert decision.inference.n_success == 0
    assert len(decision.inference.failures) == len(cases)


def test_promotion_manifest_records_calibration_and_failures(tmp_path):
    cases = _heldout_cases(tmp_path)
    # Mix: one valid, rest broken.
    by_text = {
        tuple(o.text for o in c.observations): c.gold_report for c in cases
    }
    first_key = tuple(o.text for o in cases[0].observations)

    def _partial(messages):
        payload = json.loads(messages[1]["content"])
        key = tuple(o["text"] for o in payload["observations"])
        if key == first_key:
            return by_text[key].model_dump_json()
        raise RuntimeError("partial failure")

    inferencer = SituationEngineInferencer(_partial)
    judge = _fake_judge({
        "accuracy": 18, "groundedness": 18, "calibration": 18,
        "abstention": 18, "actionability": 18, "total": 90,
        "rationale": "good",
    })
    manifest = tmp_path / "manifest.json"
    decision = asyncio.run(evaluate_and_promote(
        cases=cases, inferencer=inferencer, judge=judge,
        floor=85, checkpoint_path=tmp_path, manifest_path=manifest,
    ))
    # Inference failures must always block promotion regardless of judge.
    assert decision.promoted is False
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["inference"]["n_success"] == 1
    assert payload["inference"]["n_failures"] == len(cases) - 1
    assert payload["calibration"]["n"] == 1
