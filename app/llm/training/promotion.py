"""Checkpoint promotion gate.

A trained checkpoint is **promoted** only when its mean LLM-judge score
on the held-out scenarios is at or above the configured floor (default
85/100, per ``docs/intelligence_migration.md``). This module is the
single source of truth for that gate; both the CLI in
``scripts/promote_checkpoint.py`` and CI use it through
:func:`evaluate_and_promote`.

Calibration metrics (ECE, Brier) are computed alongside the judge score
and recorded in the release manifest but do not gate promotion on their
own — the judge is the contract.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from app.evals.scenario_eval import AggregateEvalReport, ScenarioJudge
from app.evals.scenario_loader import ScenarioCase
from app.llm.training.calibration import (
    CalibrationReport,
    compute_calibration,
)
from app.llm.training.inferencer import (
    InferenceRunReport,
    SituationEngineInferencer,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromotionDecision:
    """The verdict + the data that justified it."""

    promoted: bool
    floor: int
    judge_report: AggregateEvalReport
    calibration: CalibrationReport
    inference: InferenceRunReport
    manifest_path: Optional[Path]


def _hash_directory(path: Path) -> str:
    """Stable SHA-256 of the file tree at ``path`` (sorted, content only)."""
    h = hashlib.sha256()
    if not path.exists():
        return ""
    if path.is_file():
        h.update(path.read_bytes())
        return h.hexdigest()
    for entry in sorted(path.rglob("*")):
        if entry.is_file():
            h.update(entry.relative_to(path).as_posix().encode("utf-8"))
            h.update(b"\0")
            h.update(entry.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


async def evaluate_and_promote(
    cases: List[ScenarioCase],
    inferencer: SituationEngineInferencer,
    judge: ScenarioJudge,
    *,
    floor: int = 85,
    checkpoint_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> PromotionDecision:
    """Run inference + judge + calibration; gate on the judge mean."""
    inference_report = inferencer.run(cases)

    pairs_for_judge = [
        (case, inference_report.predictions[case.scenario_id])
        for case in cases
        if case.scenario_id in inference_report.predictions
    ]
    if not pairs_for_judge:
        logger.error("no scenarios produced a valid SituationReport")
        empty_judge = await judge.score_many([])
        empty_cal = compute_calibration([])
        return PromotionDecision(
            promoted=False,
            floor=floor,
            judge_report=empty_judge,
            calibration=empty_cal,
            inference=inference_report,
            manifest_path=None,
        )

    judge_report = await judge.score_many(pairs_for_judge)
    calibration = compute_calibration(
        [(pred, case.gold_report) for case, pred in pairs_for_judge]
    )

    promoted = (
        judge_report.mean_total >= floor
        and not inference_report.failures
    )

    written_manifest_path: Optional[Path] = None
    if manifest_path is not None:
        written_manifest_path = _write_manifest(
            manifest_path=manifest_path,
            promoted=promoted,
            floor=floor,
            judge_report=judge_report,
            calibration=calibration,
            inference_report=inference_report,
            checkpoint_path=checkpoint_path,
        )

    return PromotionDecision(
        promoted=promoted,
        floor=floor,
        judge_report=judge_report,
        calibration=calibration,
        inference=inference_report,
        manifest_path=written_manifest_path,
    )


def _write_manifest(
    *,
    manifest_path: Path,
    promoted: bool,
    floor: int,
    judge_report: AggregateEvalReport,
    calibration: CalibrationReport,
    inference_report: InferenceRunReport,
    checkpoint_path: Optional[Path],
) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "decided_at": datetime.now(timezone.utc).isoformat(),
        "promoted": promoted,
        "floor": floor,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": (
            _hash_directory(checkpoint_path) if checkpoint_path else None
        ),
        "judge": {
            "n_scenarios": judge_report.n_scenarios,
            "mean_total": round(judge_report.mean_total, 3),
            "mean_accuracy": round(judge_report.mean_accuracy, 3),
            "mean_groundedness": round(judge_report.mean_groundedness, 3),
            "mean_calibration": round(judge_report.mean_calibration, 3),
            "mean_abstention": round(judge_report.mean_abstention, 3),
            "mean_actionability": round(judge_report.mean_actionability, 3),
            "floor_pass": judge_report.floor_pass,
            "per_scenario": [
                {
                    "scenario_id": r.scenario_id,
                    "total": r.score.total,
                }
                for r in judge_report.per_scenario
            ],
        },
        "calibration": {
            "n": calibration.n,
            "accuracy": round(calibration.accuracy, 3),
            "mean_confidence": round(calibration.mean_confidence, 3),
            "ece": round(calibration.expected_calibration_error, 3),
            "brier": round(calibration.brier_score, 3),
            "reliability_table": [asdict(b) for b in calibration.reliability_table],
        },
        "inference": {
            "n_total": inference_report.n_total,
            "n_success": inference_report.n_success,
            "n_failures": len(inference_report.failures),
            "failures": [
                {"scenario_id": f.scenario_id, "error_kind": f.error_kind,
                 "detail": f.detail}
                for f in inference_report.failures
            ],
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote promotion manifest to %s", manifest_path)
    return manifest_path
