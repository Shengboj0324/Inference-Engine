"""Calibration metrics for SituationReport confidence.

The promotion gate scores correctness via the LLM-as-judge rubric in
``app.evals.scenario_eval``. Calibration is a separate, deterministic
signal computed against the gold reports directly:

- ``expected_calibration_error`` (ECE)
- ``brier_score``
- ``reliability_table`` (bucketed forecast vs. observed accuracy)

"Correctness" for calibration purposes is defined as an exact match on
``signal_type`` AND ``abstain`` AND ``severity`` between the candidate
and the gold report. This is a strict definition; we want the model to
be well-calibrated against the same signal the judge ultimately scores.

The module deliberately has no torch / numpy dependencies on its hot
path so it can run on CPU inside CI without GPU wheels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from app.intelligence.situation_report import SituationReport


@dataclass(frozen=True)
class CalibrationBucket:
    """One row of the reliability diagram."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    observed_accuracy: float


@dataclass(frozen=True)
class CalibrationReport:
    """Aggregate calibration result over a (candidate, gold) collection."""

    n: int
    accuracy: float
    mean_confidence: float
    expected_calibration_error: float
    brier_score: float
    reliability_table: Tuple[CalibrationBucket, ...]


def _is_correct(candidate: SituationReport, gold: SituationReport) -> bool:
    """Strict exact-match definition used for calibration scoring."""
    return (
        candidate.signal_type == gold.signal_type
        and candidate.abstain == gold.abstain
        and candidate.severity == gold.severity
    )


def _bucket_edges(n_buckets: int) -> List[Tuple[float, float]]:
    if n_buckets < 1:
        raise ValueError("n_buckets must be >= 1")
    width = 1.0 / n_buckets
    edges: List[Tuple[float, float]] = []
    for i in range(n_buckets):
        lo = i * width
        hi = 1.0 if i == n_buckets - 1 else (i + 1) * width
        edges.append((lo, hi))
    return edges


def _bucket_index(confidence: float, edges: Sequence[Tuple[float, float]]) -> int:
    # confidence is in [0, 1]; assign 1.0 to the final bucket.
    if confidence >= 1.0:
        return len(edges) - 1
    for idx, (lo, hi) in enumerate(edges):
        if lo <= confidence < hi:
            return idx
    return len(edges) - 1


def compute_calibration(
    pairs: Sequence[Tuple[SituationReport, SituationReport]],
    *,
    n_buckets: int = 10,
) -> CalibrationReport:
    """Compute calibration metrics for an iterable of (candidate, gold) pairs.

    Args:
        pairs: Each pair is ``(candidate_report, gold_report)``.
        n_buckets: Number of equal-width confidence buckets.

    Returns:
        A ``CalibrationReport``. If ``pairs`` is empty, all fields are zero.
    """
    n = len(pairs)
    if n == 0:
        return CalibrationReport(
            n=0,
            accuracy=0.0,
            mean_confidence=0.0,
            expected_calibration_error=0.0,
            brier_score=0.0,
            reliability_table=(),
        )

    edges = _bucket_edges(n_buckets)
    bucket_correct = [0] * n_buckets
    bucket_total = [0] * n_buckets
    bucket_conf_sum = [0.0] * n_buckets

    total_correct = 0
    total_conf = 0.0
    brier_sum = 0.0
    for candidate, gold in pairs:
        conf = float(candidate.calibrated_confidence)
        if not 0.0 <= conf <= 1.0:
            raise ValueError(
                f"calibrated_confidence out of range [0,1]: {conf}"
            )
        correct = 1 if _is_correct(candidate, gold) else 0
        total_correct += correct
        total_conf += conf
        brier_sum += (conf - correct) ** 2
        idx = _bucket_index(conf, edges)
        bucket_total[idx] += 1
        bucket_correct[idx] += correct
        bucket_conf_sum[idx] += conf

    ece = 0.0
    table: List[CalibrationBucket] = []
    for idx, (lo, hi) in enumerate(edges):
        count = bucket_total[idx]
        if count == 0:
            table.append(CalibrationBucket(lo, hi, 0, 0.0, 0.0))
            continue
        mean_conf = bucket_conf_sum[idx] / count
        observed = bucket_correct[idx] / count
        ece += (count / n) * abs(mean_conf - observed)
        table.append(CalibrationBucket(lo, hi, count, mean_conf, observed))

    return CalibrationReport(
        n=n,
        accuracy=total_correct / n,
        mean_confidence=total_conf / n,
        expected_calibration_error=ece,
        brier_score=brier_sum / n,
        reliability_table=tuple(table),
    )
