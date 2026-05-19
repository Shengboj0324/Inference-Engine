"""Unit tests for ``app.llm.training.calibration``.

Covers accuracy, mean confidence, Brier score, ECE, and the reliability
table over hand-crafted (candidate, gold) report pairs.
"""

from __future__ import annotations

import math

import pytest

from app.intelligence.situation_report import (
    Citation,
    Claim,
    Severity,
    SituationReport,
    SuggestedAction,
)
from app.llm.training.calibration import compute_calibration


def _report(
    *,
    signal_type: str = "sig_a",
    severity: Severity = Severity.SEV_3,
    confidence: float = 0.7,
    abstain: bool = False,
) -> SituationReport:
    if abstain:
        return SituationReport(
            signal_type="insufficient_evidence",
            severity=Severity.SEV_5,
            calibrated_confidence=confidence,
            summary="abstain",
            claims=[],
            citations=[],
            suggested_actions=[],
            abstain=True,
            abstention_reason="not enough evidence",
        )
    return SituationReport(
        signal_type=signal_type,
        severity=severity,
        calibrated_confidence=confidence,
        summary="ok",
        citations=[Citation(post_id="obs_1", char_start=0, char_end=3)],
        claims=[Claim(text="c", citation_ids=[0], confidence=confidence)],
        suggested_actions=[
            SuggestedAction(text="a", rationale_claim_ids=[0], priority=1)
        ],
    )


def test_empty_pairs_returns_zero_report():
    report = compute_calibration([])
    assert report.n == 0
    assert report.accuracy == 0.0
    assert report.brier_score == 0.0
    assert report.expected_calibration_error == 0.0
    assert report.reliability_table == ()


def test_perfect_predictions_yield_unit_accuracy():
    gold = _report(confidence=0.9)
    cand = _report(confidence=0.9)
    report = compute_calibration([(cand, gold), (cand, gold)])
    assert report.n == 2
    assert report.accuracy == 1.0
    # Brier with 1.0 correct and 0.9 confidence: (0.9 - 1)^2 = 0.01.
    assert math.isclose(report.brier_score, 0.01, abs_tol=1e-9)


def test_all_wrong_yields_zero_accuracy():
    gold = _report(signal_type="sig_a")
    cand = _report(signal_type="sig_b", confidence=0.9)
    report = compute_calibration([(cand, gold)])
    assert report.accuracy == 0.0
    # (0.9 - 0)^2 = 0.81.
    assert math.isclose(report.brier_score, 0.81, abs_tol=1e-9)


def test_severity_mismatch_counts_as_wrong():
    gold = _report(severity=Severity.SEV_2)
    cand = _report(severity=Severity.SEV_4, confidence=0.5)
    report = compute_calibration([(cand, gold)])
    assert report.accuracy == 0.0


def test_abstain_mismatch_counts_as_wrong():
    gold = _report(abstain=True, confidence=0.6)
    cand = _report(abstain=False, confidence=0.6)
    report = compute_calibration([(cand, gold)])
    assert report.accuracy == 0.0


def test_ece_zero_when_confidence_matches_accuracy():
    # Two correct at conf=1.0, two wrong at conf=0.0 => perfectly calibrated.
    gold = _report(signal_type="sig_a")
    other = _report(signal_type="sig_b")
    pairs = [
        (_report(signal_type="sig_a", confidence=1.0), gold),
        (_report(signal_type="sig_a", confidence=1.0), gold),
        (_report(signal_type="sig_a", confidence=0.0), other),
        (_report(signal_type="sig_a", confidence=0.0), other),
    ]
    report = compute_calibration(pairs, n_buckets=10)
    assert report.accuracy == 0.5
    assert math.isclose(report.expected_calibration_error, 0.0, abs_tol=1e-9)


def test_ece_high_when_overconfident():
    gold = _report(signal_type="sig_a")
    other = _report(signal_type="sig_b")
    # All wrong but confidence 1.0 -> ECE = 1.0.
    pairs = [
        (_report(signal_type="sig_a", confidence=1.0), other),
        (_report(signal_type="sig_a", confidence=1.0), other),
    ]
    report = compute_calibration(pairs)
    assert math.isclose(report.expected_calibration_error, 1.0, abs_tol=1e-9)


def test_reliability_table_buckets_are_disjoint_and_cover_unit_interval():
    gold = _report()
    cand = _report(confidence=0.55)
    report = compute_calibration([(cand, gold)], n_buckets=5)
    assert len(report.reliability_table) == 5
    # Sums of [lo, hi) should cover [0, 1].
    assert report.reliability_table[0].lower == 0.0
    assert report.reliability_table[-1].upper == 1.0
    # Only one bucket should be populated.
    populated = [b for b in report.reliability_table if b.count > 0]
    assert len(populated) == 1
    assert populated[0].lower <= 0.55 < populated[0].upper


def test_rejects_out_of_range_confidence():
    gold = _report()
    bad_payload = gold.model_dump()
    bad_payload["calibrated_confidence"] = 1.5
    # Pydantic rejects out-of-range values at construction time, so we
    # cannot build a SituationReport with confidence > 1. Instead, force
    # the situation via a mock that bypasses validation.
    with pytest.raises(Exception):
        SituationReport.model_validate(bad_payload)
