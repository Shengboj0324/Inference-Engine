"""Tests for the structural CitationVerifier."""

import pytest

from app.evals.scenario_loader import Observation
from app.intelligence.citation_verifier import (
    CitationGroundingError,
    CitationVerifier,
)
from app.intelligence.situation_report import (
    Citation,
    Claim,
    Severity,
    SituationReport,
)


def _obs(observation_id: str, text: str) -> Observation:
    return Observation(
        observation_id=observation_id,
        text=text,
        source="unit_test",
        timestamp=None,
    )


def _report(citations, *, abstain: bool = False) -> SituationReport:
    if abstain:
        return SituationReport(
            signal_type="insufficient_evidence",
            severity=Severity.SEV_5,
            calibrated_confidence=0.5,
            summary="No basis for a finding.",
            citations=[],
            claims=[],
            suggested_actions=[],
            abstain=True,
            abstention_reason="not enough evidence",
        )
    return SituationReport(
        signal_type="security_concern",
        severity=Severity.SEV_3,
        calibrated_confidence=0.7,
        summary="A summary.",
        citations=citations,
        claims=[Claim(text="A claim.", citation_ids=[0], confidence=0.7)],
    )


def test_happy_path_accepts_valid_spans():
    verifier = CitationVerifier()
    observations = [_obs("obs_1", "hello world")]
    report = _report([Citation(post_id="obs_1", char_start=0, char_end=5)])
    verifier.verify_report(report, observations)


def test_rejects_unknown_post_id():
    verifier = CitationVerifier()
    observations = [_obs("obs_1", "hello world")]
    report = _report([Citation(post_id="not_there", char_start=0, char_end=5)])
    with pytest.raises(CitationGroundingError) as exc_info:
        verifier.verify_report(report, observations)
    assert any(f.post_id == "not_there" for f in exc_info.value.failures)


def test_rejects_char_end_past_text():
    verifier = CitationVerifier()
    observations = [_obs("obs_1", "hi")]
    report = _report([Citation(post_id="obs_1", char_start=0, char_end=10)])
    with pytest.raises(CitationGroundingError) as exc_info:
        verifier.verify_report(report, observations)
    assert "exceeds text length" in exc_info.value.failures[0].reason


def test_rejects_char_start_past_text():
    verifier = CitationVerifier()
    observations = [_obs("obs_1", "hi")]
    # char_start=5 past end; we must also satisfy citation.char_end > char_start
    report = _report([Citation(post_id="obs_1", char_start=5, char_end=6)])
    with pytest.raises(CitationGroundingError) as exc_info:
        verifier.verify_report(report, observations)
    reasons = [f.reason for f in exc_info.value.failures]
    assert any("char_start" in r or "char_end" in r for r in reasons)


def test_rejects_whitespace_only_span():
    verifier = CitationVerifier()
    observations = [_obs("obs_1", "a    b")]
    # Span "   " (indices 1..4) is all whitespace
    report = _report([Citation(post_id="obs_1", char_start=1, char_end=4)])
    with pytest.raises(CitationGroundingError) as exc_info:
        verifier.verify_report(report, observations)
    assert "whitespace" in exc_info.value.failures[0].reason


def test_abstain_skips_verification():
    verifier = CitationVerifier()
    observations = [_obs("obs_1", "hi")]
    report = _report([], abstain=True)
    verifier.verify_report(report, observations)


def test_collects_multiple_failures():
    verifier = CitationVerifier()
    observations = [_obs("obs_1", "hi")]
    report = SituationReport(
        signal_type="security_concern",
        severity=Severity.SEV_3,
        calibrated_confidence=0.7,
        summary="A summary.",
        citations=[
            Citation(post_id="obs_1", char_start=0, char_end=2),
            Citation(post_id="missing", char_start=0, char_end=2),
            Citation(post_id="obs_1", char_start=0, char_end=99),
        ],
        claims=[
            Claim(text="A claim.", citation_ids=[0, 1, 2], confidence=0.5),
        ],
    )
    with pytest.raises(CitationGroundingError) as exc_info:
        verifier.verify_report(report, observations)
    assert len(exc_info.value.failures) == 2
    indices = {f.citation_index for f in exc_info.value.failures}
    assert indices == {1, 2}
