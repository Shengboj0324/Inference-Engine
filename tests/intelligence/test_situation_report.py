"""Tests for the SituationReport schema."""

import pytest
from pydantic import ValidationError

from app.intelligence.situation_report import (
    Citation,
    Claim,
    Severity,
    SituationReport,
    SuggestedAction,
)


def _valid_kwargs(**overrides):
    base = dict(
        signal_type="security_concern",
        severity=Severity.SEV_2,
        calibrated_confidence=0.81,
        summary="A summary grounded in observations.",
        citations=[Citation(post_id="obs_1", char_start=0, char_end=10)],
        claims=[Claim(text="A claim.", citation_ids=[0], confidence=0.8)],
        suggested_actions=[
            SuggestedAction(text="Investigate.", rationale_claim_ids=[0], priority=2)
        ],
    )
    base.update(overrides)
    return base


def test_citation_rejects_empty_span():
    with pytest.raises(ValidationError):
        Citation(post_id="obs", char_start=5, char_end=5)


def test_citation_rejects_inverted_span():
    with pytest.raises(ValidationError):
        Citation(post_id="obs", char_start=10, char_end=3)


def test_valid_positive_report():
    r = SituationReport(**_valid_kwargs())
    assert r.signal_type == "security_concern"
    assert r.severity is Severity.SEV_2
    assert not r.abstain


def test_abstain_requires_reason():
    with pytest.raises(ValidationError):
        SituationReport(
            signal_type="unclear",
            severity=Severity.SEV_5,
            calibrated_confidence=0.2,
            summary="no evidence",
            abstain=True,
        )


def test_abstain_forbids_claims():
    with pytest.raises(ValidationError):
        SituationReport(
            signal_type="unclear",
            severity=Severity.SEV_5,
            calibrated_confidence=0.2,
            summary="no evidence",
            abstain=True,
            abstention_reason="no corroborating source",
            claims=[Claim(text="x", citation_ids=[0], confidence=0.5)],
            citations=[Citation(post_id="o", char_start=0, char_end=2)],
        )


def test_abstain_with_reason_validates():
    r = SituationReport(
        signal_type="unclear",
        severity=Severity.SEV_5,
        calibrated_confidence=0.2,
        summary="evidence missing",
        abstain=True,
        abstention_reason="no named affected party",
    )
    assert r.abstain is True
    assert r.claims == []


def test_non_abstain_requires_claims():
    with pytest.raises(ValidationError):
        SituationReport(**_valid_kwargs(claims=[], citations=[]))


def test_non_abstain_requires_citations():
    with pytest.raises(ValidationError):
        SituationReport(**_valid_kwargs(
            citations=[],
            claims=[Claim(text="c", citation_ids=[0], confidence=0.5)],
        ))


def test_claim_citation_id_out_of_range():
    with pytest.raises(ValidationError):
        SituationReport(**_valid_kwargs(
            citations=[Citation(post_id="o", char_start=0, char_end=2)],
            claims=[Claim(text="c", citation_ids=[5], confidence=0.5)],
        ))


def test_action_rationale_claim_id_out_of_range():
    with pytest.raises(ValidationError):
        SituationReport(**_valid_kwargs(
            suggested_actions=[
                SuggestedAction(
                    text="x", rationale_claim_ids=[7], priority=1
                )
            ],
        ))


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        SituationReport(**_valid_kwargs(calibrated_confidence=1.5))
    with pytest.raises(ValidationError):
        Claim(text="x", citation_ids=[0], confidence=-0.1)


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        SituationReport(**_valid_kwargs(), unexpected="value")


def test_round_trip_via_json():
    r = SituationReport(**_valid_kwargs())
    blob = r.model_dump_json()
    restored = SituationReport.model_validate_json(blob)
    assert restored == r
