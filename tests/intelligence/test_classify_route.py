"""Unit tests for the Phase 3 classify route plumbing.

The HTTP path itself is exercised by the public-API contract suite; this
module pins the helper layer that converts engine exceptions into HTTP
errors and the RawObservation → engine Observation adapter used by
``/api/v1/signals/situation``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.api.routes.classify import (
    ClassifyRequest,
    ClassifyResponse,
    _engine_failure_to_http,
)
from app.api.routes.signals import _raw_to_observation
from app.domain.raw_models import RawObservation
from app.intelligence.citation_verifier import CitationGroundingError
from app.intelligence.situation_engine import GenerationError, OutputParseError
from app.core.models import MediaType, SourcePlatform


def _raw(title: str = "Headline", body: str = "Body text") -> RawObservation:
    return RawObservation(
        user_id=uuid4(),
        source_platform=SourcePlatform.REDDIT,
        source_id=f"t3_{uuid4().hex[:8]}",
        source_url="https://reddit.com/r/x/comments/abc/post",
        author="someone",
        title=title,
        raw_text=body,
        media_type=MediaType.TEXT,
        published_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# ClassifyRequest schema
# ---------------------------------------------------------------------------


def test_classify_request_rejects_empty_observations():
    with pytest.raises(Exception):
        ClassifyRequest(observations=[])


def test_classify_request_rejects_out_of_range_min_confidence():
    with pytest.raises(Exception):
        ClassifyRequest(
            observations=[{
                "observation_id": "o1",
                "text": "hi",
                "source": "unit",
                "timestamp": None,
            }],
            min_confidence=1.5,
        )


# ---------------------------------------------------------------------------
# _engine_failure_to_http
# ---------------------------------------------------------------------------


def test_citation_failure_maps_to_422():
    exc = _engine_failure_to_http(CitationGroundingError([]))
    assert exc.status_code == 422
    assert exc.detail["error"] == "citation_grounding_failed"


def test_parse_failure_maps_to_422():
    exc = _engine_failure_to_http(OutputParseError("bad json"))
    assert exc.status_code == 422
    assert exc.detail["error"] == "output_parse_failed"


def test_generation_failure_maps_to_502():
    exc = _engine_failure_to_http(GenerationError("provider down"))
    assert exc.status_code == 502
    assert exc.detail["error"] == "generator_failed"


def test_unknown_failure_maps_to_500():
    exc = _engine_failure_to_http(RuntimeError("???"))
    assert exc.status_code == 500
    assert exc.detail["error"] == "engine_error"


# ---------------------------------------------------------------------------
# _raw_to_observation
# ---------------------------------------------------------------------------


def test_raw_to_observation_joins_title_and_body():
    raw = _raw(title="Outage", body="Region us-east-1 down")
    obs = _raw_to_observation(raw)
    assert obs.observation_id == str(raw.id)
    assert "Outage" in obs.text
    assert "us-east-1 down" in obs.text
    assert obs.source == "reddit"
    assert obs.timestamp is not None


def test_raw_to_observation_handles_missing_body():
    raw = _raw(title="Title only", body="")
    obs = _raw_to_observation(raw)
    assert obs.text == "Title only"


def test_raw_to_observation_handles_missing_title():
    raw = _raw(title="", body="Body only")
    obs = _raw_to_observation(raw)
    assert obs.text == "Body only"


def test_raw_to_observation_falls_back_to_placeholder():
    raw = _raw(title="", body="")
    obs = _raw_to_observation(raw)
    assert obs.text == "(no text)"


# ---------------------------------------------------------------------------
# ClassifyResponse
# ---------------------------------------------------------------------------


def test_classify_response_round_trips_minimal_payload():
    from app.intelligence.situation_report import (
        Citation,
        Claim,
        Severity,
        SituationReport,
    )

    report = SituationReport(
        signal_type="opportunity",
        severity=Severity.SEV_3,
        calibrated_confidence=0.8,
        summary="ok",
        citations=[Citation(post_id="o1", char_start=0, char_end=3)],
        claims=[Claim(text="x", citation_ids=[0], confidence=0.8)],
    )
    payload = ClassifyResponse(
        report=report, used_fallback=False, redaction_count=0
    )
    assert payload.report.signal_type == "opportunity"
    assert payload.used_fallback is False
