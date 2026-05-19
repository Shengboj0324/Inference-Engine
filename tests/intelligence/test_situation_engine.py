"""Tests for the Phase 3 SituationEngine runtime orchestrator."""

from __future__ import annotations

import json
from typing import Dict, List

import pytest

from app.evals.scenario_loader import Observation
from app.intelligence.citation_verifier import CitationGroundingError
from app.intelligence.situation_engine import (
    AsyncSituationEngine,
    GenerationError,
    OutputParseError,
    SituationEngine,
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


def _report_for(observations: List[Observation]) -> SituationReport:
    first = observations[0]
    return SituationReport(
        signal_type="security_concern",
        severity=Severity.SEV_3,
        calibrated_confidence=0.7,
        summary="A grounded summary.",
        citations=[
            Citation(
                post_id=first.observation_id,
                char_start=0,
                char_end=min(8, len(first.text)),
            )
        ],
        claims=[Claim(text="A claim.", citation_ids=[0], confidence=0.7)],
    )


def _abstain_report() -> SituationReport:
    return SituationReport(
        signal_type="insufficient_evidence",
        severity=Severity.SEV_5,
        calibrated_confidence=0.4,
        summary="No basis for a finding.",
        abstain=True,
        abstention_reason="too sparse",
    )


def _make_generator(payload: SituationReport):
    captured: Dict[str, object] = {}

    def _generate(messages: List[Dict[str, str]]) -> str:
        captured["messages"] = messages
        return json.dumps(payload.model_dump(mode="json"))

    return _generate, captured


def test_analyze_returns_engine_result_on_happy_path():
    observations = [_obs("obs_1", "hello world example")]
    generate, _ = _make_generator(_report_for(observations))
    engine = SituationEngine(generate=generate)
    result = engine.analyze(observations)
    assert result.report.signal_type == "security_concern"
    assert result.redaction_count == 0


def test_pii_scrubbed_before_generator_sees_text():
    observations = [_obs("obs_1", "contact me at user@example.com tomorrow")]
    generate, captured = _make_generator(_report_for(observations))
    engine = SituationEngine(generate=generate)
    result = engine.analyze(observations)
    user_msg = captured["messages"][1]["content"]
    assert "user@example.com" not in user_msg
    assert result.redaction_count >= 1


def test_generator_exception_wrapped():
    def _boom(_messages):
        raise RuntimeError("provider down")

    engine = SituationEngine(generate=_boom)
    with pytest.raises(GenerationError) as exc_info:
        engine.analyze([_obs("obs_1", "anything at all")])
    assert "provider down" in str(exc_info.value)


def test_malformed_output_raises_output_parse_error():
    def _gen(_messages):
        return "not json"

    engine = SituationEngine(generate=_gen)
    with pytest.raises(OutputParseError):
        engine.analyze([_obs("obs_1", "anything at all")])


def test_citation_pointing_to_missing_post_id_raises():
    observations = [_obs("obs_1", "hello world example")]
    bad_report = _report_for(observations).model_copy(
        update={"citations": [Citation(post_id="ghost", char_start=0, char_end=4)]}
    )
    generate, _ = _make_generator(bad_report)
    engine = SituationEngine(generate=generate)
    with pytest.raises(CitationGroundingError):
        engine.analyze(observations)


def test_abstain_report_skips_citation_check():
    observations = [_obs("obs_1", "hello world example")]
    generate, _ = _make_generator(_abstain_report())
    engine = SituationEngine(generate=generate)
    result = engine.analyze(observations)
    assert result.report.abstain is True


def test_empty_observations_rejected():
    engine = SituationEngine(generate=lambda _m: "{}")
    with pytest.raises(ValueError):
        engine.analyze([])


@pytest.mark.asyncio
async def test_async_engine_awaits_generator():
    observations = [_obs("obs_1", "hello world example")]
    payload = _report_for(observations)

    async def _gen(_messages):
        return json.dumps(payload.model_dump(mode="json"))

    engine = AsyncSituationEngine(generate=_gen)
    result = await engine.analyze(observations)
    assert result.report.signal_type == "security_concern"


@pytest.mark.asyncio
async def test_router_generator_translates_messages_and_returns_content():
    from app.intelligence.situation_engine import build_router_generator

    captured: Dict[str, object] = {}

    class _StubResponse:
        def __init__(self, content: str) -> None:
            self.content = content

    class _StubRouter:
        async def generate(self, *, messages, strategy, temperature, max_tokens, **kw):
            captured["messages"] = messages
            captured["temperature"] = temperature
            captured["max_tokens"] = max_tokens
            captured["extra"] = kw
            return _StubResponse(content="{\"x\": 1}")

    generate = build_router_generator(_StubRouter(), temperature=0.0, max_tokens=512)
    raw = await generate([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ])
    assert raw == "{\"x\": 1}"
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 512
    roles = [m.role for m in captured["messages"]]
    assert roles == ["system", "user"]
    contents = [m.content for m in captured["messages"]]
    assert contents == ["sys", "usr"]


# ---------------------------------------------------------------------------
# Optional re-routing (fallback)
# ---------------------------------------------------------------------------


def _make_low_conf_report(observations: List[Observation]) -> SituationReport:
    base = _report_for(observations)
    return base.model_copy(update={"calibrated_confidence": 0.20})


def _make_high_conf_report(observations: List[Observation]) -> SituationReport:
    base = _report_for(observations)
    return base.model_copy(update={"calibrated_confidence": 0.95})


def test_min_confidence_outside_range_rejected():
    with pytest.raises(ValueError):
        SituationEngine(generate=lambda _m: "{}", min_confidence=1.5)


def test_fallback_invoked_when_primary_confidence_below_threshold():
    observations = [_obs("obs_1", "hello world example")]
    primary, _ = _make_generator(_make_low_conf_report(observations))
    fallback, fb_capture = _make_generator(_make_high_conf_report(observations))
    engine = SituationEngine(
        generate=primary, fallback_generate=fallback, min_confidence=0.5
    )
    result = engine.analyze(observations)
    assert result.used_fallback is True
    assert result.report.calibrated_confidence == pytest.approx(0.95)
    assert result.primary_report is not None
    assert result.primary_report.calibrated_confidence == pytest.approx(0.20)
    assert "messages" in fb_capture


def test_fallback_skipped_when_primary_confidence_meets_threshold():
    observations = [_obs("obs_1", "hello world example")]
    primary, _ = _make_generator(_make_high_conf_report(observations))

    def _fallback(_messages):
        raise AssertionError("fallback must not run when primary passes")

    engine = SituationEngine(
        generate=primary, fallback_generate=_fallback, min_confidence=0.5
    )
    result = engine.analyze(observations)
    assert result.used_fallback is False
    assert result.primary_report is None


def test_fallback_invoked_when_primary_generator_raises():
    observations = [_obs("obs_1", "hello world example")]

    def _primary(_messages):
        raise RuntimeError("provider down")

    fallback, _ = _make_generator(_make_high_conf_report(observations))
    engine = SituationEngine(
        generate=_primary, fallback_generate=fallback, min_confidence=0.0
    )
    result = engine.analyze(observations)
    assert result.used_fallback is True
    assert result.primary_report is None


def test_fallback_failure_keeps_primary_if_primary_succeeded():
    observations = [_obs("obs_1", "hello world example")]
    primary, _ = _make_generator(_make_low_conf_report(observations))

    def _fallback(_messages):
        raise RuntimeError("fallback offline")

    engine = SituationEngine(
        generate=primary, fallback_generate=_fallback, min_confidence=0.9
    )
    result = engine.analyze(observations)
    assert result.used_fallback is False
    assert result.report.calibrated_confidence == pytest.approx(0.20)


@pytest.mark.asyncio
async def test_async_engine_invokes_fallback_below_threshold():
    observations = [_obs("obs_1", "hello world example")]
    low = _make_low_conf_report(observations)
    high = _make_high_conf_report(observations)

    async def _primary(_messages):
        return json.dumps(low.model_dump(mode="json"))

    async def _fallback(_messages):
        return json.dumps(high.model_dump(mode="json"))

    engine = AsyncSituationEngine(
        generate=_primary, fallback_generate=_fallback, min_confidence=0.5
    )
    result = await engine.analyze(observations)
    assert result.used_fallback is True
    assert result.report.calibrated_confidence == pytest.approx(0.95)
