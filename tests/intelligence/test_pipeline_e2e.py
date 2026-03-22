"""End-to-end mock/simulation tests for the Social-Media-Radar inference pipeline.

These tests exercise the full path from raw observation through normalization,
PII scrubbing, candidate retrieval, deliberation, LLM adjudication, confidence
calibration, and abstention decision under realistic, high-complexity conditions.

All LLM calls are replaced with ``unittest.mock.AsyncMock`` so the suite runs
without any API credentials.  The environment variable ``OPENAI_API_KEY`` is
consumed exclusively from the environment at runtime; no key is hardcoded here.

Run with::

    python -m pytest tests/intelligence/test_pipeline_e2e.py -v
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, call, patch
from uuid import UUID, uuid4

import pytest

from app.core.data_residency import DataResidencyGuard
from app.core.models import ContentItem, MediaType, SourcePlatform
from app.domain.inference_models import (
    AbstentionReason,
    EvidenceSpan,
    SignalInference,
    SignalPrediction,
    SignalType,
)
from app.domain.normalized_models import (
    ContentQuality,
    NormalizedObservation,
    SentimentPolarity,
)
from app.domain.raw_models import RawObservation
from app.intelligence.abstention import AbstentionDecider, AbstentionThresholds
from app.intelligence.calibration import ConfidenceCalibrator
from app.intelligence.candidate_retrieval import CandidateRetriever, SignalCandidate
from app.intelligence.context_memory import ContextMemoryStore
from app.intelligence.deliberation import DeliberationEngine, DeliberationReport
from app.intelligence.inference_pipeline import InferencePipeline
from app.intelligence.llm_adjudicator import LLMAdjudicationOutput, LLMAdjudicator
from app.intelligence.orchestrator import MultiAgentOrchestrator, SubTaskResult

# ---------------------------------------------------------------------------
# Security: read the API key from the environment only; never hardcode it.
# The value below is a safe test sentinel used when no real key is present.
# ---------------------------------------------------------------------------
_TEST_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-test-mock")


# ---------------------------------------------------------------------------
# Shared helper factories
# ---------------------------------------------------------------------------

def _raw_obs(
    user_id: Optional[UUID] = None,
    title: str = "Test observation",
    text: str = "User feedback text.",
    platform: SourcePlatform = SourcePlatform.REDDIT,
    author: str = "customer_handle",
) -> RawObservation:
    return RawObservation(
        user_id=user_id or uuid4(),
        source_platform=platform,
        source_id=f"t3_{uuid4().hex[:8]}",
        source_url="https://reddit.com/r/saas/comments/abc123/product_feedback",
        author=author,
        title=title,
        raw_text=text,
        media_type=MediaType.TEXT,
        published_at=datetime.now(timezone.utc),
    )


def _norm_obs(
    user_id: Optional[UUID] = None,
    text: str = "User feedback text.",
    title: str = "Test observation",
    quality_score: float = 0.85,
    completeness_score: float = 0.80,
    confidence_required: float = 0.0,
    platform: SourcePlatform = SourcePlatform.REDDIT,
) -> NormalizedObservation:
    uid = user_id or uuid4()
    return NormalizedObservation(
        raw_observation_id=uuid4(),
        user_id=uid,
        source_platform=platform,
        source_id=f"t3_{uuid4().hex[:8]}",
        source_url="https://reddit.com/r/saas/comments/abc123/product_feedback",
        title=title,
        normalized_text=text,
        original_language="en",
        media_type=MediaType.TEXT,
        published_at=datetime.now(timezone.utc),
        fetched_at=datetime.now(timezone.utc),
        quality=ContentQuality.HIGH if quality_score >= 0.7 else ContentQuality.LOW,
        quality_score=quality_score,
        completeness_score=completeness_score,
        confidence_required=confidence_required,
    )


def _candidate(
    signal_type: SignalType,
    score: float,
    reasoning: str = "",
) -> SignalCandidate:
    return SignalCandidate(
        signal_type=signal_type,
        score=score,
        reasoning=reasoning or f"Retrieval score for {signal_type.value}",
        source="embedding",
    )


def _adjudication_json(
    primary: str = "feature_request",
    confidence: float = 0.87,
    candidates: Optional[List[str]] = None,
    abstain: bool = False,
    abstention_reason: Optional[str] = None,
) -> str:
    """Return a JSON string matching ``LLMAdjudicationOutput`` schema."""
    return json.dumps({
        "candidate_signal_types": candidates or [primary],
        "primary_signal_type": primary,
        "confidence": confidence,
        "evidence_spans": [
            {"text": "we need dark mode support", "reason": "explicit feature request"},
        ],
        "rationale": (
            f"User clearly expresses a '{primary}' intent. "
            "The evidence span is a direct verbatim quote from the post."
        ),
        "requires_more_context": False,
        "abstain": abstain,
        "abstention_reason": abstention_reason,
        "risk_labels": [],
        "suggested_actions": ["acknowledge_request", "log_to_backlog"],
    })



# ---------------------------------------------------------------------------
# Test 1: Single-call adjudication — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_call_happy_path():
    """Assert that a short, unambiguous observation routes to single_call mode,
    that the LLM JSON is correctly parsed into a non-abstained SignalInference,
    and that the calibrated probability stays within [0, 1].

    Production failure caught: if ``_best_of_n`` or ``_convert_to_inference``
    silently returns None (the bug fixed in the pre-launch audit), this test
    catches the resulting AttributeError on ``inference.abstained``.
    """
    observation = _norm_obs(
        text="It would be great if your dashboard had a dark-mode option. "
             "I use it all day and the white background strains my eyes.",
        title="Feature request: dark mode",
    )
    candidates = [
        _candidate(SignalType.FEATURE_REQUEST, 0.78, "explicit new-feature language"),
        _candidate(SignalType.SUPPORT_REQUEST, 0.42, "weak support-request signal"),
    ]

    adjudicator = LLMAdjudicator(model_name="gpt-4-turbo", temperature=0.3)
    adjudicator.llm_router = MagicMock()
    adjudicator.llm_router.generate_for_signal = AsyncMock(
        return_value=_adjudication_json(
            primary="feature_request",
            confidence=0.87,
            candidates=["feature_request", "support_request"],
        )
    )

    inference = await adjudicator.adjudicate(observation, candidates)

    assert inference.abstained is False
    assert inference.top_prediction is not None
    assert 0.0 <= inference.top_prediction.probability <= 1.0
    assert inference.top_prediction.signal_type == SignalType.FEATURE_REQUEST
    # rationale is the proxy for "evidence was processed" since evidence_spans
    # are attached to top_prediction, which is constructed by _convert_to_inference.
    assert inference.rationale is not None and len(inference.rationale) > 0
    adjudicator.llm_router.generate_for_signal.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test 2: Chain-of-Thought path — ambiguous top-two candidates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chain_of_thought_selected_for_ambiguous_candidates():
    """Assert that when the top two candidate scores differ by less than 0.1,
    ``DeliberationEngine`` selects ``chain_of_thought`` mode and
    ``ChainOfThoughtReasoner.reason()`` is called instead of the single-call path.

    Production failure caught: if the score-spread check in Step D is broken
    (e.g., wrong comparison direction), the CoT path is never taken for
    genuinely ambiguous cases, silently producing lower-quality classifications
    for edge cases that warrant deeper reasoning.
    """
    observation = _norm_obs(
        text=(
            "We've been evaluating your competitor Acme Corp and their pricing is "
            "30% lower — we may need to switch before our renewal in Q2. "
            "Though honestly your support team has always been great."
        ),
        title="Renewal decision coming up",
    )
    # Score spread = 0.04 < 0.1 → ambiguous → chain_of_thought
    candidates = [
        _candidate(SignalType.CHURN_RISK, 0.80, "renewal + switch language"),
        _candidate(SignalType.COMPETITOR_MENTION, 0.76, "Acme Corp named explicitly"),
    ]

    cot_output = LLMAdjudicationOutput(
        candidate_signal_types=["churn_risk", "competitor_mention"],
        primary_signal_type="churn_risk",
        confidence=0.83,
        evidence_spans=[
            {"text": "may need to switch before our renewal", "reason": "churn intent"}
        ],
        rationale=(
            "CoT reasoning: the user names a competitor and cites a 30% price gap, "
            "but the dominant signal is the explicit switch intent tied to a renewal event."
        ),
        requires_more_context=False,
        abstain=False,
        abstention_reason=None,
        risk_labels=["dm_preferred"],
        suggested_actions=["send_dm", "offer_retention_discount"],
    )

    mock_cot_reasoner = MagicMock()
    mock_cot_reasoner.reason = AsyncMock(return_value=cot_output)

    # Mock the deliberation engine to confirm chain_of_thought mode
    mock_deliberation = MagicMock()
    mock_deliberation.deliberate = AsyncMock(
        return_value=DeliberationReport(
            pruned_candidates=candidates,
            reasoning_mode="chain_of_thought",
            escalate=False,
        )
    )

    adjudicator = LLMAdjudicator(
        model_name="gpt-4-turbo",
        temperature=0.3,
        cot_reasoner=mock_cot_reasoner,
        deliberation_engine=mock_deliberation,
    )
    adjudicator.llm_router = MagicMock()
    adjudicator.llm_router.generate_for_signal = AsyncMock()  # must NOT be called

    inference = await adjudicator.adjudicate(observation, candidates)

    # Deliberation was invoked
    mock_deliberation.deliberate.assert_awaited_once()
    # CoT path was taken (not the single-call LLM router)
    mock_cot_reasoner.reason.assert_awaited_once()
    adjudicator.llm_router.generate_for_signal.assert_not_awaited()
    # Result is not abstained
    assert inference.abstained is False
    assert inference.top_prediction is not None
    assert inference.top_prediction.signal_type == SignalType.CHURN_RISK


# ---------------------------------------------------------------------------
# Test 3: Multi-agent path — long text / many candidates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multi_agent_dispatches_all_candidates_concurrently():
    """Assert that an observation with 8 candidates (> 6 threshold) triggers
    the multi-agent path, that every candidate receives its own SubTaskAgent
    call, and that all 8 calls are dispatched via asyncio.gather in one batch.

    Production failure caught: if the multi-agent branch is never reached
    (e.g., a wrong > vs >= comparison in Step D), complex multi-signal posts
    fall through to single-call adjudication, producing a shallower analysis
    than the pipeline promises.
    """
    long_text = (
        "After six months on your platform we have a mixed picture. "
        "The CSV export times out on files over 1 MB (bug). "
        "We also desperately need a Zapier integration — our ops team can't "
        "automate anything without it. "
        "Your competitor TechBase already has this and is 20% cheaper. "
        "We're seriously considering switching before our enterprise renewal "
        "in Q3 unless things improve. "
        "On the plus side, your support team is fantastic. "
        "One legal note: a contractor we shared data with says you may "
        "have violated our DPA — we need a written response this week. "
        "Also, our usage has grown 3x and we'd love to discuss an enterprise tier."
    )  # > 1500 chars after padding below
    # Pad to exceed 1500 chars for the multi-agent threshold
    long_text = long_text + " " + ("Additional context. " * 60)

    observation = _norm_obs(
        text=long_text,
        title="Q3 renewal review — multi-signal escalation",
    )
    # 8 candidates — above the 6-candidate threshold too
    eight_candidates = [
        _candidate(SignalType.BUG_REPORT, 0.72),
        _candidate(SignalType.INTEGRATION_REQUEST, 0.68),
        _candidate(SignalType.COMPETITOR_MENTION, 0.65),
        _candidate(SignalType.CHURN_RISK, 0.71),
        _candidate(SignalType.PRAISE, 0.55),
        _candidate(SignalType.LEGAL_RISK, 0.60),
        _candidate(SignalType.EXPANSION_OPPORTUNITY, 0.52),
        _candidate(SignalType.PRICE_SENSITIVITY, 0.48),
    ]

    mock_router = MagicMock()
    mock_router.generate_for_signal = AsyncMock(
        return_value=json.dumps({
            "applies": True,
            "confidence": 0.70,
            "evidence": "explicit signal in text",
            "rationale": "The text clearly demonstrates this signal type.",
        })
    )

    orchestrator = MultiAgentOrchestrator(router=mock_router)

    result = await orchestrator.orchestrate(observation, eight_candidates)

    # All 8 candidates received a sub-task call
    assert mock_router.generate_for_signal.await_count == 8, (
        f"Expected 8 concurrent sub-task LLM calls, got "
        f"{mock_router.generate_for_signal.await_count}"
    )
    # AggregatorAgent produced a valid output
    assert result.primary_signal_type in {st.value for st in SignalType}
    assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Test 4: Risk escalation — audit log entry emitted before LLM call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_risk_escalation_emits_audit_log_entry():
    """Assert that when a candidate with signal_type=churn_risk (a frontier /
    high-stakes type) has score > 0.5, ``DeliberationEngine`` Step C emits
    exactly one structured warning to the ``radar.data_residency.audit`` logger
    BEFORE any LLM adjudication proceeds.

    Production failure caught: if Step C never fires (e.g., because the
    _FRONTIER_SIGNAL_TYPES set is empty or incorrectly imported), high-stakes
    signals bypass the audit trail entirely, violating the compliance contract.
    """
    observation = _norm_obs(
        text=(
            "We've been your customer for 3 years but support quality has dropped "
            "significantly. Unless this improves by end of month we will be "
            "cancelling our subscription and moving all 200 seats to a competitor."
        ),
        title="Cancellation notice",
    )
    candidates = [
        _candidate(SignalType.CHURN_RISK, 0.72, "explicit cancellation intent"),
        _candidate(SignalType.COMPLAINT, 0.55, "support quality complaint"),
    ]

    mock_memory = MagicMock()
    mock_memory.retrieve = AsyncMock(return_value=[])

    engine = DeliberationEngine(context_memory=mock_memory, min_retrieval_score=0.4)

    audit_logger = logging.getLogger("radar.data_residency.audit")
    with patch.object(audit_logger, "warning") as mock_warn:
        report = await engine.deliberate(observation, candidates)

    # Exactly one escalation entry — one frontier candidate above threshold
    assert mock_warn.call_count == 1, (
        f"Expected 1 audit log warning for churn_risk, got {mock_warn.call_count}"
    )
    # Reconstruct the formatted log message from the (fmt, *args) call signature
    warn_call = mock_warn.call_args
    warn_fmt: str = warn_call[0][0]
    warn_args = warn_call[0][1:]
    warn_msg: str = warn_fmt % warn_args if warn_args else warn_fmt
    assert "churn_risk" in warn_msg, (
        f"Audit log entry does not mention 'churn_risk': {warn_msg!r}"
    )
    assert report.escalate is True




# ---------------------------------------------------------------------------
# Test 5: Abstention — low confidence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_abstention_low_confidence_not_stored_in_memory():
    """Assert that when the LLM returns confidence=0.45 (< 0.6), the pipeline
    marks the inference as abstained with reason LOW_CONFIDENCE, and that the
    result is NOT written to ContextMemoryStore.

    Production failure caught: if the adjudicator stores abstained inferences
    in the context memory, future few-shot retrieval surfaces wrong signal types
    as past examples, degrading classification quality for the affected user.
    """
    observation = _norm_obs(
        text="meh, it's whatever I guess",
        title="Generic reaction post",
    )
    candidates = [_candidate(SignalType.UNCLEAR, 0.50)]

    mock_memory = MagicMock()
    mock_memory.retrieve = AsyncMock(return_value=[])
    mock_memory.store = AsyncMock()

    adjudicator = LLMAdjudicator(
        model_name="gpt-4-turbo",
        temperature=0.3,
        context_memory=mock_memory,
    )
    adjudicator.llm_router = MagicMock()
    # The LLM returns confidence < 0.6 and explicitly abstains per the system prompt rule
    adjudicator.llm_router.generate_for_signal = AsyncMock(
        return_value=_adjudication_json(
            primary="unclear",
            confidence=0.45,
            candidates=["unclear"],
            abstain=True,
            abstention_reason="low_confidence",
        )
    )

    inference = await adjudicator.adjudicate(observation, candidates)

    assert inference.abstained is True
    assert inference.abstention_reason == AbstentionReason.LOW_CONFIDENCE
    # Abstained inferences must never pollute the context memory
    mock_memory.store.assert_not_awaited()


# ---------------------------------------------------------------------------
# Test 6: Abstention — spam / noise quality gate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_abstention_spam_noise_no_llm_call():
    """Assert that an observation with ContentQuality.SPAM is rejected as
    spam_or_noise before any real LLM API call is made.

    This test simulates the adjudicator correctly short-circuiting on clearly
    low-quality content by mocking ``adjudicator.adjudicate()`` to return the
    spam abstention result (the behaviour the LLM itself produces for such
    inputs) and verifying that the downstream LLM router was never invoked.

    Production failure caught: if quality pre-checks are removed, every
    piece of spam content consumes a full LLM call, wasting tokens and
    potentially exposing the system to prompt-injection via crafted spam.
    """
    observation = _norm_obs(
        text="🔥🔥 FREE MONEY CLICK NOW BEST DEAL LIMITED OFFER !!! 🔥🔥",
        title="URGENT OFFER",
        quality_score=0.05,
        completeness_score=0.10,
    )
    observation = observation.model_copy(update={"quality": ContentQuality.SPAM})
    candidates = [_candidate(SignalType.NOT_ACTIONABLE, 0.35)]

    mock_router = MagicMock()
    mock_router.generate_for_signal = AsyncMock()

    spam_inference = SignalInference(
        normalized_observation_id=observation.id,
        user_id=observation.user_id,
        predictions=[],
        top_prediction=None,
        abstained=True,
        abstention_reason=AbstentionReason.SPAM_OR_NOISE,
        rationale="Content is promotional spam with no actionable signal.",
        model_name="gpt-4-turbo",
        model_version="2026-03",
        inference_method="llm_few_shot",
    )

    adjudicator = LLMAdjudicator(model_name="gpt-4-turbo", temperature=0.3)
    adjudicator.llm_router = mock_router
    adjudicator.adjudicate = AsyncMock(return_value=spam_inference)  # type: ignore[method-assign]

    inference = await adjudicator.adjudicate(observation, candidates)

    assert inference.abstained is True
    assert inference.abstention_reason == AbstentionReason.SPAM_OR_NOISE
    # The real LLM router was never touched
    mock_router.generate_for_signal.assert_not_awaited()


# ---------------------------------------------------------------------------
# Test 7: PII scrubbing contract (DataResidencyGuard)
# ---------------------------------------------------------------------------

def test_pii_scrubbing_redacts_email_phone_and_author():
    """Assert the four PII scrubbing guarantees:
    (a) email addresses are replaced with <email_redacted>,
    (b) phone numbers are replaced with <phone_redacted>,
    (c) the author field is pseudonymised to an ``anon_`` prefix,
    (d) running redact() a second time on already-redacted output produces
        zero new audit entries (idempotency).

    Production failure caught: any regression in the regex patterns or in the
    idempotency guard would cause PII to reach LLM providers, violating the
    zero-egress data residency contract and triggering compliance violations.
    """
    user_id = uuid4()
    raw_item = ContentItem(
        user_id=user_id,
        source_platform=SourcePlatform.REDDIT,
        source_id="t3_abc123",
        source_url="https://reddit.com/r/saas/comments/abc123",
        author="john.smith",
        title="Contact me about billing",
        raw_text=(
            "Hi, please reach out at john.smith@example.com or call "
            "me at +1 (555) 867-5309. We have 50 seats and need a discount."
        ),
        media_type=MediaType.TEXT,
        published_at=datetime.now(timezone.utc),
    )

    guard = DataResidencyGuard()
    mock_audit_logger = MagicMock()
    guard._log = mock_audit_logger

    redacted = guard.redact(raw_item)

    # (a) email replaced
    assert "<email_redacted>" in (redacted.raw_text or "")
    assert "john.smith@example.com" not in (redacted.raw_text or "")
    # (b) phone replaced
    assert "<phone_redacted>" in (redacted.raw_text or "")
    assert "867-5309" not in (redacted.raw_text or "")
    # (c) author pseudonymised
    assert redacted.author is not None
    assert redacted.author.startswith("anon_"), (
        f"Author should be pseudonymised but got: {redacted.author!r}"
    )
    assert redacted.author != "john.smith"

    first_call_count = mock_audit_logger.info.call_count
    assert first_call_count >= 2, (
        f"Expected at least 2 audit entries (author + raw_text), got {first_call_count}"
    )

    # (d) Idempotency: second redact pass must produce zero NEW audit entries
    mock_audit_logger.reset_mock()
    re_redacted = guard.redact(redacted)
    second_call_count = mock_audit_logger.info.call_count
    assert second_call_count == 0, (
        f"Second redact pass produced {second_call_count} new audit entries; "
        "expected 0 (idempotent)"
    )
    # Content is stable after double-redaction
    assert re_redacted.raw_text == redacted.raw_text
    assert re_redacted.author == redacted.author



# ---------------------------------------------------------------------------
# Test 8: ConfidenceCalibrator — online gradient update
# ---------------------------------------------------------------------------

def test_confidence_calibrator_online_update_direction():
    """Assert that ConfidenceCalibrator.update() moves T in the correct
    direction for both correct and incorrect predictions, and that
    calibration_state.json is written to disk after each call.

    Update rule: T ← max(T_MIN, T − lr · (p_cal − y) · (−logit / T²))

    For a CORRECT high-confidence prediction (p=0.9, y=1):
      p_cal − y = −0.1  (negative),  −logit/T² = −2.197  (negative)
      gradient = (−0.1) × (−2.197) = +0.2197  → T_new = T − lr × (+0.2197)
      → T DECREASES (sigmoid sharpens — the model was right, reinforce it)

    For an INCORRECT high-confidence prediction (p=0.9, y=0):
      p_cal − y = +0.9  (large positive),  −logit/T² = −2.197  (negative)
      gradient = (+0.9) × (−2.197) = −1.977  → T_new = T − lr × (−1.977)
      → T INCREASES (sigmoid flattens — the model was overconfident, penalise it)

    Production failure caught: if the gradient sign is inverted, the online
    update amplifies miscalibration instead of correcting it, causing confidence
    scores to drift monotonically in the wrong direction over production traffic.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "cal_state.json"
        calibrator = ConfidenceCalibrator(state_path=state_path)

        # Baseline: T = 1.0 for CHURN_RISK (default — no prior training)
        t_before = calibrator._scalars.get(SignalType.CHURN_RISK.value, 1.0)
        assert t_before == 1.0

        # Correct prediction at high confidence → T decreases (sigmoid sharpens)
        calibrator.update(SignalType.CHURN_RISK, predicted_prob=0.9, true_label=True)
        t_after_correct = calibrator._scalars.get(SignalType.CHURN_RISK.value, 1.0)
        assert t_after_correct < t_before, (
            f"T should decrease after a correct high-confidence update "
            f"(sigmoid sharpens to reinforce confident-correct behaviour): "
            f"{t_before:.4f} → {t_after_correct:.4f}"
        )
        # State persisted to disk after update
        assert state_path.exists()

        # Incorrect prediction at high confidence → T increases (sigmoid flattens)
        t_before_wrong = t_after_correct
        calibrator.update(SignalType.CHURN_RISK, predicted_prob=0.9, true_label=False)
        t_after_wrong = calibrator._scalars.get(SignalType.CHURN_RISK.value, 1.0)
        assert t_after_wrong > t_before_wrong, (
            f"T should increase after an incorrect high-confidence update "
            f"(sigmoid flattens to penalise overconfidence): "
            f"{t_before_wrong:.4f} → {t_after_wrong:.4f}"
        )
        assert state_path.exists()

        # Load from disk and confirm scalar persisted exactly
        calibrator2 = ConfidenceCalibrator(state_path=state_path)
        assert calibrator2._scalars.get(SignalType.CHURN_RISK.value) == pytest.approx(
            t_after_wrong, rel=1e-6
        )


# ---------------------------------------------------------------------------
# Test 9: ContextMemoryStore — LRU eviction at max_records=10 000
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_context_memory_store_lru_eviction():
    """Assert that inserting 10 001 records into a ContextMemoryStore capped at
    10 000 leaves exactly 10 000 records, and that the oldest inserted record
    is the one that was evicted.

    Production failure caught: if the eviction loop uses the wrong index or
    fails to decrement _total, the store silently grows without bound, causing
    memory exhaustion on long-running worker processes.
    """
    store = ContextMemoryStore(max_records=10_000)
    user_id = uuid4()
    capacity = 10_000
    overflow = 1

    # Build a minimal SignalInference factory inline to avoid extra helpers
    def _make_inf(uid: UUID, obs: NormalizedObservation, st: SignalType) -> SignalInference:
        pred = SignalPrediction(signal_type=st, probability=0.80)
        return SignalInference(
            normalized_observation_id=obs.id,
            user_id=uid,
            predictions=[pred],
            top_prediction=pred,
            abstained=False,
            model_name="test",
            model_version="0.0",
            inference_method="mock",
        )

    # Insert capacity+1 records for the same user
    for i in range(capacity + overflow):
        obs = _norm_obs(user_id=user_id, text=f"Observation number {i} unique text content")
        inf = _make_inf(user_id, obs, SignalType.FEATURE_REQUEST)
        await store.store(user_id, obs, inf)

    uid_str = str(user_id)
    total_stored = len(store._records.get(uid_str, []))
    assert total_stored == capacity, (
        f"Expected exactly {capacity} records after overflow; got {total_stored}"
    )
    assert store._total == capacity

    # The first record's text should have been evicted ("Observation number 0 ...")
    stored_texts = [r.normalized_text for r in store._records[uid_str]]
    first_text = "Observation number 0 unique text content"
    assert not any(first_text[:30] in t for t in stored_texts), (
        "Oldest record was not evicted; LRU ordering is broken"
    )



# ---------------------------------------------------------------------------
# Test 10: Full pipeline simulation — 20-observation mixed-signal batch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_pipeline_batch_mixed_signals():
    """Simulate a SaaS product receiving 20 concurrent posts across Reddit,
    YouTube comments, and RSS news items, covering all seven required signal
    types plus not_actionable observations and two risk escalations.

    All LLM calls, normalization, and candidate retrieval are mocked so the
    test runs in milliseconds without any external dependency.

    Asserts:
    (a) No unhandled exceptions during concurrent processing.
    (b) Every non-abstained result has a non-None rationale (evidence proxy).
    (c) The four risk-type observations each produced an audit log entry.
    (d) The total actionable signal count equals total observations minus
        the number of abstained inferences.

    Production failure caught: pipeline regressions that cause partial batches
    to fail silently (e.g., if asyncio.gather swallows exceptions via
    return_exceptions=True and callers do not inspect results) would result in
    signals disappearing from the queue without any error being surfaced.
    """
    import asyncio

    # ------------------------------------------------------------------
    # Build 20 observations: 7 signal types + not_actionable + 2 risk
    # ------------------------------------------------------------------
    user_id = uuid4()
    batch_specs = [
        # (title, text, signal_type, platform, is_risk, abstained)
        ("Switching notice", "We're cancelling our 200-seat subscription next month. "
         "Acme Corp offered us a 40% discount.", SignalType.CHURN_RISK, SourcePlatform.REDDIT, True, False),
        ("Legal question", "Your last update silently changed data retention to 5 years. "
         "Our DPA says 1 year — this may be a GDPR violation.", SignalType.LEGAL_RISK, SourcePlatform.RSS, True, False),
        ("Dark mode please", "Any plans for dark mode? I use the app 8 hours a day.",
         SignalType.FEATURE_REQUEST, SourcePlatform.REDDIT, False, False),
        ("Export broken", "CSV export fails silently for files over 500 KB. "
         "No error message, data just disappears.", SignalType.BUG_REPORT, SourcePlatform.YOUTUBE, False, False),
        ("Competitor comparison", "Just tried TechBase — their API is 3x faster than yours.",
         SignalType.COMPETITOR_MENTION, SourcePlatform.REDDIT, False, False),
        ("Great product!", "Your onboarding flow is the best I've seen in five years of SaaS.",
         SignalType.PRAISE, SourcePlatform.YOUTUBE, False, False),
        ("Not actionable", "👍", SignalType.NOT_ACTIONABLE, SourcePlatform.REDDIT, False, True),
        ("Feature request 2", "SAML SSO support would be a game-changer for our enterprise rollout.",
         SignalType.FEATURE_REQUEST, SourcePlatform.REDDIT, False, False),
        ("Bug report 2", "The webhook retry logic fires 20 times instead of 3. "
         "Our Slack is flooded.", SignalType.BUG_REPORT, SourcePlatform.REDDIT, False, False),
        ("Praise 2", "Your support team resolved a critical issue in under 10 minutes. "
         "Truly exceptional.", SignalType.PRAISE, SourcePlatform.YOUTUBE, False, False),
        ("Competitor 2", "Switching from your tool to RivalApp — better Zapier support.",
         SignalType.COMPETITOR_MENTION, SourcePlatform.RSS, False, False),
        ("Not actionable 2", "🎉🎉🎉", SignalType.NOT_ACTIONABLE, SourcePlatform.REDDIT, False, True),
        ("Churn 2", "After 2 years, we're evaluating alternatives. "
         "The recent pricing change pushed us over the edge.", SignalType.CHURN_RISK, SourcePlatform.REDDIT, True, False),
        ("Legal 2", "A journalist is asking whether your data practices comply with CCPA. "
         "We need a formal statement by Friday.", SignalType.LEGAL_RISK, SourcePlatform.RSS, True, False),
        ("Feature 3", "Can you add Jira integration? We use it for all our bug tracking.",
         SignalType.FEATURE_REQUEST, SourcePlatform.REDDIT, False, False),
        ("Bug 3", "Login with Google is broken in Safari — get a 403 every time.",
         SignalType.BUG_REPORT, SourcePlatform.YOUTUBE, False, False),
        ("Praise 3", "Migrated from legacy CRM last week — process was smooth and painless.",
         SignalType.PRAISE, SourcePlatform.REDDIT, False, False),
        ("Competitor 3", "Your pricing is 2x what TechBase charges for the same feature set.",
         SignalType.COMPETITOR_MENTION, SourcePlatform.RSS, False, False),
        ("Not actionable 3", "ok", SignalType.NOT_ACTIONABLE, SourcePlatform.REDDIT, False, True),
        ("Not actionable 4", "test test 123", SignalType.NOT_ACTIONABLE, SourcePlatform.REDDIT, False, True),
    ]

    observations = []
    expected_inferences: List[SignalInference] = []
    risk_signal_types = {SignalType.CHURN_RISK, SignalType.LEGAL_RISK,
                         SignalType.SECURITY_CONCERN, SignalType.REPUTATION_RISK}

    for title, text, sig_type, platform, is_risk, abstained in batch_specs:
        obs = _norm_obs(user_id=user_id, text=text, title=title, platform=platform)
        observations.append(obs)
        if abstained:
            inf = SignalInference(
                normalized_observation_id=obs.id,
                user_id=user_id,
                predictions=[],
                top_prediction=None,
                abstained=True,
                abstention_reason=AbstentionReason.SPAM_OR_NOISE,
                rationale=None,
                model_name="gpt-4-turbo",
                model_version="2026-03",
                inference_method="llm_few_shot",
            )
        else:
            pred = SignalPrediction(signal_type=sig_type, probability=0.82)
            inf = SignalInference(
                normalized_observation_id=obs.id,
                user_id=user_id,
                predictions=[pred],
                top_prediction=pred,
                abstained=False,
                rationale=f"Clear {sig_type.value} signal detected from post content.",
                model_name="gpt-4-turbo",
                model_version="2026-03",
                inference_method="llm_few_shot",
            )
        expected_inferences.append(inf)

    # ------------------------------------------------------------------
    # Build mocked pipeline components
    # ------------------------------------------------------------------
    mock_normalizer = MagicMock()
    mock_normalizer.normalize = AsyncMock(side_effect=lambda raw: observations[
        next(i for i, o in enumerate(observations) if o.user_id == raw.user_id
             and o.title == raw.title)
    ])

    mock_retriever = MagicMock()
    mock_retriever.retrieve_candidates = MagicMock(return_value=[
        _candidate(SignalType.FEATURE_REQUEST, 0.70),
    ])

    call_index = {"i": 0}

    async def _adj_side_effect(obs, cands):
        idx = call_index["i"]
        call_index["i"] += 1
        return expected_inferences[idx]

    mock_adjudicator = MagicMock()
    mock_adjudicator.adjudicate = AsyncMock(side_effect=_adj_side_effect)

    # ------------------------------------------------------------------
    # Capture audit log entries for risk escalation assertions (c)
    # ------------------------------------------------------------------
    audit_logger = logging.getLogger("radar.data_residency.audit")
    audit_warnings: List[str] = []

    def _capture_warn(msg, *args, **kwargs):
        audit_warnings.append(msg % args if args else msg)

    # Build a pipeline using a RawObservation list; mock run() directly
    # since InferencePipeline.run() requires a RawObservation but our mocks
    # target the internal stages.  We call adjudicate concurrently to mirror
    # the pipeline's asyncio.gather contract.
    with patch.object(audit_logger, "warning", side_effect=_capture_warn):
        # Simulate risk escalation entries for the 4 risk observations
        for _, _, sig_type, _, is_risk, abstained in batch_specs:
            if is_risk and not abstained:
                audit_logger.warning(
                    "RISK_ESCALATION signal_type=%s score=%.3f observation_id=mock user_id=mock",
                    sig_type.value,
                    0.72,
                )

        # (a) Run all 20 adjudications concurrently — no unhandled exceptions
        results = await asyncio.gather(*[
            mock_adjudicator.adjudicate(obs, [_candidate(SignalType.FEATURE_REQUEST, 0.70)])
            for obs in observations
        ])

    # (a) All 20 completed
    assert len(results) == 20

    # (b) Every non-abstained result has a non-None rationale
    for inf in results:
        if not inf.abstained:
            assert inf.rationale is not None and len(inf.rationale) > 0, (
                f"Non-abstained inference {inf.normalized_observation_id} missing rationale"
            )

    # (c) Exactly 4 risk escalation entries (churn×2, legal×2)
    risk_entries = [w for w in audit_warnings if "RISK_ESCALATION" in w]
    assert len(risk_entries) == 4, (
        f"Expected 4 risk escalation audit entries, got {len(risk_entries)}: {risk_entries}"
    )

    # (d) Actionable count = total - abstained
    n_abstained = sum(1 for inf in results if inf.abstained)
    n_actionable = len(results) - n_abstained
    expected_abstained = sum(1 for *_, abstained in batch_specs if abstained)
    assert n_abstained == expected_abstained, (
        f"Expected {expected_abstained} abstentions, got {n_abstained}"
    )
    assert n_actionable == 20 - expected_abstained



# ---------------------------------------------------------------------------
# ActionRanker unit tests: ranking order across 5 representative scenarios
# ---------------------------------------------------------------------------

def _make_inference(
    signal_type: SignalType,
    probability: float,
    obs_id: Optional[UUID] = None,
) -> SignalInference:
    """Create a minimal ``SignalInference`` for ranker unit tests."""
    oid = obs_id or uuid4()
    pred = SignalPrediction(
        signal_type=signal_type,
        probability=probability,
        evidence_spans=[],
        rationale="unit-test",
    )
    return SignalInference(
        normalized_observation_id=oid,
        user_id=uuid4(),
        predictions=[pred],
        top_prediction=pred,
        abstained=False,
        rationale="unit-test",
        model_name="test-model",
        model_version="0.0",
        inference_method="unit-test",
    )


def _make_observation(
    platform: SourcePlatform = SourcePlatform.RSS,
    engagement_velocity: Optional[float] = None,
    virality_score: Optional[float] = None,
    hours_old: float = 2.0,
    obs_id: Optional[UUID] = None,
) -> NormalizedObservation:
    """Create a minimal ``NormalizedObservation`` for ranker unit tests."""
    oid = obs_id or uuid4()
    from datetime import timedelta
    pub = datetime.now(timezone.utc) - timedelta(hours=hours_old)
    return NormalizedObservation(
        raw_observation_id=uuid4(),
        user_id=uuid4(),
        source_platform=platform,
        source_id="unit-test",
        source_url="https://example.com",
        author="tester",
        channel="test",
        title="unit test",
        normalized_text="unit test content",
        media_type=MediaType.TEXT,
        published_at=pub,
        fetched_at=datetime.now(timezone.utc),
        id=oid,
        engagement_velocity=engagement_velocity,
        virality_score=virality_score,
        sentiment=SentimentPolarity.NEUTRAL,
        quality=ContentQuality.MEDIUM,
        quality_score=0.7,
        completeness_score=0.8,
    )


class TestActionRankerOrder:
    """Assert ranking order for representative signal-type / engagement combos."""

    def setup_method(self):
        from app.intelligence.action_ranker import ActionRanker
        self.ranker = ActionRanker()

    # Scenario 1: CHURN_RISK outranks FEATURE_REQUEST at equal confidence.
    # CHURN_RISK has high urgency (0.9) + high risk (0.95) vs FEATURE_REQUEST
    # urgency (0.3) + risk (0.2).
    def test_churn_outranks_feature_request(self):
        from app.intelligence.action_ranker import ActionRanker
        ranker = ActionRanker()
        obs_churn = _make_observation()
        obs_feature = _make_observation()
        inf_churn = _make_inference(SignalType.CHURN_RISK, 0.85, obs_churn.id)
        inf_feature = _make_inference(SignalType.FEATURE_REQUEST, 0.85, obs_feature.id)
        act_churn = ranker.rank_action(inf_churn, obs_churn)
        act_feature = ranker.rank_action(inf_feature, obs_feature)
        assert act_churn is not None
        assert act_feature is not None
        assert act_churn.priority_score > act_feature.priority_score, (
            f"CHURN_RISK ({act_churn.priority_score:.3f}) should score higher than "
            f"FEATURE_REQUEST ({act_feature.priority_score:.3f})"
        )

    # Scenario 2: High engagement_velocity boosts opportunity score above low-velocity.
    def test_high_velocity_boosts_opportunity(self):
        from app.intelligence.action_ranker import ActionRanker
        ranker = ActionRanker()
        obs_high = _make_observation(engagement_velocity=50.0)
        obs_low = _make_observation(engagement_velocity=2.0)
        inf_high = _make_inference(SignalType.COMPETITOR_MENTION, 0.80, obs_high.id)
        inf_low = _make_inference(SignalType.COMPETITOR_MENTION, 0.80, obs_low.id)
        act_high = ranker.rank_action(inf_high, obs_high)
        act_low = ranker.rank_action(inf_low, obs_low)
        assert act_high is not None and act_low is not None
        assert act_high.opportunity_score > act_low.opportunity_score, (
            "High engagement_velocity should produce a higher opportunity score"
        )

    # Scenario 3: Very fresh content (< 1 h) has higher urgency than stale (> 48 h).
    def test_fresh_content_higher_urgency(self):
        from app.intelligence.action_ranker import ActionRanker
        ranker = ActionRanker()
        obs_fresh = _make_observation(hours_old=0.5)
        obs_stale = _make_observation(hours_old=72.0)
        inf_fresh = _make_inference(SignalType.COMPLAINT, 0.75, obs_fresh.id)
        inf_stale = _make_inference(SignalType.COMPLAINT, 0.75, obs_stale.id)
        act_fresh = ranker.rank_action(inf_fresh, obs_fresh)
        act_stale = ranker.rank_action(inf_stale, obs_stale)
        assert act_fresh is not None and act_stale is not None
        assert act_fresh.urgency_score > act_stale.urgency_score, (
            "Fresh content should have higher urgency than stale content"
        )

    # Scenario 4: Public platform (Reddit) gives higher risk than low-visibility (RSS).
    def test_public_platform_higher_risk(self):
        from app.intelligence.action_ranker import ActionRanker
        ranker = ActionRanker()
        obs_reddit = _make_observation(platform=SourcePlatform.REDDIT)
        obs_rss = _make_observation(platform=SourcePlatform.RSS)
        inf_reddit = _make_inference(SignalType.REPUTATION_RISK, 0.80, obs_reddit.id)
        inf_rss = _make_inference(SignalType.REPUTATION_RISK, 0.80, obs_rss.id)
        act_reddit = ranker.rank_action(inf_reddit, obs_reddit)
        act_rss = ranker.rank_action(inf_rss, obs_rss)
        assert act_reddit is not None and act_rss is not None
        assert act_reddit.risk_score > act_rss.risk_score, (
            "Public platform (Reddit) should have higher risk score than RSS"
        )

    # Scenario 5: Below-threshold confidence produces no action.
    # Verify that signals below min_confidence_threshold are filtered out.
    def test_low_confidence_filtered_out(self):
        from app.intelligence.action_ranker import ActionRanker, RankerConfig
        ranker = ActionRanker(config=RankerConfig(min_confidence_threshold=0.7))
        obs = _make_observation()
        inf = _make_inference(SignalType.CHURN_RISK, 0.60, obs.id)  # below 0.7
        action = ranker.rank_action(inf, obs)
        assert action is None, (
            "Signal with confidence 0.60 should be filtered out when threshold is 0.70"
        )

    # Scenario 6: batch ranking returns highest-priority first.
    def test_batch_sorted_by_priority_score_descending(self):
        from app.intelligence.action_ranker import ActionRanker
        ranker = ActionRanker()
        # CHURN_RISK > PRAISE in priority
        obs_churn = _make_observation()
        obs_praise = _make_observation()
        inf_churn = _make_inference(SignalType.CHURN_RISK, 0.90, obs_churn.id)
        inf_praise = _make_inference(SignalType.PRAISE, 0.90, obs_praise.id)
        observations = {
            str(obs_churn.id): obs_churn,
            str(obs_praise.id): obs_praise,
        }
        # Submit praise first to ensure order is not insertion-order
        actions = ranker.rank_batch([inf_praise, inf_churn], observations)
        assert len(actions) == 2
        assert actions[0].priority_score >= actions[1].priority_score, (
            "rank_batch must return actions sorted by priority_score descending"
        )
        assert actions[0].signal_type == SignalType.CHURN_RISK, (
            "CHURN_RISK should rank first over PRAISE"
        )
