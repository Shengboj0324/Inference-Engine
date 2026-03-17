"""Unit tests for signal classifier.

Tests the core signal classification logic including:
- Pattern matching
- LLM classification
- Signal creation
- Scoring integration
"""

import pytest
from datetime import datetime
from uuid import uuid4

from app.core.models import ContentItem, SourcePlatform, MediaType
from app.core.signal_models import SignalType, ActionType, ResponseTone
from app.intelligence.signal_classifier import SignalClassifier


@pytest.fixture
def classifier():
    """Create signal classifier without LLM for testing."""
    return SignalClassifier(use_llm=False, min_confidence=0.7)


@pytest.fixture
def lead_opportunity_item():
    """Create content item representing lead opportunity."""
    user_id = uuid4()
    return ContentItem(
        id=uuid4(),
        user_id=user_id,
        source_platform=SourcePlatform.REDDIT,
        source_id="reddit_123",
        source_url="https://reddit.com/r/saas/comments/123",
        title="Looking for alternatives to Slack",
        raw_text="We're a team of 20 and looking for alternatives to Slack. "
                 "Need something with better pricing and integrations. "
                 "Any recommendations?",
        author="tech_startup_ceo",
        media_type=MediaType.TEXT,
        published_at=datetime.utcnow(),
    )


@pytest.fixture
def competitor_weakness_item():
    """Create content item representing competitor weakness."""
    user_id = uuid4()
    return ContentItem(
        id=uuid4(),
        user_id=user_id,
        source_platform=SourcePlatform.REDDIT,
        source_id="reddit_456",
        source_url="https://reddit.com/r/saas/comments/456",
        title="Terrible customer support from Zendesk",
        raw_text="Been waiting 3 days for a response from Zendesk support. "
                 "This is ridiculous for a paid product. "
                 "Their support is terrible and pricing is too high.",
        author="frustrated_user",
        media_type=MediaType.TEXT,
        published_at=datetime.utcnow(),
    )


@pytest.fixture
def product_confusion_item():
    """Create content item representing product confusion."""
    user_id = uuid4()
    return ContentItem(
        id=uuid4(),
        user_id=user_id,
        source_platform=SourcePlatform.REDDIT,
        source_id="reddit_789",
        source_url="https://reddit.com/r/programming/comments/789",
        title="How do I configure SSO in Okta?",
        raw_text="I'm confused about how to setup SSO with Okta. "
                 "The documentation is unclear. Can anyone help?",
        author="developer123",
        media_type=MediaType.TEXT,
        published_at=datetime.utcnow(),
    )


class TestPatternMatching:
    """Test pattern matching functionality."""
    
    def test_lead_opportunity_pattern_match(self, classifier, lead_opportunity_item):
        """Test that lead opportunity patterns are detected."""
        matches = classifier._pattern_match(lead_opportunity_item)
        
        assert len(matches) > 0
        assert matches[0][0] == SignalType.LEAD_OPPORTUNITY
        assert matches[0][1] > 0.5  # Confidence should be reasonable
    
    def test_competitor_weakness_pattern_match(self, classifier, competitor_weakness_item):
        """Test that competitor weakness patterns are detected."""
        matches = classifier._pattern_match(competitor_weakness_item)
        
        assert len(matches) > 0
        assert SignalType.COMPETITOR_WEAKNESS in [m[0] for m in matches]
    
    def test_product_confusion_pattern_match(self, classifier, product_confusion_item):
        """Test that product confusion patterns are detected."""
        matches = classifier._pattern_match(product_confusion_item)
        
        assert len(matches) > 0
        assert SignalType.PRODUCT_CONFUSION in [m[0] for m in matches]
    
    def test_no_match_for_irrelevant_content(self, classifier):
        """Test that irrelevant content doesn't match."""
        user_id = uuid4()
        item = ContentItem(
            id=uuid4(),
            user_id=user_id,
            source_platform=SourcePlatform.REDDIT,
            source_id="reddit_999",
            source_url="https://reddit.com/r/food/comments/999",
            title="Just had a great lunch",
            raw_text="The weather is nice today. Had sushi for lunch.",
            author="random_user",
            media_type=MediaType.TEXT,
            published_at=datetime.utcnow(),
        )

        matches = classifier._pattern_match(item)
        assert len(matches) == 0


class TestSignalCreation:
    """Test signal creation from classified content."""
    
    @pytest.mark.asyncio
    async def test_create_lead_opportunity_signal(self, classifier, lead_opportunity_item):
        """Test creating signal from lead opportunity."""
        user_id = uuid4()
        
        signal = await classifier.classify_content(lead_opportunity_item, user_id)
        
        assert signal is not None
        assert signal.signal_type == SignalType.LEAD_OPPORTUNITY
        assert signal.user_id == user_id
        assert signal.source_platform == "reddit"
        assert signal.recommended_action == ActionType.REPLY_PUBLIC
        assert signal.suggested_tone == ResponseTone.HELPFUL
        assert 0.0 <= signal.action_score <= 1.0
        assert 0.0 <= signal.urgency_score <= 1.0
        assert 0.0 <= signal.impact_score <= 1.0
        assert signal.expires_at is not None
    
    @pytest.mark.asyncio
    async def test_create_competitor_weakness_signal(self, classifier, competitor_weakness_item):
        """Test creating signal from competitor weakness."""
        user_id = uuid4()
        
        signal = await classifier.classify_content(competitor_weakness_item, user_id)
        
        assert signal is not None
        assert signal.signal_type == SignalType.COMPETITOR_WEAKNESS
        assert signal.recommended_action == ActionType.CREATE_CONTENT
        assert signal.suggested_tone == ResponseTone.PROFESSIONAL


class TestSignalMetadata:
    """Test signal metadata generation."""
    
    @pytest.mark.asyncio
    async def test_signal_title_generation(self, classifier, lead_opportunity_item):
        """Test that signal title is generated correctly."""
        signal = await classifier.classify_content(lead_opportunity_item, uuid4())
        
        assert signal is not None
        assert len(signal.title) > 0
        assert len(signal.title) <= 200
        assert "Lead Opportunity" in signal.title
    
    @pytest.mark.asyncio
    async def test_signal_context_generation(self, classifier, lead_opportunity_item):
        """Test that business context is generated."""
        signal = await classifier.classify_content(lead_opportunity_item, uuid4())

        assert signal is not None
        assert len(signal.context) > 0
        assert "conversion" in signal.context.lower() or "opportunity" in signal.context.lower()


# ===========================================================================
# ChainOfThoughtReasoner tests (Enhancement 1)
# ===========================================================================

import json
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from app.domain.inference_models import SignalType as InferenceSignalType
from app.domain.normalized_models import NormalizedObservation
from app.core.models import SourcePlatform as SP, MediaType as MT
from app.intelligence.candidate_retrieval import SignalCandidate
from app.intelligence.cot_reasoner import ChainOfThoughtReasoner
from app.intelligence.llm_adjudicator import LLMAdjudicationOutput


def _make_observation(**kwargs):
    defaults = dict(
        raw_observation_id=uuid4(),
        user_id=uuid4(),
        source_platform=SP.REDDIT,
        source_id="test-123",
        source_url="https://reddit.com/r/test/123",
        title="Test post",
        normalized_text="Customer is looking to switch products urgently.",
        media_type=MT.TEXT,
        published_at=datetime.utcnow(),
        fetched_at=datetime.utcnow(),
    )
    defaults.update(kwargs)
    return NormalizedObservation(**defaults)


def _make_candidate(signal_type, score=0.8):
    return SignalCandidate(
        signal_type=signal_type,
        score=score,
        reasoning="test reasoning",
        source="test",
    )


def _valid_adjudication_json(**overrides):
    base = dict(
        candidate_signal_types=["churn_risk"],
        primary_signal_type="churn_risk",
        confidence=0.85,
        evidence_spans=[{"text": "switch products", "reason": "intent signal"}],
        rationale="Clear churn risk.",
        requires_more_context=False,
        abstain=False,
        abstention_reason=None,
        risk_labels=["high_value"],
        suggested_actions=["reach_out"],
    )
    base.update(overrides)
    return base


class TestChainOfThoughtReasoner:
    """Tests for ChainOfThoughtReasoner 4-step multi-turn scaffold."""

    def _make_router_mock(self, responses):
        """Return a mock router whose generate_for_signal yields responses in order."""
        mock_router = MagicMock()
        mock_router.generate_for_signal = AsyncMock(side_effect=responses)
        return mock_router

    @pytest.mark.asyncio
    async def test_all_four_steps_called(self):
        """Router must be called exactly 4 times (one per CoT step)."""
        final_json = json.dumps(_valid_adjudication_json())
        responses = ["decompose result", "retrieve result", "synth result", final_json]
        router = self._make_router_mock(responses)
        reasoner = ChainOfThoughtReasoner(router=router)
        obs = _make_observation()
        candidates = [_make_candidate(InferenceSignalType.CHURN_RISK)]
        await reasoner.reason(obs, candidates, InferenceSignalType.CHURN_RISK)
        assert router.generate_for_signal.call_count == 4

    @pytest.mark.asyncio
    async def test_returns_valid_adjudication_output(self):
        """Final step output must validate against LLMAdjudicationOutput."""
        final_json = json.dumps(_valid_adjudication_json())
        responses = ["step1", "step2", "step3", final_json]
        router = self._make_router_mock(responses)
        reasoner = ChainOfThoughtReasoner(router=router)
        obs = _make_observation()
        candidates = [_make_candidate(InferenceSignalType.CHURN_RISK)]
        result = await reasoner.reason(obs, candidates)
        assert isinstance(result, LLMAdjudicationOutput)
        assert result.primary_signal_type == "churn_risk"
        assert result.confidence == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_malformed_verify_raises_value_error(self):
        """When the Verify step returns non-JSON, a ValueError is raised."""
        responses = ["step1", "step2", "step3", "NOT JSON AT ALL"]
        router = self._make_router_mock(responses)
        reasoner = ChainOfThoughtReasoner(router=router)
        obs = _make_observation()
        candidates = [_make_candidate(InferenceSignalType.COMPLAINT)]
        with pytest.raises(ValueError, match="valid JSON"):
            await reasoner.reason(obs, candidates)

    @pytest.mark.asyncio
    async def test_signal_type_routing_passed_to_router(self):
        """The signal_type argument is forwarded to every generate_for_signal call."""
        final_json = json.dumps(_valid_adjudication_json())
        responses = ["d", "r", "s", final_json]
        router = self._make_router_mock(responses)
        reasoner = ChainOfThoughtReasoner(router=router)
        obs = _make_observation()
        candidates = [_make_candidate(InferenceSignalType.LEGAL_RISK)]
        await reasoner.reason(obs, candidates, InferenceSignalType.LEGAL_RISK)
        for call in router.generate_for_signal.call_args_list:
            assert call.kwargs.get("signal_type") == InferenceSignalType.LEGAL_RISK


# ===========================================================================
# MultiAgentOrchestrator tests (Enhancement 3)
# ===========================================================================

from app.intelligence.orchestrator import (
    AggregatorAgent,
    MultiAgentOrchestrator,
    SubTaskAgent,
    SubTaskResult,
)


class TestAggregatorAgent:
    """Tests for AggregatorAgent weighted-vote aggregation."""

    def test_selects_highest_confidence_type(self):
        """Winner is the signal type with the highest total confidence."""
        agg = AggregatorAgent()
        results = [
            SubTaskResult(InferenceSignalType.CHURN_RISK, 0.9, "churn text", "rationale"),
            SubTaskResult(InferenceSignalType.COMPLAINT, 0.3, "complaint text", "rationale"),
        ]
        output = agg.aggregate(results)
        assert output.primary_signal_type == "churn_risk"

    def test_empty_results_produces_abstention(self):
        """Empty result list must produce an abstention."""
        agg = AggregatorAgent()
        output = agg.aggregate([])
        assert output.abstain is True

    def test_low_confidence_triggers_abstention(self):
        """Very low confidence across all sub-tasks triggers abstain."""
        agg = AggregatorAgent()
        results = [
            SubTaskResult(InferenceSignalType.PRAISE, 0.05, "", ""),
            SubTaskResult(InferenceSignalType.COMPLAINT, 0.05, "", ""),
        ]
        output = agg.aggregate(results)
        assert output.abstain is True

    def test_candidate_types_ordered_by_confidence(self):
        """candidate_signal_types list is ordered highest confidence first."""
        agg = AggregatorAgent()
        results = [
            SubTaskResult(InferenceSignalType.COMPLAINT, 0.4, "c", "r"),
            SubTaskResult(InferenceSignalType.CHURN_RISK, 0.8, "c", "r"),
        ]
        output = agg.aggregate(results)
        assert output.candidate_signal_types[0] == "churn_risk"


class TestMultiAgentOrchestrator:
    """Tests for MultiAgentOrchestrator concurrent dispatch and PII scrubbing."""

    def _mock_router(self, response_json: dict):
        mock = MagicMock()
        mock.generate_for_signal = AsyncMock(return_value=json.dumps(
            {"applies": True, "confidence": 0.7, "evidence": "text", "rationale": "ok"}
        ))
        return mock

    @pytest.mark.asyncio
    async def test_orchestrator_dispatches_one_task_per_candidate(self):
        """One SubTaskAgent call is made per candidate."""
        router = self._mock_router({})
        orch = MultiAgentOrchestrator(router=router)
        obs = _make_observation(normalized_text="x" * 1600)
        candidates = [
            _make_candidate(InferenceSignalType.CHURN_RISK),
            _make_candidate(InferenceSignalType.COMPLAINT),
            _make_candidate(InferenceSignalType.LEGAL_RISK),
        ]
        result = await orch.orchestrate(obs, candidates)
        assert router.generate_for_signal.call_count == 3

    @pytest.mark.asyncio
    async def test_orchestrator_returns_adjudication_output(self):
        """orchestrate() always returns an LLMAdjudicationOutput."""
        router = self._mock_router({})
        orch = MultiAgentOrchestrator(router=router)
        obs = _make_observation()
        candidates = [_make_candidate(InferenceSignalType.CHURN_RISK)]
        result = await orch.orchestrate(obs, candidates)
        assert isinstance(result, LLMAdjudicationOutput)

    @pytest.mark.asyncio
    async def test_orchestrator_scrubs_pii_from_text(self):
        """PII in normalized_text must be scrubbed before sub-task receives it."""
        captured_prompts = []

        async def capture_call(**kwargs):
            for msg in kwargs.get("messages", []):
                captured_prompts.append(msg.content)
            return json.dumps(
                {"applies": False, "confidence": 0.0, "evidence": "", "rationale": ""}
            )

        router = MagicMock()
        router.generate_for_signal = AsyncMock(side_effect=capture_call)
        orch = MultiAgentOrchestrator(router=router)
        pii_text = "Contact john.doe@example.com or call 555-867-5309 for info."
        obs = _make_observation(normalized_text=pii_text)
        candidates = [_make_candidate(InferenceSignalType.COMPLAINT)]
        await orch.orchestrate(obs, candidates)
        for prompt in captured_prompts:
            assert "john.doe@example.com" not in prompt
            assert "555-867-5309" not in prompt


# ===========================================================================
# ContextMemoryStore tests (Enhancement 5)
# ===========================================================================

from app.domain.inference_models import SignalInference, SignalPrediction
from app.intelligence.context_memory import ContextMemoryStore, MemoryRecord, _bow_embed


def _make_inference(signal_type: InferenceSignalType, confidence: float = 0.8) -> SignalInference:
    pred = SignalPrediction(signal_type=signal_type, probability=confidence)
    return SignalInference(
        normalized_observation_id=uuid4(),
        user_id=uuid4(),
        predictions=[pred],
        top_prediction=pred,
        abstained=False,
        model_name="test",
        model_version="2026-03",
        inference_method="test",
    )


class TestContextMemoryStore:
    """Tests for ContextMemoryStore store/retrieve lifecycle."""

    @pytest.mark.asyncio
    async def test_empty_store_returns_empty_list(self):
        """retrieve() on an empty store returns []."""
        store = ContextMemoryStore(embed_fn=_bow_embed)
        result = await store.retrieve(uuid4(), "churn risk", top_k=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_store_then_retrieve_returns_record(self):
        """A stored inference is returned by retrieve() for the same user."""
        store = ContextMemoryStore(embed_fn=_bow_embed)
        user_id = uuid4()
        obs = _make_observation(user_id=user_id, normalized_text="customer wants to cancel")
        inf = _make_inference(InferenceSignalType.CHURN_RISK)
        await store.store(user_id, obs, inf)
        results = await store.retrieve(user_id, "cancel subscription", top_k=3)
        assert len(results) == 1
        assert results[0].signal_type == InferenceSignalType.CHURN_RISK

    @pytest.mark.asyncio
    async def test_user_isolation(self):
        """Records for user A are not returned when querying for user B."""
        store = ContextMemoryStore(embed_fn=_bow_embed)
        user_a, user_b = uuid4(), uuid4()
        obs_a = _make_observation(user_id=user_a, normalized_text="bug in the software")
        inf_a = _make_inference(InferenceSignalType.BUG_REPORT)
        await store.store(user_a, obs_a, inf_a)
        results_b = await store.retrieve(user_b, "bug in the software", top_k=5)
        assert results_b == []

    @pytest.mark.asyncio
    async def test_scores_are_between_0_and_1(self):
        """Cosine similarity scores are in [0, 1]."""
        store = ContextMemoryStore(embed_fn=_bow_embed)
        user_id = uuid4()
        obs = _make_observation(user_id=user_id, normalized_text="praise for the product")
        inf = _make_inference(InferenceSignalType.PRAISE)
        await store.store(user_id, obs, inf)
        results = await store.retrieve(user_id, "praise", top_k=1)
        assert 0.0 <= results[0].score <= 1.0

    @pytest.mark.asyncio
    async def test_top_k_limits_results(self):
        """retrieve(top_k=2) returns at most 2 records."""
        store = ContextMemoryStore(embed_fn=_bow_embed)
        user_id = uuid4()
        for text, st in [
            ("churn signal one", InferenceSignalType.CHURN_RISK),
            ("churn signal two", InferenceSignalType.CHURN_RISK),
            ("churn signal three", InferenceSignalType.CHURN_RISK),
        ]:
            obs = _make_observation(user_id=user_id, normalized_text=text)
            inf = _make_inference(st)
            await store.store(user_id, obs, inf)
        results = await store.retrieve(user_id, "churn", top_k=2)
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_abstained_inference_not_stored(self):
        """Abstained inferences (no top_prediction) are silently ignored."""
        store = ContextMemoryStore(embed_fn=_bow_embed)
        user_id = uuid4()
        obs = _make_observation(user_id=user_id, normalized_text="unclear content")
        abstained_inf = SignalInference(
            normalized_observation_id=uuid4(),
            user_id=user_id,
            predictions=[],
            top_prediction=None,
            abstained=True,
            abstention_reason=None,
            model_name="test",
            model_version="2026-03",
            inference_method="test",
        )
        await store.store(user_id, obs, abstained_inf)
        results = await store.retrieve(user_id, "unclear", top_k=5)
        assert results == []


# ===========================================================================
# DeliberationEngine tests (Enhancement 6)
# ===========================================================================

from app.intelligence.deliberation import DeliberationEngine, _FRONTIER_SIGNAL_TYPES


def _mock_memory_store(memories=None):
    """Build a ContextMemoryStore mock that returns ``memories`` from retrieve()."""
    store = MagicMock(spec=ContextMemoryStore)
    store.retrieve = AsyncMock(return_value=memories or [])
    return store


class TestDeliberationEngine:
    """Tests for the four deliberation steps (A / B / C / D)."""

    @pytest.mark.asyncio
    async def test_step_b_prunes_zero_history_low_score(self):
        """Step B removes candidates with zero history AND score < 0.4."""
        memory_store = _mock_memory_store(memories=[])
        engine = DeliberationEngine(context_memory=memory_store, min_retrieval_score=0.4)
        obs = _make_observation(normalized_text="short text")
        candidates = [
            _make_candidate(InferenceSignalType.CHURN_RISK, score=0.9),   # kept (high score)
            _make_candidate(InferenceSignalType.LEGAL_RISK, score=0.2),   # pruned (zero hist, low)
        ]
        report = await engine.deliberate(obs, candidates)
        types = [c.signal_type for c in report.pruned_candidates]
        assert InferenceSignalType.CHURN_RISK in types
        assert InferenceSignalType.LEGAL_RISK not in types

    @pytest.mark.asyncio
    async def test_step_b_keeps_high_score_zero_history(self):
        """Step B keeps a candidate with no history if score >= 0.4."""
        memory_store = _mock_memory_store(memories=[])
        engine = DeliberationEngine(context_memory=memory_store, min_retrieval_score=0.4)
        obs = _make_observation()
        candidates = [_make_candidate(InferenceSignalType.PRAISE, score=0.5)]
        report = await engine.deliberate(obs, candidates)
        assert len(report.pruned_candidates) == 1

    @pytest.mark.asyncio
    async def test_step_b_never_returns_empty(self):
        """If all candidates would be pruned, the full list is preserved."""
        memory_store = _mock_memory_store(memories=[])
        engine = DeliberationEngine(context_memory=memory_store, min_retrieval_score=0.9)
        obs = _make_observation()
        candidates = [_make_candidate(InferenceSignalType.COMPLAINT, score=0.1)]
        report = await engine.deliberate(obs, candidates)
        assert len(report.pruned_candidates) == 1

    @pytest.mark.asyncio
    async def test_step_c_escalates_frontier_signal(self):
        """Step C sets escalate=True when a frontier type has score > 0.5."""
        memory_store = _mock_memory_store(memories=[])
        engine = DeliberationEngine(context_memory=memory_store)
        obs = _make_observation()
        candidates = [_make_candidate(InferenceSignalType.CHURN_RISK, score=0.8)]
        report = await engine.deliberate(obs, candidates)
        assert report.escalate is True

    @pytest.mark.asyncio
    async def test_step_c_no_escalation_below_threshold(self):
        """Step C does NOT escalate when frontier score <= 0.5."""
        memory_store = _mock_memory_store(memories=[])
        engine = DeliberationEngine(context_memory=memory_store)
        obs = _make_observation()
        candidates = [_make_candidate(InferenceSignalType.CHURN_RISK, score=0.4)]
        report = await engine.deliberate(obs, candidates)
        assert report.escalate is False

    @pytest.mark.asyncio
    async def test_step_d_single_call_default(self):
        """Short text + 1 candidate + no special conditions → single_call."""
        memory_store = _mock_memory_store(memories=[])
        engine = DeliberationEngine(context_memory=memory_store)
        obs = _make_observation(normalized_text="short text", confidence_required=0.0)
        candidates = [_make_candidate(InferenceSignalType.PRAISE, score=0.8)]
        report = await engine.deliberate(obs, candidates)
        assert report.reasoning_mode == "single_call"

    @pytest.mark.asyncio
    async def test_step_d_multi_agent_long_text(self):
        """Text length > 1500 chars selects multi_agent."""
        memory_store = _mock_memory_store(memories=[])
        engine = DeliberationEngine(context_memory=memory_store)
        obs = _make_observation(normalized_text="x" * 1501)
        candidates = [_make_candidate(InferenceSignalType.COMPLAINT, score=0.8)]
        report = await engine.deliberate(obs, candidates)
        assert report.reasoning_mode == "multi_agent"

    @pytest.mark.asyncio
    async def test_step_d_chain_of_thought_close_scores(self):
        """Two candidates with scores within 0.1 of each other → chain_of_thought."""
        memory_store = _mock_memory_store(memories=[])
        engine = DeliberationEngine(context_memory=memory_store)
        obs = _make_observation(normalized_text="medium text", confidence_required=0.0)
        candidates = [
            _make_candidate(InferenceSignalType.COMPLAINT, score=0.80),
            _make_candidate(InferenceSignalType.CHURN_RISK, score=0.75),
        ]
        report = await engine.deliberate(obs, candidates)
        assert report.reasoning_mode == "chain_of_thought"

    @pytest.mark.asyncio
    async def test_step_d_chain_of_thought_high_confidence_required(self):
        """confidence_required > 0.85 triggers chain_of_thought."""
        memory_store = _mock_memory_store(memories=[])
        engine = DeliberationEngine(context_memory=memory_store)
        obs = _make_observation(normalized_text="normal length text", confidence_required=0.9)
        candidates = [_make_candidate(InferenceSignalType.PRAISE, score=0.95)]
        report = await engine.deliberate(obs, candidates)
        assert report.reasoning_mode == "chain_of_thought"

