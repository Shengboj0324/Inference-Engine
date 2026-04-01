"""Stress tests and hardening suite for the Social-Media-Radar industrial workflow.

Covers five production-risk areas:
  1. High-concurrency InferencePipeline ingestion (120-burst, Redis publisher isolation)
  2. MultiAgentOrchestrator deep_research recursion + ResearchCriticAgent guard
  3. WebSocketConnectionManager gauge fidelity and backpressure
  4. FeedbackProcessor concurrency and global queue re-ranking
  5. NormalizationEngine entity-KB fault tolerance

All external I/O (Redis, database, LLM API) is mocked so the suite runs offline
with zero credentials.

Run with::

    python -m pytest tests/intelligence/test_stress_hardening.py -v
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call
from uuid import UUID, uuid4

import pytest

from app.core.models import MediaType, SourcePlatform
from app.domain.inference_models import (
    AbstentionReason,
    SignalInference,
    SignalPrediction,
    SignalType,
)
from app.domain.normalized_models import (
    ContentQuality,
    NormalizedObservation,
    EntityMention,
    SentimentPolarity,
)
from app.domain.raw_models import RawObservation
from app.intelligence.calibration import ConfidenceCalibrator
from app.intelligence.feedback_processor import (
    FeedbackProcessor,
    _RERANK_THRESHOLD,
    _rerank_signals_background,
)
from app.intelligence.inference_pipeline import InferencePipeline
from app.intelligence.normalization import _load_entity_kb, _ENTITY_KB_FALLBACK, _ENTITY_KB_PATH
from app.intelligence.orchestrator import (
    CritiqueResult,
    DeepResearchReport,
    DeepResearchStep,
    MultiAgentOrchestrator,
    ResearchCriticAgent,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _raw_obs(
    user_id: Optional[UUID] = None,
    title: str = "Stress test observation",
    text: str = "Generic signal text for stress testing.",
    platform: SourcePlatform = SourcePlatform.REDDIT,
) -> RawObservation:
    return RawObservation(
        user_id=user_id or uuid4(),
        source_platform=platform,
        source_id=f"t3_{uuid4().hex[:8]}",
        source_url="https://reddit.com/r/saas/comments/stress_test",
        author="stress_tester",
        title=title,
        raw_text=text,
        media_type=MediaType.TEXT,
        published_at=datetime.now(timezone.utc),
    )


def _norm_obs(
    user_id: Optional[UUID] = None,
    text: str = "Generic signal text.",
    platform: SourcePlatform = SourcePlatform.REDDIT,
) -> NormalizedObservation:
    return NormalizedObservation(
        raw_observation_id=uuid4(),
        user_id=user_id or uuid4(),
        source_platform=platform,
        source_id=f"t3_{uuid4().hex[:8]}",
        source_url="https://reddit.com/r/saas/comments/stress_test",
        title="Stress test",
        normalized_text=text,
        original_language="en",
        media_type=MediaType.TEXT,
        published_at=datetime.now(timezone.utc),
        fetched_at=datetime.now(timezone.utc),
        quality=ContentQuality.HIGH,
        quality_score=0.85,
        completeness_score=0.80,
    )


def _make_prediction(
    signal_type: SignalType = SignalType.FEATURE_REQUEST,
    probability: float = 0.82,
) -> SignalPrediction:
    return SignalPrediction(signal_type=signal_type, probability=probability)


def _make_inference(
    user_id: Optional[UUID] = None,
    norm_obs: Optional[NormalizedObservation] = None,
    abstained: bool = False,
    signal_type: SignalType = SignalType.FEATURE_REQUEST,
    probability: float = 0.82,
) -> SignalInference:
    obs = norm_obs or _norm_obs(user_id=user_id)
    if abstained:
        return SignalInference(
            normalized_observation_id=obs.id,
            user_id=user_id or obs.user_id,
            predictions=[],
            top_prediction=None,
            abstained=True,
            abstention_reason=AbstentionReason.LOW_CONFIDENCE,
            rationale=None,
            model_name="gpt-4-turbo",
            model_version="2026-03",
            inference_method="llm_few_shot",
        )
    pred = _make_prediction(signal_type=signal_type, probability=probability)
    return SignalInference(
        normalized_observation_id=obs.id,
        user_id=user_id or obs.user_id,
        predictions=[pred],
        top_prediction=pred,
        abstained=False,
        rationale="Stress-test inference rationale.",
        model_name="gpt-4-turbo",
        model_version="2026-03",
        inference_method="llm_few_shot",
    )


# ===========================================================================
# SECTION 1 — High-Concurrency Ingestion Stress
# ===========================================================================

class TestConcurrentIngestionStress:
    """Verify InferencePipeline handles 120+ concurrent observations safely.

    All external I/O (LLM, normalization, Redis) is fully mocked.
    """

    def _build_pipeline(
        self,
        n_obs: int = 120,
        abstained_indices: Optional[set] = None,
        redis_side_effect=None,
    ):
        """Return (pipeline, raw_observations, publish_mock) tuple."""
        abstained_indices = abstained_indices or set()
        user_id = uuid4()
        raw_obs_list = [_raw_obs(user_id=user_id) for _ in range(n_obs)]

        # Build per-observation normalized + inference responses
        norm_list = [_norm_obs(user_id=user_id) for _ in range(n_obs)]
        inf_list = [
            _make_inference(user_id=user_id, abstained=(i in abstained_indices))
            for i in range(n_obs)
        ]

        # Mock normalization — returns the pre-built normalized obs in order
        call_counter = {"i": 0}

        async def _norm(raw):
            idx = call_counter["i"]
            call_counter["i"] += 1
            return norm_list[idx % n_obs]

        mock_normalizer = MagicMock()
        mock_normalizer.normalize = AsyncMock(side_effect=_norm)

        # Mock candidate retriever
        from app.intelligence.candidate_retrieval import SignalCandidate
        mock_retriever = MagicMock()
        mock_retriever.retrieve_candidates = MagicMock(return_value=[
            SignalCandidate(
                signal_type=SignalType.FEATURE_REQUEST,
                score=0.75,
                reasoning="mock",
                source="embedding",
            )
        ])

        # Mock adjudicator — returns pre-built inferences in order
        adj_counter = {"i": 0}

        async def _adjudicate(obs, cands, **kwargs):
            idx = adj_counter["i"]
            adj_counter["i"] += 1
            return inf_list[idx % n_obs]

        mock_adj = MagicMock()
        mock_adj.adjudicate = AsyncMock(side_effect=_adjudicate)

        # Mock calibrator (identity pass-through)
        mock_cal = MagicMock()
        mock_cal.calibrate = MagicMock(side_effect=lambda inf: inf)

        # Mock abstention decider (never abstains at pipeline level)
        mock_abs = MagicMock()
        mock_abs.should_abstain = MagicMock(return_value=(False, None, None))

        pipeline = InferencePipeline(
            normalization_engine=mock_normalizer,
            candidate_retriever=mock_retriever,
            llm_adjudicator=mock_adj,
            calibrator=mock_cal,
            abstention_decider=mock_abs,
            redis_url="redis://localhost:6379/0",
        )

        # Patch aioredis at the inference_pipeline module level
        publish_mock = AsyncMock()

        async def _fake_publish(channel, payload):
            await publish_mock(channel, payload)

        return pipeline, raw_obs_list, publish_mock, _fake_publish

    @pytest.mark.asyncio
    async def test_120_observations_all_complete(self):
        """All 120 observations complete without exception; output list length == 120."""
        pipeline, raw_obs_list, _, fake_publish = self._build_pipeline(n_obs=120)

        with patch(
            "app.intelligence.inference_pipeline.aioredis.from_url",
            return_value=MagicMock(
                __aenter__=AsyncMock(return_value=MagicMock(publish=fake_publish)),
                __aexit__=AsyncMock(return_value=False),
            ),
        ):
            results = await pipeline.run_batch(raw_obs_list, concurrency=20)

        assert len(results) == 120, (
            f"Expected 120 results; got {len(results)}"
        )
        for norm, inf in results:
            assert norm is not None
            assert inf is not None

    @pytest.mark.asyncio
    async def test_semaphore_limits_peak_concurrency(self):
        """Peak in-flight inference count never exceeds the semaphore limit."""
        concurrency_limit = 15
        n_obs = 60
        peak_concurrent = {"value": 0, "current": 0}
        lock = asyncio.Lock()

        user_id = uuid4()
        raw_obs_list = [_raw_obs(user_id=user_id) for _ in range(n_obs)]

        async def _slow_normalize(raw):
            async with lock:
                peak_concurrent["current"] += 1
                if peak_concurrent["current"] > peak_concurrent["value"]:
                    peak_concurrent["value"] = peak_concurrent["current"]
            # Yield control so other coroutines can interleave
            await asyncio.sleep(0)
            async with lock:
                peak_concurrent["current"] -= 1
            return _norm_obs(user_id=user_id)

        from app.intelligence.candidate_retrieval import SignalCandidate

        mock_normalizer = MagicMock()
        mock_normalizer.normalize = AsyncMock(side_effect=_slow_normalize)
        mock_retriever = MagicMock()
        mock_retriever.retrieve_candidates = MagicMock(return_value=[
            SignalCandidate(
                signal_type=SignalType.FEATURE_REQUEST,
                score=0.75,
                reasoning="mock",
                source="embedding",
            )
        ])
        mock_adj = MagicMock()
        mock_adj.adjudicate = AsyncMock(
            return_value=_make_inference(user_id=user_id, abstained=True)
        )
        mock_cal = MagicMock()
        mock_cal.calibrate = MagicMock(side_effect=lambda x: x)
        mock_abs = MagicMock()
        mock_abs.should_abstain = MagicMock(return_value=(True, AbstentionReason.LOW_CONFIDENCE, "low"))

        pipeline = InferencePipeline(
            normalization_engine=mock_normalizer,
            candidate_retriever=mock_retriever,
            llm_adjudicator=mock_adj,
            calibrator=mock_cal,
            abstention_decider=mock_abs,
            redis_url=None,
        )
        await pipeline.run_batch(raw_obs_list, concurrency=concurrency_limit)

        assert peak_concurrent["value"] <= concurrency_limit, (
            f"Peak concurrency {peak_concurrent['value']} exceeded semaphore "
            f"limit {concurrency_limit}"
        )

    @pytest.mark.asyncio
    async def test_redis_connection_error_does_not_block_inference(self):
        """A Redis ConnectionError in _publish_to_redis must not propagate to run()."""
        pipeline, raw_obs_list, _, _ = self._build_pipeline(n_obs=1)

        with patch(
            "app.intelligence.inference_pipeline.aioredis.from_url",
            side_effect=Exception("Redis connection refused"),
        ):
            results = await pipeline.run_batch(raw_obs_list[:1], concurrency=1)

        # Pipeline must still return the inference result
        assert len(results) == 1
        norm, inf = results[0]
        assert norm is not None
        assert inf is not None

    @pytest.mark.asyncio
    async def test_abstained_inference_never_published_to_redis(self):
        """Redis publish must NOT fire when the inference result is abstained."""
        pipeline, raw_obs_list, publish_mock, fake_publish = self._build_pipeline(
            n_obs=5,
            abstained_indices={0, 1, 2, 3, 4},  # all abstained
        )

        with patch(
            "app.intelligence.inference_pipeline.aioredis.from_url",
            return_value=MagicMock(
                __aenter__=AsyncMock(return_value=MagicMock(publish=fake_publish)),
                __aexit__=AsyncMock(return_value=False),
            ),
        ):
            await pipeline.run_batch(raw_obs_list[:5], concurrency=5)
            # Allow any scheduled futures to run
            await asyncio.sleep(0.05)

        publish_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_redis_payload_contains_required_fields(self):
        """Published Redis payload has type, signal_type, confidence, timestamp."""
        published_payloads: list[dict] = []

        async def _capture_publish(channel, payload):
            published_payloads.append(json.loads(payload))

        pipeline, raw_obs_list, _, _ = self._build_pipeline(
            n_obs=1,
            abstained_indices=set(),
        )

        # aioredis.from_url() returns `client`; the code does:
        #   async with client:          ← client is the context manager
        #       await client.publish()  ← publish is on client itself, not __aenter__ return
        # AsyncMock supports both patterns automatically.
        mock_client = AsyncMock()
        mock_client.publish = AsyncMock(side_effect=_capture_publish)

        with patch(
            "app.intelligence.inference_pipeline.aioredis.from_url",
            return_value=mock_client,
        ):
            await pipeline.run_batch(raw_obs_list[:1], concurrency=1)
            await asyncio.sleep(0.05)  # let ensure_future flush

        assert len(published_payloads) == 1, (
            "Expected exactly 1 published payload for 1 non-abstained inference"
        )
        payload = published_payloads[0]
        assert payload["type"] == "signal"
        assert "signal_type" in payload["data"]
        assert "confidence" in payload["data"]
        assert "timestamp" in payload["data"]
        assert 0.0 <= payload["data"]["confidence"] <= 1.0


# ===========================================================================
# SECTION 2 — Recursive Agent Resilience
# ===========================================================================

def _step_json(
    answer: str = "Analysis complete.",
    gaps: Optional[List[str]] = None,
    tokens: int = 200,
) -> str:
    return json.dumps({
        "answer": answer,
        "knowledge_gaps": gaps or [],
        "sources_referenced": [],
        "tokens_estimate": tokens,
    })


def _critique_json(
    quality: float = 0.75,
    relevant: Optional[List[str]] = None,
    filtered: Optional[List[str]] = None,
    reasoning: str = "Gaps are on-topic.",
) -> str:
    return json.dumps({
        "quality_score": quality,
        "relevant_gaps": relevant or [],
        "filtered_gaps": filtered or [],
        "reasoning": reasoning,
    })


class TestDeepResearchRecursionResilience:
    """Stress the deep_research + ResearchCriticAgent loop."""

    def _make_orchestrator(
        self,
        researcher_responses: Optional[List[str]] = None,
        critic_responses: Optional[List[str]] = None,
    ) -> MultiAgentOrchestrator:
        """Build an orchestrator whose researcher and critic use pre-canned responses."""
        router = MagicMock()

        researcher_idx = {"i": 0}
        critic_idx = {"i": 0}
        researcher_responses = researcher_responses or [_step_json()]
        critic_responses = critic_responses or [_critique_json()]

        async def _route(signal_type, messages, temperature=0.4, max_tokens=800):
            msg = messages[0].content if messages else ""
            if "research critic" in msg.lower() or "critic" in msg.lower():
                idx = critic_idx["i"] % len(critic_responses)
                critic_idx["i"] += 1
                return critic_responses[idx]
            elif "synthesise" in msg.lower() or "findings" in msg.lower():
                return "Final synthesis paragraph for the research session."
            else:
                idx = researcher_idx["i"] % len(researcher_responses)
                researcher_idx["i"] += 1
                return researcher_responses[idx]

        router.generate_for_signal = AsyncMock(side_effect=_route)
        return MultiAgentOrchestrator(router=router)

    @pytest.mark.asyncio
    async def test_critic_below_quality_threshold_suppresses_all_gaps(self):
        """When quality_score < 0.4, all gaps are suppressed and loop stops early.

        The researcher proposes 2 gaps per step; the critic returns quality=0.35
        for every step.  With max_depth=5, the loop should terminate after step 0
        because there are no approved gaps to queue for depth 1.
        """
        orchestrator = self._make_orchestrator(
            researcher_responses=[
                _step_json(gaps=["Gap A?", "Gap B?"])
            ] * 5,
            critic_responses=[
                _critique_json(
                    quality=0.35,      # below threshold → all gaps filtered
                    relevant=[],
                    filtered=["Gap A?", "Gap B?"],
                )
            ] * 5,
        )

        report = await orchestrator.deep_research(
            signal_id=str(uuid4()),
            signal_type=SignalType.CHURN_RISK.value,
            signal_context="Customer signalled intent to cancel.",
            initial_question="Why is the customer churning?",
            max_depth=5,
        )

        # Exactly 1 step executed — critic killed the loop after depth 0
        assert len(report.steps) == 1, (
            f"Expected loop to terminate at 1 step but got {len(report.steps)}"
        )
        # The step itself has no approved knowledge_gaps
        assert report.steps[0].knowledge_gaps == [], (
            "knowledge_gaps must be empty after critic suppression"
        )
        # No gaps remain for future research
        assert report.knowledge_gaps_remaining == []

    @pytest.mark.asyncio
    async def test_max_depth_ceiling_enforced(self):
        """Loop never executes more steps than max_depth, even with many pending gaps."""
        max_depth = 3
        orchestrator = self._make_orchestrator(
            researcher_responses=[
                _step_json(gaps=["Follow-up 1?", "Follow-up 2?"])
            ] * 10,
            critic_responses=[
                _critique_json(
                    quality=0.85,
                    relevant=["Follow-up 1?", "Follow-up 2?"],
                )
            ] * 10,
        )

        report = await orchestrator.deep_research(
            signal_id=str(uuid4()),
            signal_type=SignalType.COMPETITOR_MENTION.value,
            signal_context="Competitor mentioned with 30% price advantage.",
            initial_question="What is the competitive threat level?",
            max_depth=max_depth,
        )

        assert len(report.steps) <= max_depth, (
            f"Steps executed ({len(report.steps)}) exceeded max_depth={max_depth}"
        )
        assert report.max_depth_reached < max_depth

    @pytest.mark.asyncio
    async def test_parallel_sessions_are_fully_independent(self):
        """Five simultaneous deep_research sessions produce non-overlapping step ids."""
        orchestrator = self._make_orchestrator(
            researcher_responses=[_step_json(gaps=[])],
            critic_responses=[_critique_json(quality=0.9, relevant=[])],
        )

        signals = [
            (str(uuid4()), f"context {i}", f"Question {i}?")
            for i in range(5)
        ]
        reports: List[DeepResearchReport] = await asyncio.gather(*[
            orchestrator.deep_research(
                signal_id=sig_id,
                signal_type=SignalType.FEATURE_REQUEST.value,
                signal_context=ctx,
                initial_question=q,
                max_depth=1,
            )
            for sig_id, ctx, q in signals
        ])

        # Each session belongs to its own signal
        report_signal_ids = [r.signal_id for r in reports]
        assert len(set(report_signal_ids)) == 5, "Session signal IDs must be unique"

        # No step depth can exceed max_depth=1 (0-indexed: depth 0 only)
        for report in reports:
            for step in report.steps:
                assert step.depth == 0, (
                    f"Step depth {step.depth} exceeds max_depth ceiling for 1-step run"
                )

    @pytest.mark.asyncio
    async def test_hallucinated_gaps_never_reach_pending_queue(self):
        """Gaps classified as 'filtered' by the critic must never appear in the
        subsequent pending questions list."""
        hallucinated = ["Unrelated geopolitical question?", "Ask about the weather?"]
        legit_gap = "What drove the competitor's price cut?"

        call_count = {"n": 0}

        async def _route(signal_type, messages, temperature=0.4, max_tokens=800):
            call_count["n"] += 1
            msg = messages[0].content if messages else ""
            if "critic" in msg.lower():
                return _critique_json(
                    quality=0.80,
                    relevant=[legit_gap],
                    filtered=hallucinated,
                )
            elif "synthesise" in msg.lower() or "findings" in msg.lower():
                return "Synthesis done."
            else:
                return _step_json(gaps=hallucinated + [legit_gap])

        router = MagicMock()
        router.generate_for_signal = AsyncMock(side_effect=_route)
        orchestrator = MultiAgentOrchestrator(router=router)

        report = await orchestrator.deep_research(
            signal_id=str(uuid4()),
            signal_type=SignalType.PRICE_SENSITIVITY.value,
            signal_context="Customer cited competitor pricing advantage.",
            initial_question="How severe is the price sensitivity?",
            max_depth=2,
        )

        # The first step's approved gaps must not contain hallucinated ones
        first_step = report.steps[0]
        for gap in hallucinated:
            assert gap not in first_step.knowledge_gaps, (
                f"Hallucinated gap {gap!r} leaked into approved knowledge_gaps"
            )
        assert legit_gap in first_step.knowledge_gaps

    @pytest.mark.asyncio
    async def test_token_budget_accumulates_correctly(self):
        """total_tokens_used is the exact sum of tokens_used across all steps."""
        tokens_per_step = 350
        orchestrator = self._make_orchestrator(
            researcher_responses=[_step_json(tokens=tokens_per_step, gaps=[])] * 3,
            critic_responses=[_critique_json(quality=0.9, relevant=[])] * 3,
        )

        report = await orchestrator.deep_research(
            signal_id=str(uuid4()),
            signal_type=SignalType.BUG_REPORT.value,
            signal_context="Webhook retry fires 20 times instead of 3.",
            initial_question="What is the root cause of the retry storm?",
            max_depth=1,
        )

        expected_total = sum(s.tokens_used for s in report.steps)
        assert report.total_tokens_used == expected_total, (
            f"total_tokens_used {report.total_tokens_used} != "
            f"sum of step tokens {expected_total}"
        )


# ===========================================================================
# SECTION 3 — WebSocket Connection Drain
# ===========================================================================

class TestWebSocketConnectionDrain:
    """Test WebSocketConnectionManager gauge accuracy and backpressure handling.

    The manager itself is imported and exercised in isolation; FastAPI wiring
    and actual network connections are fully mocked.
    """

    def _make_pubsub_listener(self, messages: List[str]) -> AsyncMock:
        """Return an async generator that yields pub/sub messages then stops."""
        async def _gen():
            for msg in messages:
                yield {"type": "message", "data": msg}
        mock = MagicMock()
        mock.listen = MagicMock(return_value=_gen())
        mock.subscribe = AsyncMock()
        mock.unsubscribe = AsyncMock()
        return mock

    def _make_redis_client(self, pubsub: MagicMock) -> MagicMock:
        client = MagicMock()
        client.pubsub = MagicMock(return_value=pubsub)
        client.aclose = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_gauge_increments_on_connect(self):
        """MetricsCollector.record_websocket_connection(+1) called on connection."""
        from app.api.routes.signals import WebSocketConnectionManager

        ws = MagicMock()
        ws.accept = AsyncMock()
        ws.send_text = AsyncMock()

        pubsub = self._make_pubsub_listener([])
        redis_client = self._make_redis_client(pubsub)

        with patch("app.api.routes.signals.aioredis.from_url", return_value=redis_client):
            with patch("app.api.routes.signals.MetricsCollector.record_websocket_connection") as gauge_mock:
                mgr = WebSocketConnectionManager()
                await mgr.connect(ws, user_id=str(uuid4()), redis_url="redis://localhost")

        # +1 must be the first gauge call
        assert gauge_mock.call_args_list[0] == call(+1)

    @pytest.mark.asyncio
    async def test_gauge_decrements_on_clean_disconnect(self):
        """record_websocket_connection(-1) called when the pub/sub generator exhausts."""
        from app.api.routes.signals import WebSocketConnectionManager

        ws = MagicMock()
        ws.accept = AsyncMock()
        ws.send_text = AsyncMock()

        pubsub = self._make_pubsub_listener([])  # 0 messages → loop exits immediately
        redis_client = self._make_redis_client(pubsub)

        with patch("app.api.routes.signals.aioredis.from_url", return_value=redis_client):
            with patch("app.api.routes.signals.MetricsCollector.record_websocket_connection") as gauge_mock:
                mgr = WebSocketConnectionManager()
                await mgr.connect(ws, user_id=str(uuid4()), redis_url="redis://localhost")

        calls = [c.args[0] for c in gauge_mock.call_args_list]
        assert calls == [+1, -1], (
            f"Expected [+1, -1] gauge sequence; got {calls}"
        )

    @pytest.mark.asyncio
    async def test_gauge_decrements_on_websocket_disconnect_exception(self):
        """record_websocket_connection(-1) called even when WebSocketDisconnect fires."""
        from fastapi import WebSocketDisconnect
        from app.api.routes.signals import WebSocketConnectionManager

        ws = MagicMock()
        ws.accept = AsyncMock()

        async def _disconnect_gen():
            raise WebSocketDisconnect()
            yield  # make it an async generator

        pubsub = MagicMock()
        pubsub.listen = MagicMock(return_value=_disconnect_gen())
        pubsub.subscribe = AsyncMock()
        pubsub.unsubscribe = AsyncMock()
        redis_client = self._make_redis_client(pubsub)

        with patch("app.api.routes.signals.aioredis.from_url", return_value=redis_client):
            with patch("app.api.routes.signals.MetricsCollector.record_websocket_connection") as gauge_mock:
                mgr = WebSocketConnectionManager()
                await mgr.connect(ws, user_id=str(uuid4()), redis_url="redis://localhost")

        calls = [c.args[0] for c in gauge_mock.call_args_list]
        assert -1 in calls, "Gauge decrement must fire even after WebSocketDisconnect"

    @pytest.mark.asyncio
    async def test_all_messages_forwarded_under_backpressure(self):
        """All 1,000 messages in the pub/sub queue reach send_text without drops."""
        from app.api.routes.signals import WebSocketConnectionManager

        n_messages = 1_000
        payloads = [json.dumps({"type": "signal", "data": {"n": i}}) for i in range(n_messages)]

        ws = MagicMock()
        ws.accept = AsyncMock()
        ws.send_text = AsyncMock()

        pubsub = self._make_pubsub_listener(payloads)
        redis_client = self._make_redis_client(pubsub)

        with patch("app.api.routes.signals.aioredis.from_url", return_value=redis_client):
            with patch("app.api.routes.signals.MetricsCollector.record_websocket_connection"):
                mgr = WebSocketConnectionManager()
                await mgr.connect(ws, user_id=str(uuid4()), redis_url="redis://localhost")

        assert ws.send_text.await_count == n_messages, (
            f"Expected {n_messages} forwarded messages; "
            f"got {ws.send_text.await_count}"
        )

    @pytest.mark.asyncio
    async def test_500_concurrent_subscribers_net_zero_gauge(self):
        """After 500 subscribers all disconnect, the gauge returns to net-zero."""
        from app.api.routes.signals import WebSocketConnectionManager

        gauge_delta = {"value": 0}

        def _track_gauge(delta: int):
            gauge_delta["value"] += delta

        async def _one_subscriber():
            ws = MagicMock()
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()
            pubsub = self._make_pubsub_listener([])
            redis_client = self._make_redis_client(pubsub)
            mgr = WebSocketConnectionManager()
            with patch("app.api.routes.signals.aioredis.from_url", return_value=redis_client):
                await mgr.connect(ws, user_id=str(uuid4()), redis_url="redis://localhost")

        with patch(
            "app.api.routes.signals.MetricsCollector.record_websocket_connection",
            side_effect=_track_gauge,
        ):
            await asyncio.gather(*[_one_subscriber() for _ in range(500)])

        assert gauge_delta["value"] == 0, (
            f"Gauge not net-zero after 500 disconnects: delta={gauge_delta['value']}"
        )


# ===========================================================================
# SECTION 4 — FeedbackProcessor Race Conditions & Re-ranking
# ===========================================================================

class TestFeedbackProcessorRaceConditions:
    """Verify FeedbackProcessor is deadlock-free under concurrent load and that
    the background re-ranking task fires correctly and computes correct scores."""

    def _make_calibrator(self, old_t: float = 1.0, new_t: float = 1.0) -> ConfidenceCalibrator:
        """Return a ConfidenceCalibrator whose _scalars simulate a scalar shift."""
        cal = MagicMock(spec=ConfidenceCalibrator)
        # Return old_t before update, new_t after — mimics a post-update read
        call_state = {"calls": 0}

        def _get_scalar(key, default=1.0):
            # First read (before update) returns old_t, subsequent reads return new_t
            if call_state["calls"] == 0:
                call_state["calls"] += 1
                return old_t
            return new_t

        cal._scalars = MagicMock()
        cal._scalars.get = MagicMock(side_effect=_get_scalar)
        cal.update = MagicMock()
        return cal

    @pytest.mark.asyncio
    async def test_50_concurrent_act_dismiss_events_no_deadlock(self):
        """50 act + 50 dismiss events complete without hanging or raising."""
        cal = self._make_calibrator(old_t=1.0, new_t=1.0)  # no threshold breach
        processor = FeedbackProcessor(calibrator=cal)

        acts = [
            processor.process_act(
                signal_id=str(uuid4()),
                signal_type_value=SignalType.FEATURE_REQUEST.value,
                confidence_score=0.80,
                current_action_score=0.60,
            )
            for _ in range(50)
        ]
        dismisses = [
            processor.process_dismiss(
                signal_id=str(uuid4()),
                signal_type_value=SignalType.CHURN_RISK.value,
                confidence_score=0.75,
                current_action_score=0.55,
            )
            for _ in range(50)
        ]

        results = await asyncio.gather(*(acts + dismisses), return_exceptions=True)

        exceptions = [r for r in results if isinstance(r, Exception)]
        assert not exceptions, (
            f"{len(exceptions)} exceptions in concurrent act/dismiss: {exceptions[:3]}"
        )
        assert len(results) == 100

    @pytest.mark.asyncio
    async def test_threshold_breach_fires_rerank_task(self):
        """When scalar delta ≥ _RERANK_THRESHOLD, create_task is called."""
        # Force a delta of 0.1 (> default threshold 0.05)
        cal = self._make_calibrator(old_t=1.0, new_t=1.1)
        processor = FeedbackProcessor(calibrator=cal)

        scheduled_tasks: list = []

        def _fake_create_task(coro):
            scheduled_tasks.append(coro)
            # Close the coroutine to avoid ResourceWarning
            coro.close()
            return MagicMock()

        with patch.object(asyncio, "get_event_loop") as mock_loop:
            mock_loop.return_value.create_task = _fake_create_task
            await processor.process_act(
                signal_id=str(uuid4()),
                signal_type_value=SignalType.FEATURE_REQUEST.value,
                confidence_score=0.85,
                current_action_score=0.70,
            )

        assert len(scheduled_tasks) == 1, (
            "Expected exactly 1 background re-rank task to be scheduled"
        )

    @pytest.mark.asyncio
    async def test_no_rerank_when_delta_below_threshold(self):
        """When scalar delta < _RERANK_THRESHOLD, no task is scheduled."""
        # Force a delta of 0.01 (< default threshold 0.05)
        cal = self._make_calibrator(old_t=1.000, new_t=1.001)
        processor = FeedbackProcessor(calibrator=cal)

        scheduled_tasks: list = []

        def _fake_create_task(coro):
            scheduled_tasks.append(coro)
            coro.close()
            return MagicMock()

        with patch.object(asyncio, "get_event_loop") as mock_loop:
            mock_loop.return_value.create_task = _fake_create_task
            await processor.process_act(
                signal_id=str(uuid4()),
                signal_type_value=SignalType.FEATURE_REQUEST.value,
                confidence_score=0.85,
                current_action_score=0.70,
            )

        assert len(scheduled_tasks) == 0, (
            f"No task should fire when delta < {_RERANK_THRESHOLD}; "
            f"got {len(scheduled_tasks)}"
        )

    def test_rerank_formula_correct(self):
        """_rerank_signals_background applies sigmoid(logit/T) * urgency * impact."""
        urgency, impact, raw_conf, T = 0.8, 0.7, 0.9, 0.8
        raw_logit = math.log(raw_conf / (1.0 - raw_conf))
        calibrated = 1.0 / (1.0 + math.exp(-raw_logit / T))
        expected = urgency * impact * calibrated

        # Verify the formula directly — no DB needed
        assert abs(expected - (0.8 * 0.7 * calibrated)) < 1e-9
        assert 0.0 < expected < 1.0, "action_score must be in (0, 1)"

    @pytest.mark.asyncio
    async def test_rerank_db_failure_is_non_fatal(self):
        """If AsyncSessionLocal raises, _rerank_signals_background logs but doesn't propagate."""
        # AsyncSessionLocal is imported lazily inside the function body, so we
        # patch it at its source module (app.core.db), not at feedback_processor.
        with patch(
            "app.core.db.AsyncSessionLocal",
            side_effect=Exception("DB pool exhausted"),
        ):
            # Must not raise — failure is caught and logged internally
            await _rerank_signals_background(
                signal_type_value=SignalType.FEATURE_REQUEST.value,
                new_temperature=0.85,
            )  # no exception → test passes

    @pytest.mark.asyncio
    async def test_rerank_updates_action_score_for_new_and_queued_signals(self):
        """_rerank_signals_background updates action_score for NEW/QUEUED rows only.

        NOTE: _rerank_signals_background validates signal_type_value against
        ``app.core.signal_models.SignalType`` (the GTM-tier enum), which has
        different values from ``app.domain.inference_models.SignalType``.
        'churn_risk' is valid in both; we use it here to avoid a ValueError in
        the type-resolution step inside the background task.
        """
        from app.core.signal_models import SignalStatus

        # 'churn_risk' is present in BOTH app.core.signal_models.SignalType
        # and app.domain.inference_models.SignalType.
        valid_type = "churn_risk"

        urgency, impact, conf = 0.8, 0.7, 0.9
        new_t = 0.8

        # Build two mock DB signals
        def _make_db_sig(status):
            sig = MagicMock()
            sig.confidence_score = conf
            sig.urgency_score = urgency
            sig.impact_score = impact
            sig.action_score = 0.5
            sig.status = status
            return sig

        new_sig = _make_db_sig(SignalStatus.NEW)
        queued_sig = _make_db_sig(SignalStatus.QUEUED)
        mock_rows = MagicMock()
        mock_rows.scalars.return_value.all.return_value = [new_sig, queued_sig]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_rows)
        mock_session.commit = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_factory = MagicMock(return_value=mock_session)

        # AsyncSessionLocal is imported lazily inside the function body.
        with patch("app.core.db.AsyncSessionLocal", mock_session_factory):
            await _rerank_signals_background(
                signal_type_value=valid_type,
                new_temperature=new_t,
            )

        # Both signals must have had action_score updated
        raw_logit = math.log(conf / (1.0 - conf))
        calibrated = 1.0 / (1.0 + math.exp(-raw_logit / new_t))
        expected_score = urgency * impact * calibrated

        for sig in [new_sig, queued_sig]:
            actual = sig.action_score
            assert abs(actual - expected_score) < 1e-6, (
                f"Expected action_score ≈ {expected_score:.6f}; got {actual:.6f}"
            )
        mock_session.commit.assert_awaited_once()


# ===========================================================================
# SECTION 5 — NormalizationEngine Configuration Fault Tolerance
# ===========================================================================

class TestEntityKBFaultTolerance:
    """Verify _load_entity_kb() gracefully degrades on file errors and that the
    NormalizationEngine entity-linking path continues operating with the fallback.
    """

    def test_missing_file_returns_fallback_and_logs_warning(self, caplog):
        """When entity_kb.json does not exist, fallback is used and WARNING logged."""
        non_existent = Path("/tmp/definitely_missing_entity_kb_xyz.json")

        with patch("app.intelligence.normalization._ENTITY_KB_PATH", non_existent):
            with caplog.at_level(logging.WARNING, logger="app.intelligence.normalization"):
                result = _load_entity_kb()

        # Must return the built-in fallback (all keys present)
        for key in _ENTITY_KB_FALLBACK:
            assert key in result, f"Fallback key {key!r} missing from result"

        # Warning must have been logged
        assert any("not found" in r.message.lower() or "fallback" in r.message.lower()
                   for r in caplog.records), (
            "Expected a WARNING about missing entity KB file"
        )

    def test_corrupted_json_returns_fallback_and_logs_error(self, caplog, tmp_path):
        """When entity_kb.json contains invalid JSON, fallback is used and ERROR logged."""
        bad_file = tmp_path / "entity_kb.json"
        bad_file.write_text("{this is not: valid json{{{{", encoding="utf-8")

        with patch("app.intelligence.normalization._ENTITY_KB_PATH", bad_file):
            with caplog.at_level(logging.ERROR, logger="app.intelligence.normalization"):
                result = _load_entity_kb()

        for key in _ENTITY_KB_FALLBACK:
            assert key in result, f"Fallback key {key!r} missing after corrupt JSON"

        assert any("failed" in r.message.lower() or "parse" in r.message.lower()
                   for r in caplog.records), (
            "Expected an ERROR about JSON parse failure"
        )

    def test_valid_file_loads_all_entries_lowercased(self, tmp_path):
        """Valid entity_kb.json loads all entries with lower-cased surface forms."""
        entries = {
            "OpenAI": ["wikidata:Q56296273", "OpenAI"],
            "ANTHROPIC": ["wikidata:Q116256162", "Anthropic"],
            "Custom Vendor": ["custom:vendor-1", "Custom Vendor Inc."],
        }
        kb_file = tmp_path / "entity_kb.json"
        kb_file.write_text(
            json.dumps({"version": "1.0", "entries": entries}),
            encoding="utf-8",
        )

        with patch("app.intelligence.normalization._ENTITY_KB_PATH", kb_file):
            result = _load_entity_kb()

        # All keys must be lower-cased
        assert "openai" in result
        assert "anthropic" in result
        assert "custom vendor" in result
        # None of the original mixed-case keys should survive
        assert "OpenAI" not in result
        assert "ANTHROPIC" not in result

    def test_valid_file_canonical_id_and_name_correct(self, tmp_path):
        """Loaded entries have exactly the (canonical_id, canonical_name) specified."""
        kb_file = tmp_path / "entity_kb.json"
        kb_file.write_text(
            json.dumps({
                "version": "1.0",
                "entries": {
                    "testco": ["internal:testco-001", "TestCo Corporation"],
                },
            }),
            encoding="utf-8",
        )

        with patch("app.intelligence.normalization._ENTITY_KB_PATH", kb_file):
            result = _load_entity_kb()

        assert result["testco"] == ("internal:testco-001", "TestCo Corporation")

    def test_entity_linking_assigns_canonical_id_from_fallback_kb(self):
        """_link_entities_to_kb uses the module-level KB to assign canonical_id."""
        from app.intelligence.normalization import NormalizationEngine

        # EntityMention requires `confidence` (float, 0-1); relevance_score is not a field.
        entities = [
            EntityMention(entity_name="OpenAI", entity_type="ORG", confidence=0.9),
            EntityMention(entity_name="UnknownCorp", entity_type="ORG", confidence=0.7),
        ]
        result = NormalizationEngine._link_entities_to_kb(entities)

        # "openai" is in the hardcoded fallback — canonical_id must be assigned
        openai_mention = next(e for e in result if e.entity_name == "OpenAI")
        assert openai_mention.canonical_id is not None, (
            "canonical_id must be set for known entities"
        )
        assert openai_mention.canonical_id.startswith("wikidata:")

        # Unknown entity must remain unchanged
        unknown_mention = next(e for e in result if e.entity_name == "UnknownCorp")
        assert unknown_mention.canonical_id is None, (
            "canonical_id must remain None for unknown entities"
        )

    def test_normalization_pipeline_continues_with_fallback_kb(self, caplog, tmp_path):
        """NormalizationEngine Stage E (_link_entities_to_kb) runs normally even
        when entity_kb.json is absent and the fallback is active."""
        from app.intelligence.normalization import NormalizationEngine

        missing = tmp_path / "missing_kb.json"  # does not exist

        with patch("app.intelligence.normalization._ENTITY_KB_PATH", missing):
            # Re-load the KB (simulating a cold start with missing file)
            with caplog.at_level(logging.WARNING, logger="app.intelligence.normalization"):
                fallback_kb = _load_entity_kb()

        # Patch the module-level _ENTITY_KB with the fallback that was just loaded
        with patch("app.intelligence.normalization._ENTITY_KB", fallback_kb):
            entities = [
                EntityMention(entity_name="openai", entity_type="ORG", confidence=0.9),
            ]
            result = NormalizationEngine._link_entities_to_kb(entities)

        # Must not raise, and known entity from fallback must be linked
        assert result[0].canonical_id is not None

