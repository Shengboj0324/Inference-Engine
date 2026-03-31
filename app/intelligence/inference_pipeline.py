"""Inference pipeline orchestrator.

This module orchestrates the complete inference pipeline:
1. Normalization: RawObservation -> NormalizedObservation
2. Candidate Retrieval: Find similar signals
3. LLM Adjudication: Structured signal classification
4. Calibration: Probability calibration
5. Abstention: Decide whether to abstain

This is the main entry point for the Phase 2 inference system.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import UUID

import redis.asyncio as aioredis

from app.domain.raw_models import RawObservation
from app.domain.normalized_models import NormalizedObservation
from app.domain.inference_models import SignalInference, UserContext
from app.core.models import PlatformAuthStatus
from app.intelligence.normalization import NormalizationEngine
from app.intelligence.candidate_retrieval import CandidateRetriever
from app.intelligence.llm_adjudicator import LLMAdjudicator
from app.intelligence.calibration import Calibrator, ConfidenceCalibrator
from app.intelligence.abstention import AbstentionDecider
from app.intelligence.cot_reasoner import ChainOfThoughtReasoner
from app.intelligence.orchestrator import MultiAgentOrchestrator
from app.intelligence.context_memory import ContextMemoryStore
from app.intelligence.deliberation import DeliberationEngine
from app.intelligence.reranker import Reranker

logger = logging.getLogger(__name__)


class InferencePipeline:
    """End-to-end inference pipeline orchestrator.

    Chains together all pipeline components:
    - Normalization
    - Candidate retrieval
    - LLM adjudication (with optional E1–E6 enhancements)
    - ECE calibration (aggregate-level, Stage D)
    - Abstention

    The five enhancement components (E1–E3, E5–E6) are injected into
    ``LLMAdjudicator`` and default to ``None``, preserving full backward
    compatibility.  When ``None``, adjudication falls back to the original
    single-call path.
    """

    def __init__(
        self,
        normalization_engine: Optional[NormalizationEngine] = None,
        candidate_retriever: Optional[CandidateRetriever] = None,
        llm_adjudicator: Optional[LLMAdjudicator] = None,
        calibrator: Optional[Calibrator] = None,
        abstention_decider: Optional[AbstentionDecider] = None,
        # Enhancement components (E1–E3, E5–E6) — all optional
        confidence_calibrator: Optional[ConfidenceCalibrator] = None,
        cot_reasoner: Optional[ChainOfThoughtReasoner] = None,
        orchestrator: Optional[MultiAgentOrchestrator] = None,
        context_memory: Optional[ContextMemoryStore] = None,
        deliberation_engine: Optional[DeliberationEngine] = None,
        # Reranker (Stage 2b) — optional
        reranker: Optional[Reranker] = None,
        # RAG document pool — pre-populated NormalizedObservations used as
        # retrieval context by the reranker.  Empty list disables reranking.
        rag_document_pool: Optional[List[NormalizedObservation]] = None,
        # Redis signal publisher
        redis_url: Optional[str] = None,
    ):
        """Initialize inference pipeline.

        Args:
            normalization_engine: Normalization engine (creates default if None).
            candidate_retriever: Candidate retriever (creates default if None).
            llm_adjudicator: Pre-built LLM adjudicator.  When provided, the
                five enhancement params below are ignored (the caller is
                responsible for injecting them into the adjudicator).
            calibrator: Aggregate ECE calibrator (creates default if None).
            abstention_decider: Abstention decider (creates default if None).
            confidence_calibrator: Per-SignalType temperature-scaling
                calibrator (E2).  Applied inside adjudication, not after.
            cot_reasoner: Chain-of-Thought reasoner (E1).
            orchestrator: Multi-agent orchestrator (E3).
            context_memory: Per-user vector memory store (E5).
            deliberation_engine: Pre-adjudication deliberation engine (E6).
            redis_url: Redis connection URL for the WebSocket signal broadcaster.
                When set, every non-abstained inference result is published to
                ``signals:{user_id}`` so ``WebSocketConnectionManager`` clients
                receive live updates.  Reads ``settings.redis_url`` by default.
        """
        # Lazily resolve redis_url from settings when not explicitly provided.
        if redis_url is None:
            try:
                from app.core.config import settings as _settings
                redis_url = _settings.redis_url
            except Exception:
                redis_url = None
        self._redis_url: Optional[str] = redis_url
        self.normalization_engine = normalization_engine or NormalizationEngine(
            enable_translation=False,
            enable_entity_extraction=False,
            enable_embedding_generation=True,
        )

        self.candidate_retriever = candidate_retriever or CandidateRetriever(top_k=5)

        # Auto-load ConfidenceCalibrator from the training artifact when no
        # explicit instance is supplied and the state file exists on disk.
        # This closes the gap between the training run (which writes
        # training/calibration_state.json) and the serving path (which must
        # use those learned temperature scalars rather than defaulting to T=1.0).
        if confidence_calibrator is None:
            _default_state = Path("training/calibration_state.json")
            if _default_state.exists():
                try:
                    confidence_calibrator = ConfidenceCalibrator(state_path=_default_state)
                    logger.info(
                        "InferencePipeline: loaded ConfidenceCalibrator from %s",
                        _default_state,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "InferencePipeline: failed to load ConfidenceCalibrator from %s: %s",
                        _default_state, exc,
                    )
            else:
                logger.info(
                    "InferencePipeline: calibration_state.json not found at %s; "
                    "ConfidenceCalibrator will default to T=1.0 for all signal types",
                    _default_state,
                )

        self.llm_adjudicator = llm_adjudicator or LLMAdjudicator(
            model_name="gpt-4-turbo",
            temperature=0.3,
            confidence_calibrator=confidence_calibrator,
            cot_reasoner=cot_reasoner,
            orchestrator=orchestrator,
            context_memory=context_memory,
            deliberation_engine=deliberation_engine,
        )

        self.calibrator = calibrator or Calibrator(method="temperature")

        self.abstention_decider = abstention_decider or AbstentionDecider()

        # Reranker (Stage 2b) — None means reranking is skipped
        self.reranker: Optional[Reranker] = reranker
        self.rag_document_pool: List[NormalizedObservation] = rag_document_pool or []

        logger.info("InferencePipeline initialized with all components")
    
    async def run(
        self,
        raw_observation: RawObservation,
        skip_normalization: bool = False,
        normalized_observation: Optional[NormalizedObservation] = None,
        user_context: Optional[UserContext] = None,
    ) -> Tuple[NormalizedObservation, SignalInference]:
        """Run the complete inference pipeline.

        Args:
            raw_observation: Raw observation from connector
            skip_normalization: Skip normalization if already done
            normalized_observation: Pre-normalized observation (if skip_normalization=True)
            user_context: Optional per-user strategic priorities injected into
                Stage 3 (LLM Adjudication) to bias urgency / impact scoring
                against the user's tracked competitors and focus areas.

        Returns:
            Tuple of (normalized_observation, signal_inference)
        """
        logger.info(f"Running inference pipeline for observation {raw_observation.id}")
        
        # Stage 1: Normalization
        if skip_normalization and normalized_observation:
            normalized = normalized_observation
            logger.debug("Skipped normalization (using provided)")
        else:
            logger.debug("Stage 1: Normalization")
            normalized = await self.normalization_engine.normalize(raw_observation)
        
        # Stage 2: Candidate Retrieval (sync operation)
        logger.debug("Stage 2: Candidate Retrieval")
        candidates = self.candidate_retriever.retrieve_candidates(normalized)
        logger.debug(f"Retrieved {len(candidates)} candidates")

        # Stage 2b: Reranker — only when enabled in StrategicPriorities and
        # a non-empty RAG document pool is available.
        sp = user_context.strategic_priorities if user_context else None
        reranker_enabled = sp.reranker_enabled if sp is not None else True
        reranker_top_k = sp.reranker_top_k if sp is not None else 10

        rag_pool = getattr(self, "rag_document_pool", []) or []
        reranked_pool: List[NormalizedObservation] = []
        if getattr(self, "reranker", None) is not None and reranker_enabled and rag_pool:
            logger.debug("Stage 2b: Reranker — pool=%d docs", len(rag_pool))
            query_text = normalized.normalized_text or normalized.title or ""
            rag_top_k = sp.rag_top_k if sp is not None else 20
            effective_top_k = min(reranker_top_k, rag_top_k)
            _active_reranker = getattr(self, "reranker", None)
            reranked_pool = _active_reranker.rerank(
                query=query_text,
                candidates=rag_pool,
                top_k=effective_top_k,
            )
            logger.debug(
                "Stage 2b: Reranker returned %d docs (requested %d)",
                len(reranked_pool),
                effective_top_k,
            )

        # Stage 3: LLM Adjudication (user_context injects strategic priorities)
        logger.debug("Stage 3: LLM Adjudication")
        inference = await self.llm_adjudicator.adjudicate(
            normalized, candidates, user_context=user_context
        )
        logger.debug(f"LLM adjudication complete: abstained={inference.abstained}")
        
        # Stage 4: Calibration
        logger.debug("Stage 4: Calibration")
        inference = self.calibrator.calibrate(inference)
        logger.debug("Calibration complete")
        
        # Stage 5: Abstention Decision
        logger.debug("Stage 5: Abstention Decision")
        should_abstain, reason, explanation = self.abstention_decider.should_abstain(
            inference, normalized, candidates
        )
        
        if should_abstain and not inference.abstained:
            # Override inference with abstention
            inference.abstained = True
            inference.abstention_reason = reason
            inference.rationale = f"{inference.rationale or ''}\n\nAbstention: {explanation}"
            logger.debug(f"Abstention triggered: {reason.value if reason else 'unknown'}")

        logger.info(
            f"Pipeline complete for {raw_observation.id}: "
            f"signal={inference.top_prediction.signal_type.value if inference.top_prediction else 'none'}, "
            f"confidence={inference.top_prediction.probability if inference.top_prediction else 0.0:.2f}, "
            f"abstained={inference.abstained}"
        )

        # Stage 5b: Populate ResponseArtifacts
        # Attach source citations for RAG pool docs, image/video URLs from
        # platform_metadata, and document links for news-connector sources.
        # Wrapped in try/except because artifact population is a value-add step
        # and must never crash the pipeline even on unexpected Pydantic errors.
        try:
            self._populate_artifacts(raw_observation, normalized, reranked_pool, inference)
        except Exception as _art_exc:
            logger.warning(
                "InferencePipeline: artifact population failed (obs=%s): %s",
                raw_observation.id,
                _art_exc,
            )

        # Stage 6: Publish to Redis for WebSocket broadcast (non-blocking)
        if not inference.abstained and raw_observation.user_id:
            asyncio.ensure_future(
                self._publish_to_redis(raw_observation.user_id, inference)
            )

        # Stage 7: High-confidence thread expansion (Req 4, non-blocking)
        # Fires a high-priority Celery task when the classified probability
        # exceeds the ingestion threshold so the full conversation thread is
        # ingested asynchronously.
        if not inference.abstained and inference.top_prediction is not None:
            self._maybe_trigger_thread_expansion(raw_observation, inference)

        return normalized, inference

    @staticmethod
    def _maybe_trigger_thread_expansion(
        raw_observation: "RawObservation",
        inference: "SignalInference",
    ) -> None:
        """Fire ``expand_conversation_thread`` when thresholds are exceeded.

        Thresholds (Req 4):
        - ``top_prediction.probability > 0.8`` (maps to confidence_score)
        - The signal type is inherently high-impact (CHURN_RISK, COMPETITOR_MENTION,
          ALTERNATIVE_SEEKING, EXPANSION_OPPORTUNITY, UPSELL_OPPORTUNITY) **or** the
          prediction has high confidence AND the probability > 0.85, acting as a
          proxy for impact_score when the DB score is not yet available.

        All failures are logged at WARNING level and swallowed — thread expansion
        is strictly a value-add operation and must never block the critical path.

        Args:
            raw_observation: The raw observation that produced the inference.
            inference: The classified ``SignalInference``.
        """
        from app.domain.inference_models import SignalType
        _HIGH_IMPACT_TYPES = {
            SignalType.CHURN_RISK,
            SignalType.COMPETITOR_MENTION,
            SignalType.ALTERNATIVE_SEEKING,
            SignalType.EXPANSION_OPPORTUNITY,
            SignalType.UPSELL_OPPORTUNITY,
            SignalType.REPUTATION_RISK,
        }
        _CONFIDENCE_THRESHOLD = 0.8
        _HIGH_IMPACT_PROXY_THRESHOLD = 0.85

        pred = inference.top_prediction
        if pred is None or pred.probability < _CONFIDENCE_THRESHOLD:
            return
        is_high_impact = (
            pred.signal_type in _HIGH_IMPACT_TYPES
            or pred.probability >= _HIGH_IMPACT_PROXY_THRESHOLD
        )
        if not is_high_impact:
            return
        # Celery task import is deferred to avoid circular imports at module load
        try:
            from app.ingestion.tasks import expand_conversation_thread
            expand_conversation_thread.apply_async(
                kwargs={
                    "signal_id": str(raw_observation.id),
                    "source_id": raw_observation.source_id or "",
                    "platform": raw_observation.source_platform.value
                    if hasattr(raw_observation.source_platform, "value")
                    else str(raw_observation.source_platform),
                    "user_id": str(raw_observation.user_id) if raw_observation.user_id else "",
                },
                queue="high_priority",
            )
            logger.info(
                "InferencePipeline: thread expansion queued for obs=%s type=%s conf=%.3f",
                raw_observation.id,
                pred.signal_type.value,
                pred.probability,
            )
        except Exception as exc:
            logger.warning(
                "InferencePipeline: failed to queue thread expansion for obs=%s: %s",
                raw_observation.id, exc,
            )

    # ── NEWS_CONNECTOR_PLATFORMS ──────────────────────────────────────────────
    # Used by _populate_artifacts to decide whether to emit a document_link.
    _NEWS_PLATFORMS = frozenset({"abc_news", "reuters", "nytimes", "wsj", "bbc"})

    @staticmethod
    def _populate_artifacts(
        raw: "RawObservation",
        normalized: "NormalizedObservation",
        reranked_pool: List["NormalizedObservation"],
        inference: "SignalInference",
    ) -> None:
        """Populate ``inference.artifacts`` with structured response artifacts.

        Adds (in order):
        1. ``source_citation`` for every NormalizedObservation in *reranked_pool*.
        2. ``image_url`` / ``video_url`` from ``raw.platform_metadata``.
        3. ``document_link`` when the source platform is a news connector.
        4. A fallback ``source_citation`` from ``normalized.source_url`` when
           the reranked pool is empty and a URL is available.

        All failures are silently skipped — artifact population is a value-add
        step and must never raise.
        """
        from app.domain.inference_models import ResponseArtifact  # local to avoid circular

        artifacts = []

        # 1. RAG source citations
        for rag_doc in reranked_pool:
            url = getattr(rag_doc, "source_url", None) or ""
            if url:
                artifacts.append(
                    ResponseArtifact(
                        artifact_type="source_citation",
                        content=url,
                        label=getattr(rag_doc, "title", None) or url[:80],
                        source_platform=getattr(rag_doc, "source_platform", None),
                        confidence=0.8,
                        published_at=getattr(rag_doc, "published_at", None),
                    )
                )

        # 2. Image / video URLs from platform_metadata
        meta = raw.platform_metadata or {}
        for img_key in ("image_url", "thumbnail_url"):
            img_url = meta.get(img_key)
            if img_url and isinstance(img_url, str):
                artifacts.append(
                    ResponseArtifact(
                        artifact_type="image_url",
                        content=img_url,
                        label="Image from " + raw.source_platform.value,
                        source_platform=raw.source_platform,
                        confidence=0.9,
                    )
                )
                break
        vid_url = meta.get("video_url") or meta.get("media_url")
        if vid_url and isinstance(vid_url, str):
            artifacts.append(
                ResponseArtifact(
                    artifact_type="video_url",
                    content=vid_url,
                    label="Video from " + raw.source_platform.value,
                    source_platform=raw.source_platform,
                    confidence=0.9,
                )
            )

        # 3. Document link for news connector sources
        platform_val = (
            raw.source_platform.value
            if hasattr(raw.source_platform, "value")
            else str(raw.source_platform)
        )
        if platform_val.lower() in InferencePipeline._NEWS_PLATFORMS and raw.source_url:
            artifacts.append(
                ResponseArtifact(
                    artifact_type="document_link",
                    content=raw.source_url,
                    label=raw.title or raw.source_url[:80],
                    source_platform=raw.source_platform,
                    confidence=0.95,
                    published_at=raw.published_at,
                )
            )

        # 4. Fallback citation from normalized source URL
        if not reranked_pool and normalized.source_url and not any(
            a.artifact_type == "source_citation" for a in artifacts
        ):
            artifacts.append(
                ResponseArtifact(
                    artifact_type="source_citation",
                    content=normalized.source_url,
                    label=normalized.title or normalized.source_url[:80],
                    source_platform=getattr(normalized, "source_platform", None),
                    confidence=0.7,
                )
            )

        inference.artifacts = artifacts

    async def _publish_to_redis(
        self,
        user_id: UUID,
        inference: SignalInference,
        platform_auth_status: Optional[PlatformAuthStatus] = None,
    ) -> None:
        """Publish a non-abstained inference result to the user's Redis channel.

        The message is consumed by :class:`WebSocketConnectionManager` in
        ``app/api/routes/signals.py`` and forwarded to every WebSocket client
        subscribed to ``signals:{user_id}``.

        The payload is a JSON object with ``type="signal"`` so the WebSocket
        client can distinguish signal events from keepalive pings:

        .. code-block:: json

            {
              "type": "signal",
              "data": {
                "inference_id": "<uuid>",
                "signal_type": "<value>",
                "confidence": 0.91,
                "rationale": "...",
                "timestamp": "<iso8601>"
              }
            }

        Failures are logged at WARNING level and never propagate to callers —
        a Redis outage must never degrade inference correctness.

        Args:
            user_id: UUID of the user who owns this observation.
            inference: Completed, non-abstained ``SignalInference``.
        """
        if not self._redis_url:
            return
        top = inference.top_prediction
        if not top:
            return
        channel = f"signals:{user_id}"
        payload = json.dumps({
            "type": "signal",
            "data": {
                "inference_id": str(inference.id),
                "signal_type": top.signal_type.value,
                "confidence": round(top.probability, 4),
                "rationale": (inference.rationale or "")[:300],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                # OAuth status of the source platform — frontend uses this to
                # prompt re-authentication when a token has expired mid-session.
                "platform_auth_status": (
                    platform_auth_status.value
                    if platform_auth_status is not None
                    else PlatformAuthStatus.CONNECTED.value
                ),
                # Structured response artifacts serialised for the frontend
                # renderer (citation cards, image thumbnails, video previews).
                "artifacts": [
                    a.model_dump(mode="json") for a in inference.artifacts
                ],
            },
        })
        try:
            client: aioredis.Redis = aioredis.from_url(
                self._redis_url, decode_responses=True
            )
            async with client:
                await client.publish(channel, payload)
            logger.debug(
                "InferencePipeline: published signal %s to %s",
                top.signal_type.value,
                channel,
            )
        except Exception as exc:
            logger.warning(
                "InferencePipeline: Redis publish failed for user %s: %s",
                user_id,
                exc,
            )
    
    async def run_batch(
        self,
        raw_observations: list[RawObservation],
        concurrency: int = 5,
        user_context: Optional[UserContext] = None,
    ) -> List[Tuple[NormalizedObservation, SignalInference]]:
        """Run pipeline on a batch of observations concurrently.

        Processes observations concurrently up to ``concurrency`` at a time using
        :class:`asyncio.Semaphore`, preserving input order in the output list.
        Failed observations are logged but do not abort the batch; their
        corresponding slot in the output list is omitted.

        Args:
            raw_observations: List of raw observations to process.
            concurrency: Maximum number of parallel pipeline executions (default 5).

        Returns:
            List of (normalized_observation, signal_inference) tuples in the same
            order as ``raw_observations``, excluding any that raised exceptions.

        Raises:
            ValueError: If ``concurrency`` is less than 1.
        """
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")

        semaphore = asyncio.Semaphore(concurrency)

        async def _run_one(raw_obs: RawObservation) -> Optional[Tuple[NormalizedObservation, SignalInference]]:
            async with semaphore:
                try:
                    return await self.run(raw_obs, user_context=user_context)
                except Exception as e:
                    logger.error(
                        f"Error processing observation {raw_obs.id}: {e}",
                        exc_info=True,
                    )
                    return None

        # Gather preserves input order; None sentinels are filtered out.
        raw_results = await asyncio.gather(*[_run_one(obs) for obs in raw_observations])
        results = [r for r in raw_results if r is not None]

        logger.info(
            f"Batch processing complete: {len(results)}/{len(raw_observations)} successful "
            f"(concurrency={concurrency})"
        )
        return results

