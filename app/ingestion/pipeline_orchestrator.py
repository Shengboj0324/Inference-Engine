"""Pipeline orchestrator for end-to-end content processing.

This module provides the PipelineOrchestrator class that coordinates the complete
pipeline from content ingestion to workflow execution:

1. ContentIngestor: Fetch content from multiple sources
2. NormalizationEngine: Standardize content format
3. EnrichmentService: Augment with AI metadata
4. SignalClassifier: Detect actionable signals
5. WorkflowOrchestrator: Execute business workflows

This is the main integration point between Phase 4 (Ingestion) and Phase 3 (Workflows).
"""

import logging
from typing import Dict, List, Optional, Tuple
from uuid import UUID
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import ContentItem
from app.core.signal_models import ActionableSignal
from app.core.db_models import ContentItemDB, ActionableSignalDB
from app.domain.raw_models import RawObservation
from app.domain.normalized_models import NormalizedObservation
from app.domain.inference_models import SignalInference
from app.domain.action_models import ActionableSignal as DomainActionableSignal
from app.ingestion.content_ingestor import ContentIngestor
from app.ingestion.normalization_engine import NormalizationEngine
from app.ingestion.enrichment_service import EnrichmentService
from app.intelligence.signal_classifier import SignalClassifier
from app.intelligence.inference_pipeline import InferencePipeline
from app.intelligence.action_pipeline import ActionPipeline
from app.workflows.orchestrator import WorkflowOrchestrator

logger = logging.getLogger(__name__)


class PipelineMetrics:
    """Metrics for pipeline operations."""

    def __init__(self):
        """Initialize metrics."""
        self.total_items_fetched = 0
        self.total_items_normalized = 0
        self.total_items_enriched = 0
        self.total_signals_detected = 0
        self.total_workflows_executed = 0
        self.pipeline_failures = 0

    def record_pipeline_run(
        self,
        fetched: int = 0,
        normalized: int = 0,
        enriched: int = 0,
        signals: int = 0,
        workflows: int = 0,
        failed: bool = False,
    ):
        """Record pipeline run metrics."""
        if not failed:
            self.total_items_fetched += fetched
            self.total_items_normalized += normalized
            self.total_items_enriched += enriched
            self.total_signals_detected += signals
            self.total_workflows_executed += workflows
        else:
            self.pipeline_failures += 1

    def get_summary(self) -> Dict:
        """Get metrics summary."""
        return {
            "total_items_fetched": self.total_items_fetched,
            "total_items_normalized": self.total_items_normalized,
            "total_items_enriched": self.total_items_enriched,
            "total_signals_detected": self.total_signals_detected,
            "total_workflows_executed": self.total_workflows_executed,
            "pipeline_failures": self.pipeline_failures,
        }


class PipelineOrchestrator:
    """End-to-end pipeline orchestrator from ingestion to workflow execution.

    This class coordinates all phases of content processing:
    - Phase 4: Ingestion (fetch, normalize, enrich)
    - Phase 1: Intelligence (classify signals)
    - Phase 3: Workflows (execute actions)

    Features:
    - Concurrent processing of multiple items
    - Graceful error handling at each stage
    - Comprehensive metrics tracking
    - Database persistence at each stage
    """

    def __init__(
        self,
        db_session: AsyncSession,
        content_ingestor: Optional[ContentIngestor] = None,
        normalization_engine: Optional[NormalizationEngine] = None,
        enrichment_service: Optional[EnrichmentService] = None,
        signal_classifier: Optional[SignalClassifier] = None,
        workflow_orchestrator: Optional[WorkflowOrchestrator] = None,
        inference_pipeline: Optional[InferencePipeline] = None,
        action_pipeline: Optional[ActionPipeline] = None,
    ):
        """Initialize pipeline orchestrator.

        Args:
            db_session: Database session.
            content_ingestor: Content ingestor instance (created if None).
            normalization_engine: Legacy normalization engine (created if None).
                Only used by the legacy ``run_full_pipeline()`` path.
            enrichment_service: Enrichment service instance (created if None).
                Only used by the legacy ``run_full_pipeline()`` path.
            signal_classifier: Legacy signal classifier (created if None).
                Only used by the legacy ``run_full_pipeline()`` path.
            workflow_orchestrator: Workflow orchestrator instance (created if None).
            inference_pipeline: New-stack InferencePipeline (created lazily if
                None).  Used by ``run_unified_pipeline()``.
            action_pipeline: New-stack ActionPipeline (created lazily if None).
                Used by ``run_unified_pipeline()``.
        """
        self.db = db_session

        # Legacy components (kept for backward compatibility)
        self.content_ingestor = content_ingestor or ContentIngestor(db_session)
        self.normalization_engine = normalization_engine or NormalizationEngine()
        self.enrichment_service = enrichment_service or EnrichmentService()
        self.signal_classifier = signal_classifier or SignalClassifier()
        self.workflow_orchestrator = workflow_orchestrator or WorkflowOrchestrator()

        # New canonical serving path components
        self._inference_pipeline: Optional[InferencePipeline] = inference_pipeline
        self._action_pipeline: Optional[ActionPipeline] = action_pipeline

        # Metrics
        self.metrics = PipelineMetrics()

        logger.info("PipelineOrchestrator initialized with all components")

    async def run_full_pipeline(
        self,
        user_id: UUID,
        execute_workflows: bool = True,
        store_content: bool = True,
        store_signals: bool = True,
    ) -> Dict:
        """Run the complete pipeline for a user.

        Args:
            user_id: User ID to process content for
            execute_workflows: Whether to execute workflows for detected signals
            store_content: Whether to store content items in database
            store_signals: Whether to store signals in database

        Returns:
            Pipeline execution summary
        """
        logger.info(f"Starting full pipeline for user {user_id}")
        start_time = asyncio.get_event_loop().time()

        try:
            # Stage 1: Fetch content
            content_items = await self.content_ingestor.fetch_from_sources(user_id)
            logger.info(f"Fetched {len(content_items)} items")

            if not content_items:
                logger.info("No content items fetched, pipeline complete")
                return self._build_summary(0, 0, 0, 0, 0, start_time)

            # Stage 2: Normalize content
            normalized_items = await self._normalize_items(content_items)
            logger.info(f"Normalized {len(normalized_items)} items")

            # Stage 3: Enrich content
            enriched_items = await self._enrich_items(normalized_items)
            logger.info(f"Enriched {len(enriched_items)} items")

            # Store content items if requested
            if store_content:
                await self._store_content_items(enriched_items)

            # Stage 4: Classify signals
            signals = await self._classify_signals(enriched_items, user_id)
            logger.info(f"Detected {len(signals)} actionable signals")

            # Store signals if requested
            if store_signals and signals:
                await self._store_signals(signals)
                logger.info(f"Stored {len(signals)} signals")

            # Stage 5: Execute workflows
            workflows_executed = 0
            if execute_workflows and signals:
                workflows_executed = await self._execute_workflows(signals)
                logger.info(f"Executed {workflows_executed} workflows")

            # Record metrics
            self.metrics.record_pipeline_run(
                fetched=len(content_items),
                normalized=len(normalized_items),
                enriched=len(enriched_items),
                signals=len(signals),
                workflows=workflows_executed,
            )

            return self._build_summary(
                len(content_items),
                len(normalized_items),
                len(enriched_items),
                len(signals),
                workflows_executed,
                start_time,
            )

        except Exception as e:
            logger.error(f"Pipeline failed for user {user_id}: {e}", exc_info=True)
            self.metrics.record_pipeline_run(failed=True)
            raise

    async def _normalize_items(self, items: List[ContentItem]) -> List[ContentItem]:
        """Normalize content items.

        Args:
            items: Content items to normalize

        Returns:
            Normalized content items
        """
        normalized = []
        for item in items:
            try:
                normalized_item = self.normalization_engine.normalize(item)
                normalized.append(normalized_item)
            except Exception as e:
                logger.error(f"Failed to normalize item {item.id}: {e}")
                # Continue with unnormalized item
                normalized.append(item)

        return normalized

    async def _enrich_items(self, items: List[ContentItem]) -> List[ContentItem]:
        """Enrich content items with AI metadata.

        Args:
            items: Content items to enrich

        Returns:
            Enriched content items
        """
        enriched = []
        for item in items:
            try:
                enriched_item = await self.enrichment_service.enrich(item)
                enriched.append(enriched_item)
            except Exception as e:
                logger.error(f"Failed to enrich item {item.id}: {e}")
                # Continue with unenriched item
                enriched.append(item)

        return enriched

    async def _classify_signals(
        self,
        items: List[ContentItem],
        user_id: UUID,
    ) -> List[ActionableSignal]:
        """Classify content items into actionable signals.

        Args:
            items: Content items to classify
            user_id: User ID for signal ownership

        Returns:
            List of detected signals
        """
        signals = []
        for item in items:
            try:
                signal = await self.signal_classifier.classify_content(item, user_id)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Failed to classify item {item.id}: {e}")

        return signals

    async def _execute_workflows(self, signals: List[ActionableSignal]) -> int:
        """Execute workflows for signals.

        Args:
            signals: Actionable signals

        Returns:
            Number of workflows executed
        """
        executed = 0
        for signal in signals:
            try:
                execution = await self.workflow_orchestrator.execute_for_signal(signal)
                if execution:
                    executed += 1
            except Exception as e:
                logger.error(f"Failed to execute workflow for signal {signal.id}: {e}")

        return executed

    async def _store_content_items(self, items: List[ContentItem]):
        """Store content items in database.

        Args:
            items: Content items to store
        """
        for item in items:
            try:
                # Convert to DB model
                db_item = ContentItemDB(
                    id=item.id,
                    user_id=item.user_id,
                    source_platform=item.source_platform,
                    source_id=item.source_id,
                    source_url=item.source_url,
                    author=item.author,
                    channel=item.channel,
                    title=item.title,
                    raw_text=item.raw_text,
                    media_type=item.media_type,
                    media_urls=item.media_urls,
                    published_at=item.published_at,
                    fetched_at=item.fetched_at,
                    lang=item.lang,
                    topics=item.topics,
                    embedding=item.embedding,
                    metadata_=item.metadata,
                )
                self.db.add(db_item)
            except Exception as e:
                logger.error(f"Failed to store content item {item.id}: {e}")

        await self.db.commit()

    async def _store_signals(self, signals: List[ActionableSignal]):
        """Store signals in database.

        Args:
            signals: Actionable signals to store
        """
        for signal in signals:
            try:
                # Convert to DB model
                db_signal = ActionableSignalDB(
                    id=signal.id,
                    user_id=signal.user_id,
                    signal_type=signal.signal_type,
                    source_item_ids=signal.source_item_ids,
                    source_platform=signal.source_platform,
                    source_url=signal.source_url,
                    source_author=signal.source_author,
                    title=signal.title,
                    description=signal.description,
                    context=signal.context,
                    urgency_score=signal.urgency_score,
                    impact_score=signal.impact_score,
                    confidence_score=signal.confidence_score,
                    action_score=signal.action_score,
                    recommended_action=signal.recommended_action,
                    suggested_channel=signal.suggested_channel,
                    suggested_tone=signal.suggested_tone,
                    draft_response=signal.draft_response,
                    draft_post=signal.draft_post,
                    draft_dm=signal.draft_dm,
                    positioning_angle=signal.positioning_angle,
                    status=signal.status,
                    assigned_to=signal.assigned_to,
                    created_at=signal.created_at,
                    expires_at=signal.expires_at,
                    acted_at=signal.acted_at,
                    outcome_feedback=signal.outcome_feedback,
                    metadata_=signal.metadata,
                )
                self.db.add(db_signal)
            except Exception as e:
                logger.error(f"Failed to store signal {signal.id}: {e}")

        await self.db.commit()

    def _build_summary(
        self,
        fetched: int,
        normalized: int,
        enriched: int,
        signals: int,
        workflows: int,
        start_time: float,
    ) -> Dict:
        """Build pipeline execution summary.

        Args:
            fetched: Number of items fetched
            normalized: Number of items normalized
            enriched: Number of items enriched
            signals: Number of signals detected
            workflows: Number of workflows executed
            start_time: Pipeline start time

        Returns:
            Summary dictionary
        """
        duration = asyncio.get_event_loop().time() - start_time

        return {
            "status": "success",
            "items_fetched": fetched,
            "items_normalized": normalized,
            "items_enriched": enriched,
            "signals_detected": signals,
            "workflows_executed": workflows,
            "duration_seconds": duration,
            "ingestion_metrics": self.content_ingestor.get_metrics(),
            "normalization_metrics": self.normalization_engine.get_metrics(),
            "enrichment_metrics": self.enrichment_service.get_metrics(),
            "pipeline_metrics": self.metrics.get_summary(),
        }

    def get_metrics(self) -> Dict:
        """Get pipeline metrics.

        Returns:
            Dictionary of metrics
        """
        return self.metrics.get_summary()

    # ------------------------------------------------------------------
    # New canonical serving path via InferencePipeline + ActionPipeline
    # ------------------------------------------------------------------

    @property
    def inference_pipeline(self) -> InferencePipeline:
        """Lazy-initialise and return the canonical InferencePipeline.

        Returns:
            The shared ``InferencePipeline`` instance.
        """
        if self._inference_pipeline is None:
            self._inference_pipeline = InferencePipeline()
            logger.info("PipelineOrchestrator: lazily created InferencePipeline")
        return self._inference_pipeline

    @property
    def action_pipeline(self) -> ActionPipeline:
        """Lazy-initialise and return the canonical ActionPipeline.

        Returns:
            The shared ``ActionPipeline`` instance.
        """
        if self._action_pipeline is None:
            self._action_pipeline = ActionPipeline()
            logger.info("PipelineOrchestrator: lazily created ActionPipeline")
        return self._action_pipeline

    async def run_unified_pipeline(
        self,
        user_id: UUID,
        concurrency: int = 5,
    ) -> Dict:
        """Run the canonical intelligence serving path for a user.

        This is the authoritative execution path that replaces the legacy
        ``NormalizationEngine → EnrichmentService → SignalClassifier`` chain
        with the new ``InferencePipeline → ActionPipeline`` stack.

        Execution flow
        --------------
        1. Fetch content items via ``ContentIngestor``.
        2. Bridge each ``ContentItem`` to a ``RawObservation`` (domain layer).
        3. Run all observations through ``InferencePipeline`` concurrently.
        4. Run all non-abstained inferences through ``ActionPipeline``.
        5. Return a structured execution summary.

        Args:
            user_id: User ID to process content for.
            concurrency: Maximum concurrent ``InferencePipeline.run()`` calls.

        Returns:
            Execution summary dict with keys ``items_fetched``,
            ``inferences_produced``, ``actions_produced``,
            ``abstentions``, ``duration_seconds``.

        Raises:
            Exception: Any un-recoverable error from the ingestor propagates;
                per-item failures are logged and skipped.
        """
        logger.info("run_unified_pipeline: starting for user %s", user_id)
        start_time = asyncio.get_event_loop().time()

        # Stage 1: Fetch
        content_items = await self.content_ingestor.fetch_from_sources(user_id)
        logger.info("run_unified_pipeline: fetched %d items", len(content_items))

        if not content_items:
            return {
                "status": "success",
                "items_fetched": 0,
                "inferences_produced": 0,
                "actions_produced": 0,
                "abstentions": 0,
                "duration_seconds": asyncio.get_event_loop().time() - start_time,
            }

        # Stage 2: Bridge ContentItem → RawObservation
        raw_observations: List[RawObservation] = [
            self._content_item_to_raw_observation(item) for item in content_items
        ]

        # Stage 3: Inference (concurrently via InferencePipeline.run_batch)
        pipeline_results: List[Tuple[NormalizedObservation, SignalInference]] = (
            await self.inference_pipeline.run_batch(
                raw_observations, concurrency=concurrency
            )
        )
        logger.info("run_unified_pipeline: produced %d inferences", len(pipeline_results))

        # Stage 4: Action ranking for non-abstained inferences
        obs_by_id: Dict[str, NormalizedObservation] = {
            str(norm.id): norm for norm, _ in pipeline_results
        }
        inferences: List[SignalInference] = [inf for _, inf in pipeline_results]
        non_abstained = [inf for inf in inferences if not inf.abstained]
        abstentions = len(inferences) - len(non_abstained)

        actions: List[DomainActionableSignal] = await self.action_pipeline.process_batch(
            non_abstained, obs_by_id
        )
        logger.info("run_unified_pipeline: produced %d actions", len(actions))

        duration = asyncio.get_event_loop().time() - start_time
        self.metrics.record_pipeline_run(
            fetched=len(content_items),
            normalized=len(pipeline_results),
            signals=len(actions),
        )

        return {
            "status": "success",
            "items_fetched": len(content_items),
            "inferences_produced": len(inferences),
            "actions_produced": len(actions),
            "abstentions": abstentions,
            "duration_seconds": duration,
        }

    @staticmethod
    def _content_item_to_raw_observation(item: ContentItem) -> RawObservation:
        """Bridge a legacy ``ContentItem`` to a domain-layer ``RawObservation``.

        Maps every overlapping field directly.  Platform-specific metadata
        stored in ``ContentItem.metadata`` is passed through as
        ``RawObservation.platform_metadata`` so that engagement signals
        (upvotes, shares, etc.) remain available to the normalization engine.

        Args:
            item: Legacy ``ContentItem`` produced by ``ContentIngestor``.

        Returns:
            ``RawObservation`` ready for ``InferencePipeline.run()``.
        """
        return RawObservation(
            id=item.id,
            user_id=item.user_id,
            source_platform=item.source_platform,
            source_id=item.source_id,
            source_url=item.source_url,
            author=item.author,
            channel=item.channel,
            title=item.title,
            raw_text=item.raw_text,
            media_type=item.media_type,
            media_urls=item.media_urls,
            published_at=item.published_at,
            fetched_at=item.fetched_at,
            platform_metadata=item.metadata,
        )
