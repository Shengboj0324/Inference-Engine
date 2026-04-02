"""Canonical entry point for the auto-research pipeline.

Usage::

    from app.research.auto_research_pipeline import AutoResearchPipeline

    pipeline = AutoResearchPipeline()
    report = await pipeline.run("AI safety", tenant_id="default")

"""

from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ResearchReport:
    """Complete output of one ``AutoResearchPipeline.run()`` call.

    Attributes
    ----------
    query:
        The research query that produced this report.
    tenant_id:
        Tenant context for multi-tenant deployments.
    grounded_summaries:
        ``GroundedSummary`` objects produced by ``GroundedSummaryBuilder``.
    evidence_chunks:
        ``ChunkRecord`` objects retrieved from the ``ChunkStore`` for the query.
    quality_gate_outcomes:
        ``QualityGateResult`` objects from the inline quality gate step.
    watchlist_gap_count:
        Number of watched entities with no coverage after this run.
    slo_health_status:
        Overall SLO status reported by the ``PipelineHealthMonitor``.
    generated_at:
        UTC timestamp when the report was produced.
    wall_s:
        Total wall-clock time in seconds for ``run()``.
    chunks_indexed:
        Number of chunks written to the store during this run.
    confidence_score_mean:
        Mean ``confidence_score`` across all grounded summaries
        (``0.0`` when no summaries were produced).
    quality_gate_rejection_rate:
        Fraction of quality-gate evaluations that failed (``0.0`` when none).
    """

    query: str
    tenant_id: str
    grounded_summaries: List[Any] = field(default_factory=list)
    evidence_chunks: List[Any] = field(default_factory=list)
    quality_gate_outcomes: List[Any] = field(default_factory=list)
    watchlist_gap_count: int = 0
    slo_health_status: Any = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    wall_s: float = 0.0
    chunks_indexed: int = 0
    confidence_score_mean: float = 0.0
    quality_gate_rejection_rate: float = 0.0


class AutoResearchPipeline:
    """Canonical end-to-end research pipeline.

    Composes all enhanced subsystems — source acquisition, multimodal analysis,
    event-first memory, output quality gating — into a single coroutine that
    accepts a free-text query and returns a ``ResearchReport``.

    All injected dependencies are optional.  When ``None`` is passed, sensible
    defaults are created automatically so the pipeline is usable out-of-the-box.

    All shared mutable state is protected by ``threading.Lock``.

    Args:
        pipeline:              ``IndexingPipeline`` for routing and indexing.
                               A default instance (with quality gate enabled) is
                               created when ``None``.
        watchlist_graph:       ``WatchlistGraph`` for coverage tracking.
        health_monitor:        ``PipelineHealthMonitor`` for SLO observability.
        acquisition_scheduler: ``AcquisitionScheduler`` for source scheduling.
        quality_gate:          ``QualityGate`` appended to the pipeline if no
                               gate is already attached to *pipeline*.
        multimodal_analyzer:   ``MultimodalAnalyzer`` for image/video evidence.
        min_confidence_threshold: Default quality gate threshold (``0.60``).

    Raises:
        ValueError: If ``min_confidence_threshold`` is outside [0, 1].
    """

    def __init__(
        self,
        pipeline:              Optional[Any] = None,
        watchlist_graph:       Optional[Any] = None,
        health_monitor:        Optional[Any] = None,
        acquisition_scheduler: Optional[Any] = None,
        quality_gate:          Optional[Any] = None,
        multimodal_analyzer:   Optional[Any] = None,
        min_confidence_threshold: float = 0.60,
    ) -> None:
        if not (0.0 <= min_confidence_threshold <= 1.0):
            raise ValueError(
                f"'min_confidence_threshold' must be in [0, 1], "
                f"got {min_confidence_threshold!r}"
            )
        # _lock guards _acquisition_scheduler, _health_monitor, _watchlist_graph
        # attributes when AutoResearchPipeline is shared across threads.
        self._lock  = threading.Lock()
        self._min_confidence = min_confidence_threshold

        # -- Build defaults lazily to avoid heavy imports at module load time --
        self._health_monitor        = health_monitor
        self._watchlist_graph       = watchlist_graph
        self._acquisition_scheduler = acquisition_scheduler
        self._multimodal_analyzer   = multimodal_analyzer

        # Resolve the quality gate
        self._quality_gate = quality_gate or self._make_default_quality_gate()

        # Resolve the IndexingPipeline (inject quality gate if not already set)
        if pipeline is not None:
            self._pipeline = pipeline
        else:
            self._pipeline = self._make_default_pipeline()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, query: str, tenant_id: str = "default") -> ResearchReport:
        """Execute the full research cycle for *query*.

        Steps
        -----
        1. Acquire content items via the ``AcquisitionScheduler`` (or generate
           synthetic items from the query when no scheduler is attached).
        2. Process the batch through ``IndexingPipeline.process_batch()``,
           which indexes chunks, clusters events, de-duplicates across source
           families, and runs the quality gate.
        3. Build a ``GroundedSummary`` for the query topic.
        4. Retrieve evidence chunks from the ``ChunkStore``.
        5. Collect watchlist gap count and SLO health status.
        6. Assemble and return a ``ResearchReport``.

        Any individual step failure is caught, logged at WARNING level, and
        must not abort the overall run.

        Args:
            query:     Free-text research query (non-empty).
            tenant_id: Tenant context (defaults to ``"default"``).

        Returns:
            ``ResearchReport`` — never raises.

        Raises:
            ValueError: If *query* is empty or *tenant_id* is empty.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("'query' must be a non-empty string")
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("'tenant_id' must be a non-empty string")

        t0 = time.perf_counter()
        report = ResearchReport(query=query, tenant_id=tenant_id)

        # ── Step 1: acquire content items ─────────────────────────────
        with self._lock:
            scheduler = self._acquisition_scheduler
            wg        = self._watchlist_graph
            mon       = self._health_monitor
        items = self._acquire_items(query, scheduler=scheduler)

        # ── Step 2: process batch ─────────────────────────────────────
        indexing_result = None
        try:
            indexing_result = await self._pipeline.process_batch(items, tenant_id=tenant_id)
            report.chunks_indexed = indexing_result.stats.chunks_indexed
            report.quality_gate_outcomes = list(indexing_result.quality_gate_results)
        except Exception as exc:
            logger.warning("AutoResearchPipeline.run: process_batch failed: %s", exc)

        # ── Step 3: build grounded summary ────────────────────────────
        if indexing_result is not None:
            try:
                summary = self._pipeline.build_grounded_summary(
                    indexing_result, topic=query, tenant_id=tenant_id
                )
                if summary is not None:
                    report.grounded_summaries = [summary]
                    report.confidence_score_mean = float(summary.confidence_score)
            except Exception as exc:
                logger.warning(
                    "AutoResearchPipeline.run: build_grounded_summary failed: %s", exc
                )

        # ── Step 4: retrieve evidence chunks ──────────────────────────
        try:
            store = self._pipeline._store  # type: ignore[attr-defined]
            all_chunks = []
            if indexing_result is not None:
                for pr in indexing_result.pipeline_results:
                    chunks = store.get_by_observation(str(pr.content_item_id))
                    all_chunks.extend(chunks)
            report.evidence_chunks = all_chunks
        except Exception as exc:
            logger.warning("AutoResearchPipeline.run: chunk retrieval failed: %s", exc)

        # ── Step 5a: watchlist gap count ──────────────────────────────
        if wg is not None:
            try:
                cov = wg.coverage_report()
                report.watchlist_gap_count = cov.nodes_at_risk
            except Exception as exc:
                logger.warning(
                    "AutoResearchPipeline.run: watchlist_graph.coverage_report failed: %s",
                    exc,
                )

        # ── Step 5b: SLO health status ────────────────────────────────
        if mon is not None:
            try:
                health = mon.health_report()
                report.slo_health_status = health.overall_status
            except Exception as exc:
                logger.warning(
                    "AutoResearchPipeline.run: health_monitor.health_report failed: %s",
                    exc,
                )

        # ── Step 6: compute derived metrics ───────────────────────────
        if report.quality_gate_outcomes:
            failed = sum(1 for q in report.quality_gate_outcomes if not q.passed)
            report.quality_gate_rejection_rate = failed / len(report.quality_gate_outcomes)

        report.wall_s = time.perf_counter() - t0
        logger.info(
            "AutoResearchPipeline.run: query=%r tenant=%r "
            "wall_s=%.2f chunks=%d summaries=%d gate_outcomes=%d",
            query, tenant_id, report.wall_s, report.chunks_indexed,
            len(report.grounded_summaries), len(report.quality_gate_outcomes),
        )
        return report

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pipeline(self) -> Any:
        """The underlying ``IndexingPipeline``."""
        return self._pipeline

    @property
    def health_monitor(self) -> Optional[Any]:
        """The attached ``PipelineHealthMonitor``."""
        return self._health_monitor

    @property
    def watchlist_graph(self) -> Optional[Any]:
        """The attached ``WatchlistGraph``."""
        return self._watchlist_graph

    @property
    def acquisition_scheduler(self) -> Optional[Any]:
        """The attached ``AcquisitionScheduler``."""
        return self._acquisition_scheduler

    @property
    def quality_gate(self) -> Optional[Any]:
        """The attached ``QualityGate``."""
        return self._quality_gate

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_default_quality_gate(self) -> Any:
        try:
            from app.ingestion.indexing_pipeline import QualityGate
            return QualityGate(min_confidence=self._min_confidence)
        except Exception as exc:
            logger.warning("AutoResearchPipeline: could not create QualityGate: %s", exc)
            return None

    def _make_default_pipeline(self) -> Any:
        from app.ingestion.indexing_pipeline import IndexingPipeline
        return IndexingPipeline(
            watchlist_graph=self._watchlist_graph,
            health_monitor=self._health_monitor,
            multimodal_analyzer=self._multimodal_analyzer,
            quality_gate=self._quality_gate,
        )

    def _acquire_items(self, query: str, *, scheduler: Optional[Any] = None) -> List[Any]:
        """Acquire content items for *query*.

        When an ``AcquisitionScheduler`` is attached, retrieves the top
        sources and creates synthetic ``ContentItem``-like objects seeded
        with the query text.  When no scheduler is attached, creates a
        single synthetic item directly from the query so the pipeline can
        still function in standalone mode.

        Args:
            query:     The research query.
            scheduler: Pre-resolved scheduler (avoids double lock acquisition).

        Returns:
            List of content items suitable for ``IndexingPipeline.process_batch()``.
        """
        items = []
        try:
            if scheduler is not None:
                sources = scheduler.next_batch(10)
                for src in sources:
                    items.append(_make_synthetic_item(query, source_id=src.source_id))
            if not items:
                items.append(_make_synthetic_item(query))
        except Exception as exc:
            logger.warning("AutoResearchPipeline._acquire_items: failed: %s", exc)
            items = [_make_synthetic_item(query)]
        return items


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _make_synthetic_item(query: str, source_id: str = "auto-research") -> Any:
    """Create a minimal content item from *query* for pipeline ingestion.

    Returns a ``ContentItem``-like object compatible with
    ``ContentPipelineRouter.route()``.

    Args:
        query:     Free-text research query used as the item body.
        source_id: Logical source identifier stamped on the item.

    Returns:
        A ``ContentItem`` (or duck-typed equivalent).
    """
    # Extract simple keywords from query for the ``topics`` field
    _stop = frozenset("a an the and or but in on at to for of with is are was were".split())
    _topics = [
        w.strip(".,!?;:")
        for w in query.split()
        if len(w) > 3 and w.lower() not in _stop
    ][:10]

    try:
        from app.core.models import ContentItem, SourcePlatform, MediaType
        from uuid import uuid4
        from datetime import datetime, timezone
        return ContentItem(
            user_id=uuid4(),
            source_platform=SourcePlatform.REDDIT,
            source_id=source_id,
            source_url=f"https://auto-research.internal/{source_id}",
            title=query[:200],
            raw_text=query,
            media_type=MediaType.TEXT,
            published_at=datetime.now(timezone.utc),
            topics=_topics,
        )
    except Exception as exc:
        logger.debug("_make_synthetic_item: ContentItem fallback: %s", exc)
    try:
        from app.domain.raw_models import RawObservation
        from app.core.models import SourcePlatform, MediaType
        from uuid import uuid4
        from datetime import datetime, timezone
        return RawObservation(
            user_id=uuid4(),
            source_platform=SourcePlatform.REDDIT,
            source_id=source_id,
            source_url=f"https://auto-research.internal/{source_id}",
            title=query[:200],
            raw_text=query,
            media_type=MediaType.TEXT,
            published_at=datetime.now(timezone.utc),
        )
    except Exception as exc:
        logger.debug("_make_synthetic_item: RawObservation fallback: %s", exc)
    # Final fallback: plain dict (ContentPipelineRouter will route it as-is)
    return {"source_id": source_id, "raw_text": query, "title": query[:200]}

