"""End-to-end ingestion and indexing pipeline.

``IndexingPipeline`` is the integration bus that wires together every previously
independent domain module into a single, coherent processing path:

    ContentItem(s)
        │
        ▼  ContentPipelineRouter  (source-family dispatch)
        │
        ▼  ChunkStore.chunk_text  (auto-index for RAG)
        │
        ▼  EventClusterer         (group cross-source items by event proximity)
        │
        ▼  CrossSourceDeduper     (remove within-bundle duplicates)
        │
        ▼  IndexingResult         (bundles + per-item pipeline results + stats)

Design principles
-----------------
- **Fault-isolated per item** — a failure processing item N never aborts items
  N+1…M.  Errors are recorded in ``IndexingResult.errors`` and the pipeline
  continues.
- **Async-friendly** — ``process_batch()`` is a coroutine so the caller
  controls the event loop.  CPU-bound clustering/deduping runs synchronously
  inside the coroutine (it is cheap).
- **Observable** — ``IndexingStats`` captures throughput, error counts, chunk
  counts, and timing so the ``PipelineHealthMonitor`` can consume it directly.
- **Pluggable stores** — callers may inject a shared ``ChunkStore`` instance;
  when none is supplied a private store is created (suitable for testing).

Usage::

    from app.ingestion.indexing_pipeline import IndexingPipeline

    pipeline = IndexingPipeline()
    result = await pipeline.process_batch(items)
    print(result.stats)          # IndexingStats
    print(result.bundles)        # List[EventBundle] after dedup
    print(result.pipeline_results)  # List[IntelligencePipelineResult]
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.entity_resolution.cross_source_deduper import CrossSourceDeduper
from app.entity_resolution.event_clusterer import EventClusterer
from app.entity_resolution.models import EventBundle
from app.ingestion.content_pipeline_router import ContentPipelineRouter
from app.ingestion.pipeline_result import IntelligencePipelineResult, PipelineStatus
from app.intelligence.retrieval.chunk_store import ChunkStore

logger = logging.getLogger(__name__)

# WatchlistGraph is an optional dependency — imported lazily to avoid pulling
# in the full personalization stack when not needed.
_WatchlistGraph = None

# GroundedSummaryBuilder is an optional dependency — imported lazily.
_GroundedSummaryBuilder = None
_SynthesisRequest = None
_EvidenceSource = None
_GroundedSummary = None


def _get_summary_builder_cls():
    global _GroundedSummaryBuilder, _SynthesisRequest, _EvidenceSource, _GroundedSummary
    if _GroundedSummaryBuilder is None:
        from app.summarization.grounded_summary_builder import GroundedSummaryBuilder
        from app.summarization.models import SynthesisRequest, EvidenceSource, GroundedSummary
        _GroundedSummaryBuilder = GroundedSummaryBuilder
        _SynthesisRequest = SynthesisRequest
        _EvidenceSource = EvidenceSource
        _GroundedSummary = GroundedSummary
    return _GroundedSummaryBuilder, _SynthesisRequest, _EvidenceSource

def _get_watchlist_graph_cls():
    global _WatchlistGraph
    if _WatchlistGraph is None:
        from app.personalization.watchlist_graph import WatchlistGraph
        _WatchlistGraph = WatchlistGraph
    return _WatchlistGraph


# ---------------------------------------------------------------------------
# Stats / result models
# ---------------------------------------------------------------------------

@dataclass
class IndexingStats:
    """Throughput and quality counters for one ``process_batch()`` call.

    Attributes
    ----------
    total_items:          Number of ``ContentItem``s submitted.
    routed_ok:            Items that produced SUCCESS or PARTIAL results.
    route_errors:         Items that produced FAILED results.
    chunks_indexed:       Total text chunks stored in ``ChunkStore``.
    bundles_formed:       Number of ``EventBundle``s after clustering.
    bundles_after_dedup:  Bundles after cross-source deduplication.
    duplicates_removed:   Total duplicate items removed across all bundles.
    watchlist_nodes_updated: Distinct watchlist nodes that received a new
                          coverage entry from this batch.  0 when no
                          ``WatchlistGraph`` is attached.
    wall_s:               Total wall-clock seconds for the batch.
    """
    total_items:           int   = 0
    routed_ok:             int   = 0
    route_errors:          int   = 0
    chunks_indexed:        int   = 0
    bundles_formed:        int   = 0
    bundles_after_dedup:   int   = 0
    duplicates_removed:    int   = 0
    watchlist_nodes_updated: int = 0
    wall_s:                float = 0.0

    @property
    def throughput_items_per_s(self) -> float:
        return self.total_items / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return self.route_errors / self.total_items if self.total_items > 0 else 0.0


@dataclass
class IndexingResult:
    """Aggregated output of one ``IndexingPipeline.process_batch()`` call.

    Attributes
    ----------
    bundles:          Deduplicated ``EventBundle``s formed from this batch.
    pipeline_results: Per-item ``IntelligencePipelineResult`` objects
                      (one per input ``ContentItem``, including failed ones).
    stats:            Throughput and quality counters.
    errors:           ``{item_id: error_message}`` for items that failed.
    produced_at:      UTC timestamp when the result was assembled.
    """
    bundles:          List[EventBundle]               = field(default_factory=list)
    pipeline_results: List[IntelligencePipelineResult] = field(default_factory=list)
    stats:            IndexingStats                   = field(default_factory=IndexingStats)
    errors:           Dict[str, str]                  = field(default_factory=dict)
    produced_at:      datetime                        = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class IndexingPipeline:
    """End-to-end ingestion and indexing pipeline.

    Args:
        router:       ``ContentPipelineRouter`` to use.  A default instance is
                      created when ``None``.
        chunk_store:  ``ChunkStore`` to index into.  A private store is created
                      when ``None`` (useful for isolated testing).
        clusterer:    ``EventClusterer`` for grouping.  Default parameters used
                      when ``None``.
        deduper:      ``CrossSourceDeduper``.  Default parameters used when
                      ``None``.
        chunk_size:   Character length per chunk when auto-indexing.
        chunk_overlap: Character overlap between consecutive chunks.
        route_timeout_s: Per-item timeout passed to ``ContentPipelineRouter``.
        max_concurrency: Maximum number of items routed concurrently.
    """

    def __init__(
        self,
        router:           Optional[ContentPipelineRouter] = None,
        chunk_store:      Optional[ChunkStore]            = None,
        clusterer:        Optional[EventClusterer]        = None,
        deduper:          Optional[CrossSourceDeduper]    = None,
        watchlist_graph:  Optional[Any]                   = None,
        health_monitor:   Optional[Any]                   = None,
        trust_scorer:     Optional[Any]                   = None,
        chunk_size:       int   = 800,
        chunk_overlap:    int   = 100,
        route_timeout_s:  float = 60.0,
        max_concurrency:  int   = 8,
    ) -> None:
        self._router    = router    or ContentPipelineRouter()
        self._store     = chunk_store or ChunkStore()
        self._clusterer = clusterer or EventClusterer()
        self._deduper   = deduper   or CrossSourceDeduper()
        # Optional WatchlistGraph — coverage is updated automatically after
        # each batch when provided.  Accepts any object with a
        # ``record_coverage_from_result(result)`` method so tests can inject
        # a plain mock without importing the real WatchlistGraph.
        self._watchlist_graph = watchlist_graph
        # Optional PipelineHealthMonitor — receives watchlist gap count after
        # each batch.  Accepts any object with
        # ``record_watchlist_gap_count(n: int)`` method.
        self._health_monitor  = health_monitor
        # Optional SourceTrustScorer — when provided, the composite trust score
        # for each result's source_family is stamped into ChunkRecord metadata
        # and into the health monitor (very low trust triggers a warning).
        self._trust_scorer    = trust_scorer
        # Per-tenant ChunkStore registry.  The injected ``chunk_store`` (or the
        # auto-created default) becomes the "default" tenant's store.
        self._tenant_stores: Dict[str, ChunkStore] = {"default": self._store}
        self._chunk_size      = chunk_size
        self._chunk_overlap   = chunk_overlap
        self._route_timeout   = route_timeout_s
        self._max_concurrency = max_concurrency
        # Semaphore is created lazily inside process_batch() so that no event
        # loop is required at construction time (required for Python 3.9 compat).
        self._semaphore: Optional[asyncio.Semaphore] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def process_batch(
        self,
        items: List[Any],
        tenant_id: str = "default",
    ) -> IndexingResult:
        """Route, index, cluster, and deduplicate a batch of ``ContentItem``s.

        Args:
            items:     List of ``ContentItem`` objects (any length including empty).
            tenant_id: Tenant identifier used to select (or create) the per-tenant
                       ``ChunkStore`` partition for this batch.  Defaults to
                       ``"default"`` for single-tenant deployments.

        Returns:
            ``IndexingResult`` — never raises; errors are recorded per-item.
        """
        # Create the semaphore lazily so construction works without a running loop.
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrency)

        # Ensure this tenant has a ChunkStore partition
        store = self._get_or_create_tenant_store(tenant_id)

        t0 = time.perf_counter()
        result = IndexingResult(stats=IndexingStats(total_items=len(items)))

        if not items:
            result.stats.wall_s = time.perf_counter() - t0
            return result

        # ── Phase 1: route all items concurrently ─────────────────────
        pipeline_results = await self._route_all(items, result, tenant_id=tenant_id)
        result.pipeline_results = pipeline_results

        # ── Phase 2: auto-index chunks into per-tenant ChunkStore ──────
        self._index_chunks(pipeline_results, result, store=store)

        # ── Phase 2b: auto-update WatchlistGraph coverage ──────────────
        if self._watchlist_graph is not None:
            self._update_watchlist(pipeline_results, result)

        # ── Phase 3: build clusterer input dicts ──────────────────────
        cluster_inputs = self._build_cluster_inputs(pipeline_results)

        # ── Phase 4: cluster into EventBundles ────────────────────────
        bundles = self._cluster(cluster_inputs, result)

        # ── Phase 5: deduplicate within each bundle ────────────────────
        bundles = self._deduplicate(bundles, result)

        # ── Phase 6: report watchlist gap count to health monitor ──────
        if self._watchlist_graph is not None and self._health_monitor is not None:
            self._report_watchlist_health()

        result.bundles = bundles
        result.stats.wall_s = time.perf_counter() - t0
        logger.info(
            "IndexingPipeline: %d items → %d bundles | "
            "%d chunks | errors=%d | %.2fs",
            result.stats.total_items,
            result.stats.bundles_after_dedup,
            result.stats.chunks_indexed,
            result.stats.route_errors,
            result.stats.wall_s,
        )
        return result

    @property
    def chunk_store(self) -> ChunkStore:
        """The default (``"default"`` tenant) ``ChunkStore``."""
        return self._store

    @property
    def trust_scorer(self) -> Optional[Any]:
        """The ``SourceTrustScorer`` injected at construction, or ``None``."""
        return self._trust_scorer

    def tenant_store(self, tenant_id: str) -> Optional[ChunkStore]:
        """Return the ``ChunkStore`` for *tenant_id*, or ``None`` if not yet created."""
        return self._tenant_stores.get(tenant_id)

    def tenant_ids(self) -> List[str]:
        """Return all tenant IDs that have an active ``ChunkStore`` partition."""
        return list(self._tenant_stores.keys())

    def _get_or_create_tenant_store(self, tenant_id: str) -> ChunkStore:
        """Return (or lazily create) the ``ChunkStore`` for *tenant_id*."""
        if tenant_id not in self._tenant_stores:
            self._tenant_stores[tenant_id] = ChunkStore()
        return self._tenant_stores[tenant_id]

    @property
    def watchlist_graph(self) -> Optional[Any]:
        """The attached ``WatchlistGraph`` (``None`` when not configured)."""
        return self._watchlist_graph

    @property
    def health_monitor(self) -> Optional[Any]:
        """The attached ``PipelineHealthMonitor`` (``None`` when not configured)."""
        return self._health_monitor

    # ------------------------------------------------------------------
    # Phase 1: concurrent routing
    # ------------------------------------------------------------------

    async def _route_all(
        self,
        items: List[Any],
        result: IndexingResult,
        *,
        tenant_id: str = "default",
    ) -> List[IntelligencePipelineResult]:
        async def _route_one(item: Any) -> IntelligencePipelineResult:
            async with self._semaphore:
                pr = await self._router.route(
                    item,
                    timeout_s=self._route_timeout,
                    tenant_id=tenant_id,
                )
                return pr

        tasks = [asyncio.create_task(_route_one(item)) for item in items]
        pipeline_results: List[IntelligencePipelineResult] = []
        for task in asyncio.as_completed(tasks):
            try:
                pr = await task
                pipeline_results.append(pr)
                if pr.status == PipelineStatus.FAILED:
                    result.stats.route_errors += 1
                    result.errors[str(pr.content_item_id)] = (
                        "; ".join(pr.extraction_warnings) or "Unknown failure"
                    )
                else:
                    result.stats.routed_ok += 1
            except Exception as exc:  # noqa: BLE001
                result.stats.route_errors += 1
                logger.exception("IndexingPipeline: unexpected routing error: %s", exc)
        return pipeline_results

    # ------------------------------------------------------------------
    # Phase 6: watchlist health reporting
    # ------------------------------------------------------------------

    def _report_watchlist_health(self) -> None:
        """Compute gap count from the WatchlistGraph and send to the HealthMonitor.

        Errors are caught and logged so a bad graph or monitor never aborts the
        indexing batch.
        """
        try:
            report = self._watchlist_graph.coverage_report()
            self._health_monitor.record_watchlist_gap_count(report.nodes_at_risk)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "IndexingPipeline: failed to report watchlist health: %s", exc
            )

    # ------------------------------------------------------------------
    # Phase 2: auto-indexing
    # ------------------------------------------------------------------

    def _update_watchlist(
        self,
        pipeline_results: List[IntelligencePipelineResult],
        result: IndexingResult,
    ) -> None:
        """Push coverage observations into the attached ``WatchlistGraph``.

        For every actionable pipeline result, calls
        ``watchlist_graph.record_coverage_from_result(pr)`` which matches the
        result's entities against watched nodes and records a coverage entry.

        Errors are caught and logged so a misbehaving graph never aborts the
        indexing batch.

        The counter ``result.stats.watchlist_nodes_updated`` is incremented by
        the number of *distinct* watched nodes that received at least one new
        coverage entry in this call.
        """
        graph = self._watchlist_graph
        if graph is None:
            return

        # Snapshot currently-watched node IDs so we can count distinct updates
        try:
            watched_before = set(n.node_id for n in graph.watched_nodes())
        except Exception:
            watched_before = set()

        updated: set = set()
        for pr in pipeline_results:
            if not pr.is_actionable():
                continue
            try:
                graph.record_coverage_from_result(pr)
                # Track which watched nodes this result covered
                for entity in pr.entities:
                    if entity in watched_before:
                        updated.add(entity)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "IndexingPipeline: WatchlistGraph update failed for item %s: %s",
                    pr.content_item_id, exc,
                )

        result.stats.watchlist_nodes_updated += len(updated)

    def _index_chunks(
        self,
        pipeline_results: List[IntelligencePipelineResult],
        result: IndexingResult,
        *,
        store: Optional[ChunkStore] = None,
    ) -> None:
        target_store = store or self._store
        for pr in pipeline_results:
            if not pr.is_actionable():
                continue
            text = pr.all_text_for_chunking().strip()
            if not text:
                continue
            # ── Trust score for this source family ──────────────────────
            trust_score: Optional[float] = None
            if self._trust_scorer is not None:
                try:
                    ts = self._trust_scorer.score(pr.source_family)
                    trust_score = ts.composite
                except Exception as exc:
                    logger.warning(
                        "IndexingPipeline: trust_scorer.score failed for "
                        "source_family=%r item=%s: %s",
                        pr.source_family, pr.content_item_id, exc,
                    )
            try:
                chunk_meta: Dict[str, Any] = {
                    "signal_type": pr.signal_type,
                    "confidence":  pr.confidence,
                    "result_id":   str(pr.result_id),
                    "tenant_id":   pr.tenant_id,
                }
                if trust_score is not None:
                    chunk_meta["trust_score"] = trust_score
                ids = target_store.chunk_text(
                    observation_id=str(pr.content_item_id),
                    text=text,
                    source_family=pr.source_family,
                    chunk_size=self._chunk_size,
                    overlap=self._chunk_overlap,
                    metadata=chunk_meta,
                )
                result.stats.chunks_indexed += len(ids)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "IndexingPipeline: chunk indexing failed for item %s: %s",
                    pr.content_item_id, exc,
                )

    # ------------------------------------------------------------------
    # Phase 3: build clusterer input dicts
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cluster_inputs(
        pipeline_results: List[IntelligencePipelineResult],
    ) -> List[Dict[str, Any]]:
        """Convert pipeline results to the dict schema EventClusterer expects."""
        cluster_inputs: List[Dict[str, Any]] = []
        for pr in pipeline_results:
            if not pr.is_actionable():
                continue
            cluster_inputs.append({
                "source_id":    str(pr.content_item_id),
                "title":        pr.summary[:200] if pr.summary else f"Item {pr.content_item_id}",
                # EventClusterer._parse_dt() requires an ISO-8601 string
                "published_at": pr.produced_at.isoformat(),
                "entities":     pr.entities,
                "platform":     pr.source_family,
                "trust_score":  pr.confidence,
                # Extras for deduper / digest modes
                "claims":       pr.claims,
                "source_ids":   [str(pr.content_item_id)],
                "signal_type":  pr.signal_type or "unknown",
                "importance":   pr.confidence,
            })
        return cluster_inputs

    # ------------------------------------------------------------------
    # Phase 4: clustering
    # ------------------------------------------------------------------

    def _cluster(
        self, cluster_inputs: List[Dict[str, Any]], result: IndexingResult
    ) -> List[EventBundle]:
        if not cluster_inputs:
            return []
        try:
            bundles = self._clusterer.cluster(cluster_inputs)
            result.stats.bundles_formed = len(bundles)
            return bundles
        except Exception as exc:  # noqa: BLE001
            logger.error("IndexingPipeline: clustering failed: %s", exc)
            result.stats.bundles_formed = 0
            return []

    # ------------------------------------------------------------------
    # Grounded summary synthesis
    # ------------------------------------------------------------------

    def build_grounded_summary(
        self,
        indexing_result: "IndexingResult",
        topic: str,
        *,
        summary_builder: Optional[Any] = None,
        min_source_trust: float = 0.0,
        max_claims: int = 10,
        who_it_affects: Optional[List[str]] = None,
    ) -> Optional[Any]:
        """Build a ``GroundedSummary`` from an ``IndexingResult``.

        Converts each actionable ``IntelligencePipelineResult`` into an
        ``EvidenceSource`` (using ``self.trust_scorer`` when available), then
        calls :class:`~app.summarization.grounded_summary_builder.GroundedSummaryBuilder`
        to produce a ``GroundedSummary`` that contains attribution, contradictions,
        and uncertainty annotations.

        Args:
            indexing_result: The result of a prior :meth:`process_batch` call.
            topic:           Short label describing the batch topic (non-empty).
            summary_builder: Optional pre-built ``GroundedSummaryBuilder``.
                             A default instance is created if not provided.
            min_source_trust: Sources whose trust score falls below this
                              threshold are excluded.  In [0, 1].  Default 0.0
                              (include all).
            max_claims:      Maximum number of claims to extract.
            who_it_affects:  Optional stakeholder list forwarded to the summary.

        Returns:
            A ``GroundedSummary``, or ``None`` if no actionable results exist.

        Raises:
            TypeError:  If *indexing_result* is not an ``IndexingResult`` or
                        *topic* is not a non-empty string.
            ValueError: If *min_source_trust* is outside [0, 1].
        """
        if not isinstance(indexing_result, IndexingResult):
            raise TypeError(
                f"'indexing_result' must be IndexingResult, got {type(indexing_result)!r}"
            )
        if not isinstance(topic, str) or not topic.strip():
            raise TypeError("'topic' must be a non-empty string")
        if not (0.0 <= min_source_trust <= 1.0):
            raise ValueError(
                f"'min_source_trust' must be in [0, 1], got {min_source_trust!r}"
            )

        BuilderCls, SynthesisRequestCls, EvidenceSourceCls = _get_summary_builder_cls()

        # Build EvidenceSource objects from actionable pipeline results
        sources = []
        for pr in indexing_result.pipeline_results:
            if not pr.is_actionable():
                continue
            snippet = pr.all_text_for_chunking().strip() or pr.summary or ""
            # Determine trust score
            trust = max(0.0, min(1.0, pr.confidence)) if pr.confidence else 0.5
            if self._trust_scorer is not None:
                try:
                    ts = self._trust_scorer.score(pr.source_family)
                    trust = ts.composite
                except Exception as exc:
                    logger.warning(
                        "IndexingPipeline.build_grounded_summary: trust_scorer "
                        "failed for source=%r item=%s: %s",
                        pr.source_family, pr.content_item_id, exc,
                    )
            src = EvidenceSourceCls(
                source_id=str(pr.content_item_id),
                title=pr.summary[:200] if pr.summary else topic,
                platform=pr.source_family,
                trust_score=trust,
                published_at=pr.produced_at,
                content_snippet=snippet[:2000],
            )
            sources.append(src)

        if not sources:
            logger.warning(
                "IndexingPipeline.build_grounded_summary: no actionable results "
                "for topic=%r; returning None", topic,
            )
            return None

        request = SynthesisRequestCls(
            topic=topic,
            sources=sources,
            min_source_trust=min_source_trust,
            max_claims=max_claims,
            who_it_affects=who_it_affects or [],
        )

        builder = summary_builder or BuilderCls()
        try:
            summary = builder.build(request)
            logger.info(
                "IndexingPipeline.build_grounded_summary: topic=%r "
                "sources=%d confidence=%.3f",
                topic, len(sources), summary.confidence_score,
            )
            return summary
        except Exception as exc:
            logger.error(
                "IndexingPipeline.build_grounded_summary: builder.build failed "
                "for topic=%r: %s", topic, exc,
            )
            return None

    # ------------------------------------------------------------------
    # Phase 5: deduplication
    # ------------------------------------------------------------------

    def _deduplicate(
        self, bundles: List[EventBundle], result: IndexingResult
    ) -> List[EventBundle]:
        if not bundles:
            return []
        try:
            dedup_results = self._deduper.deduplicate_batch(bundles)
            total_removed = sum(len(dr.removed_ids) for dr in dedup_results)
            result.stats.duplicates_removed  = total_removed
            result.stats.bundles_after_dedup = len(dedup_results)
            # Rebuild bundles with duplicate_ids updated from dedup results
            deduped_bundles: List[EventBundle] = []
            for bundle, dr in zip(bundles, dedup_results):
                deduped_bundles.append(
                    bundle.model_copy(update={"duplicate_ids": list(dr.removed_ids)})
                )
            return deduped_bundles
        except Exception as exc:  # noqa: BLE001
            logger.error("IndexingPipeline: deduplication failed: %s", exc)
            result.stats.bundles_after_dedup = len(bundles)
            result.stats.duplicates_removed  = 0
            return bundles

