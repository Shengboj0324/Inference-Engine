# AutoResearchPipeline — Multi-Phase Research Orchestration

## Overview

`AutoResearchPipeline` (`app/research/auto_research_pipeline.py`) is an async orchestrator that sequences every inference subsystem into a single `run()` coroutine and returns a fully-typed `ResearchReport`. It is the entry point for end-to-end signal research: acquire → ingest → summarise → gate → audit.

## `run()` — Seven Steps

```python
report = await pipeline.run(query: str, tenant_id: str = "default") -> ResearchReport
```

Raises `ValueError` if `query` or `tenant_id` is empty. All shared subsystem references are snapshotted under `threading.Lock()` at the top of `run()` to make the method safe for concurrent callers.

| Step | Action | Failure handling |
|---|---|---|
| 1 | `AcquisitionScheduler.next_batch(n)` — fetch prioritised source batch | `logger.warning`; empty batch → skip ingestion |
| 2 | `IndexingPipeline.process_batch(items, tenant_id)` — 7-phase ingestion | `logger.error`; `IndexingResult` still returned with `errors` dict populated |
| 3 | `build_grounded_summary(result, topic=query, tenant_id=tenant_id)` — citation-backed summary | `logger.warning`; `grounded_summaries` left empty |
| 4 | `QualityGate.evaluate()` per summary — filter below `min_confidence` | All outcomes collected into `quality_gate_outcomes` regardless of pass/fail |
| 5 | `WatchlistGraph.coverage_report(query)` — count unwatched entities in results | `logger.debug`; `watchlist_gap_count` defaults to `0` |
| 6 | `PipelineHealthMonitor.health_report()` — read SLO status | `logger.debug`; `slo_health_status` defaults to `None` |
| 7 | Assemble and return `ResearchReport` | Always returns; never raises after input validation |

## `ResearchReport` Contract

```python
@dataclass
class ResearchReport:
    query: str
    tenant_id: str
    chunks_indexed: int                         # ≥ 0
    confidence_score_mean: float                # ∈ [0.0, 1.0]
    quality_gate_rejection_rate: float          # ∈ [0.0, 1.0]
    quality_gate_outcomes: List[QualityGateResult]
    watchlist_gap_count: int                    # ≥ 0
    slo_health_status: Optional[SLOStatus]      # None if monitor not wired
    grounded_summaries: List[GroundedSummary]
    wall_s: float                               # total run() wall time in seconds
    generated_at: datetime                      # UTC-aware
```

`confidence_score_mean` is the arithmetic mean of `GroundedSummary.confidence_score` across all summaries that passed the quality gate. `quality_gate_rejection_rate = rejected / total` over all evaluated summaries.

## Wiring All Subsystems

```python
from app.research.auto_research_pipeline import AutoResearchPipeline
from app.ingestion.indexing_pipeline import IndexingPipeline, QualityGate
from app.source_intelligence.source_registry import AcquisitionScheduler, SourceRegistryStore
from app.intelligence.multimodal import MultimodalAnalyzer
from app.intelligence.health_monitor import PipelineHealthMonitor
from app.personalization.watchlist_graph import WatchlistGraph
from app.entity_resolution.cross_source_deduper import CrossSourceDeduper
from app.intelligence.retrieval.chunk_store import ChunkStore

store = ChunkStore()
pipeline = IndexingPipeline(
    chunk_store=store,
    multimodal_analyzer=MultimodalAnalyzer(),
    watchlist_graph=WatchlistGraph(user_id="user-uuid"),
    health_monitor=PipelineHealthMonitor(cb_open_threshold=5),
    deduper=CrossSourceDeduper(title_threshold=0.65),
    quality_gate=QualityGate(min_confidence=0.60),
)

research = AutoResearchPipeline(
    pipeline=pipeline,
    acquisition_scheduler=AcquisitionScheduler(SourceRegistryStore()),
    watchlist_graph=WatchlistGraph(user_id="user-uuid"),
    health_monitor=PipelineHealthMonitor(cb_open_threshold=5),
    quality_gate=QualityGate(min_confidence=0.60),
    multimodal_analyzer=MultimodalAnalyzer(),
)

report = await research.run("AI safety alignment 2025", tenant_id="safety-lab")
```

All parameters are optional. Unset subsystems are skipped gracefully; the corresponding `ResearchReport` field is set to its zero value.

## Per-Tenant Isolation

`process_batch(items, tenant_id)` indexes chunks into `IndexingPipeline._tenant_stores[tenant_id]` (a per-tenant `ChunkStore` SQLite partition, created on first access). `build_grounded_summary` must receive the same `tenant_id` to resolve the correct partition — passing the wrong tenant returns an empty `source_attributions` list because the default store contains no chunks for that tenant.

## `GroundedSummary` Fields Relevant to Research Output

| Field | Type | Description |
|---|---|---|
| `what_happened` | `str` | Primary narrative synthesised from `ChunkRecord` text |
| `why_it_matters` | `str \| None` | Contextual significance |
| `source_attributions` | `List[EvidenceSource]` | Text citations (`observation_id`) + multimodal citations (`mm-img-*` / `mm-vid-*`) |
| `contradictions` | `List[str]` | Inter-source conflicts detected during synthesis |
| `uncertainty_annotations` | `List[str]` | Explicit epistemic hedges |
| `overall_uncertainty_score` | `float` | Aggregate uncertainty in [0, 1] |
| `confidence_score` | `float` | Summary-level confidence after calibration |
| `source_count` | `int` | Number of distinct `EvidenceSource` objects |

