# Source Acquisition — AcquisitionScheduler, SourceRegistryStore, and WatchlistGraph

## SourceRegistryStore and SourceSpec

`SourceRegistryStore` (`app/source_intelligence/source_registry.py`) holds the registered source pool. Each source is a `SourceSpec`:

```python
@dataclass
class SourceSpec:
    source_id: str
    platform: SourcePlatform     # enum; 19 values including ARXIV, GITHUB_RELEASES, REDDIT, YOUTUBE
    family: SourceFamily         # RESEARCH | DEVELOPER_RELEASE | SOCIAL | NEWS | MEDIA
    priority: float              # ∈ [0.0, 1.0]; determines fetch order
    authority_score: float = 0.5 # used by AcquisitionScheduler trust gate
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### `next_batch(n, min_priority=0.0) → List[SourceSpec]`

Returns the top-n sources sorted by descending `priority`, filtered to `priority ≥ min_priority`. Raises `ValueError` if `n ≤ 0`. All reads are protected by `threading.Lock()`.

```python
reg = SourceRegistryStore()
reg.register(SourceSpec("arxiv-cs", SourcePlatform.ARXIV, SourceFamily.RESEARCH, priority=0.85))
reg.register(SourceSpec("openai-blog", SourcePlatform.RSS, SourceFamily.NEWS, priority=0.95))

top3 = reg.next_batch(3, min_priority=0.5)   # sorted: openai-blog, arxiv-cs
ok   = reg.update_priority("arxiv-cs", 0.99)  # returns False if source_id unknown
```

---

## AcquisitionScheduler

`AcquisitionScheduler` wraps `SourceRegistryStore` with three operational safety mechanisms.

### 1. Exponential Back-off

`record_failure(source_id)` increments the failure counter and sets a cooldown window:

```
cooldown = min(base_backoff_s × 2^(failures−1), max_backoff_s)
eligible_after = now() + cooldown
```

`is_eligible(source_id)` returns `False` while `now() < eligible_after` OR `failures ≥ max_retries`. `record_success(source_id)` resets both counters to zero.

Default values: `base_backoff_s=30.0`, `max_retries=5`, `max_backoff_s=3600.0`.

### 2. Trust Gate

Sources whose `authority_score < min_authority_threshold` are excluded from `next_batch()` results regardless of their `priority`. Default `min_authority_threshold=0.0` (disabled). Set to e.g. `0.4` to exclude low-authority sources automatically.

### 3. PipelineHealthMonitor Integration

`record_failure` and `record_success` optionally forward events to a `PipelineHealthMonitor` instance. Monitor exceptions are caught and logged at `WARNING` level — a broken monitor never propagates back to the scheduler.

### Thread Safety

All scheduler state (`_failure_counts`, `_cooldown_until`, `_success_counts`) is protected by `threading.Lock()`. `next_batch()` takes a snapshot of the eligible source list under the lock.

### API

```python
from app.source_intelligence.source_registry import AcquisitionScheduler, SourceRegistryStore
from app.intelligence.health_monitor import PipelineHealthMonitor

scheduler = AcquisitionScheduler(
    registry=SourceRegistryStore(),
    base_backoff_s=30.0,
    max_retries=5,
    min_authority_threshold=0.4,
    health_monitor=PipelineHealthMonitor(cb_open_threshold=5),
)

batch = scheduler.next_batch(10)            # top-10 eligible sources
scheduler.record_failure("arxiv-cs")        # doubles cooldown, forwards to monitor
scheduler.record_success("openai-blog")     # resets back-off

print(scheduler.failure_count("arxiv-cs"))  # ≥ 0; thread-safe read
print(scheduler.is_eligible("arxiv-cs"))    # False until cooldown expires
```

---

## PipelineHealthMonitor

`PipelineHealthMonitor` (`app/intelligence/health_monitor.py`) tracks per-connector SLO compliance and exposes a circuit-breaker state machine per source.

### Circuit Breaker States

```
CLOSED → (N consecutive failures) → OPEN → (timeout_s seconds) → HALF_OPEN → (success) → CLOSED
                                                                             → (failure) → OPEN
```

`cb_open_threshold` (default 5) is the number of consecutive failures that trips OPEN. `health_report()` returns `SLOStatus.GREEN` (all CLOSED), `YELLOW` (any HALF_OPEN), or `RED` (any OPEN).

```python
monitor = PipelineHealthMonitor(cb_open_threshold=5, cb_half_open_timeout_s=60.0)
monitor.record_connector_failure("arxiv-cs")   # increments; may trip circuit
monitor.record_connector_success("arxiv-cs")   # resets to CLOSED
status = monitor.health_report()               # SLOStatus enum
```

---

## WatchlistGraph

`WatchlistGraph` (`app/personalization/watchlist_graph.py`) maintains a per-user directed graph of watched entities. It feeds into `IndexingPipeline` Phase 4 (coverage recording) and into `AutoResearchPipeline` Step 5 (gap counting).

```python
wg = WatchlistGraph(user_id="user-uuid")
wg.watch("openai",   node_type="entity")
wg.watch("deepmind", node_type="entity")

# After indexing, check which watched entities appeared in results
report = wg.coverage_report(query="AI safety alignment")
# report.gap_count: int — number of watched entities with no matching chunk
# report.covered: List[str]
# report.missing: List[str]
```

`IndexingPipeline` calls `wg.record_coverage(source_family, bundle_id)` in Phase 4 for each `EventBundle` produced by `EventClusterer`. `AutoResearchPipeline` reads `coverage_report().gap_count` and writes it to `ResearchReport.watchlist_gap_count`.

---

## Acquisition → Ingestion Flow

```
SourceRegistryStore.next_batch(n)
         │  sorted by priority, filtered by min_priority
         │  trust gate: authority_score ≥ min_authority_threshold
         │  back-off gate: is_eligible(source_id)
         ▼
AcquisitionScheduler.next_batch(n)  →  List[SourceSpec]
         │
   (fetch content per SourceSpec — connector layer)
         │
         ▼
 List[ContentItem]
         │
IndexingPipeline.process_batch(items, tenant_id)
         │
  Phase 1: route via ContentPipelineRouter
  Phase 2: chunk + MultimodalAnalyzer
  Phase 3: SourceTrustScorer
  Phase 4: EventClusterer + WatchlistGraph.record_coverage()
  Phase 5: CrossSourceDeduper.deduplicate_cross_bundle()
  Phase 6: GroundedSummaryBuilder
  Phase 7: QualityGate
         │
         ▼
   IndexingResult → ResearchReport (via AutoResearchPipeline)
```

