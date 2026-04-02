# Six-Tier Test Campaign

4,616 tests, 0 failures. Each tier targets a distinct failure mode.

## Run the Full Suite

```bash
python -m pytest tests/ --ignore=tests/llm/test_load.py -q
# → 4616 passed in ~94s

# Live LLM tests (requires OPENAI_API_KEY)
python -m pytest tests/llm/test_load.py -v
```

---

## Tier 1 — Unit / Functional (903 tests)

**File:** `tests/unit/test_campaign_tier1.py`

One class per subsystem, 20 parametrised rounds each. Tests are pure-Python, no I/O.

| Class | Subsystem | Key assertions |
|---|---|---|
| `TestSourceRegistryStore` | `SourceRegistryStore`, `SourceSpec.priority` | `next_batch` order, `update_priority` return value, `min_priority` filter |
| `TestAcquisitionScheduler` | `AcquisitionScheduler` | exponential back-off doubling, `is_eligible` gate, trust threshold, thread safety |
| `TestConfidenceCalibrator` | `ConfidenceCalibrator` | `sigmoid(logit/T)` value, online update direction, federated blend convergence |
| `TestQualityGate` | `QualityGate`, `QualityGateResult` | pass/fail at threshold boundary, `ValueError` on bad `min_confidence` |
| `TestMultimodalAnalyzer` | `MultimodalAnalyzer` | `has_visual_content` on all three source types, `to_evidence_sources` dict keys |
| `TestCrossSourceDeduper` | `CrossSourceDeduper` | partition invariant, trust-weighted primary selection, empty-list contract |
| `TestIndexingPipeline` | `IndexingPipeline` | `chunks_indexed ≥ 1`, `multimodal_evidence` in chunk metadata, per-tenant store isolation |
| `TestAutoResearchPipeline` | `AutoResearchPipeline`, `ResearchReport` | all `ResearchReport` fields, `ValueError` on empty query, thread snapshot correctness |

---

## Tier 2 — Realistic Data Ingestion (285 tests)

**File:** `tests/unit/test_campaign_tier2.py`

Five cross-platform scenarios (OpenAI, DeepMind, Anthropic, NVIDIA, Meta), 20 rounds each. Uses real news-text articles and `ContentItem` with image/video `metadata` and `media_urls`. Asserts `chunks_indexed ≥ 1`, multimodal citation presence (`source_id.startswith("mm-")`), quality gate rejection rate, and `GroundedSummary.source_attributions` non-empty.

---

## Tier 3 — High-Volume / Stress (127 tests)

**File:** `tests/unit/test_campaign_tier3.py`

| Test class | Scenario | Guard |
|---|---|---|
| `TestT3ABatchSizes` | 50 / 100 / 500 / 1,000-item batches | 60 s / 120 s wall-clock |
| `TestT3BConcurrentPipeline` | 8 / 16 / 32 threads calling `AutoResearchPipeline.run()` | 30 s deadlock guard |
| `TestT3CAcquisitionSchedulerStress` | 200 sources, 50 threads, 10,000 total scheduler iterations | `failure_count ≥ 0` invariant |
| `TestT3DCrossSourceDeduperStress` | 500 bundles, 20 concurrent `deduplicate_cross_bundle` callers | partition invariant, `priority ∈ [0, 1]` |
| `TestT3EMemoryStability` | 20 × 100-item batches with `gc.collect()` between rounds | object count growth < 500,000 |

---

## Tier 4 — End-to-End Integration (220 tests)

**File:** `tests/unit/test_campaign_tier4.py`

All subsystems wired (`AcquisitionScheduler`, `MultimodalAnalyzer`, `PipelineHealthMonitor`, `WatchlistGraph`, `QualityGate(0.60)`, `CrossSourceDeduper`, `IndexingPipeline`). Five realistic queries, 20 rounds each, asserted against the full `ResearchReport` contract:

- `chunks_indexed ≥ 1`
- `confidence_score_mean ∈ [0.0, 1.0]`
- `wall_s > 0`
- `quality_gate_rejection_rate ∈ [0.0, 1.0]`
- `slo_health_status ∈ {GREEN, YELLOW, RED, None}`
- `watchlist_gap_count ≥ 0`
- `generated_at.tzinfo is not None` (UTC-aware)

`TestT4CrossQueryIntegration` runs all 5 queries sequentially within a single shared pipeline instance, verifying no state leak between calls and that at least one `GroundedSummary` is produced per run.

---

## Tier 5 — Response Quality Rubric (800 tests)

**File:** `tests/unit/test_campaign_tier5.py`

Seven criteria applied to every `GroundedSummary`. 100% pass rate required. 5 queries × 20 rounds per criterion.

| Criterion | Field checked | Pass condition |
|---|---|---|
| C1 Attribution | `source_count` | `≥ 1` |
| C2 Confidence | `confidence_score` | `∈ [0.3, 1.0]` |
| C3 Non-empty summary | `what_happened` or `why_it_matters` | `len(stripped) ≥ 20` |
| C4 No hallucinated citations | `source_attributions[*].source_id` | each ID ∈ store's `observation_ids()` or starts with `mm-` |
| C5 Contradictions field | `contradictions` | `hasattr` |
| C6 Uncertainty field | `uncertainty_annotations` or `overall_uncertainty_score` | `hasattr` |
| C7 Multimodal citation | `source_attributions[*].source_id` | at least one starts with `mm-img-` or `mm-vid-` when image URL in input |

C4 uses `ChunkStore.observation_ids()` and `get_by_observation()` to enumerate ground-truth IDs, never `store._chunks` (which does not exist in the SQLite-backed implementation). C7 verifies the D-007/D-008 fixes: duck-typed `_extract_media_urls()` and per-tenant store lookup in `build_grounded_summary(tenant_id=…)`.

---

## Tier 6 — Static Inspection + Boundary Smoke Tests (22 tests)

**File:** `tests/unit/test_campaign_tier6.py`

**Static (AST-level, parametrised over 5 files):**
- `S1` — No bare `except … : pass` handlers
- `S2` — Every `threading.Lock()` creation accompanied by `with self._lock`
- `S3` — No unused `import math`

**Boundary conditions:**
- `B1` — `QualityGate(min_confidence=0.0).evaluate(score=0.001)` → `passed=True`
- `B2` — `AcquisitionScheduler(max_retries=1)`: `is_eligible` → `True`, after one `record_failure` → `False`
- `B3` — `SourceRegistryStore.next_batch(0)` raises `ValueError`
- `B4` — `CrossSourceDeduper.deduplicate_cross_bundle([])` → `([], [])`
- `B5` — `IndexingPipeline.process_batch([])` → `chunks_indexed=0`, `errors={}`, `bundles=[]`
- `B6` — `AutoResearchPipeline.run("")` raises `ValueError`
- `B7` — `ResearchReport.confidence_score_mean` initialises in `[0, 1]`

---

## Defects Found During Campaign

| ID | Component | Defect | Fix |
|---|---|---|---|
| D-001 | `IndexingPipeline._record_bundle_coverage` | Missing `source_family` kwarg | Added kwarg derived from bundle items |
| D-002 | `source_registry.py` | Unused `import math` | Removed |
| D-003 | `AcquisitionScheduler.next_batch` | Direct `_registry._store` access bypassed lock | Replaced with `len(self._registry)` |
| D-004 | `AutoResearchPipeline.run` | Two bare `except: pass` blocks | Changed to `logger.debug` |
| D-005 | `AutoResearchPipeline.run` | `self._lock` created but never acquired | `with self._lock` at top of `run()` |
| D-006 | Test helpers | `SourcePlatform.TWITTER` does not exist | Mapped to `SourcePlatform.REDDIT` |
| D-007 | `MultimodalAnalyzer.to_evidence_sources` | Only read `platform_metadata`; `ContentItem.metadata` and `media_urls` silently skipped | Added `_extract_media_urls()` duck-typed helper |
| D-008 | `IndexingPipeline.build_grounded_summary` | Always read `self._store` (default); per-tenant chunks not found | Added `tenant_id=` param; reads `_tenant_stores[tenant_id]` |

