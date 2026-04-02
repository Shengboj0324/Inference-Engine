# Architecture

## Pipeline Stages

```
ContentItem / RawObservation
        │
  ┌─────▼──────────────────────────────────────────────────┐
  │  Stage A  NormalizationEngine                          │
  │  title+body merge · language detection · spaCy NER    │
  │  engagement/freshness features · embedding (1536-dim)  │
  │  MultimodalAnalyzer._extract_media_urls()              │
  │    → platform_metadata | ContentItem.metadata |        │
  │       ContentItem.media_urls (ext-classified)          │
  └─────┬──────────────────────────────────────────────────┘
        │
  ┌─────▼──────────────────────────────────────────────────┐
  │  DataResidencyGuard                                    │
  │  author → SHA-256 pseudonym (anon_<16 hex>)           │
  │  PII URL params → <redacted>                           │
  │  email/phone tokens in text                            │
  │  RedactionAuditEntry written before text proceeds     │
  │  verify_clean() raises DataResidencyViolationError     │
  └─────┬──────────────────────────────────────────────────┘
        │
  ┌─────▼──────────────────────────────────────────────────┐
  │  Stage B  CandidateRetriever                           │
  │  HNSW similarity (pgvector) against ExemplarBank       │
  │  entity-conditioned vocabulary rules                   │
  │  per-platform base-rate priors                         │
  │  output: List[SignalCandidate] (top-k, ranked)         │
  └─────┬──────────────────────────────────────────────────┘
        │
  ┌─────▼──────────────────────────────────────────────────┐
  │  E6  DeliberationEngine  (4 steps)                     │
  │  A: ContextMemoryStore → top-5 historical observations │
  │  B: prune candidates below retrieval score threshold   │
  │  C: escalate risk types (score>0.5) → audit log        │
  │  D: select reasoning mode →                            │
  │       len>1500 or candidates>6 → multi_agent           │
  │       top-2 within 0.1 or conf_req>0.85 → cot          │
  │       else → single_call                               │
  └─────┬──────────────────────────────────────────────────┘
        │
   ┌────┴───────────────────────────────┐
   │                                    │
   ▼                                    ▼
E1 ChainOfThoughtReasoner    E3 MultiAgentOrchestrator
   step-by-step scratchpad       sub-tasks with per-task PII scrub
   │                                    │
   └────────────────┬───────────────────┘
                    │
  ┌─────────────────▼──────────────────────────────────────┐
  │  Stage C  LLMAdjudicator                               │
  │  structured JSON-schema prompt                         │
  │  few-shot context from E5 ContextMemoryStore           │
  │  LLMRouter: _FRONTIER_SIGNAL_TYPES → GPT-4o tier      │
  │             remaining 14 → fine-tuned / Ollama tier    │
  │  fallback: exponential back-off + circuit breaker      │
  └─────┬──────────────────────────────────────────────────┘
        │
  ┌─────▼──────────────────────────────────────────────────┐
  │  E2  ConfidenceCalibrator                              │
  │  sigmoid(logit / T_eff)                                │
  │  T_eff = α·T_user + (1−α)·T_global  (federated)       │
  │  α → 0.7 after 500 confirmed analyst outcomes          │
  │  E4 FeedbackStore: one gradient step per correction    │
  │    T ← max(T_MIN, T − lr·(p_cal−y)·(−logit/T²))      │
  └─────┬──────────────────────────────────────────────────┘
        │
  ┌─────▼──────────────────────────────────────────────────┐
  │  Stage D  AbstentionDecider                            │
  │  7 typed AbstentionReason enum values                  │
  │  default threshold: 0.7 (post-calibration)             │
  │  abstentions logged, never enter signal queue          │
  └─────┬──────────────────────────────────────────────────┘
        │
  ┌─────▼──────────────────────────────────────────────────┐
  │  IndexingPipeline.process_batch()  — 7 phases          │
  │  1. route → IntelligencePipelineResult                 │
  │  2. chunk; stamp multimodal_evidence in ChunkRecord    │
  │     item_map passed for ContentItem fallback lookup    │
  │  3. SourceTrustScorer → trust_score per chunk          │
  │  4. EventClusterer + WatchlistGraph coverage           │
  │  5. CrossSourceDeduper.deduplicate_cross_bundle()      │
  │     Jaccard similarity; trust-weighted primary         │
  │  6. GroundedSummaryBuilder → source_attributions,      │
  │     contradictions, uncertainty_annotations            │
  │  7. QualityGate(min_confidence) → QualityGateResult   │
  │  per-tenant ChunkStore partitions (SQLite in-proc)     │
  └─────┬──────────────────────────────────────────────────┘
        │
  ┌─────▼──────────────────────────────────────────────────┐
  │  AutoResearchPipeline.run()  — async orchestrator      │
  │  1. AcquisitionScheduler.next_batch()                  │
  │     priority sort · trust gate · exponential back-off  │
  │     PipelineHealthMonitor circuit breaking             │
  │  2. process_batch(items, tenant_id)                    │
  │  3. build_grounded_summary(…, tenant_id)               │
  │     reads _tenant_stores[tenant_id] (not default)      │
  │  4. QualityGate evaluation                             │
  │  5. WatchlistGraph.coverage_report() → gap_count       │
  │  6. PipelineHealthMonitor.health_report() → SLOStatus  │
  │  7. emit ResearchReport                                │
  └─────┬──────────────────────────────────────────────────┘
        │
  PostgreSQL + pgvector · REST API · SSE stream
  ActionableSignal · ResearchReport · GroundedSummary
```

---

## Core Data Models

### `ContentItem`
```python
id: UUID
source_platform: SourcePlatform  # enum; 19 values
source_id: str
source_url: str
title: str
raw_text: str
media_type: MediaType            # TEXT | IMAGE | VIDEO | MIXED
media_urls: List[str]            # duck-typed by MultimodalAnalyzer
published_at: datetime           # UTC
topics: List[str]
metadata: Dict[str, Any]         # image_url / video_url keys consumed by MultimodalAnalyzer
embedding: Optional[List[float]] # 1536-dim (text-embedding-3-large default)
```

### `NormalizedObservation`
Output of `NormalizationEngine`. Adds `language`, `entities`, `engagement_score`, `freshness_score`, `embedding` to `ContentItem` fields. Stored in PostgreSQL as `NormalizedObservationDB` with `embedding vector(1536)`.

### `ChunkRecord`
```python
chunk_id: str
observation_id: str
text: str
metadata: Dict[str, Any]   # includes multimodal_evidence: List[Dict]
trust_score: Optional[float]
tenant_id: str
```

### `IntelligencePipelineResult`
Output of `ContentPipelineRouter.route()`. Carries `content_item_id`, `source_family`, `is_actionable()`, `all_text_for_chunking()`, `summary`.

### `ResearchReport`
```python
query: str
tenant_id: str
chunks_indexed: int
confidence_score_mean: float          # ∈ [0, 1]
quality_gate_rejection_rate: float    # ∈ [0, 1]
quality_gate_outcomes: List[QualityGateResult]
watchlist_gap_count: int
slo_health_status: Optional[SLOStatus]
grounded_summaries: List[GroundedSummary]
wall_s: float
generated_at: datetime                # UTC
```

---

## Technology Stack

| Layer | Component |
|---|---|
| API | FastAPI + Uvicorn (async); SSE via `EventSourceResponse` |
| Task queue | Celery + Redis Streams; Beat for 15-min ingestion cron |
| Database | PostgreSQL 15 + pgvector (`vector(1536)`); async via asyncpg |
| In-proc vector store | `ChunkStore` — SQLite per tenant (zero-config) |
| Object storage | MinIO (local) / S3-compatible |
| Embeddings | `text-embedding-3-large` (default); BGE-m3 (OSS fallback) |
| LLM (frontier) | GPT-4o / Claude 3.5 Sonnet |
| LLM (fine-tuned) | GPT-4o-mini fine-tuned or Ollama (`llama3.1:8b`) |
| NLP | spaCy `en_core_web_sm`; fallback 512-dim BoW |
| Monitoring | Prometheus + Grafana (`deployment/grafana/`) |
| Containerisation | Docker Compose; Kubernetes manifests in `deployment/kubernetes/` |

