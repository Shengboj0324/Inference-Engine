# Repo-Specific Refactor Plan for Social-Media-Radar

## Purpose

This document is a repo-specific engineering refactor plan for the uploaded **Social-Media-Radar-main** codebase. It is written for software engineering and ML engineering execution. The objective is to convert the current system from a digest-centric aggregation stack into a **calibrated inference-and-action system** that can handle noisy, ambiguous, multilingual, adversarial, and commercially meaningful inputs with much higher reliability.

This plan is based on direct inspection of the uploaded repository, especially:

- `app/core/models.py`
- `app/core/db_models.py`
- `app/core/ranking.py`
- `app/api/routes/search.py`
- `app/api/routes/sources.py`
- `app/api/routes/digest.py`
- `app/intelligence/digest_engine.py`
- `app/intelligence/cluster_summarizer.py`
- `app/intelligence/__init__.py`
- `app/ingestion/tasks.py`
- `app/mcp_server/server.py`
- `app/connectors/base.py`
- `app/connectors/registry.py`
- selected connector implementations such as `app/connectors/reddit.py`
- `app/api/main.py`
- `pyproject.toml`

---

## Executive verdict

The repo has a solid amount of infrastructure and ambition, but the core runtime contract is still unstable and the intelligence stack is still **digest-oriented, heuristic, and under-specified for agent-grade actioning**.

The most important conclusion is this:

> The project should not be extended by adding more disconnected features. It should be refactored into a layered system with strict contracts, typed inference artifacts, calibrated ML, LLM adjudication, and action policies.

At a high level, the current repo has four strengths:

1. good substrate for multi-source ingestion
2. a unified content abstraction (`ContentItem`)
3. a usable LLM subsystem
4. enough scaffolding to evolve into a stronger intelligence system

But it also has four serious weaknesses:

1. **schema drift** across Pydantic, SQLAlchemy, routes, and task workers
2. **digest-first architecture** instead of signal/action-first architecture
3. **heuristic scoring and brittle orchestration** in critical paths
4. **missing typed contracts** for inference, actionability, uncertainty, and evaluation

The repo is not ready yet for unpredictable real-world inputs in the form required by a high-trust AI action system.

---

## What the final system should become

The target product architecture should be this:

**Raw multi-platform content** -> **normalized observation** -> **semantic candidate retrieval** -> **ML/LLM signal inference** -> **confidence calibration** -> **action ranking** -> **policy-constrained response planning** -> **human/agent execution** -> **feedback and learning**

The core product object should no longer be just `Cluster` or `DigestResponse`.

The new first-class runtime objects should be:

- `NormalizedObservation`
- `SignalInference`
- `ActionableSignal`
- `ResponsePlan`
- `ExecutionOutcome`

That is the architectural center of gravity the repo currently lacks.

---

# Part I. Detailed repo findings from the old files

## 1. Domain model drift is real and already hurting correctness

### 1.1 `app/core/models.py`

`ContentItem` is the primary cross-platform Pydantic object. It is still too thin for the system you want to build.

Current fields are mostly ingestion-oriented:

- source identity
- title/body/media
- topics/lang/embedding
- metadata

What is missing for a calibrated signal/action engine:

- normalized language and translation fields
- thread context and parent/child relationships
- author/account features
- engagement summary in canonical form
- extraction fields for entities, products, brands, competitors
- moderation/risk fields
- uncertainty/confidence fields
- signal-level or action-level derived state

### 1.2 `app/core/db_models.py`

There is a mismatch between the route/service assumptions and actual database models.

Concrete examples:

- `ContentItemDB` stores JSON in `metadata_ = mapped_column("metadata", JSON, default=dict)` while other parts of the code read `db_item.metadata` as if it were a normal mapped attribute.
- `PlatformConfigDB` has no `feeds` column, but `app/api/routes/sources.py` reads and writes `config.feeds` and `platform_config.feeds`.
- `PlatformConfigDB` also has no `last_fetch_time`, but `app/ingestion/tasks.py` reads `config.last_fetch_time`.
- There is no `SourceConfig` ORM model, but `app/mcp_server/server.py` imports `SourceConfig` from `app.core.db_models`.

This is a structural issue, not a minor bug. It means the codebase has multiple incompatible mental models of the same persistence layer.

### 1.3 Why this matters

Once contracts drift at the model layer, every upper layer becomes unreliable:

- API serialization becomes fragile
- ingestion workers silently diverge from DB schema
- MCP and route code become dead-on-arrival
- evaluation becomes meaningless because the runtime state is inconsistent

### required refactor

Create a domain contract package and split models into clear layers:

- `app/domain/raw_models.py`
- `app/domain/normalized_models.py`
- `app/domain/inference_models.py`
- `app/domain/action_models.py`
- `app/persistence/sql_models.py`
- `app/schemas/api/*.py`

Do not reuse one Pydantic model for every layer.

---

## 2. Search route is currently contract-broken

### file: `app/api/routes/search.py`

This route is one of the clearest correctness failures in the repo.

It constructs `ContentItem(...)` with fields that do not exist or are incomplete relative to the model.

Examples from the current implementation:

- passes `engagement_score=db_item.engagement_score`, but `ContentItem` has no `engagement_score`
- does not pass `media_type`, which is required on `ContentItem`
- does not pass `channel`, `lang`, `embedding`, or normalized metadata consistently
- reads `db_item.engagement_score`, which does not exist on `ContentItemDB`

This route should be treated as broken until rewritten.

### required refactor

Replace ad hoc DB-to-Pydantic construction with explicit mappers:

- `ContentItemMapper.from_db(db_item) -> ContentItemReadModel`
- `NormalizedObservationMapper.from_db(db_item, thread_ctx, account_ctx) -> NormalizedObservation`

Also split the route contract:

- `/search/raw` for retrieval of content items
- `/search/signals` for retrieval of inferred signals
- `/search/opportunities` for ranked opportunity/action objects

### implementation note

Introduce a dedicated read schema, for example:

```python
class ContentItemRead(BaseModel):
    id: UUID
    user_id: UUID
    source_platform: SourcePlatform
    source_id: str
    source_url: str
    author: str | None
    channel: str | None
    title: str
    raw_text: str | None
    media_type: MediaType
    media_urls: list[str]
    published_at: datetime
    fetched_at: datetime
    topics: list[str]
    lang: str | None
    metadata: dict[str, Any]
```

No route should directly instantiate domain models from ORM rows without a mapper.

---

## 3. Sources route and MCP server are out of sync with the persistence layer

### files

- `app/api/routes/sources.py`
- `app/mcp_server/server.py`
- `app/core/db_models.py`
- `app/connectors/registry.py`

### findings

#### 3.1 `app/api/routes/sources.py`

This file assumes:

- `PlatformConfigDB.feeds` exists
- top-level `get_connector()` exists in `app.connectors.registry`

But in the actual repo:

- `PlatformConfigDB` has no `feeds`
- `app/connectors/registry.py` exposes `ConnectorRegistry.get_connector(...)`, not a matching top-level helper with the same signature used in routes
- the route also creates `PlatformConfigDB(..., feeds=[])`, which is invalid against the current SQLAlchemy model

#### 3.2 `app/mcp_server/server.py`

This file imports and uses:

- `SourceConfig` from `app.core.db_models`
- `app.intelligence.clustering.ContentClusterer`

Neither aligns with the actual repo structure:

- there is no `SourceConfig` model
- there is no `app.intelligence.clustering` module matching that import

The MCP layer is therefore not just incomplete; it is materially disconnected from the codebase.

### required refactor

Unify source configuration around a single persistence and service API.

Create:

- `app/services/source_config_service.py`
- `app/schemas/api/source_config.py`
- `app/connectors/factory.py`

Then define a clean service contract:

```python
class SourceConfigService:
    async def list_user_sources(self, user_id: UUID) -> list[SourceConfigRead]: ...
    async def upsert_source(self, user_id: UUID, req: SourceConfigWrite) -> SourceConfigRead: ...
    async def test_source(self, user_id: UUID, platform: SourcePlatform) -> SourceConnectionStatus: ...
```

MCP tools must call service methods, not manually construct ORM objects.

---

## 4. Ingestion task layer has async/sync and schema inconsistencies

### file: `app/ingestion/tasks.py`

This file needs hardening before any advanced agent logic is added.

### findings

#### 4.1 async connector methods are used from synchronous Celery tasks without awaiting

`BaseConnector.fetch_content()` is async. In `fetch_source_content`, the code calls:

```python
result = connector.fetch_content(
    since=since, max_items=settings.max_items_per_fetch
)
```

That returns a coroutine, not a result, unless explicitly awaited or run in an event loop.

That means the ingestion path is currently conceptually incorrect.

#### 4.2 `config.last_fetch_time` is referenced but does not exist on `PlatformConfigDB`

This makes incremental fetch logic incomplete.

#### 4.3 metadata field mismatch

`ContentItemDB` stores `metadata_`, but task code writes `metadata=item.metadata` when constructing ORM rows.

#### 4.4 duplicate/content idempotency is missing

There is no clear deduplication or upsert policy for repeated source items. For a production signal engine, ingestion must be idempotent.

### required refactor

Split ingestion into three explicit layers:

- connector extraction
- normalization/enrichment
- persistence

Recommended package changes:

- `app/ingestion/fetch_jobs.py`
- `app/ingestion/normalize.py`
- `app/ingestion/persist.py`
- `app/ingestion/checkpointing.py`

Refactor Celery workers to either:

- use async task execution properly, or
- wrap async connector calls through a well-defined sync adapter

Example shape:

```python
class ConnectorRunner:
    async def fetch(self, connector: BaseConnector, since: datetime | None) -> FetchResult: ...

class ItemPersistenceService:
    async def upsert_items(self, items: list[NormalizedObservation]) -> PersistReport: ...
```

Also add a unique content key:

- `(user_id, source_platform, source_id)`

This should be enforced in the DB and used for upserts.

---

## 5. Current ranking and clustering are too primitive for the target product

### file: `app/core/ranking.py`

This file is functional as a prototype but not sufficient for calibrated inference.

### findings

#### 5.1 `RelevanceScorer`

The scorer is still a weighted heuristic:

- embedding similarity
- topic overlap
- recency
- engagement score from metadata

This is acceptable as a baseline, but not as the decision core for a product that must prioritize real commercial opportunities.

It is also still tied to a `UserInterestProfile` model designed for digest personalization rather than action prioritization.

#### 5.2 `ContentClusterer`

The clusterer creates `Cluster` objects with:

- simplistic topic extraction from the most common topic token
- static `relevance_score = 0.5`
- no confidence measure
- no cluster-level uncertainty
- no temporal burst modeling
- no differentiation between lead capture, churn risk, competitor weakness, support escalation, or trend opportunity

This is digest logic, not action logic.

### required refactor

Keep a baseline ranking module, but move business logic out of `core/ranking.py`.

Replace with:

- `app/inference/candidate_generation.py`
- `app/inference/signal_classifier.py`
- `app/inference/calibrator.py`
- `app/ranking/action_ranker.py`
- `app/ranking/explainer.py`

The current `RelevanceScorer` can survive as a fallback feature generator, but not as the main ranking system.

### target ML stack

#### stage 1: candidate generation

- embedding retrieval against labeled exemplars
- topic and entity expansion
- temporal burst candidates

#### stage 2: classification

Train a supervised classifier for signal type:

- lead_capture
- competitor_displacement
- churn_risk
- product_confusion
- trend_to_content
- support_escalation
- misinformation_risk
- ignore

Pragmatic first models:

- sentence embeddings + LightGBM / XGBoost
- linear head over transformer embeddings

#### stage 3: calibrated action scoring

Train a ranking model over features such as:

- signal probabilities
n- source credibility
- account features
- engagement velocity
- competitor relevance
- historical conversion/engagement outcomes
- reply risk
- freshness decay

Use calibration methods:

- Platt scaling
- isotonic regression
- temperature scaling for neural classifiers

---

## 6. Digest engine is orchestrating the wrong product primitive

### file: `app/intelligence/digest_engine.py`

This engine is coherent for a digest application, but it is the wrong architectural centerpiece for the product direction you want.

### findings

- it fetches items
- scores relevance
- clusters them
- summarizes clusters
- ranks clusters
- returns a digest

That is a good reporting pipeline, not a calibrated signal-to-action system.

There is also a correctness smell:

- `_score_items` assigns `item.relevance_score = ...` onto `ContentItem`, but `ContentItem` does not declare that field in `app/core/models.py`
- with Pydantic v2, this is not something you should depend on; the contract is unclear and brittle
- `_fetch_content_items` reads `db_item.metadata` even though the ORM model defines `metadata_`

### required refactor

Do not delete `DigestEngine`; demote it.

Move the center of gravity to a new orchestration layer:

- `app/pipelines/observation_pipeline.py`
- `app/pipelines/signal_pipeline.py`
- `app/pipelines/action_pipeline.py`

And define `DigestEngine` as a downstream consumer of `ActionableSignal` or `SignalInference`, not the primary engine.

New orchestration shape:

```python
class ObservationPipeline:
    async def run(self, raw_items: list[ContentItemDB]) -> list[NormalizedObservation]: ...

class SignalPipeline:
    async def run(self, observations: list[NormalizedObservation]) -> list[SignalInference]: ...

class ActionPipeline:
    async def run(self, inferences: list[SignalInference]) -> list[ActionableSignal]: ...
```

A digest should be generated from ranked signals and opportunities, not from raw cluster summaries.

---

## 7. Cluster summarizer is LLM-rich but not action-grounded

### file: `app/intelligence/cluster_summarizer.py`

This file is one of the more mature LLM components in the repo. It has:

- improved prompts
- JSON outputs
- optional ensemble support
- fallback summaries

That is useful. But it is still optimizing for **summary quality**, not **decision quality**.

### findings

The summarizer does not yet produce the fields required for trustworthy actioning, such as:

- signal type probabilities
- business-value estimation
- action recommendations
- uncertainty rationale
- abstention recommendation
- policy/risk category
- response plan requirements

### required refactor

Keep the summarizer code and prompt discipline, but repurpose it into a broader inference stack.

Create:

- `app/inference/llm_adjudicator.py`
- `app/inference/response_planner.py`
- `app/inference/policy_checker.py`

The current summarizer should become one tool among several, not the primary intelligence artifact generator.

Suggested structured output from adjudication:

```json
{
  "signal_type": "competitor_displacement",
  "signal_probs": {
    "lead_capture": 0.11,
    "competitor_displacement": 0.78,
    "churn_risk": 0.06,
    "ignore": 0.05
  },
  "confidence": 0.73,
  "needs_more_context": false,
  "evidence_spans": ["..."],
  "rationale": "...",
  "actionability": 0.82,
  "recommended_action": "public_reply",
  "risk_flags": ["comparative_claims"],
  "abstain": false
}
```

---

## 8. `app/intelligence/__init__.py` is too eager

This file eagerly imports `DigestEngine` and `ClusterSummarizer`.

That may look harmless, but in a large codebase it increases import coupling and causes cascading dependency loading.

### required refactor

Turn package init files into light export surfaces or remove them as convenience import hubs.

Prefer direct imports from concrete modules.

Example:

- bad: `from app.intelligence import DigestEngine`
- better: `from app.pipelines.digest_pipeline import DigestPipeline`

This reduces accidental import-time failures and improves test isolation.

---

## 9. Connector layer is structurally fine, but not normalized enough for agent-grade inference

### files

- `app/connectors/base.py`
- `app/connectors/reddit.py`
- `app/connectors/registry.py`

### findings

The connector abstraction itself is decent. It has:

- shared config
- shared fetch result model
- health and retry concepts
- helper creation of `ContentItem`

But the output contract is still too ingestion-centric and too weak for robust downstream inference.

### missing concepts

- canonical thread relationships
- canonical account identity schema
- canonical engagement schema
- source reliability and provenance confidence
- extraction diagnostics
- ingestion checkpoint metadata
- language detection and translation status

### required refactor

Add a normalized observation layer between connector outputs and downstream inference.

New models:

```python
class CanonicalEngagement(BaseModel):
    likes: int | None = None
    comments: int | None = None
    shares: int | None = None
    views: int | None = None
    score: float | None = None

class NormalizedObservation(BaseModel):
    observation_id: UUID
    user_id: UUID
    content_id: UUID
    source_platform: SourcePlatform
    source_id: str
    source_url: str
    parent_source_id: str | None = None
    thread_id: str | None = None
    author_handle: str | None = None
    author_display_name: str | None = None
    author_followers: int | None = None
    title: str | None = None
    body_text: str | None = None
    full_text: str
    translated_text: str | None = None
    language: str | None = None
    media_type: MediaType
    published_at: datetime
    canonical_engagement: CanonicalEngagement
    raw_metadata: dict[str, Any]
```

This becomes the input to inference.

---

## 10. API main app still reflects the old product shape

### file: `app/api/main.py`

The mounted routes are:

- auth
- sources
- digest
- search
- llm

This is still the surface of an ingestion + digest + LLM utility system.

### required refactor

The final product should expose a queue-first action surface.

New route groups should be:

- `/api/v1/observations`
- `/api/v1/signals`
- `/api/v1/opportunities`
- `/api/v1/actions`
- `/api/v1/playbooks`
- `/api/v1/evaluations`
- `/api/v1/feedback`

The old `/digest` routes can remain, but they should be clearly secondary and generated from the signal/action layer.

---

# Part II. Repo-specific target architecture

## 11. New package structure

A practical refactor target:

```text
app/
  api/
    routes/
      auth.py
      observations.py
      signals.py
      opportunities.py
      actions.py
      feedback.py
      digest.py
  domain/
    enums.py
    raw_models.py
    normalized_models.py
    inference_models.py
    action_models.py
  persistence/
    sql_models.py
    repositories/
      content_repo.py
      source_repo.py
      signal_repo.py
      action_repo.py
  ingestion/
    connectors/
    fetch_jobs.py
    normalize.py
    translate.py
    entity_enrich.py
    persist.py
    checkpointing.py
  inference/
    features.py
    candidate_generation.py
    signal_classifier.py
    llm_adjudicator.py
    calibrator.py
    abstention.py
    explainer.py
  ranking/
    action_ranker.py
    playbook_ranker.py
  planning/
    response_planner.py
    channel_planner.py
    action_policy.py
  execution/
    workflow_types.py
    action_handlers.py
    human_review.py
  services/
    source_config_service.py
    observation_service.py
    signal_service.py
    opportunity_service.py
    feedback_service.py
  pipelines/
    observation_pipeline.py
    signal_pipeline.py
    action_pipeline.py
    digest_pipeline.py
  eval/
    datasets/
    offline_metrics.py
    calibration.py
    safety_metrics.py
    replay.py
```

This is a substantial reorganization, but it is the correct one.

---

## 12. New first-class domain objects

### 12.1 `NormalizedObservation`

Purpose: canonicalized source content with enough context for ML and LLM inference.

### 12.2 `SignalInference`

Purpose: the model’s best current belief about what the content means.

Suggested fields:

```python
class SignalInference(BaseModel):
    inference_id: UUID
    observation_id: UUID
    candidate_signal_types: dict[str, float]
    top_signal_type: str
    calibrated_confidence: float
    actionability_score: float
    evidence_strength: float
    context_completeness: float
    risk_score: float
    abstain: bool
    abstain_reason: str | None
    evidence_spans: list[str]
    rationale: str
    model_versions: dict[str, str]
    created_at: datetime
```

### 12.3 `ActionableSignal`

Purpose: a signal that is strong enough to surface to the user.

Suggested fields:

```python
class ActionableSignal(BaseModel):
    signal_id: UUID
    inference_id: UUID
    signal_type: str
    priority_score: float
    urgency_score: float
    impact_score: float
    risk_score: float
    recommended_action: str
    recommended_channel: str | None
    recommended_owner: str | None
    playbook_id: str | None
    expires_at: datetime | None
    status: str
```

### 12.4 `ResponsePlan`

Purpose: the structured plan before generation.

Suggested fields:

- goal
- target audience
- reply mode (public reply / DM / post / internal escalation)
- constraints
- prohibited claims
- desired tone
- CTA
- evidence references

---

# Part III. Exact refactors by old file

## 13. File-by-file refactor instructions

### 13.1 `app/core/models.py`

#### keep

- `MediaType`
- `SourcePlatform` (though likely move to `app/domain/enums.py`)

#### split

- keep `ContentItem` only if renamed to `RawContentItem` or `IngestedContentRecord`
- do not keep it as the universal runtime model

#### add

- `NormalizedObservation`
- `SignalInference`
- `ActionableSignal`
- `ResponsePlan`
- `ExecutionOutcome`

#### remove anti-pattern

Do not dynamically attach undeclared fields such as `relevance_score` to base models.

---

### 13.2 `app/core/db_models.py`

#### immediate changes

- rename to `app/persistence/sql_models.py`
- align ORM field names with actual code usage
- stop using `metadata_` if the rest of the code expects `metadata`; either use explicit property aliases or make the ORM naming consistent
- add unique constraints for content identity
- add ingestion checkpoints
- add source health and last fetch fields
- add signal/action tables

#### new tables

- `normalized_observations`
- `signal_inferences`
- `actionable_signals`
- `response_plans`
- `execution_outcomes`
- `model_decisions`
- `feedback_events`

#### source config fields to add or explicitly remove from the whole codebase

Choose one direction:

1. **minimal path**: remove all `feeds` assumptions from routes and services
2. **richer path**: add feed/channel subscription persistence explicitly

For the product you want, the richer path is better.

Suggested additions to `PlatformConfigDB`:

- `last_fetch_time`
- `last_successful_fetch`
- `last_error`
- `health_status`
- `feed_filters`
- `channel_filters`

---

### 13.3 `app/api/routes/search.py`

#### immediate rewrite

- stop constructing `ContentItem` inline from ORM rows
- use read mappers
- split retrieval from ranking/inference

#### recommended endpoint redesign

- `POST /api/v1/search/raw`
- `POST /api/v1/search/signals`
- `POST /api/v1/search/opportunities`

#### search ranking

Use a hybrid search strategy:

- lexical retrieval
- embedding retrieval
- recency decay
- platform priors
- signal score priors

This should not rely only on `OpenAIEmbeddingClient` and pgvector ordering.

---

### 13.4 `app/api/routes/digest.py`

#### keep but downgrade

This can remain as a secondary reporting surface.

#### refactor

- digest generation should consume ranked `ActionableSignal` objects
- provide both narrative digest and operations digest

Example output modes:

- executive digest
- market opportunity digest
- support risk digest
- competitor movement digest

---

### 13.5 `app/api/routes/sources.py`

#### rewrite against actual service layer

Remove direct ORM manipulation and connector assumptions.

Route should become a thin wrapper over `SourceConfigService`.

#### minimum endpoint set

- list configured platforms
- upsert platform credentials/settings
- test source
- list feeds/channels
- enable/disable source

#### current bugs to eliminate

- `feeds` usage on ORM model that does not support it
- incorrect connector factory import/usage
- weak connection validation path

---

### 13.6 `app/ingestion/tasks.py`

#### mandatory changes

- fix async/sync model
- add source checkpoint persistence
- add dedup/upsert behavior
- add observation enrichment stage

#### split task responsibilities

- `fetch_source_content`
- `normalize_source_items`
- `persist_observations`
- `compute_inferences`
- `refresh_action_queue`

Avoid large monolithic worker tasks.

---

### 13.7 `app/core/ranking.py`

#### keep only as baseline utilities

- embedding similarity helpers
- simple cluster utilities if still needed for digests

#### remove from core product decision path

Action ranking should live in a separate module with trained models and explicit calibration.

---

### 13.8 `app/intelligence/digest_engine.py`

#### demote to `digest_pipeline.py`

Replace the current engine-centric architecture with pipelines driven by normalized observations and signal inference.

#### keep

- orchestration discipline
- fallback summary concept

#### remove as primary artifact

- cluster-first worldview
- direct assignment of runtime-only fields onto thin models

---

### 13.9 `app/intelligence/cluster_summarizer.py`

#### repurpose

Reuse prompt, JSON parsing, ensemble ideas, and fallback strategies.

#### new role

- `llm_adjudicator.py`
- `response_planner.py`
- `market_brief_generator.py`

---

### 13.10 `app/intelligence/__init__.py`

#### simplify

Do not re-export heavyweight orchestration components from package init.

---

### 13.11 `app/mcp_server/server.py`

#### rewrite almost completely

Current MCP code references objects and modules that do not exist in the current repo.

The MCP layer should become a thin adapter over stable services:

- search opportunities
- list sources
- get signal queue
- generate response plan
- mark action outcome

It should not perform direct ORM writes or import speculative modules.

---

### 13.12 `app/connectors/registry.py`

#### keep but add explicit factory API

Provide one official connector construction interface and use it everywhere.

For example:

```python
class ConnectorFactory:
    @classmethod
    def create(cls, platform: SourcePlatform, cfg: ConnectorConfig, user_id: UUID) -> BaseConnector: ...
```

Then remove inconsistent route-level imports.

---

# Part IV. The calibrated inference and action system

## 14. Inference pipeline design

### stage A. observation normalization

Inputs:

- raw connector outputs
- thread relationships
- account metadata
- translation results
- entity extraction

Outputs:

- `NormalizedObservation`

### stage B. candidate generation

Methods:

- embedding nearest neighbors against labeled exemplars
- lexical rules as weak features only
- temporal burst detection
- competitor/product entity expansion

Outputs:

- candidate signal types with weak priors

### stage C. supervised classifier

Initial practical model:

- sentence transformer embeddings
- LightGBM or XGBoost multiclass classifier

Predictions:

- signal type probabilities
- ignore probability

### stage D. LLM adjudicator

Used for:

- ambiguity resolution
- policy-sensitive interpretation
- reasoning over thread context
- response planning

Not used as the sole classifier.

### stage E. calibration and abstention

Inputs:

- classifier probabilities
- LLM confidence proxy
- source quality
- context completeness
- disagreement features

Outputs:

- calibrated confidence
- abstain / escalate / monitor-only

### stage F. action ranking

Train a separate ranking model for:

- likely business value
- urgency
- conversion potential
- safe actionability

### stage G. response planning and generation

Pipeline:

- infer intent/risk
- produce response plan
- generate variants
- critique
- revise
- policy check
- channel fit check
- final rank

---

## 15. ML implementation roadmap

### 15.1 dataset creation

You need a labeled dataset from the product domain. Without this, the system will never become reliably calibrated.

Create labels for:

- signal type
- actionability
- urgency
- risk
- should engage?
- recommended action type
- acceptable channel
- confidence band

### 15.2 baseline models

#### classifier baseline

- text embedding model: `sentence-transformers` or equivalent
- classifier: LightGBM/XGBoost multiclass

#### action ranker baseline

- gradient boosted trees on engineered features

#### calibration

- isotonic regression or temperature scaling

### 15.3 online learning later

Once enough action/outcome data exists:

- bandit layer for action recommendations
- re-ranking from outcome feedback
- response variant selection from execution outcomes

But do not start with RL. Start with offline supervised learning and calibrated ranking.

---

## 16. Uncertainty and trust architecture

To achieve user trust, the system must know when **not** to over-assert.

Every surfaced signal should include:

- calibrated confidence
- evidence strength
- context completeness
- action safety
- abstention reason if applicable

### abstention states to support

- `no_clear_signal`
- `needs_thread_context`
- `multi_label_ambiguous`
- `unsafe_to_engage`
- `low_confidence_monitor_only`

This is mandatory for a trustworthy AI agent.

---

## 17. Evaluation framework that this repo currently lacks

Add a dedicated `app/eval/` package.

### offline evaluation metrics

#### classification

- macro F1
- per-class precision/recall
- confusion matrix
- abstention precision

#### calibration

- Expected Calibration Error
- Brier score
- reliability diagrams

#### ranking

- NDCG@k
- MRR
- precision@k

#### response quality

- policy violation rate
- hallucination rate
- acceptance/edit distance by humans
- channel-fit success rate

#### business outcome proxies

- reply rate
- conversion rate
- escalation rate
- false action rate

### adversarial evaluation set

Include:

- sarcasm
- multilingual/code-switched text
- indirect alternative-seeking
- subtle competitor complaints
- long noisy threads
- screenshot OCR artifacts
- vague support frustration
- legal/political/high-risk conversations

Until this suite exists, claims of agent robustness are not credible.

---

# Part V. Exact implementation phases

## 18. Phase 0: stop-the-bleeding hardening

Duration: 1 sprint

### goals

- restore contract integrity
- eliminate broken route/ORM assumptions
- stabilize ingestion

### tasks

1. rewrite `search.py` mapper path
2. rewrite `sources.py` against actual ORM model or service layer
3. remove `SourceConfig` references from MCP server
4. fix `last_fetch_time` strategy in ingestion
5. fix async connector execution in Celery
6. standardize metadata field access
7. add content uniqueness constraint and upsert behavior
8. remove dynamic undeclared field mutation from domain models

### deliverable

A repo that is internally consistent and testable.

---

## 19. Phase 1: domain model split

Duration: 1 sprint

### goals

- separate raw, normalized, inference, and action models

### tasks

1. add `domain/normalized_models.py`
2. add `domain/inference_models.py`
3. add `domain/action_models.py`
4. add mappers from ORM -> raw -> normalized
5. update APIs to return dedicated read schemas

### deliverable

Typed contracts for every layer.

---

## 20. Phase 2: candidate generation and baseline classifier

Duration: 1-2 sprints

### goals

- move from digest clustering to actionable signal inference

### tasks

1. build labeled exemplar library
2. implement embedding retrieval candidate generator
3. implement multiclass classifier baseline
4. add candidate explanation traces
5. persist `SignalInference`

### deliverable

First end-to-end signal inference path.

---

## 21. Phase 3: calibration, abstention, and action ranking

Duration: 1 sprint

### goals

- make predictions trustworthy enough to surface to users

### tasks

1. add calibrator module
2. add abstention policy module
3. build ranking model for `ActionableSignal`
4. persist queue state
5. expose `/signals` and `/opportunities` APIs

### deliverable

Usable opportunity queue with calibrated confidence.

---

## 22. Phase 4: response planning and generation

Duration: 1 sprint

### goals

- turn signals into concrete user-ready actions

### tasks

1. implement `ResponsePlan`
2. add plan -> draft -> critique -> revise -> rank flow
3. add playbooks by signal type
4. add policy/risk checking
5. add human review states

### deliverable

Action-ready output, not just insights.

---

## 23. Phase 5: feedback loop and evaluation

Duration: continuous

### goals

- learn from outcomes and quantify robustness

### tasks

1. log outcome events
2. add human feedback UI/API
3. build replay evaluation
4. retrain ranking/classification periodically
5. add dashboards for calibration and false-action rate

### deliverable

A system that measurably improves rather than merely grows in code size.

---

# Part VI. Final product form

## 24. What the final deliverable should look like

The product should appear in three forms.

### 24.1 primary product: queue-first SaaS application

The primary UI should be a ranked **Action Queue**, not a dashboard full of passive analytics.

Each item should show:

- signal type
- why it matters
- confidence
- urgency
- source thread/post
- recommended action
- draft response/content
- risk notes
- owner/state

### 24.2 secondary product: digest and strategy views

Derived from the queue and signal store:

- daily market brief
- competitor weakness report
- support risk brief
- content opportunity brief

### 24.3 platform/API form

APIs for:

- source management
- observation retrieval
- signal retrieval
- opportunity queue
- response planning
- feedback submission
- evaluation/metrics

This is the correct final deliverable shape for engineering.

---

# Part VII. Non-negotiable engineering standards

## 25. Standards required for this repo to reach “peak performance and user trust”

### contract standards

- no direct ORM-to-domain leakage
- no undeclared field mutation on Pydantic models
- no route-specific schema invention
- one official source configuration model

### ML standards

- every model version logged
- calibration measured continuously
- abstention supported by design
- no heuristic confidence presented as real confidence

### orchestration standards

- typed pipeline stages
- typed execution outcomes
- explicit retries and failure reasons
- event logging for every decision boundary

### testing standards

- unit tests for mappers and policies
- integration tests for ingestion -> inference -> queue
- replay tests on labeled data
- adversarial benchmark set

### product standards

- queue-first UX
- explanation available for each surfaced opportunity
- monitor-only state for ambiguous content
- human override support everywhere that matters

---

# Part VIII. Immediate engineering task list

## 26. First 20 repo-specific tasks to open as tickets

1. Replace ad hoc `ContentItem` construction in `app/api/routes/search.py` with a mapper.
2. Remove `engagement_score` usage from `search.py` until backed by schema.
3. Add explicit `media_type` population to all response schemas.
4. Standardize access to DB metadata field and stop mixing `metadata` vs `metadata_` semantics.
5. Add `last_fetch_time` and related source checkpoint fields, or remove all assumptions that they exist.
6. Fix `app/ingestion/tasks.py` to correctly execute async connector fetches.
7. Add content dedup/upsert path keyed by `(user_id, source_platform, source_id)`.
8. Remove `feeds` assumptions from `sources.py` or add explicit persistence support for them.
9. Replace direct connector registry route usage with a service/factory layer.
10. Delete or fully rewrite broken `SourceConfig` references in `app/mcp_server/server.py`.
11. Replace stale MCP imports for nonexistent modules.
12. Split `app/core/models.py` into layered domain models.
13. Introduce `NormalizedObservation` and migrate ingestion outputs to it.
14. Introduce `SignalInference` persistence and schema.
15. Build a baseline signal taxonomy and label guide.
16. Implement embedding-based candidate generation.
17. Implement baseline multiclass signal classifier.
18. Implement confidence calibration and abstention logic.
19. Add `/api/v1/signals` and `/api/v1/opportunities` endpoints.
20. Build offline evaluation harness for classification, ranking, and calibration.

---

# Final conclusion

The old files show a repo that is ambitious and partly sophisticated, but still organized around the wrong primitive: **digest generation over content clusters**.

Your next leap should not be more broad features. It should be a repo-wide shift toward:

- strict contracts
- normalized observations
- calibrated signal inference
- ranked actionable opportunities
- policy-constrained response planning
- evaluation-driven trust

That is how this codebase becomes a real AI and ML action engine instead of a feature-rich but brittle aggregation system.

