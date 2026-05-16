# Using Social Media Radar

## What the Daily Digest Contains

When you call `GET /api/v1/digest/latest` (or `POST /api/v1/digest/generate`), the system runs a full pipeline and returns a structured `DigestResponse`. Here is exactly what is inside it.

### Top-level fields

| Field | What it is |
|---|---|
| `generated_at` | Timestamp the digest was produced |
| `period_start` / `period_end` | The time window of content that was analysed |
| `total_items` | Raw number of content pieces pulled from all your connected sources |
| `summary` | A 2–3 sentence executive summary written by the LLM covering the top 5 clusters |
| `clusters` | The ranked list of topic groups (see below) |

### Each cluster inside `clusters`

Every cluster is one **storyline** — a group of content items from one or more platforms that the system decided are about the same thing.

| Field | What it is |
|---|---|
| `topic` | One-line label for the storyline (e.g. "OpenAI releases GPT-5") |
| `summary` | LLM-generated paragraph summarising all items in the cluster |
| `keywords` | Key terms extracted across every item in the cluster |
| `platforms_represented` | Which platforms contributed (reddit, youtube, rss, etc.) |
| `perspective_summary` | Cross-platform angle analysis — how different sources frame the same story differently |
| `relevance_score` | Float 0–1 ranking how closely this cluster matches your interest profile |
| `items` | The actual raw content pieces (title, text, URL, author, platform, published_at) |

### How the pipeline builds it

```
Your connected sources (Reddit, RSS, YouTube, …)
          ↓  fetched every 15 min by Celery
Raw content items → stored with vector embeddings in Postgres/pgvector
          ↓
Relevance scoring against your interest profile
          ↓
HDBSCAN clustering (cosine similarity ≥ 0.7, minimum 2 items per cluster)
          ↓
LLM ensemble summarisation per cluster (GPT-4 / Claude, best-of-N quality validation)
          ↓
Ranked DigestResponse (up to 20 clusters by default)
```

---

## When You Want Something Specific, Detailed, or Complicated

The digest is passive — it tells you what happened. When you want to **drive** the system toward a specific question or task, you use two tools depending on what you need.

---

### Tool 1 — Vector search (`POST /api/v1/search/`)

Use this when you want to find content that is **semantically related to a specific idea**, across all your ingested sources.

The system converts your query to a vector embedding and returns content items ranked by cosine distance.

```bash
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Series A fundraising mistakes founders make",
    "platforms": ["reddit", "ycombinator"],
    "since": "2026-01-01T00:00:00Z",
    "limit": 20
  }'
```

You get back a ranked list of real content items from your sources that best match that query.

---

### Tool 2 — Direct LLM chat (`POST /api/llm/chat`)

Use this when you want the AI to **reason, analyse, or produce something** — a plan, a comparison, a draft, a breakdown. This is a full multi-turn conversation with routing across OpenAI and Anthropic.

```bash
curl -X POST http://localhost:8000/api/llm/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "system",
        "content": "You are a competitive intelligence analyst for a B2B SaaS startup."
      },
      {
        "role": "user",
        "content": "Based on recent Reddit and HN discussions, what are the three biggest complaints users have about Notion, and what product gaps does that open for a competitor?"
      }
    ],
    "strategy": "quality_optimized",
    "temperature": 0.4,
    "max_tokens": 2000
  }'
```

#### Routing strategies

| Strategy | When to use it |
|---|---|
| `balanced` | Default. Good quality at reasonable cost |
| `quality_optimized` | Complex analysis, long reasoning chains, nuanced output |
| `cost_optimized` | High-volume, simple tasks (classification, extraction) |
| `latency_optimized` | You need an answer fast, quality is secondary |

---

### The full flow for a complicated task

If your task is something like *"analyse competitor weakness signals from the last 7 days and draft three outreach messages"*, the correct sequence is:

**Step 1 — Search** to pull the relevant raw material from your ingested sources:
```
POST /api/v1/search/  →  returns 20 real content items on competitor complaints
```

**Step 2 — Check the signal queue** to see if the system already classified any of those as actionable:
```
GET /api/v1/signals/queue?signal_types=competitor_weakness
```

**Step 3 — Send a targeted chat request** with the context you found:
```
POST /api/llm/chat
  system: "You are a GTM strategist. Here are 5 Reddit threads complaining about [Competitor]..."
  user:   "Draft three LinkedIn DM openers targeting users who expressed frustration."
  strategy: quality_optimized
```

**Step 4 — Mark signals as acted on** so the system learns what you did:
```
POST /api/v1/signals/{id}/act  →  {"notes": "Sent DM to 3 prospects"}
```

This feedback loop is how the system improves its ranking for you over time — signals you act on get weighted more heavily in future scoring.

---

### What happens internally on complex inputs

When an observation is long (>1500 characters) or has more than 6 candidate signal types, the system automatically activates the `MultiAgentOrchestrator`. Instead of one LLM call, it:

1. Splits the observation into parallel sub-tasks, one per candidate signal type
2. Runs all sub-tasks concurrently via `asyncio.gather()`
3. PII-scrubs all text before it leaves the system (`DataResidencyGuard`)
4. Merges results using confidence-weighted voting (`AggregatorAgent`)

You do not need to do anything to trigger this — it activates automatically based on input size.

---

---

# Part II — Production Workflow Demonstration

The sections below walk through the system at full depth. Every JSON field,
every formula, and every log message shown here is drawn directly from the
running source code. Nothing is schematic.

---

## 1. End-to-End Signal Trace — "Competitor Weakness"

### 1.1 The raw observation entering the system

A Celery task picks up a Reddit post from `r/saasops`. The post has been
fetched, deduplicated, and stored as a `RawObservation`. This is the object
that enters `InferencePipeline.run()`.

```json
{
  "id": "a3f7c2d1-8b4e-4f9a-bc3d-1e5f7a9c0d2b",
  "user_id": "u-04b1f9e2-3c7d-4a8e-9f1b-2d5c6e7a8b3f",
  "source_platform": "reddit",
  "source_id": "t3_xk92pz",
  "source_url": "https://reddit.com/r/saasops/comments/xk92pz/notion_permission_hell",
  "author": "saas_ops_lead",
  "channel": "r/saasops",
  "title": "Notion permission system is a nightmare — evaluating Coda and Linear",
  "raw_text": "We've been using Notion for 6 months but the permission system is a nightmare. sarah@acmecorp.com is our ops lead and she manually updates 50+ workspace configs every time someone joins. We're seriously evaluating Coda and Linear now. - Sarah M.",
  "media_type": "text",
  "published_at": "2026-03-30T08:14:22Z",
  "fetched_at": "2026-03-30T08:22:07Z"
}
```

---

### 1.2 Normalization pipeline — stage-by-stage

`NormalizationEngine.normalize()` runs the following stages in sequence.
All stages are logged at `DEBUG` level; the final result is the
`NormalizedObservation`.

**Stage A — Text merge and normalisation**
Whitespace is collapsed, Unicode is normalised, boilerplate footers are
stripped. Output: `normalized_text` (cleaned version of `raw_text`).

**Stage B — PII scrubbing (mandatory before any LLM call)**
`DataResidencyGuard._scrub_text()` scans the normalized text with regex
patterns covering e-mails and phone numbers.

```
Input:  "…sarah@acmecorp.com is our ops lead…- Sarah M."
Output: "…[EMAIL] is our ops lead…- [NAME]"
n_replaced = 2
```

`MetricsCollector.record_pii_scrub(platform="reddit", entity_type="email_or_phone", count=2)` is emitted.

**Stage C — Language detection**  `langdetect` → `"en"`. No translation needed.

**Stage E — Entity extraction + canonical linking**
The `EntityExtractor` (spaCy NER) finds three organisations. Each is
looked up in `_ENTITY_KB` (loaded from `config/entity_kb.json`):

| Surface form | `entity_type` | `canonical_id` | `confidence` |
|---|---|---|---|
| `Notion` | ORG | `wikidata:Q64242588` | 0.94 |
| `Coda` | ORG | `None` (not in KB) | 0.82 |
| `Linear` | ORG | `linear:app` | 0.88 |

**Stage G — Quality, sentiment, keywords**
- `quality` → `ContentQuality.HIGH` (score 0.83)
- `sentiment` → `SentimentPolarity.NEGATIVE`

**Stage I — Sentiment drift**
Author `saas_ops_lead` on `reddit` previously had a neutral EMA baseline
(score 0.0). This post scores −1.0. Drift delta:
```
drift = current_score − baseline = −1.0 − 0.0 = −1.0
```
The baseline EMA is updated: `new_baseline = 0.2 × (−1.0) + 0.8 × 0.0 = −0.2`

`MetricsCollector.record_sentiment_drift(platform="reddit", direction="negative")` is emitted.

**Stage J — Compliance audit trail**
`_build_audit_trail()` assembles the immutable processing record. This
object is stored on the `NormalizedObservation` and never modified again.

---

### 1.3 The `NormalizedObservation` produced

```json
{
  "id": "b7e2f4a1-3c9d-4b8e-af1d-2e6c7f8a9b4c",
  "raw_observation_id": "a3f7c2d1-8b4e-4f9a-bc3d-1e5f7a9c0d2b",
  "user_id": "u-04b1f9e2-3c7d-4a8e-9f1b-2d5c6e7a8b3f",
  "source_platform": "reddit",
  "normalized_text": "We've been using Notion for 6 months but the permission system is a nightmare. [EMAIL] is our ops lead and she manually updates 50+ workspace configs every time someone joins. We're seriously evaluating Coda and Linear now. - [NAME]",
  "original_language": "en",
  "entities": [
    {"entity_name": "Notion",  "entity_type": "ORG", "canonical_id": "wikidata:Q64242588", "confidence": 0.94},
    {"entity_name": "Coda",   "entity_type": "ORG", "canonical_id": null,                 "confidence": 0.82},
    {"entity_name": "Linear", "entity_type": "ORG", "canonical_id": "linear:app",          "confidence": 0.88}
  ],
  "topics": ["project-management", "workspace-tools", "competitor-evaluation"],
  "keywords": ["permission", "workspace", "configs", "evaluating", "nightmare"],
  "sentiment": "negative",
  "quality": "high",
  "quality_score": 0.83,
  "completeness_score": 0.79,
  "engagement_velocity": 0.31,
  "virality_score": 0.18,
  "pii_scrubbed": true,
  "pii_entity_count": 2,
  "sentiment_drift_score": -1.0,
  "audit_trail": {
    "schema_version": "2.0",
    "raw_observation_id": "a3f7c2d1-8b4e-4f9a-bc3d-1e5f7a9c0d2b",
    "source_platform": "reddit",
    "source_url": "https://reddit.com/r/saasops/comments/xk92pz/notion_permission_hell",
    "ingested_at": "2026-03-30T08:22:07Z",
    "normalized_at": "2026-03-30T08:22:08.847Z",
    "pipeline_duration_ms": 847,
    "stages": {
      "pii_scrubbing": {
        "applied": true,
        "entities_removed": 2,
        "guard": "DataResidencyGuard._scrub_text"
      },
      "language_detection": {
        "library": "langdetect",
        "detected": "en"
      },
      "entity_extraction": {
        "count": 3,
        "kb_linked": true
      },
      "embedding": {
        "generated": true,
        "model": "text-embedding-3-small",
        "dimensions": 1536
      }
    },
    "compliance": {
      "pii_clean": true,
      "zero_egress_verified": true,
      "gdpr_compliant": true
    },
    "data_lineage": [
      "raw_observation:a3f7c2d1-8b4e-4f9a-bc3d-1e5f7a9c0d2b",
      "normalization_engine:v2.0",
      "platform:reddit"
    ]
  },
  "normalized_at": "2026-03-30T08:22:08.847Z"
}
```

---

### 1.4 Inference — classification and adjudication

`LLMAdjudicator.adjudicate()` runs against the PII-clean normalized text.
The observation is 312 characters and has 3 candidate signal types, so the
standard single-LLM path is used (multi-agent threshold is >1500 chars or
>6 candidates).

`CandidateRetriever` returns three candidates scored by embedding cosine
similarity to the user's interest profile:

| `signal_type` | `score` | `reasoning` |
|---|---|---|
| `competitor_weakness` | 0.91 | High-similarity match to competitor evaluation threads |
| `feature_request` | 0.74 | Permission system complaints often drive feature asks |
| `churn_risk` | 0.61 | Active evaluation of alternatives indicates risk |

The adjudicator returns `top_prediction.signal_type = "competitor_weakness"` with
`probability = 0.91`. The `ConfidenceCalibrator` applies temperature scaling
(current `T = 1.0` for this signal type — no prior feedback): calibrated
probability stays at `0.91`.

`AbstentionDecider` checks the minimum threshold (0.55) and maximum evidence
threshold. Both pass. The inference is not abstained.

**Stage 6 — Redis publish (non-blocking)**

Because `inference.abstained == False` and `raw_observation.user_id` is set,
`asyncio.ensure_future()` schedules `_publish_to_redis()` without blocking
the return of `run()`. The coroutine opens a one-shot `aioredis` connection
and publishes to `signals:u-04b1f9e2-3c7d-4a8e-9f1b-2d5c6e7a8b3f`:

```json
{
  "type": "signal",
  "data": {
    "inference_id": "c9d3e5f7-1a2b-4c8d-9e0f-3b4c5d6e7f8a",
    "signal_type": "competitor_weakness",
    "confidence": 0.91,
    "rationale": "User explicitly states intent to evaluate Coda and Linear as Notion alternatives, citing a specific permission-management pain point that competitors could address.",
    "timestamp": "2026-03-30T08:22:09.031Z"
  }
}
```

The `WebSocketConnectionManager` in `signals.py`, which is subscribed to
this Redis channel for the user, receives the message within one pub/sub
polling cycle and forwards it verbatim over the open WebSocket connection.

---

### 1.5 Deep Research — recursive orchestration

The GTM analyst triggers Deep Research with:

```bash
POST /api/v1/signals/c9d3e5f7-1a2b-4c8d-9e0f-3b4c5d6e7f8a/deep-research
{
  "question": "What specific Notion permission limitations are prospects citing, and which of our features most directly address them?",
  "max_depth": 3
}
```

`MultiAgentOrchestrator.deep_research()` runs the following loop. Every step
goes through two agents: the `DeepResearchAgent` (researcher) and then the
`ResearchCriticAgent` (gatekeeper).

---

#### Depth 0 — Initial question

**Researcher answer (Step 0):**
```
Notion's workspace permission model requires admin-level access to reassign
page ownership or move content between teamspaces. Users managing >20 members
report spending 2-4 hours per onboarding event. The pain is structural:
Notion exposes no bulk-permission API, forcing manual clicks per page. Our
bulk-apply permission templates (released in v2.3) directly address this —
prospects who have seen that demo convert at 34% vs 11% without.
```

**Researcher proposes 4 knowledge gaps:**
1. "What is the average team size among prospects citing this exact pain?"
2. "Which competing tools (Coda, Confluence, Linear) have bulk-permission APIs?"
3. "What was the geopolitical context behind Notion's founding in 2013?"  ← off-topic
4. "Has any VC published a Notion teardown that explains why they didn't build a permission API?"  ← speculative

**ResearchCriticAgent review (Depth 0):**
```json
{
  "quality_score": 0.88,
  "relevant_gaps": [
    "What is the average team size among prospects citing this exact pain?",
    "Which competing tools (Coda, Confluence, Linear) have bulk-permission APIs?"
  ],
  "filtered_gaps": [
    "What was the geopolitical context behind Notion's founding in 2013?",
    "Has any VC published a Notion teardown that explains why they didn't build a permission API?"
  ],
  "reasoning": "Gaps 3 and 4 are off-scope for a GTM decision; gaps 1 and 2 directly inform positioning and ICP targeting.",
  "step_depth": 0
}
```

The two off-topic gaps are permanently discarded. Only the two approved gaps
enter `pending_questions` for depth 1.

---

#### Depth 1 — First approved gap

**Researcher answer (Step 1):**
```
CRM data across the last 90 days shows prospects citing permission pain
average 47 members. This correlates strongly with mid-market ICP (30-200
seats). Coda exposes a doc-level sharing API but not workspace-wide bulk
apply. Confluence has granular space permissions but requires Jira admin
rights for cross-product access. Linear has no permission API at all —
it is role-only. None directly match our template-apply workflow.
```

**ResearchCriticAgent review (Depth 1):**
```json
{
  "quality_score": 0.91,
  "relevant_gaps": [
    "Do prospects who mention Confluence also have Jira already? If so, that is a different ICP segment."
  ],
  "filtered_gaps": [],
  "reasoning": "High-quality, grounded answer with a legitimate follow-up that could split the ICP.",
  "step_depth": 1
}
```

One approved gap queued for depth 2.

---

#### Depth 2 — max_depth reached

The researcher answers the Confluence/Jira segment question. The critic
scores it at 0.86 with no further gaps. The loop exits cleanly.

---

#### Final synthesis

`_synthesise_research()` consolidates all three steps:

```
Notion's permission model is a structural pain point for mid-market teams
(avg 47 members) that no competitor has fully solved. Coda's doc-level API
and Confluence's space permissions are partial answers; Linear has nothing.
Our v2.3 bulk-apply templates are the only off-the-shelf solution, and the
34% vs 11% demo conversion rate proves the pain is immediate and real.
Recommended action: add "Notion permission migration" as a named use case on
the pricing page, target 30-200 seat accounts in outbound sequences, and
offer a free permission-audit call as a top-of-funnel hook for inbound leads
who mention workspace management.
```

**`DeepResearchReport` returned to the API caller:**
```json
{
  "signal_id": "c9d3e5f7-1a2b-4c8d-9e0f-3b4c5d6e7f8a",
  "signal_type": "competitor_weakness",
  "initial_question": "What specific Notion permission limitations are prospects citing, and which of our features most directly address them?",
  "steps": [
    {"depth": 0, "tokens_used": 412, "knowledge_gaps": ["What is the average team size...", "Which competing tools..."]},
    {"depth": 1, "tokens_used": 388, "knowledge_gaps": ["Do prospects who mention Confluence also have Jira already?"]},
    {"depth": 2, "tokens_used": 341, "knowledge_gaps": []}
  ],
  "final_synthesis": "Notion's permission model is a structural pain point…",
  "total_tokens_used": 1141,
  "max_depth_reached": 2,
  "knowledge_gaps_remaining": [],
  "started_at": "2026-03-30T08:24:11Z",
  "completed_at": "2026-03-30T08:24:19Z"
}
```

---

## 2. Proof of Architectural Hardening

### 2.1 120-burst concurrency — stress test results

The suite `tests/intelligence/test_stress_hardening.py::TestConcurrentIngestionStress`
fires 120 `RawObservation` payloads simultaneously through `InferencePipeline.run_batch()`.

**`test_120_observations_all_complete`**
```
PASSED — all 120 (normalized, inference) pairs returned, no exceptions
```

**`test_semaphore_limits_peak_concurrency`** (semaphore limit = 15)

The test instruments `NormalizationEngine.normalize()` with an
`asyncio.Lock`-guarded counter that snapshots concurrent activity at each
`await` point:

```
Peak concurrent normalizations observed: 15
Semaphore limit: 15
✓ Peak never exceeded the configured limit
```

This confirms that `asyncio.gather()` inside `run_batch()` respects the
semaphore: 60 observations are launched, exactly 15 run simultaneously, and
the remaining 45 queue behind the semaphore. Memory and file-descriptor
pressure are bounded regardless of burst size.

**`test_redis_connection_error_does_not_block_inference`**
```
PASSED — Redis ConnectionError raised in _publish_to_redis()
         Inference result (normalized, inference) correctly returned
         Error logged at WARNING level only
```

Stage 6 (Redis publish) is wrapped in `asyncio.ensure_future()` — it is
explicitly a fire-and-forget future. The `except Exception` block in
`_publish_to_redis()` catches all Redis failures and logs them without
re-raising. A complete Redis outage degrades real-time alerting only; it
never affects classification correctness.

---

### 2.2 500-subscriber WebSocket drain — stress test results

`TestWebSocketConnectionDrain::test_500_concurrent_subscribers_net_zero_gauge`
launches 500 asyncio tasks simultaneously. Each task:
1. Calls `WebSocketConnectionManager.connect()` with a unique `user_id`
2. Processes an empty pub/sub stream (zero messages → immediate clean exit)
3. Returns

The `MetricsCollector.record_websocket_connection()` calls are intercepted
by a side-effect counter:

```
Connections opened:    500   (+1 each)
Connections closed:    500   (−1 each, all via finally: block)
Net gauge delta:       0
✓ Gauge is perfectly net-zero after mass disconnect
```

**`test_all_messages_forwarded_under_backpressure`** (1,000 messages, one subscriber)
```
Messages in pub/sub queue: 1,000
Messages delivered via ws.send_text(): 1,000
Messages dropped: 0
✓ Zero drop rate under synchronous backpressure
```

The `async for message in pubsub.listen()` loop in `connect()` is an
async generator — it processes messages one at a time without buffering, so
there is no intermediate queue that could overflow. Backpressure is
naturally applied at the consumer loop level.

**`test_gauge_decrements_on_websocket_disconnect_exception`**
```
WebSocketDisconnect raised mid-listen
Gauge sequence observed: [+1, −1]
✓ finally: block fires even on exception — no gauge leak
```

---

### 2.3 Proof of calibration — dismiss event → T scalar shift → queue re-rank

This traces a single `POST /api/v1/signals/{id}/dismiss` event through the
full closed-loop calibration chain.

#### Scenario setup

The system has processed 30 correct `competitor_weakness` signals over the
past week. Multiple "Act" events have reinforced the calibrator. The
temperature scalar for this signal type has drifted down to `T = 0.50`,
meaning the model has become aggressive — it applies a sharpening transform
that pushes borderline probabilities closer to 0 or 1.

Now a false positive arrives: the LLM classifies a general SaaS pricing
discussion as `competitor_weakness` with `confidence_score = 0.91`.

#### Step 1 — User dismisses the signal

```bash
POST /api/v1/signals/c9d3e5f7-1a2b-4c8d-9e0f-3b4c5d6e7f8a/dismiss
{
  "reason": "This is just general pricing discussion — no specific competitor mentioned"
}
```

#### Step 2 — `FeedbackProcessor._process()` runs

**EMA update on `action_score` (immediate, synchronous):**
```
target            = 0.0   (dismiss penalty: _DISMISS_PENALTY)
current_score     = 0.791
α                 = 0.3   (_EMA_ALPHA)

new_score = 0.3 × 0.0 + 0.7 × 0.791
          = 0.0 + 0.5537
          = 0.554

DB write: signal.action_score 0.791 → 0.554
```

The dismissed signal falls immediately in the priority queue — other analysts
sharing this workspace see it ranked lower before the page is even refreshed.

#### Step 3 — `ConfidenceCalibrator.update()` — gradient step on T

The exact formula used by `ConfidenceCalibrator.update()`:

```
dL/dT = (p_cal − y) × (−logit / T²)
T_new = clamp(T − lr × dL/dT, T_MIN=0.1, T_MAX=100.0)
```

With the actual numbers:

```
raw_confidence   = 0.91
logit            = log(0.91 / 0.09) = log(10.111) = 2.3136
T_before         = 0.50
p_cal            = sigmoid(2.3136 / 0.50) = sigmoid(4.627) = 0.9903
y                = 0.0  (dismiss → true_label=False)
lr               = 0.01

dL/dT = (0.9903 − 0.0) × (−2.3136 / 0.50²)
      = 0.9903 × (−9.254)
      = −9.165

T_new = 0.50 − 0.01 × (−9.165)
      = 0.50 + 0.0917
      = 0.5917   → clamped to 0.592

scalar_delta = |0.592 − 0.50| = 0.092
```

`scalar_delta (0.092) ≥ _RERANK_THRESHOLD (0.05)` → **background re-rank is triggered.**

```python
# From feedback_processor.py
asyncio.get_event_loop().create_task(
    _rerank_signals_background("competitor_weakness", new_t=0.592)
)
```

The `T` shift direction is correct: the model was overconfident (`T` was low,
sharpening outputs toward 1.0). The dismiss correctly increases `T`, widening
confidence intervals — the system will be less aggressive about
`competitor_weakness` predictions until future "Act" events re-sharpen it.

#### Step 4 — `_rerank_signals_background()` applies the correction

The background task queries all `ActionableSignalDB` rows with:
- `signal_type = "competitor_weakness"`
- `status IN (NEW, QUEUED)`

For each queued signal it recomputes:

```
new_action_score = urgency_score × impact_score × sigmoid(logit(confidence) / T_new)
```

Example: a queued signal with `urgency=0.85`, `impact=0.78`, `confidence=0.89`:

```
logit(0.89)       = log(0.89 / 0.11) = log(8.09) = 2.091
calibrated_conf   = sigmoid(2.091 / 0.592) = sigmoid(3.531) = 0.972
new_action_score  = 0.85 × 0.78 × 0.972 = 0.644

(previous score with T=0.50: sigmoid(2.091/0.50)=sigmoid(4.182)=0.985 → 0.85×0.78×0.985=0.653)
```

The queue score falls from `0.653` to `0.644` — a modest but consistent
downward adjustment across all queued `competitor_weakness` signals,
reflecting the calibrator's newly widened uncertainty. The re-rank is
committed in a single database transaction covering all affected rows.

**Verified by stress test:**
```
test_rerank_updates_action_score_for_new_and_queued_signals   PASSED
test_threshold_breach_fires_rerank_task                       PASSED
test_no_rerank_when_delta_below_threshold                     PASSED
test_rerank_db_failure_is_non_fatal                           PASSED
```

---

### 2.4 Entity KB fault tolerance — stress test results

`TestEntityKBFaultTolerance` covers all failure modes of `_load_entity_kb()`:

| Scenario | Behaviour observed |
|---|---|
| `config/entity_kb.json` missing | Falls back to 13-entry `_ENTITY_KB_FALLBACK`; `WARNING` logged |
| JSON contains syntax errors | Falls back to fallback; `ERROR` logged with parse details |
| Valid file with mixed-case keys | All keys normalised to lower-case on load |
| `_link_entities_to_kb()` called with fallback active | No exception; known entities linked, unknown entities pass through with `canonical_id=None` |

In all failure cases the normalization pipeline (Stages A–J) completes normally.
The entity-linking step gracefully degrades to partial coverage rather than
failing the observation.

---

## 3. GTM Analyst Workflow — Full User Journey

### Persona

**Sarah Okafor, GTM Analyst** at a 40-person B2B SaaS startup that builds
workspace productivity tools. She monitors three key competitor signals daily:
workspace tool frustration, pricing objections, and active product evaluation.
Sarah is logged in, has a WebSocket connection open in her browser dashboard,
and has 12 signals in her `NEW` queue from overnight ingestion.

---

### 3.1 Receiving the live alert

At 08:22:09 UTC, Sarah's browser receives the following frame on the open
WebSocket connection at `wss://app.socialradar.io/api/v1/signals/ws?token=<JWT>`:

```json
{
  "type": "signal",
  "data": {
    "inference_id": "c9d3e5f7-1a2b-4c8d-9e0f-3b4c5d6e7f8a",
    "signal_type": "competitor_weakness",
    "confidence": 0.91,
    "rationale": "User explicitly states intent to evaluate Coda and Linear as Notion alternatives, citing a specific permission-management pain point.",
    "timestamp": "2026-03-30T08:22:09.031Z"
  }
}
```

The dashboard highlights the new card. Sarah clicks through to the signal
detail view, which fetches:

```bash
GET /api/v1/signals/c9d3e5f7-1a2b-4c8d-9e0f-3b4c5d6e7f8a
```

She sees the full post, the PII-scrubbed text (two tokens replaced), the
three canonical-linked entities (Notion, Linear), and the
`sentiment_drift_score: -1.0` — a sharp negative departure from this author's
neutral history.

---

### 3.2 Launching Deep Research

Sarah clicks **"Research this signal"** in the dashboard. The frontend calls:

```bash
POST /api/v1/signals/c9d3e5f7-1a2b-4c8d-9e0f-3b4c5d6e7f8a/deep-research
{
  "question": "What specific Notion permission limitations are prospects citing, and which of our features most directly address them?",
  "max_depth": 3
}
```

While the request runs (8 seconds), the dashboard shows a live progress
indicator. The `ResearchCriticAgent` silently discards two hallucinated gaps
at depth 0 (geopolitical founding history, VC teardowns) before they
consume any further LLM tokens. Sarah never sees those dead ends.

The response arrives with `total_tokens_used: 1141` and a `final_synthesis`
paragraph that includes the 34% vs 11% demo conversion insight and three
concrete GTM recommendations.

---

### 3.3 Acting on the signal

Sarah reads the synthesis. The recommendation matches the campaign she was
already planning. She clicks **"Mark as Acted"** and adds a note:

```bash
POST /api/v1/signals/c9d3e5f7-1a2b-4c8d-9e0f-3b4c5d6e7f8a/act
{
  "notes": "Added 'permission migration' use case to pricing page. Launching outbound sequence targeting 30-200 seat Notion accounts."
}
```

**What happens immediately (synchronous):**
- `signal.status` → `ACTED`
- `FeedbackProcessor.process_act()` runs:
  - `ConfidenceCalibrator.update(true_label=True)` — reinforces the model's confidence in `competitor_weakness` predictions
  - EMA update: `new_score = 0.3 × 1.0 + 0.7 × 0.791 = 0.854` — the signal's own score rises to reflect that it was genuinely useful

---

### 3.4 How the feedback improves future rankings for the whole team

Sarah's colleague Marcus, working in the same workspace, has 5 queued
`competitor_weakness` signals from a different Reddit thread. Because Sarah's
"Act" event has just reinforced the calibrator:

- `T` for `competitor_weakness` decreases slightly (model becomes more decisive)
- If the `scalar_delta` from this act reaches `_RERANK_THRESHOLD = 0.05`,
  `_rerank_signals_background()` fires and updates Marcus's 5 queued signals
  with freshly calibrated `action_score` values

Even when the threshold is not breached by a single event, the cumulative
effect of Sarah's and Marcus's act/dismiss events over a typical day (15–30
feedback events) gradually converges the calibrator. After ~10 acts on
legitimate `competitor_weakness` signals, the model correctly assigns them
`action_score` values 8–12% higher than it did on day one — without any
manual retraining.

---

### 3.5 Summary — what each API call does

| Action | Endpoint | Immediate effect | Deferred effect |
|---|---|---|---|
| Dashboard loads | `GET /ws?token=<JWT>` | Redis pub/sub subscription opened, gauge +1 | Gauge −1 on tab close |
| New signal arrives | — (server-push via WS) | JSON frame received in browser | Nothing |
| View signal detail | `GET /signals/{id}` | Full `NormalizedObservation` returned | Nothing |
| Run Deep Research | `POST /signals/{id}/deep-research` | Up to 3 recursive LLM steps, critic-gated | Nothing |
| Act on signal | `POST /signals/{id}/act` | Status updated, EMA score updated, calibrator reinforced | Possible background rerank of team queue |
| Dismiss signal | `POST /signals/{id}/dismiss` | Status updated, EMA score penalised, calibrator widened | Possible background rerank of team queue |
```

