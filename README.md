# Social-Media-Radar

A locally-deployed inference pipeline for structured signal classification from social media content.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 593 passed](https://img.shields.io/badge/tests-593%20passed-brightgreen.svg)](./docs/TESTING_GUIDE.md)

[Architecture](#architecture) • [Deployment](#local-deployment-guide) • [Getting Started](#getting-started--your-first-signal) • [Performance](#performance-reference) • [Benefits](#expected-benefits-for-teams) • [Testing](#running-the-test-suite) • [Contributing](#contributing)

---

## Overview

Social-Media-Radar is a B2B signal detection system designed for teams that need to extract structured, actionable intelligence from social media at scale. It reads raw posts from platform connectors, normalizes them into a common schema, and classifies each observation into one of 18 defined signal types — from `churn_risk` to `integration_request` — with calibrated confidence scores and verbatim evidence spans.

The system is not a content aggregator or a general-purpose social listening tool. Its scope is narrow and deliberate: given a stream of platform observations, determine what kind of action each one warrants, with enough transparency in the prediction (confidence, evidence, rationale) that a human analyst can make a sound decision about whether to trust it. When the model is not confident enough to commit, it abstains rather than producing a low-quality classification.

All inference runs locally. No observation text, user identifier, or competitive intelligence leaves the machine unless the operator explicitly configures a cloud LLM API endpoint. PII scrubbing is enforced at the pipeline boundary, before any text reaches an LLM.

---

## Architecture

### End-to-End Pipeline

The pipeline has four labelled stages (A–D) and six enhancement components (E1–E6) that slot into the adjudication stage:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INGESTION                                                                  │
│  Platform Connectors (Reddit, YouTube, TikTok, RSS, …)                      │
│           │                                                                 │
│           ▼                                                                 │
│  Stage A  NormalizationEngine                                               │
│           Merge title/body · language detection · entity extraction         │
│           embedding generation · engagement/freshness features              │
│           │                                                                 │
│           ▼                                                                 │
│  ◆        DataResidencyGuard  ──────── PII scrub (email, phone, author)     │
│           Audit log written before any text reaches an LLM                  │
│           │                                                                 │
│           ▼                                                                 │
│  Stage B  CandidateRetriever                                                │
│           HNSW similarity to exemplar bank · entity rules                   │
│           platform priors · lightweight classifier                          │
│           → top-k SignalCandidates with weak prior scores                   │
│           │                                                                 │
│           ▼                                                                 │
│  E6       DeliberationEngine                                                │
│           Step A: landscape scan (ContextMemoryStore, top-5 similar past)  │
│           Step B: candidate pruning (zero-history + low-score removal)      │
│           Step C: risk escalation check → audit log                         │
│           Step D: reasoning mode selection                                  │
│           │                                                                 │
│           ├──── single_call ────────────────────────────────────────┐       │
│           ├──── chain_of_thought ──► E1 ChainOfThoughtReasoner      │       │
│           └──── multi_agent ───────► E3 MultiAgentOrchestrator      │       │
│                                                                     │       │
│  Stage C  LLMAdjudicator ◄──────────────────────────────────────────┘       │
│           Few-shot prompt · JSON schema enforcement · retry/repair           │
│           E5 ContextMemoryStore: inject past observations as context        │
│           E5 ContextMemoryStore: store result for future retrievals         │
│           │                                                                 │
│           ▼                                                                 │
│  E2       ConfidenceCalibrator                                              │
│           Per-SignalType temperature scaling: sigmoid(logit / T)            │
│           T learned from JSONL dataset and updated online via E4            │
│           │                                                                 │
│  E4       FeedbackStore ──────────────────────────────────────────►  DB     │
│           Human correction → calibrator.update() → T adjusted              │
│           │                                                                 │
│  Stage D  AbstentionDecider                                                 │
│           confidence < threshold · disagreement · context completeness      │
│           │                                                                 │
│           ▼                                                                 │
│  OUTPUT   ActionableSignal → PostgreSQL + REST API + SSE stream             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage A — Normalization

`NormalizationEngine` converts a `RawObservation` (the connector's output) into a `NormalizedObservation`. This involves merging title and body text, detecting the source language, optionally translating non-English content, running named-entity extraction, computing engagement and freshness features, and generating a semantic embedding. The normalized schema is fixed — every downstream component, from candidate retrieval to the LLM adjudicator, relies on the same field contract defined in `NormalizedObservation`.

After normalization and before any further processing, every observation passes through `DataResidencyGuard`. See [Data Residency and PII Scrubbing](#data-residency-and-pii-scrubbing) for details.

### Stage B — Candidate Retrieval

`CandidateRetriever` produces a ranked list of `SignalCandidate` objects — hypotheses about which signal types are plausible — without committing to a final classification. It combines three weak signals: embedding similarity against a bank of canonical exemplars (retrieved via an HNSW index), entity-conditioned rules that recognize specific vocabulary patterns, and platform-specific priors (the base rate of each signal type varies by platform). These are merged into a per-candidate score, and the top-k candidates are passed forward.

This stage exists because calling a frontier LLM with all 18 signal types and no pre-filtering is both expensive and less accurate than providing a focused, ranked shortlist. The retrieval stage is fast (sub-100ms) and narrows the hypothesis space before the costly adjudication call.

### Stage C — LLM Adjudication

`LLMAdjudicator` is the main classification stage. It assembles a structured prompt from the normalized observation text, the candidate shortlist, optional few-shot examples retrieved from `ContextMemoryStore` (E5), and the system-level taxonomy and abstention rules. It calls the LLM via `LLMRouter` (see [LLM Routing Strategy](#llm-routing-strategy)), parses the JSON response against a strict schema, and retries with a repair prompt on parse failure. The adjudicator produces evidence spans (verbatim excerpts with relevance reasoning), a structured rationale, and a confidence score.

Before adjudication, `DeliberationEngine` (E6) runs to prune the candidate list and select one of three reasoning paths. Depending on that selection, the adjudicator delegates to `ChainOfThoughtReasoner` (E1) or `MultiAgentOrchestrator` (E3) rather than issuing a single call.

### Stage D — Calibration and Abstention

The raw LLM confidence is a probability, but LLMs are systematically miscalibrated — they tend toward overconfidence for common signal types and underconfidence for rare ones. `ConfidenceCalibrator` (E2) applies a per-type temperature-scaling transform before the abstention decision is made. `AbstentionDecider` then applies threshold rules to the calibrated score. If the observation does not meet the quality bar, the pipeline records the classification as abstained with a structured reason rather than surfacing a low-confidence result to the signal queue.

---

## Deliberation Engine (E6)

Before the LLM adjudicator runs, `DeliberationEngine` executes four steps that determine how the adjudication should proceed. These steps are deliberate rather than computational shortcuts — each one exists to address a specific failure mode that appears in production classification.

**Step A — Landscape Scan.** The engine queries `ContextMemoryStore` for the five most similar past observations for the same user. This surfaces any recurring patterns (e.g., a user's customers who consistently raise the same type of complaint under different wording) that would not be visible from the current observation alone. The results are passed to the LLM adjudicator as few-shot context, effectively giving the model access to the user's signal history without requiring a long-context window or a separate retrieval pass.

**Step B — Candidate Pruning.** The retrieval stage may surface marginal candidates — signal types that score weakly and have no supporting history. Keeping them in the prompt adds noise without adding signal. Step B removes any candidate whose retrieval score falls below a configurable threshold and for which no similar past observation exists. A safety net ensures the list is never emptied entirely: if all candidates would be pruned, the full original list is preserved.

**Step C — Risk Escalation Check.** Four signal types are designated as high-stakes: `churn_risk`, `legal_risk`, `security_concern`, and `reputation_risk`. When any of these appears among the surviving candidates with a score above 0.5, the engine writes a structured entry to the audit logger (`radar.data_residency.audit`) before adjudication proceeds. This creates an immutable audit trail for every potentially high-consequence classification, independent of whether the LLM ultimately assigns that type as the primary prediction.

**Step D — Reasoning Mode Selection.** The final step selects one of three reasoning paths based on observable properties of the observation and the candidate list. If the normalized text exceeds 1,500 characters or there are more than six candidates, the multi-agent path is selected, because a single LLM call cannot reliably hold a long document and a broad hypothesis set in working memory simultaneously. If the top two candidates have retrieval scores within 0.1 of each other — an ambiguous case — or if the caller has set `confidence_required` above 0.85, chain-of-thought reasoning is selected to force the model to reason step by step before committing. All other cases fall through to a single-call adjudication, which is faster and cheaper and appropriate when the hypothesis space is narrow and the content is short.

---

## LLM Routing Strategy

`LLMRouter` implements a two-tier model selection strategy. Signal types are divided at module load time into two disjoint sets: a frontier set (`churn_risk`, `legal_risk`, `security_concern`, `reputation_risk`) and a fine-tuned set (the remaining 14 types). When the adjudicator identifies the top candidate type before calling the router, it passes that type as a routing hint. The router always sends frontier-type observations to the highest-capability model available (e.g., GPT-4o). Observations in the fine-tuned set route to a smaller, cheaper fine-tuned model when one is configured.

The rationale is error asymmetry. A false positive on `legal_risk` carries a very different cost than a false positive on `feature_request`. For the four high-stakes types, the cost of a missed or wrong classification is high enough that the per-call cost of a frontier model is justified. For the remaining 14 types, a fine-tuned smaller model trained on the JSONL dataset can achieve comparable F1 at a fraction of the inference cost. On batches of non-critical observations, this separation reduces LLM spend by roughly 70–80% without meaningful accuracy loss for the types it covers.

The frontier set is defined as a module-level `frozenset` in `app/llm/router.py` and imported by `DeliberationEngine`. There is a single source of truth: changing the set in `router.py` is sufficient to propagate the change to both routing and escalation logic.

---

## Confidence Calibration (E2)

`ConfidenceCalibrator` applies per-`SignalType` temperature scaling: given a raw log-odds score from the LLM, it computes `sigmoid(logit / T)`, where `T` is a learned scalar for that signal type. When `T = 1.0` (the default), the transform is mathematically identical to the plain sigmoid, so the system is fully functional without any training.

Temperature scaling was chosen over Platt scaling and isotonic regression for three reasons. First, it has a single parameter per class, making it tractable to update online after every human feedback event. Platt scaling's two-parameter logistic requires a held-out validation set for reliable fitting, and isotonic regression requires enough labeled examples to reliably estimate a piecewise-monotone function — neither is practical for a 18-class system with sparse feedback in the early weeks of deployment. Second, temperature scaling preserves the relative ordering of probabilities within a class, so its effect on ranking is predictable. Third, it composes cleanly with the existing sigmoid output: the calibrated probability is always in [0, 1] and needs no post-hoc clamping.

The `update()` method performs a single gradient-descent step on binary cross-entropy, adjusting `T` based on one labeled example. This online update is what `FeedbackStore` (E4) calls after every analyst correction routed through the `POST /{signal_id}/feedback` API endpoint. Batch calibration on the 107-example JSONL seed dataset takes under one second on any hardware; online updates keep the scalars current as the model sees production traffic, which would not be possible with a batch-only approach.

---

## Context Memory Store (E5)

`ContextMemoryStore` maintains a per-user vector index of past `(observation, inference)` pairs. When the adjudicator processes a new observation, it queries this store for the top-k most similar past observations by cosine similarity, and injects them as few-shot context in the prompt. This gives the LLM access to the user's specific classification history — the kinds of signals that appear in their product's community, the language patterns their customers use — without any retraining.

The store uses cosine similarity over dense vector representations. When a live embedding API is available (e.g., OpenAI embeddings), the stored vectors are full semantic embeddings. When no API is configured, the store falls back to a 512-dimensional bag-of-words representation with L2 normalization. The trade-off is accuracy versus operational dependency: the BOW fallback runs entirely offline and adds no API latency, but it misses semantic equivalences (synonyms, paraphrases) that a trained embedding model would capture. For most signal types, where classification-relevant vocabulary is domain-specific and consistent, the BOW fallback is sufficient to surface meaningfully similar past examples.

The store holds up to 10,000 records per user. When the limit is reached, least-recently-used records are evicted. Abstained inferences are not stored, since they carry no ground-truth signal type that would be useful in future few-shot context.

---

## Data Residency and PII Scrubbing

`DataResidencyGuard` enforces a zero-egress contract: no observation text that contains PII is permitted to reach an LLM call. The guard runs synchronously in the pipeline between normalization and candidate retrieval, before any text is assembled into a prompt.

The scrubbing process covers four fields. Author handles are replaced with a deterministic SHA-256-derived pseudonym of the form `anon_<16 hex chars>`, so downstream analysis can still correlate observations by author without knowing the real identity. PII query parameters in source URLs (email addresses, user IDs, tracking tokens) are replaced with the literal string `<redacted>`. Email addresses and phone numbers in the raw text body are replaced with `<email_redacted>` and `<phone_redacted>` tokens respectively. The same scrubbing is applied recursively to any metadata dictionary attached to the observation.

Every redaction generates an immutable `RedactionAuditEntry` written to the structured audit logger before the cleaned record proceeds. The guard is idempotent: passing an already-redacted observation through it again produces no new redactions and no audit entries. A `verify_clean()` method is available for use at the LLM call boundary as a final safety check — it raises a `DataResidencyViolationError` if any detectable PII pattern survives.

In the multi-agent orchestrator (E3), each `SubTaskAgent` call additionally passes the observation text through `DataResidencyGuard._scrub_text()` before constructing its sub-task prompt, providing a second layer of enforcement specific to the parallel-call path.

---

## Signal Taxonomy

The system classifies observations into 18 signal types organized into four groups:

**Customer signals** — direct feedback from users about their experience with a product or service: `support_request`, `feature_request`, `bug_report`, `complaint`, `praise`.

**Market signals** — indications of buying intent, competitive evaluation, or ecosystem gaps: `competitor_mention`, `alternative_seeking`, `price_sensitivity`, `integration_request`.

**Risk signals** — observations that may require urgent human review: `churn_risk`, `security_concern`, `legal_risk`, `reputation_risk`. These four types route to the frontier LLM tier and trigger audit log entries in the deliberation engine.

**Opportunity signals** — positive commercial indicators: `expansion_opportunity`, `upsell_opportunity`, `partnership_opportunity`.

**Meta types** — used when the classification system cannot assign a substantive type: `unclear`, `not_actionable`. These are distinct from abstention; an observation classified as `not_actionable` is a confident prediction that the content carries no actionable signal, whereas abstention means the model was not confident enough to classify it at all.

### Abstention

The pipeline abstains rather than produce a low-confidence result. Abstention is recorded as a structured outcome with one of seven reasons: `low_confidence` (calibrated probability below the threshold), `ambiguous_multi_label` (two or more types are equally likely), `insufficient_context` (the observation references a thread or conversation that is not available), `out_of_distribution` (the content is unlike anything in the training distribution), `unsafe_to_classify` (the content involves legal, political, or safety-sensitive material where a wrong classification carries high cost), `language_barrier` (translation quality was too low for reliable inference), or `spam_or_noise` (content quality below the minimum threshold).

The abstention threshold is configured in `AbstentionThresholds` and defaults to a minimum confidence of 0.7 (after calibration). The deliberation engine's `confidence_required` field can raise this threshold per-observation when the caller requires high certainty.

---

## How This Differs from OpenClaw

OpenClaw (openclaw.ai) is an open-source, locally-deployed personal AI assistant. It is a general-purpose agentic system — it executes tasks, controls browsers, sends messages, and integrates with personal productivity tools. It is not a signal classification system, and the comparison below is architectural rather than competitive: they address different problems.

The differences that are verifiable from this codebase and from OpenClaw's public documentation are as follows.

**Domain and scope.** Social-Media-Radar is purpose-built for B2B signal detection. Its entire classification apparatus — the 18-type taxonomy, the exemplar bank, the few-shot prompting strategy, the abstention conditions — is tuned for the specific task of distinguishing actionable business signals from noise in social media content. OpenClaw provides no structured signal taxonomy and publishes no signal detection accuracy metrics, because classification accuracy is not its objective.

**Calibrated confidence and abstention.** Social-Media-Radar produces calibrated probability outputs with per-type temperature scaling, evidence spans, and a structured abstention mechanism. OpenClaw passes LLM outputs directly to action with no post-hoc calibration layer. For a classification system where a wrong action (e.g., escalating a `legal_risk` that is actually a `complaint`) has real cost, calibration and abstention are not optional features — they are the primary quality mechanism.

**Multi-platform social ingestion pipeline.** Social-Media-Radar has a layered ingestion pipeline (raw observations → normalization → candidate retrieval → LLM adjudication) designed around the structure of social media content. OpenClaw has no equivalent ingestion pipeline for social platforms and no candidate retrieval stage.

**Team workflow.** Social-Media-Radar routes classified signals into a structured queue with priority scoring, role-based assignment, SSE streaming, and a human feedback loop that drives online calibration updates. OpenClaw is a single-user assistant with no concept of a shared signal queue or team workflow.

**Evaluation infrastructure.** Social-Media-Radar includes evaluation tooling for ECE (expected calibration error), macro F1, and NDCG@10. These metrics have no counterpart in OpenClaw because task completion rather than classification accuracy is OpenClaw's primary measure.

**What OpenClaw does that this system does not.** OpenClaw has persistent cross-session memory, browser control, system-level shell access, a community skill marketplace (ClawHub), and proactive scheduling (heartbeats, cron jobs). Social-Media-Radar has none of these, by design — they are outside its scope.

---

## Local Deployment Guide

This section covers every path to running the full Social-Media-Radar stack on a local machine — from a single Docker Compose command to a fully manual bare-metal setup. Read the prerequisites table first, then choose the deployment path that matches your environment.

---

### System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **CPU** | 4 cores | 8+ cores (inference is CPU-bound when using a local LLM) |
| **RAM** | 8 GB | 16 GB (required if running Ollama with a 7B+ model locally) |
| **Disk** | 10 GB free | 30 GB free (model weights + Postgres data + MinIO objects) |
| **OS** | macOS 12+, Ubuntu 20.04+, WSL2 (Windows 11) | macOS 14+ on Apple Silicon or Ubuntu 22.04 LTS |
| **Python** | 3.9 | 3.11 (recommended; ships `tomllib` and has faster asyncio) |
| **Docker** | 24.0+ with Compose v2 | Docker Desktop 4.28+ |
| **Internet** | Required at first run | Only needed for LLM API calls if not using a local model |

> **Apple Silicon note (M1/M2/M3/M4):** all packages in `requirements.txt` ship ARM64-native wheels under the version pins given. No Rosetta translation or source compilation is required.

---

### Option A — Docker Compose (Recommended for All Platforms)

This is the fastest path. A single command starts PostgreSQL 15 with pgvector, Redis 7, MinIO, the FastAPI application, a Celery worker, and Celery Beat. Database migrations run automatically before the API accepts connections.

**Prerequisites:** Docker Desktop 24.0+ or Docker Engine 24.0+ with the Compose v2 plugin (`docker compose version` must print `v2.x.x` or later).

#### Step 1 — Clone the repository

```bash
git clone https://github.com/yourusername/social-media-radar.git
cd social-media-radar
```

#### Step 2 — Generate secrets and create your `.env` file

```bash
cp .env.example .env
```

Now open `.env` and fill in the three required values. The remaining defaults work out of the box for local Docker Compose:

```bash
# Generate SECRET_KEY
python3 -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"

# Generate ENCRYPTION_KEY
python3 -c "import secrets; print('ENCRYPTION_KEY=' + secrets.token_urlsafe(32))"
```

Paste the output of each command into `.env`. Then add your LLM API key:

```
OPENAI_API_KEY=sk-...         # required for LLM adjudication
ANTHROPIC_API_KEY=sk-ant-...  # optional — enables the Anthropic routing tier
```

> **What you do NOT need to change in `.env` for Docker Compose:** `DATABASE_URL`, `REDIS_URL`, `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, and all Celery URLs. The `docker-compose.yml` overrides these with Docker-internal hostnames automatically.

**Complete list of required `.env` variables:**

| Variable | Where to get it | Example |
|---|---|---|
| `SECRET_KEY` | Generate locally (see above) | 32-byte random string |
| `ENCRYPTION_KEY` | Generate locally (see above) | 32-byte random string |
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/) | `sk-proj-…` |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) — optional | `sk-ant-…` |

#### Step 3 — Start the full stack

```bash
docker compose up
```

The first run downloads base images and builds the application image. Expect 3–5 minutes on a fast connection. On subsequent runs the images are cached; startup takes under 30 seconds.

**What happens automatically on first run:**

1. PostgreSQL container starts and passes its health check (`pg_isready`).
2. Redis container starts and passes its health check (`redis-cli ping`).
3. The `db-init` container runs `python scripts/init_db.py` (creates the `vector` extension) followed by `alembic upgrade head` (applies all schema migrations), then exits with code 0.
4. The `api`, `celery-worker`, and `celery-beat` containers start only after `db-init` completes successfully.
5. On startup, the API sends a `PING` to Redis. If it fails, the process exits with a non-zero code and Docker restarts it rather than serving traffic with a broken blacklist.

**Expected console output when healthy:**

```
radar-db-init      | INFO  [alembic.runtime.migration] Running upgrade -> 000_initial
radar-db-init      | INFO  [alembic.runtime.migration] Running upgrade 000_initial -> 001_signals
radar-db-init exited with code 0
radar-api          | INFO     Redis connectivity confirmed (PING → PONG) at redis://redis:6379/0
radar-api          | INFO     Application startup complete.
radar-api          | INFO     Uvicorn running on http://0.0.0.0:8000
radar-celery-worker | celery@... ready.
```

#### Step 4 — Run initial calibration

In a separate terminal (while the stack is running):

```bash
docker compose exec api python training/calibrate.py --epochs 5
```

Expected output:

```
Calibration complete: 535 updates, 0 skipped
State written to: training/calibration_state.json
```

This adjusts the per-signal-type temperature scalars in `ConfidenceCalibrator` using the 107-example seed dataset. Without this step the calibrator runs with `T = 1.0` for all types, which is mathematically correct but not optimally calibrated. The calibration step takes under 2 seconds on any hardware.

#### Step 5 — Verify the deployment

```bash
# Health check
curl -s http://localhost:8000/health | python3 -m json.tool
# Expected: {"status": "healthy", "database": "ok", "redis": "ok"}

# OpenAPI documentation
open http://localhost:8000/docs       # macOS
xdg-open http://localhost:8000/docs  # Linux
```

**Service port map:**

| Service | Local URL | Purpose |
|---|---|---|
| FastAPI | `http://localhost:8000` | REST API + SSE streaming |
| API docs | `http://localhost:8000/docs` | Interactive OpenAPI UI |
| MinIO console | `http://localhost:9001` | Object storage browser (admin/minioadmin) |
| PostgreSQL | `localhost:5432` | Direct DB access for inspection |
| Redis | `localhost:6379` | Cache and task broker |

#### Stopping and restarting

```bash
docker compose down          # stop all containers, preserve volumes
docker compose down -v       # stop all containers and DELETE all data (full reset)
docker compose up -d         # start in detached (background) mode
docker compose logs -f api   # tail API logs
```

---

### Option B — macOS Bare-Metal (Apple Silicon and Intel)

Use this path when you prefer to run the application process directly without Docker, or when you need to attach a debugger to the FastAPI process.

#### Step 1 — Install system dependencies

**Homebrew packages (Apple Silicon):**

```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Postgres 15 with pgvector, Redis, and MinIO
brew install postgresql@15 pgvector redis minio/stable/minio

# Add postgres to PATH (Apple Silicon path)
echo 'export PATH="/opt/homebrew/opt/postgresql@15/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Start services
brew services start postgresql@15
brew services start redis
```

**Python (3.11 recommended):**

```bash
brew install python@3.11
python3.11 --version  # should print Python 3.11.x
```

**Xcode Command Line Tools** (required for packages that need a C compiler):

```bash
xcode-select --install
```

#### Step 2 — Create a virtual environment and install dependencies

```bash
git clone https://github.com/yourusername/social-media-radar.git
cd social-media-radar

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Verify the two ARM64-critical packages:

```bash
pip show asyncpg numpy | grep -E "^(Name|Version)"
# asyncpg  >= 0.28.0  (earlier versions lack ARM64 wheels)
# numpy    >= 1.24    (first release with universal2 macOS wheels)
```

#### Step 3 — Configure environment variables

```bash
cp .env.example .env
```

Generate and insert secrets:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"  # for SECRET_KEY
python3 -c "import secrets; print(secrets.token_urlsafe(32))"  # for ENCRYPTION_KEY
```

Set the bare-metal database URL (replace `<youruser>` with your macOS username):

```
DATABASE_URL=postgresql+asyncpg://<youruser>@localhost:5432/social_radar
DATABASE_SYNC_URL=postgresql://<youruser>@localhost:5432/social_radar
REDIS_URL=redis://localhost:6379/0
```

#### Step 4 — Create the database and run migrations

```bash
# Create the database
createdb social_radar

# Enable the pgvector extension and apply migrations
python scripts/init_db.py
alembic upgrade head
```

Expected migration output:

```
INFO  [alembic.runtime.migration] Running upgrade  -> 000_initial, Initial schema
INFO  [alembic.runtime.migration] Running upgrade 000_initial -> 001_signals, Add actionable signals
```

#### Step 5 — Start MinIO (optional — required only for media storage)

```bash
mkdir -p ~/minio-data
minio server ~/minio-data --console-address :9001 &
```

Create the default bucket:

```bash
pip install minio
python3 -c "
from minio import Minio
c = Minio('localhost:9000', access_key='minioadmin', secret_key='minioadmin', secure=False)
c.make_bucket('radar-content')
print('Bucket created')
"
```

#### Step 6 — Run initial calibration

```bash
python training/calibrate.py --epochs 5
# Expected: "Calibration complete: 535 updates, 0 skipped"
```

#### Step 7 — Start the application processes

Open three terminal tabs:

```bash
# Tab 1 — FastAPI application server
source .venv/bin/activate
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

# Tab 2 — Celery worker (background ingestion tasks)
source .venv/bin/activate
celery -A app.ingestion.celery_app worker --loglevel=info

# Tab 3 — Celery Beat (scheduled ingestion every 15 minutes)
source .venv/bin/activate
celery -A app.ingestion.celery_app beat --loglevel=info
```

Alternatively, use the Makefile shortcuts:

```bash
make dev     # starts uvicorn with --reload
make worker  # starts celery worker
make beat    # starts celery beat
```

---

### Option C — Linux / Ubuntu 22.04 LTS and WSL2

#### Install system dependencies

```bash
sudo apt-get update && sudo apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    postgresql-15 postgresql-15-pgvector \
    redis-server \
    libpq-dev gcc

# Start services
sudo systemctl enable --now postgresql redis-server
```

#### Create the database

```bash
sudo -u postgres psql -c "CREATE USER radar WITH PASSWORD 'radar_password';"
sudo -u postgres psql -c "CREATE DATABASE social_radar OWNER radar;"
```

Then follow Steps 2–7 of Option B (substituting `sudo -u postgres psql` for `psql` commands where needed and using `python3.11` explicitly).

---

### Troubleshooting Common Issues

| Symptom | Cause | Fix |
|---|---|---|
| `pg_isready: command not found` | Postgres not on PATH | `export PATH="/opt/homebrew/opt/postgresql@15/bin:$PATH"` |
| `ImportError: No module named 'asyncpg'` | Wrong Python activated | `source .venv/bin/activate` |
| `FATAL: role "radar" does not exist` | DB user not created | Run `createuser radar` or use `psql -U postgres` |
| `redis.exceptions.ConnectionError` | Redis not running | `brew services start redis` or `sudo systemctl start redis` |
| `InvalidToken` on source credential decrypt | Pre-M1 credentials in DB | Run `python scripts/migrate_credentials.py` |
| `alembic.exc.CommandError: Can't locate revision` | Missing base migration | Confirm `alembic/versions/000_initial_schema.py` exists |
| Docker: `db-init exited with code 1` | Postgres not ready in time | Increase `postgres` healthcheck `retries` in `docker-compose.yml` |
| `OSError: [Errno 28] No space left on device` | Docker volumes full | `docker system prune -a --volumes` (deletes all data) |

---

## Getting Started — Your First Signal

Once the stack is running (via either Option A or B above), follow these steps to produce your first classified signal end-to-end. All examples use `curl` against the default local address. The interactive OpenAPI UI at `http://localhost:8000/docs` lets you do the same thing without leaving the browser.

---

### Step 1 — Register an account

```bash
curl -s -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "analyst@yourcompany.com",
    "password": "StrongPassword123!"
  }' | python3 -m json.tool
```

Expected response (HTTP 201):

```json
{
  "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "email": "analyst@yourcompany.com",
  "is_active": true
}
```

---

### Step 2 — Obtain a JWT access token

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "analyst@yourcompany.com", "password": "StrongPassword123!"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

echo "Token acquired: ${TOKEN:0:20}..."
```

The token is valid for 30 minutes by default (`JWT_ACCESS_TOKEN_EXPIRE_MINUTES` in `.env`). Pass it as a Bearer token in all subsequent requests.

---

### Step 3 — Connect a source platform

Add a Reddit source (no OAuth required for public subreddit scraping in development mode):

```bash
curl -s -X POST http://localhost:8000/api/v1/sources \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "platform": "reddit",
    "credentials": {
      "client_id":     "YOUR_REDDIT_CLIENT_ID",
      "client_secret": "YOUR_REDDIT_CLIENT_SECRET",
      "user_agent":    "social-media-radar/1.0"
    },
    "settings": {
      "subreddits": ["SaaS", "entrepreneur", "startups"],
      "post_limit": 25,
      "include_comments": true
    }
  }' | python3 -m json.tool
```

For an RSS feed (no credentials required):

```bash
curl -s -X POST http://localhost:8000/api/v1/sources \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "platform": "rss",
    "credentials": {},
    "settings": {
      "feed_urls": [
        "https://hnrss.org/frontpage",
        "https://feeds.feedburner.com/TechCrunch"
      ]
    }
  }' | python3 -m json.tool
```

Test the connection immediately after adding it:

```bash
curl -s http://localhost:8000/api/v1/sources/reddit/test \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
# Expected: {"status": "ok", "platform": "reddit", "latency_ms": 142}
```

**Supported platforms and their credential requirements:**

| Platform | Type | Requires OAuth / API Key |
|---|---|---|
| `reddit` | Social | Yes — [Reddit app credentials](https://www.reddit.com/prefs/apps) |
| `youtube` | Social | Yes — [Google Cloud Console](https://console.cloud.google.com/) |
| `tiktok` | Social | Yes — [TikTok for Developers](https://developers.tiktok.com/) |
| `facebook` | Social | Yes — Meta Developer App token |
| `instagram` | Social | Yes — Meta Developer App token |
| `wechat` | Social | Yes — WeChat Open Platform |
| `rss` | Generic | No — provide feed URLs directly |
| `nytimes` | News | No — public RSS feeds |
| `wsj` | News | No — public RSS feeds |
| `abc_news` | News | No — public feeds |
| `abc_news_au` | News | No — public feeds |
| `google_news` | News | No — scrape only |
| `apple_news` | News | No — scrape only |

---

### Step 4 — Trigger your first ingestion

Ingestion runs automatically every 15 minutes via Celery Beat. To trigger it immediately:

```bash
curl -s -X POST http://localhost:8000/api/v1/sources/reddit/ingest \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
# Expected: {"task_id": "abc-123", "status": "queued", "platform": "reddit"}
```

Monitor the worker logs to watch observations move through the pipeline:

```bash
# Docker Compose
docker compose logs -f celery-worker

# Bare-metal
# Watch the terminal where you ran: celery -A app.ingestion.celery_app worker
```

You will see log lines for each stage: raw observation fetched → normalized → candidate retrieval → LLM adjudication → signal persisted.

---

### Step 5 — Review your signal queue

```bash
curl -s "http://localhost:8000/api/v1/signals/queue" \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

Each signal in the response includes:

- `signal_type` — one of the 18 classified types (e.g., `churn_risk`, `feature_request`)
- `confidence` — calibrated probability in [0, 1]
- `urgency_score` — composite priority (confidence × engagement velocity × freshness)
- `evidence_spans` — verbatim text excerpts from the source post with relevance reasoning
- `rationale` — the LLM's structured reasoning chain
- `status` — `PENDING`, `ACTED`, `DISMISSED`, or `ASSIGNED`
- `source_platform`, `source_url`, `author` — provenance

Filter by signal type and urgency:

```bash
curl -s "http://localhost:8000/api/v1/signals/queue?signal_types=churn_risk,feature_request&min_urgency=0.7&limit=10" \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

Stream new signals in real time via Server-Sent Events (SSE):

```bash
curl -N -H "Authorization: Bearer $TOKEN" \
  -H "Accept: text/event-stream" \
  http://localhost:8000/api/v1/signals/stream
```

---

### Step 6 — Act on a signal

Mark a signal as acted upon with a structured response:

```bash
SIGNAL_ID="paste-signal-uuid-here"

curl -s -X POST "http://localhost:8000/api/v1/signals/${SIGNAL_ID}/act" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "responded",
    "notes": "Reached out to the customer to schedule a call.",
    "response_tone": "empathetic"
  }' | python3 -m json.tool
```

Assign a signal to a team member:

```bash
curl -s -X POST "http://localhost:8000/api/v1/signals/${SIGNAL_ID}/assign" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"assignee_id": "team-member-uuid", "role": "ANALYST"}' \
  | python3 -m json.tool
```

Dismiss a signal that is not actionable:

```bash
curl -s -X POST "http://localhost:8000/api/v1/signals/${SIGNAL_ID}/dismiss" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"reason": "Out of scope for current sprint"}' \
  | python3 -m json.tool
```

---

### Step 7 — Submit feedback to improve calibration

When the model misclassifies a signal, submit a correction. The `ConfidenceCalibrator` performs an online gradient-descent step immediately, adjusting the temperature scalar for that signal type:

```bash
curl -s -X POST "http://localhost:8000/api/v1/signals/${SIGNAL_ID}/feedback" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "predicted_type": "feature_request",
    "true_type":      "bug_report",
    "predicted_confidence": 0.81
  }' | python3 -m json.tool
```

Each feedback submission triggers one calibration update. Calibration improvements are visible in subsequent inferences without any restart.

---

### Step 8 — View team digest and statistics

Get a summary of signal activity for your team over the past 7 days:

```bash
curl -s "http://localhost:8000/api/v1/signals/team?team_id=YOUR_TEAM_UUID&days=7&requester_role=ANALYST" \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

Get aggregate counts by signal type and status:

```bash
curl -s "http://localhost:8000/api/v1/signals/stats?days=30" \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

---

## Running the Test Suite

```bash
python -m pytest tests/ --ignore=tests/llm/test_load.py -q
```

Expected output:

```
593 passed, 20 skipped in ~54s
```

The 20 skipped tests require live LLM API credentials and are excluded from the default run. They can be enabled by setting `OPENAI_API_KEY` and removing the ignore flag, or by running the load test suite directly:

```bash
python -m pytest tests/llm/test_load.py -v
```

---

## Performance Reference

All figures below are measured values from `deliverables/benchmark.py`, using 3 warm-up passes and 7 timed repetitions on an Apple M-series chip. The benchmark measures five core algorithms that run on every observation as it moves through the pipeline.

---

### Algorithm Benchmark Results

#### 1. BloomFilter — Duplicate Detection (O(1) per operation)

The Bloom filter gates every fetched URL against a set of already-processed items before any database write or LLM call is made. This is the first line of defence against ingesting duplicate content across Celery worker restarts.

| Items checked (n) | Total time (ms) | Per-operation (µs) |
|---|---|---|
| 500 | 6.1 | 12.3 |
| 1,000 | 12.2 | 12.2 |
| 5,000 | 60.9 | 12.2 |
| 10,000 | 122.2 | 12.2 |
| 50,000 | 631.6 | 12.6 |
| 100,000 | 1,300.3 | 13.0 |

**Practical meaning:** deduplication cost per URL is constant at approximately **12–13 µs** regardless of how many URLs the filter already holds. A worker ingesting 100 items every 15 minutes spends under 1.5 ms total on deduplication per fetch cycle.

#### 2. ReservoirSampler — Uniform Stream Sampling (O(n))

When a platform returns more content than `MAX_ITEMS_PER_FETCH` allows, the reservoir sampler draws a statistically unbiased sample of exactly 500 items from the stream, regardless of total stream length.

| Stream length (n) | Time (ms) | Throughput (items/ms) |
|---|---|---|
| 1,000 | 0.95 | 1,053 |
| 10,000 | 10.1 | 990 |
| 50,000 | 50.1 | 998 |
| 100,000 | 101.2 | 988 |
| 250,000 | 248.3 | 1,007 |
| 500,000 | 503.5 | 993 |

**Practical meaning:** the sampler maintains a steady throughput of approximately **1,000 items/ms**. Sampling 500 items from a 50,000-item stream takes 50 ms — completely invisible against typical network latency for the upstream platform API call.

#### 3. ConfidenceCalibrator — Online Learning Update (O(m))

The calibrator performs one gradient-descent step per feedback event, updating the temperature scalar for the affected signal type in-memory. The `_save()` disk write is patched out in this benchmark to isolate the mathematical cost.

| Updates (m) | Computation time (ms) | Per-update (µs) |
|---|---|---|
| 100 | 0.78 | 7.8 |
| 1,000 | 5.6 | 5.6 |
| 10,000 | 66.9 | 6.7 |
| 100,000 | 666.5 | 6.7 |
| 500,000 | 3,282.7 | 6.6 |

**Practical meaning:** a single analyst feedback submission triggers one calibration update at a cost of approximately **6–8 µs of computation** plus a small disk flush to `calibration_state.json`. The calibrator can absorb thousands of feedback events per second without becoming a bottleneck.

#### 4. ActionRanker — Signal Priority Scoring (O(n))

`ActionRanker.rank_batch()` scores every signal in the queue by combining confidence, engagement velocity, content freshness, and signal-type urgency weights. The result determines the order in which signals appear to analysts.

| Signals ranked (n) | Time (ms) | Per-signal (µs) |
|---|---|---|
| 10 | 0.13 | 13.3 |
| 100 | 1.33 | 13.3 |
| 1,000 | 14.1 | 14.1 |
| 5,000 | 88.9 | 17.8 |
| 10,000 | 176.5 | 17.7 |
| 50,000 | 887.5 | 17.8 |

**Practical meaning:** ranking a queue of 1,000 signals takes **14 ms**. For a typical small-to-mid-size team accumulating 200–500 signals per day, the re-ranking that happens on every `GET /signals/queue` call completes in under 3 ms.

#### 5. BFS Graph Traversal — Related Signal Discovery (O(V+E))

The BFS traversal is used internally by the `ContextMemoryStore` to identify clusters of related past observations for few-shot context injection. The benchmark uses a degree-4 ring graph as a representative topology.

| Nodes (n) | Time (ms) |
|---|---|
| 1,000 | 0.23 |
| 5,000 | 1.04 |
| 10,000 | 2.09 |
| 50,000 | 10.3 |

**Practical meaning:** scanning a context memory store of 1,000 past observations for related few-shot examples takes under **0.25 ms**.

---

### LLM Inference Throughput

LLM adjudication latency is network- and provider-dependent. Observed figures in development:

| Configuration | Approx. latency per signal | Notes |
|---|---|---|
| GPT-4o (frontier tier, streaming) | 1.5 – 4 s | Used for `churn_risk`, `legal_risk`, `security_concern`, `reputation_risk` |
| GPT-4o mini (fine-tuned, non-frontier) | 0.4 – 1.2 s | Used for the remaining 14 signal types |
| Claude 3.5 Haiku (Anthropic tier) | 0.5 – 1.5 s | Requires `ANTHROPIC_API_KEY` |
| Ollama llama3.1:8b (fully local) | 3 – 12 s | No API cost; runs on M2 Pro with 16 GB RAM |

The two-tier routing strategy (`LLMRouter`) routes 70–80% of observations to the cheaper non-frontier model when `FINE_TUNED_MODEL_ID` or `LOCAL_LLM_URL` is configured, reducing average per-signal LLM cost significantly without measurable accuracy loss on the 14 non-critical signal types.

---

### Recommended Hardware by Team Size

| Team size | Daily signal volume | Recommended setup |
|---|---|---|
| 1–3 analysts | < 200 signals/day | MacBook M2/M3, 16 GB RAM, GPT-4o mini + fine-tune |
| 4–10 analysts | 200–1,000 signals/day | Mac Studio M2 Ultra or Linux workstation, 32 GB RAM |
| 10+ analysts | 1,000+ signals/day | Dedicated server (8-core CPU, 32 GB RAM) or cloud VM; consider horizontal Celery worker scaling |

---

## Expected Benefits for Teams

---

### 1. Structured, Classified Intelligence — Not Raw Noise

Social media returns thousands of posts per day. Most of them contain no actionable signal for your business. Social-Media-Radar's 18-type taxonomy with calibrated confidence and abstention replaces that noise with a prioritised, classified queue. Each item includes verbatim evidence spans and a structured rationale — the analyst sees exactly which sentence in the original post drove the classification, and why.

**Before:** an analyst manually reads 200 Reddit posts per morning to find the 8 that are relevant.<br>
**After:** the signal queue surfaces those 8 (plus any from platforms the analyst was not checking) with confidence scores and evidence, ordered by urgency.

---

### 2. Privacy and Data Sovereignty by Default

All inference runs locally. The pipeline enforces a zero-egress contract via `DataResidencyGuard`:

- Author handles are pseudonymised (deterministic SHA-256) before any text reaches an LLM call.
- PII (email addresses, phone numbers, identifying URL parameters) is scrubbed before prompt assembly.
- Every redaction generates an immutable audit log entry.
- `verify_clean()` is called at the LLM call boundary as a final safety check.

When using a local LLM provider (Ollama), observation text never leaves the machine at all. This makes the system deployable in environments with strict data-residency requirements where cloud AI providers are prohibited.

---

### 3. Calibrated Confidence — Not Bare LLM Probability

LLMs are systematically miscalibrated: they tend toward overconfidence on common signal types and underconfidence on rare ones. `ConfidenceCalibrator` applies per-type temperature scaling, learned from the seed dataset and updated online after every analyst feedback event. The practical consequence:

- Fewer false escalations on `churn_risk` (miscalibrated overconfidence is the typical failure mode).
- Fewer missed classifications on rare types like `partnership_opportunity` (miscalibrated underconfidence).
- When the model is genuinely uncertain, it **abstains** with a structured reason rather than producing a low-quality result. Abstentions are logged separately and never surface to the signal queue.

---

### 4. 70–80% LLM Cost Reduction via Two-Tier Routing

The four risk signal types (`churn_risk`, `legal_risk`, `security_concern`, `reputation_risk`) route to the frontier model (GPT-4o or equivalent). The remaining 14 types route to a fine-tuned smaller model or a local Ollama model. In a typical B2B SaaS context, risk signals account for 15–25% of total signal volume. Routing 75–85% of observations to the cheaper tier reduces LLM spend accordingly — without any accuracy regression on the types it covers, because the fine-tuned model is trained specifically on those signal types.

---

### 5. Online Calibration — The System Gets Better With Use

Every analyst correction submitted via `POST /signals/{id}/feedback` triggers a single gradient-descent step in `ConfidenceCalibrator`. The temperature scalar for the corrected signal type is adjusted immediately, in-memory, and flushed to disk. There is no retraining cycle, no redeployment, and no minimum batch size. The first correction improves subsequent classifications on that signal type within the same session.

---

### 6. Team Workflow Built In

The signal queue is not a personal inbox. It supports:

- **Role-based assignment** — `VIEWER`, `ANALYST`, `MANAGER` roles with different field visibility
- **Team digest** — `GET /signals/team` returns aggregate counts by type and status over a configurable window, paginated at 500 signals per page with `Link: rel="next"` headers
- **Real-time streaming** — `POST /signals/stream` (SSE) pushes new signals to connected analyst clients as they are classified
- **Audit trail** — every act, dismiss, assign, and feedback event is timestamped and associated with the acting user

---

### 7. Full Offline Operation

Configure `LOCAL_LLM_URL=http://localhost:11434` and `LOCAL_LLM_MODEL=llama3.1:8b` in `.env` to run the entire pipeline — ingestion, normalisation, candidate retrieval, adjudication, calibration — without any network dependency. On an Apple M2 Pro with 16 GB unified memory, `llama3.1:8b` produces a classification in 3–12 seconds per observation, which is acceptable for asynchronous background ingestion.

The fallback embedding path (512-dimensional bag-of-words with L2 normalisation) runs with no API dependency at all, at the cost of weaker semantic similarity in candidate retrieval.

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System design and component relationships |
| [Getting Started](docs/getting-started.md) | Quickstart guide |
| [Deployment](docs/deployment.md) | Production deployment (Docker, cloud) |
| [Testing Guide](docs/TESTING_GUIDE.md) | Test organization and conventions |
| [Training](docs/TRAINING.md) | Calibration training and dataset format |
| [LLM Deployment](docs/LLM_DEPLOYMENT_GUIDE.md) | LLM provider configuration |
| [Pre-Launch Report](docs/pre_launch_report.md) | Audit findings and launch checklist |
| API Reference | `http://localhost:8000/docs` (live, requires running server) |

---

## Contributing

Bug reports and pull requests are welcome via the [issue tracker](https://github.com/yourusername/social-media-radar/issues).

For non-trivial changes, open an issue first to discuss the intended approach. Changes to the inference pipeline (normalization, candidate retrieval, adjudication, calibration) should include tests that demonstrate the change does not regress existing behavior:

```bash
# After making changes, run the full suite before submitting
python -m pytest tests/ --ignore=tests/llm/test_load.py -q
# All 593 tests should pass
```

Code style follows [Black](https://black.readthedocs.io/) formatting and [Ruff](https://docs.astral.sh/ruff/) linting. Type annotations are required for all public functions. Pydantic v2 APIs (`model_dump()`, `model_validate()`, `@field_validator`) are used throughout; v1-style APIs are not accepted.

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

---

## License

MIT License. See [LICENSE](LICENSE) for the full text.

---

**Last updated:** 2026-03-23 | **Test baseline:** 593 passed, 20 skipped