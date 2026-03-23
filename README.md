# Social-Media-Radar

A locally-deployed inference pipeline for structured signal classification from social media content.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 577 passed](https://img.shields.io/badge/tests-577%20passed-brightgreen.svg)](./docs/TESTING_GUIDE.md)

[Architecture](#architecture) • [Installation](#installation-macos--apple-silicon) • [Testing](#running-the-test-suite) • [Documentation](./docs) • [Contributing](#contributing)

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

## Installation (macOS / Apple Silicon)

The instructions below target macOS on Apple Silicon (M1/M2/M3/M4, ARM64). All listed packages ship ARM64-native wheels for this architecture under the version pins given. No Rosetta or source compilation is required.

### Prerequisites

- **Python 3.9 or later.** Python 3.11 is recommended.
- **PostgreSQL 15+** with the `pgvector` extension. On Apple Silicon, install via Homebrew 4.0 or later, which ships an ARM64-native formula: `brew install postgresql@15 pgvector`.
- **`asyncpg >= 0.28.0`** — earlier versions lack ARM64 wheels on PyPI and require a local C compile.
- **`numpy >= 1.24`** — first release with universal2 macOS wheels. Earlier versions fall back to a Rosetta x86_64 wheel (functional but ~30–40% slower).
- Xcode Command Line Tools for any packages that fall through to a local build: `xcode-select --install`.

### Step 1 — Clone and create a virtual environment

```bash
git clone https://github.com/yourusername/social-media-radar.git
cd social-media-radar
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Verify the critical version pins:

```bash
pip show asyncpg numpy | grep -E "^(Name|Version)"
# asyncpg should be >= 0.28.0
# numpy should be >= 1.24
```

### Step 3 — Configure environment variables

Copy the example configuration and fill in your values:

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

| Variable | Description |
|---|---|
| `SECRET_KEY` | Random secret — generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `ENCRYPTION_KEY` | Random secret — generate the same way |
| `OPENAI_API_KEY` | Required for LLM adjudication (`sk-…`) |
| `ANTHROPIC_API_KEY` | Optional — enables Anthropic routing tier |
| `DATABASE_URL` | Postgres connection string (pre-filled for Docker Compose) |

> **Docker Compose users:** the inline `environment:` blocks in `docker-compose.yml` override the database/Redis/MinIO URLs automatically to use Docker-internal hostnames. You only need to set the secrets and API keys in `.env`.

**One-command local stack (Docker Compose):**

```bash
cp .env.example .env   # edit .env — add SECRET_KEY, ENCRYPTION_KEY, OPENAI_API_KEY
docker compose up
```

This starts Postgres (with pgvector), Redis, MinIO, runs all migrations, and launches the API on `http://localhost:8000`.

**Manual / bare-metal setup:**

```bash
export OPENAI_API_KEY="sk-..."
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/smr"
export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export ENCRYPTION_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export ANTHROPIC_API_KEY="sk-ant-..."  # Optional
```

### Step 4 — Database setup

Start PostgreSQL, create the database, enable the `pgvector` extension, and run migrations:

```bash
# Start PostgreSQL (adjust service name for your install)
brew services start postgresql@15

# Create the database and enable the extension
psql -U postgres -c "CREATE DATABASE smr;"
psql -U postgres -d smr -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Apply all schema migrations
alembic upgrade head
```

### Step 5 — Run calibration

The `ConfidenceCalibrator` ships with all temperature scalars initialized to `T = 1.0` (mathematical identity). Running the calibration script on the seed dataset adjusts these scalars before the system processes any production traffic:

```bash
python training/calibrate.py --epochs 5
# Expected: "Calibration complete: 535 updates, 0 skipped"
# Output written to: training/calibration_state.json
```

### Step 6 — Start the development server

```bash
uvicorn app.api.main:app --reload --port 8000
```

The OpenAPI documentation is available at `http://localhost:8000/docs`.

---

## Running the Test Suite

```bash
python -m pytest tests/ --ignore=tests/llm/test_load.py -q
```

Expected output:

```
577 passed, 20 skipped in ~54s
```

The 20 skipped tests require live LLM API credentials and are excluded from the default run. They can be enabled by setting `OPENAI_API_KEY` and removing the ignore flag, or by running the load test suite directly:

```bash
python -m pytest tests/llm/test_load.py -v
```

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
# All 577 tests should pass
```

Code style follows [Black](https://black.readthedocs.io/) formatting and [Ruff](https://docs.astral.sh/ruff/) linting. Type annotations are required for all public functions. Pydantic v2 APIs (`model_dump()`, `model_validate()`, `@field_validator`) are used throughout; v1-style APIs are not accepted.

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

---

## License

MIT License. See [LICENSE](LICENSE) for the full text.

---

**Last updated:** 2026-03-18 | **Test baseline:** 577 passed, 20 skipped