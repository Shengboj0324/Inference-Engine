# SituationEngine Migration — Implementation Progress

**Scope:** Phases 0 → 3 of the migration described in
`docs/intelligence_migration.md`. This document records what was built,
the contracts each component exposes, and how the pieces fit together.
It does not cover Phase 4+ (full corpus, fine-tune run, production
cut-over), which are gated on the labelling deliverables in
`docs/labelling/deliverables.md`.

**Public API contract:** `phase6.0` (set in `app/api/main.py`,
verified by `tests/contract/test_public_api_surface.py`).

**Regression status at end of Phase 3:** `5843 passed` in the full
`pytest tests/` run (excluding live-LLM tests
`tests/llm/test_load.py` and `tests/llm/test_integration.py`, which
require external provider credentials).

---

## 1. Output Contract — `SituationReport`

File: `app/intelligence/situation_report.py` (158 LOC).

A single Pydantic v2 schema is now the only legal output shape for the
intelligence engine. Every downstream consumer (Phase 3 runtime, Phase
2 trainer, Phase 1 evaluator) imports from this module.

- `Severity` — closed `Enum` of `SEV-1` … `SEV-5`. Per-`signal_type`
  thresholds live in `docs/labelling/guidelines.md` and are owned by the
  labelling team, not by code.
- `Citation` — `post_id` + half-open `[char_start, char_end)` range with
  a `model_validator` rejecting empty / inverted spans.
- `Claim` — text + `citation_ids` (≥ 1 required) + `confidence ∈ [0,1]`.
  Every claim must point to at least one citation.
- `SuggestedAction` — text + `rationale_claim_ids` + priority 1–5.
- `SituationReport` — top-level. `extra="forbid"` (no schema drift).
  Enforces two mutually exclusive states via `model_validator`:
  - **Positive finding:** ≥ 1 claim AND ≥ 1 citation; every
    `claim.citation_ids[i]` must be in range of `citations`; every
    `action.rationale_claim_ids[i]` must be in range of `claims`.
  - **Abstention:** `abstain=True` requires `abstention_reason` and
    forbids any `claims` / `suggested_actions`.
- `schema_version` defaults to `1.0.0` (semver-pinned via regex).

---

## 2. Evaluation Infrastructure (Phase 1)

### 2.1 Scenario Loader — `app/evals/scenario_loader.py` (173 LOC)

- `Observation` — `extra="forbid"`; fields: `observation_id`, `text`,
  `source`, optional `timestamp`.
- `ScenarioMetadata` — pins author, annotators, optional adjudicator,
  `guideline_version` (semver), `created_on`, `adversarial_tags`, and
  the mandatory `pii_review_passed` + `pii_reviewer_id`.
- `ScenarioCase` — frozen dataclass binding metadata + observations +
  gold `SituationReport` + on-disk path.
- `ScenarioLoader.discover(split=…, require_pii_signoff=True)` —
  filesystem walker that validates every scenario at load time. The
  PII-signoff filter is **on** by default, so unreviewed scenarios are
  silently excluded from training and evaluation.

### 2.2 LLM-as-Judge Runner — `app/evals/scenario_eval.py` (243 LOC)

- `_JUDGE_SYSTEM_PROMPT` + `_JUDGE_RUBRIC_PROMPT` — five-dimension
  rubric (Accuracy / Groundedness / Severity Fit / Action Quality /
  Calibration), scored 0–20 each → 0–100.
- `ScenarioScore` — Pydantic schema the judge must return.
- `ScenarioJudge` — async runner that calls the judge model through
  `LLMRouter` (default: Claude 3.5 Sonnet via `RoutingStrategy.QUALITY`).
- `AggregateEvalReport` — per-dimension means + per-scenario records,
  consumed by the Phase 2 promotion gate.

### 2.3 Synthetic Smoke Corpus — `app/evals/synthetic_scenarios.py` (159 LOC)

CPU-only synthetic scenario generator used exclusively by the
stress-test harness. Hard-codes `STAGING_ONLY = True` and refuses to
write into anything other than `pytest`/`tmp_path` directories so a
synthetic case can never enter a real training run.

---

## 3. Training Pipeline (Phase 2)

### 3.1 Frozen Prompt — `app/llm/prompts/situation_engine_system.txt` (114 lines)

The system turn is loaded from disk (never embedded in Python source)
so diffs are reviewable and CI can hash it for provenance. Loading is
centralised in `app/llm/training/situation_prompt.py`:

- `load_system_prompt()` — strict; raises `MissingSystemPromptError` if
  the file is missing or empty.
- `build_training_messages(case)` and `build_inference_messages(obs, …)`
  — produce byte-identical message lists. Used by the trainer, the
  inferencer, the Phase 3 runtime, and the stress harness.
- `parse_model_output(raw)` — tolerant JSON extractor (handles
  prose-wrapped JSON) that returns a validated `SituationReport`.

### 3.2 Fine-tune Orchestrator — `app/llm/training/situation_engine_finetune.py` (379 LOC)

QLoRA-on-Llama-3.1-8B-Instruct adapter on top of `LoRATrainer`:

- `_DEFAULT_BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"`.
- Assistant-only loss masking: tokens belonging to the system + user
  turns are set to `-100` so the trainee only learns to emit the JSON
  assistant turn.
- Heavy imports (`torch`, `transformers`, `peft`, `datasets`) are
  deferred inside the methods that need them, so the **pre-flight
  check** (`run_preflight(...) -> PreflightReport`) runs CPU-only inside
  CI without GPU wheels. The preflight validates that train + val JSONL
  parse, that the system prompt loads, and that assistant lengths fall
  inside the configured token budget.

### 3.3 Dataset Builder — `scripts/build_training_set.py` (133 LOC)

CLI that walks `data/scenarios/` and emits `train.jsonl` + `val.jsonl`
under the requested output directory. The `heldout` split is
**deliberately excluded** so it can never leak into a fine-tune. The
script aborts if the frozen system prompt is missing.

### 3.4 Inferencer — `app/llm/training/inferencer.py` (165 LOC)

- `GenerateFn = Callable[[List[Dict[str, str]]], str]` — generator
  protocol used everywhere (trainer eval, runtime, tests).
- `SituationEngineInferencer` — takes any `GenerateFn` and runs it over
  a `ScenarioLoader`-discovered split, returning
  `InferenceRunReport(candidates, failures, latency_ms)`.
- `build_lora_generator(...)` — the only entry point that imports
  `transformers` + `peft`; loads a LoRA adapter on top of a base model
  and returns a `GenerateFn`.

### 3.5 Calibration — `app/llm/training/calibration.py` (149 LOC)

Deterministic, torch-free. Computes:

- `expected_calibration_error` (ECE, bucketed)
- `brier_score`
- `reliability_table: List[CalibrationBucket]`

"Correct" for calibration purposes means exact agreement on
`signal_type`, `abstain`, and `severity` between candidate and gold.

### 3.6 Promotion Gate — `app/llm/training/promotion.py` (187 LOC)

Single source of truth for "is this checkpoint shippable?". Used by
both `scripts/promote_checkpoint.py` (CLI) and CI:

- `PromotionConfig` — judge floor (default 85/100, configurable),
  manifest path, optional model/version stamps.
- `PromotionDecision` — frozen verdict carrying the
  `AggregateEvalReport`, `CalibrationReport`, and `InferenceRunReport`
  that justified it.
- `evaluate_and_promote(generator, judge, scenarios, config)` —
  orchestrates inference → calibration → judge → manifest write. Emits
  a JSON release manifest with a SHA-256 of the system prompt and the
  loader-discovered scenario IDs.
- Promotion is **judge-gated only**: ECE and Brier are recorded for
  observability but do not block. If `mean_score < floor`,
  `PromotionDecision.promoted` is `False` and the CLI exits non-zero.

### 3.7 Driver Scripts

| Script | Lines | Role |
|---|---|---|
| `scripts/build_training_set.py` | 133 | Scenarios → `train.jsonl` / `val.jsonl`. |
| `scripts/run_situation_inference.py` | 114 | Run a `GenerateFn` over a split and dump candidate reports + run report. |
| `scripts/run_scenario_eval.py` | 142 | Run the judge over a candidates folder, print and persist `AggregateEvalReport`. |
| `scripts/promote_checkpoint.py` | 109 | End-to-end inference → calibration → judge → manifest with promotion exit code. |
| `scripts/stress_test_phase2.py` | 213 | Loader → builder → preflight → inference → calibration → promotion on the synthetic corpus. Used as a CI smoke test. |
| `scripts/check_no_keyword_rules.py` | 136 | Repo-wide guard that fails the build if forbidden patterns reappear under `app/intelligence`. |
| `notebooks/train_situation_engine.ipynb` | — | A100 Colab driver that wraps `situation_engine_finetune` for the actual GPU run. |

### 3.8 Phase 2 Stress Harness

`tests/scripts/test_stress_phase2.py` (48 LOC) wraps
`scripts/stress_test_phase2.py` and verifies it reports
`ALL STAGES PASSED` end-to-end. Latest run:

```
loader: train=4, val=2, heldout=3
preflight: train_examples=4, val_examples=2, system_prompt_chars=4506
inference+calibration: n=3 accuracy=1.00 ece=0.293 brier=0.099
promotion: PASSED (mean=95.00)
```

---

## 4. Runtime Wiring (Phase 3)

### 4.1 Citation Verifier — `app/intelligence/citation_verifier.py` (142 LOC)

Pure structural grounding check. Stateless and side-effect free.

- For every citation in a non-abstaining report it verifies:
  1. `post_id` matches an `observation_id` in the input.
  2. `char_start < len(text)`.
  3. `char_end <= len(text)`.
  4. The cited span contains at least one non-whitespace character.
- Failures are collected into an ordered `List[CitationFailure]` and
  raised together as `CitationGroundingError`, so the operator sees
  every problem in one pass rather than one-at-a-time.
- Abstaining reports skip the check (they have no citations to verify
  by construction).

### 4.2 SituationEngine — `app/intelligence/situation_engine.py` (428 LOC)

Async- and sync-capable orchestrator. Three stages around a single
generator call:

1. **PII scrub.** Every `Observation.text` is passed through
   `DataResidencyGuard.scrub_text` before reaching the generator. The
   redaction count is propagated through to `EngineResult` for audit.
2. **Generate.** The scrubbed observations are wrapped via
   `build_inference_messages(...)` (Phase 2 prompt module) and submitted
   to a `GenerateFn` / `AsyncGenerateFn`. The raw completion is parsed
   by `parse_model_output(...)`.
3. **Verify grounding.** `CitationVerifier.verify_report(...)` runs the
   structural check above.

Two engine classes share the pipeline through `_Attempt`,
`_attempt_sync`, `_attempt_async`, and `_parse_and_verify`:

- `SituationEngine` — synchronous, for background workers.
- `AsyncSituationEngine` — async, for FastAPI routes and the routed
  frontier path.

`EngineResult` carries: `report`, `raw_completion`, `redaction_count`,
`used_fallback: bool`, `primary_report: Optional[SituationReport]`.

#### Optional re-routing

Both engines accept `fallback_generate` and `min_confidence`. The
escalation policy is intentionally minimal:

```python
def _should_fallback(self, attempt: _Attempt) -> bool:
    if self._fallback_generate is None:
        return False
    if attempt.error is not None:
        return True
    return attempt.report.calibrated_confidence < self._min_confidence
```

- One retry, never a loop.
- If the primary call **succeeded** but the fallback then **failed**,
  the primary report is returned with a warning log
  (`situation_engine.fallback_failed_keeping_primary`). The system
  degrades to "primary-only" rather than failing the request.
- `used_fallback` and `primary_report` are surfaced on `EngineResult`
  so operators can audit when escalation fired and what the original
  low-confidence answer was.

#### Router integration

`build_router_generator(router, *, model=None, temperature=0.0,
max_tokens=2048) -> AsyncGenerateFn` wraps an `LLMRouter` instance into
the engine's generator protocol. Imports are lazy so the LoRA-only
path (offline inference / tests) does not pay the router import cost.


### 4.3 New HTTP Surface

Three new routes were added in Phase 3 (`API_CONTRACT_VERSION` bumped
from `phase5.0` → `phase6.0`):

| Method | Path | Module | Purpose |
|---|---|---|---|
| `POST` | `/api/v1/classify` | `app/api/routes/classify.py` | Synchronous JSON: takes `observations[]` + optional routing knobs, returns `ClassifyResponse{report, used_fallback, redaction_count}`. |
| `POST` | `/api/v1/classify/stream` | `app/api/routes/classify.py` | SSE wrapper: emits `start` → (`report` \| `error`) → `done` frames. Honours `Request.is_disconnected()` for backpressure. |
| `POST` | `/api/v1/signals/situation` | `app/api/routes/signals.py` | Adapts a `RawObservation` (existing connector payload) into an engine `Observation` and runs the SituationEngine. Returns a verified `SituationReport`. |

Shared request schema (`ClassifyRequest` / `SituationRequest`):

- `observations` / `raw_observation` — input payload (Observation list
  vs. single connector record).
- `min_confidence: float ∈ [0.0, 1.0]` — re-routing threshold.
- `primary_model: Optional[str]` — routing hint for the primary call.
- `fallback_model: Optional[str]` — when set, builds a fallback
  generator using the same `LLMRouter`.

Engine exceptions are mapped to HTTP cleanly:

| Exception | HTTP | `error` code |
|---|---|---|
| `CitationGroundingError` | 422 | `citation_grounding_failed` |
| `OutputParseError` | 422 | `output_parse_failed` |
| `GenerationError` | 502 | `generator_failed` |
| anything else | 500 | `engine_error` |

The `/api/v1/signals/stream` endpoint (legacy `InferencePipeline`-backed
SSE) is unchanged and remains frozen in the API contract. The new
`/signals/situation` endpoint is purely additive.

### 4.4 Contract & Wiring Updates

- `app/api/main.py` — added `classify` to the `app.api.routes` import
  block; `app.include_router(classify.router, prefix="/api/v1", …)`
  added immediately after the `signals` router so URL ordering is
  stable. `API_CONTRACT_VERSION` advanced to `phase6.0`.
- `tests/contract/test_public_api_surface.py` — three new entries added
  to `FROZEN_ROUTES` under a `# Phase 6 — Grounded SituationEngine
  surface` header. The contract test catches both removed and silently-
  added routes, so future drift will fail CI loudly.

---

## 5. Test Coverage Added

### 5.1 Phase 1 / 2 Suites

| File | Lines | Coverage |
|---|---|---|
| `tests/intelligence/test_situation_report.py` | — | Schema invariants (grounding rules, abstention rules, severity enum, citation half-open semantics, `extra="forbid"`). |
| `tests/evals/test_scenario_infra.py` | — | `Observation`, `ScenarioMetadata`, `ScenarioLoader` walk + PII-signoff filter. |
| `tests/llm/test_situation_prompt.py` | 146 | System-prompt loading, message construction (training + inference), strict and prose-wrapped `parse_model_output`. |
| `tests/llm/test_calibration.py` | 146 | Empty input, perfect prediction, miscalibration, bucket boundary handling, ECE / Brier / reliability table arithmetic. |
| `tests/llm/test_situation_finetune_preflight.py` | 148 | CPU-only preflight: malformed JSONL, missing prompt, oversize assistant turns. |
| `tests/scripts/test_promotion_gate.py` | 139 | Promotion pass / fail, manifest write, exit-code contract. |
| `tests/scripts/test_stress_phase2.py` | 48 | Wraps `scripts/stress_test_phase2.py`; asserts `ALL STAGES PASSED`. |

### 5.2 Phase 3 Suites

| File | Lines | Coverage |
|---|---|---|
| `tests/intelligence/test_citation_verifier.py` | 124 | Missing post_id, out-of-range `char_start` / `char_end`, empty / whitespace-only spans, multi-failure aggregation, abstention bypass. |
| `tests/intelligence/test_situation_engine.py` | 279 | Happy path (sync + async), PII scrub passthrough, generator exception wrapping, malformed output, missing-post citation failure, empty-input rejection, router-generator translation, **+6 re-routing tests** (`min_confidence` validation, fallback triggered on low confidence sync/async, fallback triggered on primary exception, fallback skipped when primary passes, primary retained when fallback itself fails). |
| `tests/intelligence/test_classify_route.py` | 152 | `ClassifyRequest` validation, `_engine_failure_to_http` mapping for all four error classes, `_raw_to_observation` adapter (title+body / title-only / body-only / empty fallback), `ClassifyResponse` round-trip. |

### 5.3 Regression Summary

- `pytest tests/intelligence tests/llm tests/evals tests/scripts tests/contract` (focused): all green.
- `pytest tests/` (full suite, excluding pre-existing live-LLM tests):
  **5843 passed**, 0 failed, 1 unrelated pre-existing pydantic-v2
  deprecation warning.
- `scripts/check_no_keyword_rules.py`: `[OK]`.
- `scripts/stress_test_phase2.py`: `ALL STAGES PASSED`.

---

## 6. Documentation Added / Updated

| File | Purpose |
|---|---|
| `docs/intelligence_migration.md` | The plan-of-record for the whole migration; unchanged structurally during Phases 0–3, referenced by the loader, the judge, the promotion gate, and the engine. |
| `docs/labelling/guidelines.md` | Authoring rules for scenarios (signal taxonomy, severity definitions, citation conventions, PII handling). |
| `docs/labelling/deliverables.md` | Phase-1 exit checklist owned by the labelling team: 60 seed + 120 adversarial scenarios in train/val, 40 held-out, IAA thresholds, 100 % PII sign-off. |
| `docs/intelligence_migration_progress.md` | **This document.** |

---

## 7. What Is Still Blocked on the Labelling Team

The engineering surface from Phase 0 through Phase 3 is complete and
green. The next step (running a real fine-tune and cutting the engine
over to the trained adapter) is gated entirely on the corpus described
in `docs/labelling/deliverables.md`:

- 60 seed + 120 adversarial scenarios under
  `data/scenarios/{train,val}/`
- 40 held-out scenarios under `data/scenarios/heldout/`
- Cohen's κ ≥ 0.80 on `signal_type`, weighted κ ≥ 0.75 on `severity`
- 100 % PII sign-off (`metadata.pii_review_passed = true` +
  `pii_reviewer_id` on every scenario)

When that corpus lands, the existing pipeline runs unchanged:

```
ScenarioLoader.discover
    → scripts/build_training_set.py            # train.jsonl + val.jsonl
    → run_preflight                            # CPU validation
    → notebooks/train_situation_engine.ipynb   # QLoRA on A100
    → scripts/promote_checkpoint.py            # judge gate ≥ 85/100
    → AsyncSituationEngine + build_router_generator
    → /api/v1/classify, /api/v1/classify/stream, /api/v1/signals/situation
```

No code changes will be required between "labelled data arrives" and
"trained engine is live behind the new routes".

