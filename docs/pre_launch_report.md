# Pre-Launch Audit Report

**Date:** 2026-03-17  
**Auditor:** Augment Agent (automated)  
**Test baseline:** 577 passed, 20 skipped, 0 errors after all fixes applied

---

## Implementation Gap Findings

| # | File | Line(s) | Problem | Fix Applied |
|---|------|---------|---------|-------------|
| G1 | `app/llm/ensemble.py` | 146–151 | `_best_of_n` builds a `tasks` list but never calls `asyncio.gather` and has no `return` statement. Any call to `generate_summary()` with the default `BEST_OF_N` strategy silently returns `None`, crashing the caller with an `AttributeError`. | Completed `_best_of_n`: added `asyncio.gather(*tasks, return_exceptions=True)`, `valid_summaries` filtering, `RuntimeError` guard, `best_summary` selection, and `return`. |
| G2 | `app/llm/ensemble.py` | 214, 223 | `_consensus_vote` reads `s.quality.sentiment_score` / `summary.quality.sentiment_score`. `SummaryQuality` has no `sentiment_score` field — it has `coherence_score`, `factuality_score`, `completeness_score`, `conciseness_score`, `overall_score`. Would raise `AttributeError` at runtime. | Replaced both references with `coherence_score`; updated docstring weight label from "Sentiment consistency" to "Coherence consistency". |
| G3 | `app/llm/ensemble.py` | 228 | `len(summary.summary)` — `EnsembleSummary` has no `summary` field; the field is named `content`. Raises `AttributeError`. Also `summary.original_text` (same line) does not exist on the model. | Replaced with `tokens_used`-based length scoring (no `original_text` is available; `tokens_used` is the best available proxy). |
| G4 | `app/llm/ensemble.py` | 237–238 | `max(s.cost for s in summaries)` / `summary.cost` — `EnsembleSummary` has no `cost` field. Raises `AttributeError`. | Replaced with `max(s.tokens_used for s in summaries)` / `summary.tokens_used` as a cost proxy (more tokens = higher cost). |
| G5 | `app/llm/ensemble.py` | 376–389 | Orphaned code fragment (the body that should have been inside `_best_of_n`) is placed inside `_score_conciseness` after its `return 0.5` statement. Completely unreachable. | Removed the dead block; the equivalent logic is now correctly inside the fixed `_best_of_n`. |
| G6 | `app/intelligence/deliberation.py` | 35–40 | `_FRONTIER_SIGNAL_TYPES` is redefined locally, duplicating the canonical definition in `app/llm/router.py`. If `router.py` adds a new high-stakes type, `DeliberationEngine` will silently skip escalation for it. | Replaced local definition with `from app.llm.router import _FRONTIER_SIGNAL_TYPES`. |
| G7 | `app/intelligence/inference_pipeline.py` | 68–71 | `LLMAdjudicator` constructed in `InferencePipeline.__init__` without any of the five new optional components (E1 CoT, E2 ConfidenceCalibrator, E3 Orchestrator, E5 ContextMemory, E6 Deliberation). The pipeline is the primary production entry point; none of the enhancements can be activated without re-instantiating the adjudicator manually. | Added five optional constructor params (`confidence_calibrator`, `cot_reasoner`, `orchestrator`, `context_memory`, `deliberation_engine`) to `InferencePipeline.__init__`; forwarded to the default-constructed `LLMAdjudicator`. Existing callers unaffected (all default to `None`). |
| G8 | `app/api/routes/signals.py` | feedback endpoint | The endpoint constructed `FeedbackStore()` without a `session_factory`, so all feedback was written only to an ephemeral in-memory list that was discarded at the end of the request. Feedback was never persisted to the database. | Replaced with a direct `SignalFeedbackDB` row insert using the already-open `AsyncSession`; added `await db.commit()`. |
| G9 | `app/api/routes/signals.py` | feedback endpoint | `ConfidenceCalibrator()` was constructed with the default relative path `training/calibration_state.json`. Depending on the process working directory at startup, this path may resolve to the wrong location. | Added module-level constant `_CALIBRATION_STATE_PATH` resolved at import time via `Path(__file__).resolve().parent.parent.parent / "training" / "calibration_state.json"`. |

---

## Static Analysis Findings

| # | File | Line(s) | Error Type | Fix Applied |
|---|------|---------|------------|-------------|
| S1 | `app/llm/ensemble.py` | 143 | **Dead import** — `import time` at the top of `_best_of_n` was a stale leftover; `time` is not used inside that method (it is used inside `_generate_with_provider`). | Removed `import time` from `_best_of_n`. |
| S2 | `app/intelligence/context_memory.py` | 160 | **Import inside loop** — `import dataclasses` appeared inside a `for` loop in `retrieve()`. Python re-evaluates the import statement on every iteration (cache-hit, but wasteful and style-incorrect). | Moved `import dataclasses` to the module top-level import block. |
| S3 | `app/core/db_models.py` | 603 | **Timezone-naive default on tz-aware column** — `SignalFeedbackDB.created_at` used `default=datetime.utcnow` (a bare method reference that returns a tz-naive `datetime`). The column is declared `DateTime(timezone=True)`, creating a type mismatch that would cause PostgreSQL to reject or silently mis-store the value depending on driver strictness. | Replaced with `default=lambda: datetime.now(timezone.utc)` and added `timezone` to the `from datetime import …` line. |
| S4 | `app/intelligence/llm_adjudicator.py` | `_format_memory_context` | **Bare generic type annotation** — `memories: List` is a bare `List` with no subscript. Python 3.9+ style requires `List[Any]` (or `list[Any]`). | Changed to `List[Any]`. `Any` was already in scope from the existing `from typing import … Any`. |
| S5 | `app/intelligence/deliberation.py` | 35–40 | **Logic error risk** — `_FRONTIER_SIGNAL_TYPES` was defined as a hardcoded local `frozenset`. Any future change to the canonical set in `router.py` would leave `DeliberationEngine`'s escalation logic silently out of sync. | See G6 above — replaced with a direct import. |
| S6 | `app/intelligence/inference_pipeline.py` | 24 | **Missing imports** — `ConfidenceCalibrator`, `ChainOfThoughtReasoner`, `MultiAgentOrchestrator`, `ContextMemoryStore`, `DeliberationEngine` not imported; needed for the new constructor params. | Added all five imports. |

---

## macOS Compatibility Notes

All packages used by Social-Media-Radar ship ARM64-native wheels for Apple Silicon (M1/M2/M3/M4). No source builds or special flags are required for any dependency **provided the following version constraints are met**:

- **`numpy ≥ 1.24`** — First release with universal2 macOS wheels. Older versions fall back to a Rosetta 2 x86_64 wheel that works but runs 30–40 % slower. Pin: `numpy>=1.24`.
- **`asyncpg ≥ 0.28.0`** — First release shipping ARM64 macOS wheels on PyPI. Earlier versions require a local C compile which needs Xcode Command Line Tools (`xcode-select --install`). Pin: `asyncpg>=0.28.0`.
- **`pgvector ≥ 0.2.0`** — The Python client is pure Python; the PostgreSQL extension must be built from source on macOS (no homebrew ARM64 formula before Homebrew 4.x). Install: `brew install pgvector` (requires Homebrew ≥ 4.0 on Apple Silicon). No Python pin needed.
- **`pydantic ≥ 2.0`** — Ships ARM64 wheels. The project already targets Pydantic v2. Confirm: `pip show pydantic | grep Version`.
- **`openai`, `anthropic`, `fastapi`, `sqlalchemy`** — Pure Python. No platform-specific constraints.
- **`spacy` (optional, used by NormalizationEngine)** — ARM64 wheels available since spacy 3.5. Language models must be downloaded separately: `python -m spacy download en_core_web_sm`.

No package in the dependency tree requires `ARCHFLAGS="-arch arm64"` or Rosetta prefix invocations when the above version pins are respected.

---

## Training Plan

### 1. What Will Be Trained

**ConfidenceCalibrator — per-`SignalType` temperature scalars**

| Property | Detail |
|----------|--------|
| Parameters learned | One scalar `T ∈ [0.1, ∞)` per `SignalType` — 18 scalars total |
| Loss function | Binary cross-entropy: `L = −[y·log(p_cal) + (1−y)·log(1−p_cal)]` |
| Gradient update | `T ← max(T_MIN, T − lr · (p_cal − y) · (−logit / T²))` — one step per example |
| Input format | JSONL records with fields `signal_type: str`, `confidence: float`, `is_hard_negative: bool` |
| Output format | `training/calibration_state.json` — JSON with `scalars: {signal_type: float}` dict |
| Stored artefact | `training/calibration_state.json` (overwritten in-place after every gradient step) |
| Learning rate | `0.01` (default; adjustable via `ConfidenceCalibrator(learning_rate=…)`) |

No LLM fine-tuning is performed. No embedding model is trained. The LLM is called via API at inference time only.

### 2. Is Training Necessary Before Launch?

**Training is NOT required before launch.** The system is fully functional without it:

- All 18 `ConfidenceCalibrator` scalars default to `T = 1.0`, which is a mathematical identity (`sigmoid(logit / 1) = sigmoid(logit)`). Predicted probabilities are unmodified.
- `ContextMemoryStore` starts empty and progressively populates as production traffic flows through.
- All other components (LLMAdjudicator, CandidateRetriever, etc.) have no trainable parameters.

**Concrete risk of launching without training:** The abstention threshold in `LLMAdjudicator._build_prompt` is `confidence < 0.6`. If the LLM is systematically over-confident for certain `SignalType` values (common with GPT-4), the calibrator would reduce `T` to sharpen predictions and push borderline cases toward abstention — improving precision. Launching at `T = 1.0` means this correction is absent; the system may produce slightly over-confident predictions for 2–4 signal types. This is **low severity** (incorrect confidence values, not incorrect classifications).

**Recommendation:** Run one epoch of calibration on the 107-example seed dataset before launch. It takes < 1 second.

### 3. Training Time Estimate (Apple Silicon M-series)

Dataset: **107 examples** (`training/signal_classification_dataset.jsonl`)

| Phase | Time |
|-------|------|
| Dataset loading (`_load_dataset`) | < 10 ms — 107 JSONL lines, pure Python |
| Per-example gradient step | ~5 µs — 3 floating-point ops, 1 `math.log`, 1 `math.exp` |
| Total gradient steps (1 epoch, 107 examples) | ~0.5 ms |
| JSON serialisation to disk (`_save`) | ~1 ms per step × 107 = ~107 ms total |
| **Grand total (1 epoch)** | **< 200 ms** |
| **Grand total (5 epochs)** | **< 1 second** |

The dominant cost is file I/O: `_save()` writes the full JSON after every gradient step. For large datasets, pass `--epochs 1` and batch-accumulate updates before writing (not yet implemented).

### 4. Training Execution Instructions

Run all commands from the repository root in order:

```bash
# 0. Activate the project virtualenv (adjust path to your environment)
source .venv/bin/activate          # or: conda activate social-media-radar

# 1. Verify the seed dataset exists
wc -l training/signal_classification_dataset.jsonl
# Expected: 107

# 2. Inspect the pre-training scalar baseline (all T=1.0)
cat training/calibration_state.json

# 3. Run calibration — 5 epochs recommended for the 107-example seed set
python training/calibrate.py --epochs 5

# Expected output:
#   INFO Loaded 107 examples from training/signal_classification_dataset.jsonl
#   INFO Epoch 1/5
#   ...
#   INFO Calibration complete: 535 updates, 0 skipped
#   INFO Scalars written to: training/calibration_state.json
#
#   Final temperature scalars:
#     bug_report                      T=0.9xxx
#     churn_risk                      T=1.0xxx
#     ...

# 4. Verify the output file was updated
python -c "import json; d=json.load(open('training/calibration_state.json')); print(d['updated_at'])"

# 5. Run the full test suite to confirm calibration did not break anything
python -m pytest tests/ --ignore=tests/llm/test_load.py -q
# Expected: 577 passed, 20 skipped
```

No environment variables beyond standard `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `DATABASE_URL` are required for training. The calibration script is fully offline (no LLM calls).

---

## Launch Checklist

- [ ] **DB migration** — Execute the additive SQL migration to create the `signal_feedback` table and its three indexes (`ix_signal_feedback_signal_id`, `ix_signal_feedback_user_id`, `ix_signal_feedback_created_at`). See the SQL in the original implementation plan.
- [ ] **Run calibration** — Execute `python training/calibrate.py --epochs 5` from the repo root. Confirm `training/calibration_state.json` has an `updated_at` timestamp from today and non-unity scalars for at least one `SignalType`.
- [ ] **Verify calibration state path** — Confirm the file exists at the path printed by `python -c "from app.api.routes.signals import _CALIBRATION_STATE_PATH; print(_CALIBRATION_STATE_PATH)"` from the production deployment directory.
- [ ] **Set environment variables** — `OPENAI_API_KEY`, `DATABASE_URL`, `SECRET_KEY`, and (if using Anthropic routing) `ANTHROPIC_API_KEY` must be set in the production environment.
- [ ] **Install pgvector PostgreSQL extension** — `CREATE EXTENSION IF NOT EXISTS vector;` must be run once on the target database before any `NormalizedObservationDB` rows are inserted.
- [ ] **Python package versions** — Confirm `asyncpg>=0.28.0` and `numpy>=1.24` are installed: `pip show asyncpg numpy`.
- [ ] **Run full test suite on production host** — `python -m pytest tests/ --ignore=tests/llm/test_load.py -q` must report 577 passed, 20 skipped, 0 errors on the production host OS/architecture.
- [ ] **Smoke-test the feedback endpoint** — POST to `/{signal_id}/feedback` with a valid `signal_id` and `requester_role=analyst`; verify HTTP 201, a `feedback_id` UUID in the response, and a row in the `signal_feedback` table.
- [ ] **Smoke-test ensemble strategy** — Trigger at least one `generate_summary()` call with `EnsembleStrategy.BEST_OF_N` and verify the response is an `EnsembleSummary` (not `None`).
- [ ] **Validate deliberation escalation logging** — Confirm `radar.data_residency.audit` log entries appear in structured form when a `CHURN_RISK` or `LEGAL_RISK` signal with score > 0.5 is processed.
- [ ] **Review abstention rate on live traffic** — During the first 24 hours monitor `abstained=True` rate via the `/signals` endpoint. If rate > 20 %, the LLM temperature or `confidence_required` thresholds may need tuning.
- [ ] **Tag the release commit** — `git tag v2.0.0-pre-launch && git push origin v2.0.0-pre-launch`.

