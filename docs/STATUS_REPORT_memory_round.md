# Inference Engine — Status Report: Memory Subsystem Hardening (Round 1)

**Date:** 2026-05-22
**Scope this round:** Full-repo structure map + redundancy/conflict audit, with **depth on the memory subsystem** (the beta complaint *"insufficient memory"*). Verification attempted in-sandbox.
**Author:** Cowork agent session

---

## 1. Honest framing up front

You asked for the platform to be "completely error eliminated, all features significantly improved, and ready for enterprise deployment" in one round, and you also asked me to stay skeptical and assume the work is never finished. Those two instructions point in the same direction: **I will not claim to have done more than I verifiably did.**

This codebase is large — **1,114 Python files**, ~**92,000 lines in `app/` alone** across 25 submodules. Reading every line and credibly fixing/stress-testing all of it in a single pass is not something I can do honestly at the quality bar you set. So this round delivers a *deep, verified* slice on the agreed lead focus (memory), plus an honest map and audit around it. The remaining areas are scoped as explicit next rounds in §7.

**Verification constraint (important):** This sandbox has **no PyPI access** and only `numpy` preinstalled (no `pydantic`, `pytest`, `torch`, `transformers`, `spacy`, `playwright`). The project `.venv` is macOS-built, so its binaries don't run on the Linux sandbox. I therefore **could not run your full pytest suite or any heavy-ML stress test here.** What I *could* do — and did — is load the **real** `context_memory.py` source with minimal stand-ins for its domain-model imports and exercise the actual logic. Every claim below marked ✅ is backed by an executed check; everything else is marked as "needs run on your machine."

---

## 2. Architecture map (what the platform is)

The Inference Engine is a **FastAPI backend + Tauri desktop shell** agentic platform (internally "SMR"). The agent flow, end to end:

```
ingestion → normalization → candidate retrieval → LLM adjudication
   → calibration → abstention → personalization/ranking → output/digest
                         ↑                    ↑
                  context memory        feedback / outcomes
```

`app/` submodules by size (LOC), highest first:

| Module | LOC | Role |
|---|---|---|
| `intelligence/` | 20,671 | Core agent reasoning: orchestrator, inference pipeline, adjudication, calibration, **context_memory**, retrieval, deliberation |
| `llm/` | 9,619 | Provider clients (OpenAI/Anthropic), prompts, ensemble, training |
| `core/` | 6,917 | Config, DB, security, data residency, ranking, retry, health |
| `api/` | 6,164 | FastAPI routes (chat, classify, digest, ingest, rag, search, signals…) + middleware |
| `connectors/` | 5,766 | Source platform connectors |
| `ingestion/` | 4,856 | Content pipeline, noise filter, normalization, enrichment |
| `local/` | 4,723 | **Desktop sidecar** — SQLite stores, RAG retriever, embedding provider |
| `media/` | 4,685 | Audio/video/image intelligence |
| `workflows/` | 3,744 | Higher-level orchestrated workflows |
| `personalization/` | 2,323 | Interest graph, novelty scorer, digest ranker, feedback learner |
| (others) | — | source_intelligence, scraping, output, summarization, enterprise, evals, document_intelligence, entity_resolution, devintel, domain, research, oauth, mcp_server, desktop, monitoring |

**Client-facing surface in this repo:** the FastAPI routes (`app/api/routes/`), the output/digest formatters (`app/output/`), and the Tauri **launcher** (`app/desktop/launcher.py`, loopback-only sidecar with a per-launch shared-secret token). I found **no frontend assets** (no HTML/JS/JSX/Vue) in this repo — the actual Tauri UI lives in a separate project, so a literal "UI redesign" can't be performed from here. (See §7 for how to bring it into scope.)

**Three memory/retrieval layers coexist** (relevant to your "memory" complaint):
1. `app/intelligence/context_memory.py` — `ContextMemoryStore`, the **server-side per-user vector memory** used by `llm_adjudicator`, `deliberation`, and `inference_pipeline`. ← *this round's target.*
2. `app/intelligence/retrieval/chunk_store.py` — SQLite-backed chunk corpus (already persistent).
3. `app/local/` — desktop SQLite stores (`sqlite_store`, `rag_retriever`, `embedding_provider`, `content_store`, `dedup_store`).

---

## 3. Memory diagnosis — what was actually broken

I read `context_memory.py` in full (1,302 lines) and traced its callers. Three concrete defects directly explain *"insufficient memory"*, plus minor issues. I reproduced each against the real code before changing anything (baseline evidence in §5).

**Defect 1 — Core vector memory was never persisted (Critical).**
`persist()` serialized preferences, history, rationale, thresholds, aliases, channel prefs, and source embeddings — but **omitted `_records` and `_embeddings` entirely.** Those two buckets *are* the semantic observation memory that `retrieve()` searches. Result: every process restart silently reset per-user recall to empty, even though the API advertised durable persistence. Baseline: store 1 record → persist → reload → **0 records recalled.**

**Defect 2 — Non-deterministic fallback embedding (Critical).**
`_bow_embed()` bucketed tokens with Python's built-in `hash()`, which is **salted per process** (`PYTHONHASHSEED`, randomized by default). So the same token mapped to different vector dimensions in different processes. This corrupts *any* persisted embedding (including the rationale memory that *was* being persisted) the moment it's compared against a query embedded in a fresh process — recall silently degrades to noise after restart. Baseline: `hash("crash") % 512` = **355** under one seed, **173** under another.

**Defect 3 — Eviction wiped an arbitrary user, not the globally-oldest record (High).**
On capacity overflow, eviction popped the front of the *first non-empty bucket in dict-iteration order*. Under skewed multi-user load this evicted a **recent** record from one user while keeping a far **older** record from another. Baseline: with `max_records=4`, a recent record was evicted while a 10-hour-old backdated record survived.

**Minor:** `get_inference_history()` applied TTL only on writes (a long-idle user could read stale history); a code comment claimed the lock was "re-entrant" when it's a plain `threading.Lock`; and the module docstring overstated what `persist()` covered.

---

## 4. What I changed (file: `app/intelligence/context_memory.py`)

All changes are **additive and backward-compatible**; default `retrieve()` behavior is byte-for-byte unchanged when the new optional args are omitted.

1. **Deterministic embedding.** New `_stable_token_bucket()` uses `zlib.crc32` (stable across processes/platforms/Python versions); `_bow_embed` now uses it. Fixes Defect 2.
2. **Persist + restore the vector memory.** `persist()` now writes `records` + `embeddings` (snapshot **version "1.1"**); `load_from_disk()` rebuilds fully-typed `MemoryRecord`s and `float32` embeddings, skips records whose `signal_type` is no longer a valid enum (release drift), enforces the records/embeddings lockstep invariant, and recomputes `_total`. Fixes Defect 1. Also now persists `user_profiles` (was another silent gap affecting personalization detail-retention).
3. **Global-oldest eviction.** Eviction now selects the bucket whose front record has the smallest `created_at` and evicts that, looping until under capacity. Fixes Defect 3.
4. **Recency-aware retrieval (opt-in).** `retrieve(..., recency_half_life_days=…)` multiplies cosine similarity by `0.5 ** (age_days / half_life)` so recent memories win ties. Off by default.
5. **Retrieval filters (opt-in).** `retrieve(..., signal_types={…}, min_score=…)` for personalized/typed recall.
6. **Correctness/clarity.** Read-time TTL on `get_inference_history()`; corrected the lock comment; updated the module docstring and the v1.0→v1.1 compatibility note.

**Backward compatibility:** legacy "1.0" snapshots (no `records`/`embeddings`) load cleanly to an empty vector memory with all other state intact — verified.

**New tests added:** `tests/intelligence/test_context_memory_recall.py` — 14 tests across 6 classes, using the **real** pydantic domain models (runs anywhere numpy+pydantic exist; no torch/network).

---

## 5. Verification — what I actually ran

Because the full suite can't run here, I built a faithful harness (`mem_harness.py`) that loads the **real** `app/intelligence/context_memory.py` with minimal stand-ins only for its three domain-model imports, then exercised the genuine `ContextMemoryStore` logic.

**Baseline (before fixes):** determinism ❌, persistence ❌ (records lost on reload), eviction ❌ (evicted a recent record, kept an older one).

**After fixes — 19/19 checks passed ✅**, including:
- ✅ embedding deterministic across `PYTHONHASHSEED` (run in two subprocesses)
- ✅ `records`/`embeddings` keys persisted; record survives restart; text/type/confidence intact
- ✅ cross-process embedding recall works (stored vector still matches a later query)
- ✅ global-oldest evicted first; capacity respected
- ✅ legacy v1.0 snapshot loads; prefs restored; empty records (no crash)
- ✅ recency promotes newer record; default ranking unaffected
- ✅ `signal_types` filter and `min_score` filter behave correctly
- ✅ existing contract preserved: prefs / rationale / noise threshold / aliases persist; `user_profiles` now persist too

Also: `python3 -m compileall app/` passes cleanly (no syntax breakage anywhere in `app/`), and both edited files `py_compile` cleanly.

**Not verified here (needs your machine):** the full `pytest` suite (87 test files), and any test importing torch/transformers/spacy/playwright. Run locally:
```
pytest tests/intelligence/test_context_memory_recall.py -q          # the new tests
pytest tests/intelligence/test_comprehensive_hardening.py -q -k ContextMemory   # existing contract
```

---

## 6. Critical follow-up I deliberately did NOT do (and why)

The store is now *capable* of durable recall, but **no production code currently calls `persist()` or `load_from_disk()`** — `ContextMemoryStore` is dependency-injected into the pipeline and never flushed/restored. So the restart-durability benefit isn't realized until someone wires it in. The natural seam is the existing `lifespan` hook in `app/api/main.py` (load on startup, persist on shutdown, plus a periodic flush).

I did **not** add that this round on purpose: it touches the application lifecycle / agent flow (which you asked me to keep stable) and needs a decision on storage path and flush cadence. It's the first item in §7 and is a ~30-line, well-bounded change once you approve the approach.

---

## 7. Redundancy / conflict audit (repo-wide)

Git itself is clean — no committed virtualenvs, no `__pycache__`/`.pyc` tracked. One tracked runtime artifact: **`celerybeat-schedule`** (should be gitignored). `app/` has **no** backup/copy/`_old`/`_v2` files and only **3** unfinished-work markers — a mature tree.

I cannot *guarantee* zero redundancy across 1,114 files, but I found concrete duplicate top-level classes worth consolidating (same name, separate implementations):

| Class | Locations | Note |
|---|---|---|
| `CircuitBreaker` | `scraping/manager.py`, `llm/circuit_breaker.py`, `core/retry.py` | 3 implementations — consolidate to one in `core/`. |
| `RateLimiter` | `llm/rate_limiter.py`, `core/security.py` | 2 implementations. |
| `EmbeddingResponse` | `llm/models.py`, `llm/client_base.py` | Likely one canonical — risk of type confusion. |
| `LLMProvider` | `llm/models.py`, `llm/ensemble.py` | Verify enum vs. class collision. |
| `FeedbackRecord` | `intelligence/feedback.py`, `intelligence/feedback_store.py` | Overlapping feedback models. |
| `RelevanceScorer` | `ingestion/noise_filter.py`, `core/ranking.py` | Different scorers sharing a name — rename for clarity. |
| `NormalizationEngine` | `ingestion/normalization_engine.py`, `intelligence/normalization.py` | Different bounded contexts — confusing name, likely not true dup. |

These are **consolidation candidates**, not safe blind deletions. Each needs a usage trace before merging; I'd take them one at a time with tests.

---

## 8. Recommended next rounds (prioritized)

1. **Wire memory persistence into the lifecycle** (§6) — unlocks the durability the fixes enable. Small, high-value.
2. **Information-acquisition mismatch** (your 2nd beta complaint): audit `ingestion/` + `candidate_retrieval` + `retrieval/` recall quality using the existing `RetrievalEvaluator` (Recall@k / MRR / nDCG). This is the natural pairing with memory.
3. **Personalization depth**: `interest_graph`, `novelty_scorer`, `digest_ranker`, `feedback_learner` — close the loop from `ContextMemoryStore` preferences/profiles into ranking.
4. **Consolidate the §7 duplicates**, one class at a time with regression tests.
5. **UI / web-download**: locate the Tauri frontend project and mount it here so the client-facing surface can actually be reviewed/hardened.
6. **Establish a runnable verification baseline**: a CI lane (or a local run) that executes the full suite, since I can't here.

---

## 9. Bottom line

This round eliminated **two critical and one high-severity defect** in the server-side memory subsystem — the mechanical causes of "memory resets on restart" and "recall degrades to noise" — plus several minor issues, and added recency/typed-recall controls and a 14-test regression suite, all verified (19/19) against the real module without breaking the existing contract or the agent flow. I have **not** claimed to fix the whole platform; the honest gaps (lifecycle wiring, info-acquisition, personalization, UI, full-suite run) are scoped above. Recommend approving the §6 wiring next so the durability work pays off in production.
