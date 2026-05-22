# Math / ML Upgrade Roadmap — Memory · Acquisition · Personalization

**Date:** 2026-05-22
**Purpose:** A prioritized, concrete plan to upgrade the mathematics and ML behind the three pillars now that the answer path is open for changes. Each item states *what*, the *ML concept*, *how we verify it*, and the *notebook cell* that keeps the training notebook in sync. Items are ordered by leverage-to-risk.

**Guardrails carried from prior rounds:** every change ships behind verification (stub-harness in this sandbox, full pytest on your machine), stays backward-compatible by default, preserves the basic agent flow, and never makes unprovable "best in industry" claims. Confidence/quality numbers are *measured*, not asserted.

---

## Current state (baseline this builds on)

- **Memory** (`context_memory.py`): deterministic fallback embedding, durable records+embeddings (v1.1), global-oldest eviction, recency/typed retrieval. Fallback embedder is bag-of-words hashing.
- **Acquisition** (`candidate_retrieval.py`, `hnsw_search.py`): hybrid dense (HNSW, now with numpy fallback) + sparse TF-IDF fused by RRF (now normalised); fixed source weights (`embedding 0.4 / entity 0.3 / platform 0.3`); a cross-encoder `Reranker` exists but is not in the candidate path.
- **Personalization** (`user_persona.py` + `interest_graph.py`): persona traits via exponential-forgetting weighted mean + consistency-based confidence; interest graph via additive gradient step. Persona now injected into the live answer path.

---

## Tier 1 — highest leverage, low risk

### 1.1 Learned retrieval fusion (replace fixed source weights)
- **What:** Replace the hand-set `embedding/entity/platform` weights and the `rrf→[0,1]` heuristic with a small **logistic-regression combiner** trained on labelled `(observation → gold signal_type)` pairs: features = per-source scores (dense sim, sparse sim, RRF rank, regex hits, platform prior); target = relevance.
- **ML concept:** learning-to-rank / linear score fusion (a 1-layer model; interpretable, cheap, no GPU).
- **Verify:** Recall@k / MRR / nDCG via the existing `RetrievalEvaluator` on held-out scenarios; must beat the fixed-weight baseline (already gated in notebook §7).
- **Notebook:** extend §7 to fit the combiner on `splits['train']` and report the metric delta vs. the fixed-weight baseline.

### 1.2 Activate the cross-encoder reranker in the candidate path
- **What:** After hybrid retrieval returns top-N, rerank with the existing `Reranker` (cross-encoder) before handing candidates to adjudication.
- **ML concept:** two-stage retrieve-then-rerank (bi-encoder recall + cross-encoder precision) — the standard modern IR stack.
- **Verify:** nDCG@k uplift on held-out; latency budget check (rerank only top-N).
- **Notebook:** add a rerank step to §7 and report nDCG before/after.

### 1.3 Calibrated similarity → probability for memory recall
- **What:** Map raw cosine similarity to a **calibrated relevance probability** (isotonic or Platt scaling) so `retrieve(min_score=…)` thresholds mean the same thing across embedding models.
- **ML concept:** probability calibration (isotonic regression / Platt). Reuses the existing `Calibrator`.
- **Verify:** reliability diagram + ECE on a labelled relevance set; threshold sweeps.
- **Notebook:** new cell in §8 (memory) fitting calibration on gold rationale matches.

---

## Tier 2 — stronger personalization math

### 2.1 Beta-Binomial confidence for binary style prefs
- **What:** For traits that are effectively binary (emoji on/off, step-by-step yes/no), model the preference as a **Beta-Binomial** posterior instead of a point estimate; report the posterior mean and a credible interval.
- **ML concept:** conjugate Bayesian updating; confidence = 1 − interval width. Cleaner than the variance-penalised heuristic for binary signals.
- **Verify:** posterior coverage on simulated streams; agreement with the current estimator in the limit.
- **Notebook:** new §10 (persona) cell showing posterior tightening as evidence accrues.

### 2.2 Hierarchical (partial-pooling) priors for cold start
- **What:** Initialise a new user's traits from a **population prior** (mean trait values across users), shrinking toward the user's own evidence as it accumulates (empirical-Bayes shrinkage).
- **ML concept:** hierarchical Bayes / James–Stein shrinkage — directly fixes cold-start personalization.
- **Verify:** lower expected error on the first N turns vs. a flat 0.5 prior, on simulated users.
- **Notebook:** §10 cell estimating the population prior from many simulated personas.

### 2.3 Bandit-driven directive selection (explore/exploit)
- **What:** When confidence is mid-range, treat "which style directive to apply" as a **contextual bandit** (Thompson sampling) and learn from the next-turn implicit feedback, instead of always applying the point estimate.
- **ML concept:** Thompson sampling / contextual bandits — principled explore/exploit so the agent discovers preferences it hasn't been told.
- **Verify:** cumulative-regret simulation vs. greedy; offline replay.
- **Notebook:** §10 simulation cell plotting regret over turns.

### 2.4 Next-style prediction model
- **What:** Train a small classifier predicting the user's preferred style for the *next* turn from recent history (so we personalise proactively, not just reactively).
- **ML concept:** sequence/feature classification (logistic regression or gradient boosting on turn features).
- **Verify:** held-out accuracy / F1 on logged turns.
- **Notebook:** §10 train/eval cell once real logs exist.

---

## Tier 3 — memory & acquisition depth

### 3.1 Real embedding backend for the fallback path
- **What:** Make the production embed function a sentence-transformer (already a dependency); keep the deterministic bag-of-words only as the no-dependency fallback. Add embedding-version stamping so a model upgrade invalidates stale vectors.
- **ML concept:** dense semantic embeddings; embedding-version migration.
- **Verify:** Recall@k uplift vs. bag-of-words on held-out; migration test (stale vectors re-embedded).

### 3.2 Memory consolidation / summarization
- **What:** Periodically summarise old per-user records into compact "semantic memories" (cluster + summarise) so recall stays high without unbounded growth — beyond simple oldest-eviction.
- **ML concept:** online clustering (e.g. streaming k-means / HDBSCAN) + abstractive summarisation.
- **Verify:** recall retention after consolidation vs. raw eviction, at fixed memory budget.

### 3.3 Maximal-Marginal-Relevance (MMR) diversification
- **What:** Diversify retrieved candidates with MMR so near-duplicate evidence doesn't crowd out complementary context.
- **ML concept:** MMR (relevance − λ·redundancy).
- **Verify:** coverage/diversity metric + downstream adjudication quality.

---

## Sequencing & how the notebook stays in sync

The training notebook (`scripts/phase1_authoring/build_notebook.py` → `notebooks/train_situation_engine.ipynb`) is the single orchestration surface. Each upgrade lands as (a) reviewed code under `app/`, (b) a verification gate cell in the notebook, and (c) a metric delta reported against the prior baseline. This round already added **§7 (acquisition gate)** and **§10 (persona personalization)**; Tier-1 items extend §7/§8 and Tier-2 items extend §10. The rule: **no math upgrade is "done" until its notebook gate shows a measured, non-regressing metric.**

Recommended order: 1.1 → 1.2 → 2.2 → 1.3 → 2.1 → 2.3 → 3.1 → 3.2 → 2.4 → 3.3.
