# Personalization & Dedicated User-Memory — Capability Map & Honest Assessment

**Date:** 2026-05-22
**Scope:** Per-user personalization built on dedicated, durable user memory — communication style, reasoning style, hobbies, personal characteristics, and data-acquisition preferences. Implemented, wired into the per-user memory store, stress-tested, and assessed candidly against publicly-known approaches.

---

## 1. Honest framing (read this first)

You asked me to "guarantee our model has better memorization and stronger personalization … proved to be the best in the industry," and to "back-check with other enterprise agents like Open Claw."

I'll be straight with you, because you've asked me to stay skeptical: **I cannot *prove* "best in the industry," and I won't claim it.** Doing so would require head-to-head benchmarks against competitors' *internal* memory systems, which are not publicly accessible, plus live A/B tests and user studies on our own product. I also don't have a verifiable reference for a product called "Open Claw," so I will not invent its internals — comparing against a system I can't inspect would be fabrication, not engineering.

What I *can* do, and did: build a genuinely new, principled per-user persona-memory capability that the codebase was missing; verify its math and durability with executed tests; stress-test it at scale; and compare its **design** honestly against the *publicly-described patterns* the field uses. Where I assert an advantage, it's a design-level argument, not a measured win. §6 lists exactly what would be needed to substantiate a competitive claim.

---

## 2. What existed before vs. what's new

The `app/personalization/` package was entirely **content-ranking** personalization: an interest graph (topic weights), topic-embedding profile, feedback learner, novelty/relevance trade-off, and a digest ranker. These answer *"which signals/items should this user see?"*

**Nothing modeled the user as a person** — there was no representation of *how the agent should reason and talk to this specific user*. The `ContextMemoryStore` had only a free-form `user_profiles` dict wired to nothing. That gap is precisely what your directive targets.

**New this round:** `app/personalization/user_persona.py` — a dedicated `UserPersonaProfile` that learns, per user:

| Dimension | What it captures | Representation |
|---|---|---|
| Communication style | verbosity, formality, directness, technical depth, warmth, emoji affinity | continuous traits in [0,1] w/ confidence |
| Reasoning style | step-by-step vs. conclusion-first, evidence depth, example preference, proactivity | continuous traits in [0,1] w/ confidence |
| Hobbies / interests | free-form tags ("hiking", "jazz") | recency-decayed intensity + confidence |
| Personal characteristics | stated facts (`role=data scientist`, `timezone=US/Pacific`) | best-supported categorical value + confidence |
| Data-acquisition prefs | per-platform / per-source affinity | continuous traits in [0,1] w/ confidence |

It exposes `render_style_directive()`, which turns the **confident, non-neutral** parts of the persona into a compact natural-language instruction block the agent injects at generation time — the mechanism that actually personalizes tone and reasoning. It is wired into `ContextMemoryStore` (`get_user_persona` / `observe_user_persona` / `save_user_persona`) and persisted, exported, and erased alongside the rest of the per-user memory (GDPR-complete).

---

## 3. The math upgrade (and why it's better than what was there)

The interest graph learns weights with a naive additive step and a flat `confidence = count/10`. I left that untouched (its behavior is pinned by tests), and instead applied **upgraded math where it was missing** — the persona estimator. Each trait is an **exponential-forgetting weighted running mean with an online (West) weighted-variance estimate**:

```
Δt          = days since last observation
decay       = 0.5 ** (Δt / half_life_days)          # recency forgetting
n_eff       = n_eff * decay + strength              # effective evidence mass
lr          = strength / n_eff                      # adaptive step
value      += lr * (x - value)                      # recency-weighted mean
m2          = m2*decay + strength*(x-prev)*(x-value)# weighted variance
confidence  = (1 - 1/(1+n_eff)) * consistency       # mass AND agreement
   where consistency = 1 - min(1, sqrt(m2/n_eff) / 0.5)
```

Three properties a flat counter cannot give, each backed by a passing test:

1. **Stability with plasticity** — established traits move slowly (`lr` shrinks as `n_eff` grows), but after a long silence `n_eff` decays so fresh evidence re-shapes them. *(verified: established trait moves less per observation than a fresh one; `n_eff` drops after a 60-day gap)*
2. **Honest confidence** — confidence reflects *how much* AND *how consistent* the evidence is, so noisy/conflicting signals stay low-confidence and never leak into the agent's instructions. *(verified: alternating evidence yields strictly lower confidence than consistent evidence)*
3. **No scheduler required** — forgetting is applied lazily from `last_updated`, so the profile is always current without a cron job.

Characteristics use agreement-weighted support that only flips the stored value when contradicting evidence outweighs it *(verified: weak conflict doesn't flip, strong conflict does)*.

---

## 4. Verification & stress evidence (executed)

Because this sandbox can't run the full pytest suite (no PyPI, only numpy — see `docs/STATUS_REPORT_memory_round.md`), I loaded the **real** modules and exercised them:

- **Persona estimator** — 15/15 checks: convergence, consistency-penalized confidence, adaptive stability, recency plasticity, directive rendering, characteristic flip logic, hobby ranking, serialization round-trip.
- **`ContextMemoryStore` wiring** — 10/10 checks: observe → persist → restart → persona survives identically; GDPR export/clear include the persona.
- **No regressions** — the prior memory (19/19) and acquisition (12/12) harnesses still pass after the additions.
- **Scale/throughput** — 5,000 users × 40 observations = **200,000 updates in 0.57 s (~349k updates/s)**, **zero** numerical-bounds violations, deterministic across runs, and parallel-safe for independent profiles.

Permanent tests for your CI: `tests/unit/test_user_persona.py` (real models, no heavy deps). Run:
```
pytest tests/unit/test_user_persona.py -q
```

---

## 5. Honest design comparison vs. publicly-known approaches

This is a **design-level** comparison against patterns commonly described for assistant memory/personalization (e.g. summarized "memory" facts, embedding-based user profiles, RLHF-style preference tuning) — **not** a measured benchmark against any specific competitor.

- **Per-user, confidence-weighted, decaying preference state.** Many assistant "memory" features store extracted facts/notes and retrieve them verbatim. Our persona adds a *quantitative confidence and recency model*, so the agent only acts on preferences it's actually sure about and gracefully forgets stale ones. This is a defensible robustness advantage over plain fact-storage — but it is an argument, not a proven win.
- **Separation of "what to show" (interest graph) from "how to talk" (persona).** Treating communication/reasoning style as first-class, learned state is less common than topic personalization and is a genuine capability addition here.
- **Where we are NOT obviously ahead:** systems with large-scale RLHF/fine-tuning on real user populations, or with multimodal long-term memory, may personalize in ways this lightweight estimator does not. Our approach is interpretable and cheap, which is a different trade-off, not a strict dominance.

---

## 6. What "best in industry" would actually require (not yet done)

To substantiate any competitive superiority claim, the following are needed and are **not** in scope of this round:

1. A held-out **personalization benchmark** (e.g. predict the user's preferred style/next action) with metrics, run against our system and any baseline we can legally evaluate.
2. **Live A/B testing** in product (satisfaction, task-completion, retention) — the only real arbiter of "better personalization."
3. **Wiring the persona into the live answer path** — `render_style_directive()` must be injected into the LLM prompt in the serving loop, and `observe_user_persona()` called from turn signals. I did *not* change the agent's answer path this round (you asked to preserve the basic flow); this is the highest-value next step and is well-bounded.
4. A clarified, real reference for the competitor you mentioned ("Open Claw") so any comparison is grounded rather than invented.

---

## 7. Bottom line

This round added the missing half of personalization — a **dedicated, durable, mathematically principled model of the user as a person** — wired into per-user memory, verified (15/15 + 10/10, no regressions), and stress-tested (200k updates, zero violations). It is a real, defensible capability improvement. It is **not** a proof of being "best in the industry," and I won't pretend it is; §6 is the honest path to earning that claim. Recommend approving the answer-path wiring (§6.3) next so the learned persona actually shapes responses in production.
