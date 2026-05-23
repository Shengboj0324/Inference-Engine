# Training / Labelling / Scenario Data — Deep Analysis & Improvement Plan

**Date:** 2026-05-22
**Scope:** `data/scenarios/`, `data/labelling/`, `data/training/` — analysed against the current codebase (SituationEngine fine-tune path, the new Tier 1–3 retrieval/memory/personalization features, and the training notebook).
**Status:** This is the **plan-first** deliverable you asked for. Nothing in Part B has been implemented. I recommend the priorities in §B, and I will not start until you approve scope.

---

## A. What the data is, and how good it is

### A.1 Inventory (measured, not assumed)

- **220 scenarios** under `data/scenarios/` (plus `_template`). Each = `gold_report.json` (the labelled `SituationReport`), `observations.jsonl` (multi-source posts), `metadata.yaml` (split, annotators, adjudicator, guideline version, PII review).
- **Fine-tune set:** `train.jsonl` (144) + `val.jsonl` (36) = **180**; **heldout = 40**, and the held-out scenarios have **0 overlap** with train/val (verified) — the promotion gate is honest.
- **Format:** chat-style — `system` (the SituationReport task spec: *reason, cite spans, abstain; never classify by surface keywords*), `user` (observations JSON), `assistant` (the gold `SituationReport` JSON). This matches `ScenarioLoader` and the notebook's expectations.
- **Governance (mature):** `double_labels/` (56 inter-annotator pairs across 4 weeks), `disputes/` (6 adjudicated disagreements with written rationale), `pii_audit/` (220 — one per scenario, automated + human), `calibration_sessions/` (4 weekly notes), `pii_incidents/` (0).

### A.2 Genuine strengths (keep these)

- **Grounding is excellent:** 452 citation char-spans across all gold reports, and **0 are out-of-bounds** — every cited span actually falls inside its observation. This is the dataset's biggest asset and exactly what the "cite the evidence" system prompt needs.
- **Schema is rich and consistent:** signal_type, severity (SEV-1…5), calibrated_confidence, claims (with citation_ids + per-claim confidence), suggested_actions, abstain — `schema_version 1.0.0` throughout.
- **PII discipline:** every scenario has an automated + human PII clearance record; 0 incidents.
- **Real adjudication trail:** disputes carry the annotators' labels, the adjudicator decision, and a rationale (e.g. *"reproduction steps → bug_report not complaint"*) — high-value signal for guideline refinement.

### A.3 Concrete gaps & risks (each is evidence-backed)

| # | Finding (evidence) | Why it matters |
|---|---|---|
| **G1** | `quality_score` is **NULL for all 180** train/val records | The field exists and the pipeline can weight/filter by it, but it carries no signal today — dead governance metadata. |
| **G2** | **Class imbalance**: scenarios — security 30, legal 24, reputation/complaint/churn/bug 18, feature/expansion/competitor 12, **unclear/praise/notactionable 6**. Train mirrors it (security 22 ↔ praise/unclear/notactionable 6). | An 8B model will under-learn the rare classes; minority-class recall will lag. |
| **G3** | **Val split is missing classes**: `val.jsonl` has **no praise, no unclear, no notactionable**, and bug = 2. | The eval/promotion gate (floor 85) literally cannot measure quality on those classes — blind spots in the gate that's supposed to guarantee readiness. |
| **G4** | **Confidence labels are compressed**: `calibrated_confidence` ∈ **[0.60, 0.88]**, mean 0.748 — never < 0.6, never > 0.9. | The model can't learn calibrated *high* or *low* confidence, and Tier 1.3 calibration / ECE cannot be validated at the extremes that matter for abstention and ranking. |
| **G5** | **Scale**: 180 training examples for an 8B QLoRA fine-tune. | Overfitting risk, especially on rare classes; limited robustness to phrasing/adversarial variants. |
| **G6** | **Abstention is single-class**: abstain=true appears in **23 scenarios (10%), all `unclear`**; `notactionable` never abstains. | The model's learned refusal is entangled with one label; thin, narrow signal for calibrated abstention across genuinely-ambiguous cases. |
| **G7** | **Agreement not quantified**: raw signal-type agreement on double-labels is **50/56 = 89%**, but no Cohen's κ, no per-class confusion tracked. The sample even shows a security↔legal disagreement. | Without κ + a confusion matrix, you can't target the specific confusable pairs that cost accuracy. |
| **G8** | **Data↔new-code linkage missing**: the new **fusion** model (Tier 1.1) and **similarity calibrator** (Tier 1.3) need derived training sets the scenario data *can* produce, but these aren't persistently generated/gated; the **persona/bandit/predictor** features (Tier 2) have **no interaction-log data at all**. | The new features can't reach their measured potential without the derived datasets; persona metrics remain simulated until logs (real or synthetic) exist. |

**Bottom line:** the data is high-quality and well-governed, with best-in-class grounding — but it is **small, imbalanced, has an under-covered validation split, a compressed confidence range, and untapped governance signals**, and it doesn't yet feed the new Tier 1–3 features. None of these block fine-tuning from *running*; they cap the *accuracy and trustworthiness* of the result and of the promotion gate.

---

## B. Proposed improvement plan (enrich · fix · implement)

Ordered by leverage-to-risk. Every item ships behind a verification gate (in the notebook and/or a data-integrity test), stays reproducible, and never touches the held-out 40.

### Tier A — fixes (low risk, high trust; do first)

- **P1 — Populate `quality_score` deterministically.** Compute a per-scenario quality score from the governance signals already on disk: inter-annotator agreement, dispute/adjudication status, PII-pass, citation density, and span validity. Backfill train/val and wire the pipeline to optionally weight/filter by it.
  *Verify:* every record has a score in [0,1]; distribution reported; a regression test asserts no NULLs.
- **P2 — Repair & stratify the train/val split.** Re-stratify so **every signal_type appears in val** (and bug > 2), preserving the 80/20 ratio and leaving heldout untouched. Add a **split-integrity gate** (no class absent from val; 0 heldout leakage).
  *Verify:* notebook gate fails if any class is missing from val or any heldout id leaks.
- **P5 — Quantify agreement.** Compute **Cohen's κ** (overall + per-class) and a **confusion matrix** from `double_labels`; surface a small report. Use it to target the confusable pairs (e.g. security↔legal) with guideline clarifications.
  *Verify:* κ reported; confusion matrix rendered in the notebook; tracked over weeks.
- **P7 — Data-integrity test/CI gate.** A test that asserts: schema validity, citation spans in-bounds (currently 100% — lock it in), PII-audit present for every scenario, split isolation, and class coverage. Catches data regressions before they reach the model.
  *Verify:* added under `tests/`, runnable with the suite.

### Tier B — enrichment (raises accuracy; moderate effort)

- **P3 — Broaden the confidence range.** Author/relabel scenarios with genuinely high (>0.9) and low (<0.6) calibrated confidence, and more abstention cases, so the model learns the full range and Tier 1.3 calibration is measurable at the extremes.
  *Verify:* confidence histogram spans [0.4, 0.95]; ECE measured across bins.
- **P4 — Rebalance rare classes.** Bring praise/unclear/notactionable/competitor/feature/expansion toward parity via **guideline-consistent authored scenarios + controlled paraphrase augmentation** (anchored on the existing high-quality exemplars), with the same double-label + PII discipline.
  *Verify:* per-class train counts within a target band; per-class val recall reported on the gate.
- **P6 — Auto-derive the new-feature datasets.** From the scenarios, persistently generate and gate: the **fusion training set** (per-source features → gold signal_type) and the **calibration set** ((similarity, relevant) pairs), wired into the notebook so they're rebuilt and metric-gated each run (extends the §7/§8 cells already added).
  *Verify:* fusion Recall@k / calibration ECE gates run on real derived data, not synthetic.

### Tier C — bigger bets (discuss before committing)

- **P8 — Synthetic interaction-log generator** for persona/bandit/predictor training+eval (no real logs exist). Clearly-labelled synthetic turn streams so persona metrics become measurable rather than simulated-in-test. (Honest caveat: synthetic ≠ real users; this enables iteration, not a production claim.)
- **G5/scale — corpus growth program.** A repeatable authoring + augmentation loop to grow beyond 180, with the governance pipeline intact. Highest effort; biggest long-run accuracy lever.

---

## C. What I recommend, and the decision I need from you

I recommend doing **Tier A in full first** (P1, P2, P5, P7) — they are low-risk, raise trust immediately, and make the promotion gate honest — then **Tier B** (P3, P4, P6) for accuracy, and treating **Tier C** as a separate, scoped initiative.

**Honest caveats:** P3/P4 involve *authoring/relabelling data*, which is a labelling-policy decision, not just code — I'll generate candidates and integrity-check them, but they should pass your annotators' review before training on them. And I cannot run the actual fine-tune or `pytest` in this environment, so all gates I add are validated by the lightweight harness here and must be confirmed by one run on your machine.

**Please tell me which scope to implement** (e.g. "Tier A only", "Tier A + B", or specific Pn items), and I'll proceed exactly there — and only there.
