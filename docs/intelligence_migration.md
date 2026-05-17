# Intelligence Migration — From Regex Pipelines to a Grounded LLM Engine

**Status**: Phase 0 complete · Phase 1 next
**Owner**: Intelligence / ML
**Last reviewed**: 2026-05-17

This document is the single source of truth for the migration off the
keyword/template "Crisis Intelligence" and "Incident Intelligence" pipelines and
onto the unified, evidence-grounded LLM engine that already exists in
`app/intelligence/` and `app/llm/`.

---

## 0. The Unbreakable Rule

> **ZERO tolerance for keyword matching, hard-coded outputs, templates, or
> rule-based answering anywhere in the user-facing reasoning path.**

Every claim in every report must be:
1. produced by an LLM call (local Llama 3.1 8B or routed GPT-4o), and
2. grounded in a retrieved observation with a span-level citation.

The only places regex/rules are still allowed:
- `app/core/data_residency.py` — PII redaction (a safety filter, not a reasoner)
- `app/intelligence/normalization.py` — text cleanup before tokenisation
- format validation of LLM JSON output (Pydantic schemas)

Any PR that re-introduces a keyword registry, a Jinja-style report template, or
a hand-written rubric in the answer path must be rejected at review.

---

## 1. Diagnosis — Why We Are Rewriting This

The deleted `app/intelligence/crisis/` and `app/intelligence/incident/`
packages scored 99-100/100 on `mock_user_test.md` and
`neurobridge_mock_user_test.md` because the scorer and the producer were the
same rule set. Specifically:

- `signal_detectors.py` pattern-matched fixed phrases from the prompt.
- `report_builder.py` slotted those matches into a fixed 12-section template.
- `_neurobridge_scorer.py` / `_releaf_scorer.py` checked for the same phrases
  in the same section order.

This is a **circular validation artefact**, not intelligence. It would collapse
to near-zero on any scenario whose surface form differs from the seed prompts
(different school names, paraphrased posts, mixed languages, adversarial
spelling, new signal categories).

Phase 0 has removed approximately 3,800 LOC across:
`app/intelligence/crisis/**`, `app/intelligence/incident/**`,
`scripts/_neurobridge_scorer.py`, `scripts/_releaf_scorer.py`,
`scripts/run_neurobridge_stress_test.py`, `scripts/run_mock_user_stress_test.py`,
`tests/intelligence/test_crisis_pipeline.py`,
`tests/intelligence/test_incident_pipeline.py`,
`tests/scenarios/`.

Post-deletion: 703 passed, 20 skipped, 0 failed in
`tests/intelligence/ tests/llm/`.

---

## 2. Decisions Already Made (Frozen)

| Decision | Choice | Rationale |
|---|---|---|
| Local base model | **Llama 3.1 8B-Instruct** via Ollama | Best 8B reasoner with open weights; fits 24 GB consumer GPU at 4-bit. |
| Frontier model | **OpenAI GPT-4o** | Used by `LLMRouter` for high-stakes / long-context signals. API keys supplied by user. |
| Fine-tune hardware | **A100 40 GB on Google Colab** | QLoRA on Llama 3.1 8B fits in <30 GB at bs=4, seq=4096. |
| Fine-tune method | **QLoRA** via `app/llm/training/lora_trainer.py` | NF4 + LoRA rank 16 on attn+MLP projections. |
| Judge model | **Claude 3.5 Sonnet** (frontier, different family from the trainee) | Avoids same-family bias when scoring Llama outputs. |
| Acceptance floor | **≥ 85 / 100** on the unseen-scenario judge set | User-approved trade: real generalisation > template perfection. |
| Telemetry / egress | **Zero-egress PII guard mandatory** on every outbound call | `DataResidencyGuard.redact()` + `verify_clean()`. |

---

## 3. Target Architecture

```
RawObservation
  └─► NormalizationEngine            (app/intelligence/normalization.py)
        └─► CandidateRetriever       (candidate_retrieval.py + HNSW)
              └─► LLMAdjudicator     (llm_adjudicator.py)         ◄── JSON schema
                    ├─ if complex ──► MultiAgentOrchestrator      (orchestrator.py)
                    └─ if uncertain► ChainOfThoughtReasoner       (cot_reasoner.py)
                          └─► DeliberationEngine                  (deliberation.py)
                                └─► Calibrator + AbstentionDecider
                                      └─► ClaimVerifier (groundedness)
                                            └─► ResponseGenerator (response_generator.py)
                                                  └─► SituationReport (rendered by LLM, not template)
```

Routing (`app/llm/router.py`):
- **Tier A (GPT-4o)** — signals classified Security / Legal / PR / Safety, or
  observation length > 1500 chars, or low calibrated confidence after CoT.
- **Tier B (Llama 3.1 8B fine-tuned)** — everything else.

Memory: `ContextMemoryStore` (embedding RAG over past observations, debunked
claims, and labelled exemplars) is queried before every adjudication.

---

## 4. Phase Plan

### Phase 0 — Stop the bleeding (DONE)
Delete the regex pipelines, their scorers, drivers, and tests. Confirm core
suite green.

### Phase 1 — Data foundation (Weeks 1–4)
- Stand up labelling environment (see §5).
- Author 60 seed scenarios + 120 adversarial variants → 180 scenarios total.
- Produce gold `TrainingExample` JSONL via `app/llm/training/data_pipeline.py`.
- Build the held-out judge set (40 unseen scenarios, never seen in training).
- Implement `app/evals/scenario_eval.py` (LLM-as-judge runner; rubric stored as
  prompt, **not** as code).

### Phase 2 — Fine-tune the local model (Weeks 4–6)
- QLoRA on Llama 3.1 8B via `LoRATrainer` on A100 Colab.
- Track loss, eval loss, calibration, abstention rate per checkpoint.
- Promote checkpoint only when judge score ≥ 85 and ECE ≤ 0.08.

### Phase 3 — Wire the runtime (Weeks 6–7)
- `app/intelligence/situation_engine.py` (new) — orchestrates Normalize →
  Retrieve → Adjudicate → CoT → Deliberate → Verify → Render.
- Replace every public entry point that previously called the deleted
  `crisis/incident` packages.
- Enforce `DataResidencyGuard.verify_clean()` at the LLM boundary.

### Phase 4 — Continuous evaluation (Week 7+)
- Nightly judge run on the 40-scenario held-out set; alert if score < 85.
- Weekly drift report (embedding distance between live observations and
  training distribution).
- Quarterly red-team round: 20 fresh adversarial scenarios authored by people
  who have never seen the training set.

### Phase 5 — Hardening for industrial deployment
- Implement the remaining items from
  `Industrial_Deployment_Strict_Recommendations.md` (fail-closed degradation,
  connector governance, operator trust surfaces).

---

## 5. Internal Labelling Team — Detailed Scope

This section is operational; it tells the team exactly what to do, in what
format, at what rate, and to what quality bar.

### 5.1 Team composition (minimum 6 FTE-equivalent)

| Role | Headcount | Background | Responsibilities |
|---|---:|---|---|
| **Labelling Lead** | 1 | ML / data ops, 3+ yr labelling experience | Owns guidelines, IAA, gold set, weekly calibration; final arbiter on disputes. |
| **Domain Annotators** | 3 | Trust & safety / crisis comms / incident response background | Produce primary labels. At least one with public-school crisis experience, one with cybersecurity / breach response experience, one with corporate PR experience. |
| **Adjudicator** | 1 | Senior T&S analyst | Resolves disagreements flagged by the IAA check; cannot also be a primary annotator on the same item. |
| **PII / Privacy Reviewer** | 1 (can be part-time legal/privacy) | Privacy / compliance | Audits 100 % of items before they leave the labelling environment; signs off on the redaction log. |
| **Scenario Author** | rotates among Lead + Annotators | — | Writes new scenarios and adversarial variants (§5.5). |

No annotator may label and adjudicate the same scenario. No annotator may
author a scenario and label that same scenario.

### 5.2 Annotation guidelines (living document)

Stored at `docs/labelling/guidelines.md` (to be created in Phase 1).
Mandatory contents:

1. **Definitions** for every `SignalType` (Security, Legal, Reputational,
   Operational, Safety, etc.) with positive and negative examples.
2. **Severity rubric** (SEV-1 … SEV-5) with concrete thresholds — e.g. SEV-2 =
   "confirmed PII exposure ≤ 1,000 individuals, no SSN/financial data".
3. **Citation policy**: every claim in the gold answer must cite at least one
   observation span (`{post_id, char_start, char_end}`).
4. **Refusal / abstention policy**: when the evidence is insufficient, the
   gold label is `abstain` with a reason; annotators are explicitly forbidden
   from speculating.
5. **Style guide for gold reports**: factual, hedged where evidence is
   partial, no marketing language, no second-person ("you should…") imperatives.
6. **Versioning**: every change to the guidelines bumps a semver tag; all
   prior labels are re-checked against the new version on the next weekly
   calibration.

### 5.3 Label schema (matches `app/llm/training/data_pipeline.TrainingExample`)

Each item is a JSONL row:

```jsonc
{
  "example_id": "sha256-prefix",
  "messages": [
    {"role": "system",    "content": "<frozen system prompt v1.2>"},
    {"role": "user",      "content": "<scenario observations, PII-scrubbed>"},
    {"role": "assistant", "content": "<gold situation report, JSON>"}
  ],
  "source": "internal_label_batch_2026_05",
  "quality_score": 0.97,
  "contains_pii": false,
  "anonymized": true,
  "labels": {
    "signal_type":   "Security",
    "severity":      "SEV-2",
    "confidence":    0.88,
    "abstain":       false,
    "citations":     [{"post_id":"p_017","start":42,"end":118}, ...],
    "rationale":     "<2-4 sentence justification, used for CoT training>"
  },
  "guideline_version": "1.2.0",
  "annotator_ids":   ["A03","A07"],
  "adjudicator_id":  null
}
```

The `assistant` content is the **gold situation report** the model must learn
to produce; it is itself structured JSON validated against
`SituationReportSchema` (to be defined in Phase 3 alongside
`situation_engine.py`). Annotators write the report in a web form; the form
serialises to this schema on submit.

### 5.4 Inter-annotator agreement (IAA)

- **Target**: Cohen's κ ≥ 0.80 on `signal_type`, weighted κ ≥ 0.75 on
  `severity`, ≥ 0.85 exact match on `abstain`.
- **Measurement cadence**: every batch of 50 items, the Lead injects 10
  double-labelled items (each labelled independently by two annotators). κ is
  computed weekly.
- **Failure response**: if κ drops below target for any category, labelling
  pauses for that category, the Lead runs a 60-minute calibration session, and
  the last batch is re-labelled.
- **Adjudication**: items with annotator disagreement are routed to the
  Adjudicator; the Adjudicator's decision is the gold and is recorded with the
  disagreement metadata for later analysis.

### 5.5 Scenario authoring

| Pool | Count | Purpose |
|---|---:|---|
| **Seed scenarios** | 60 | Cover the full `SignalType × Severity` matrix at least twice. Realistic, well-formed observations. |
| **Adversarial variants** | 120 (2 per seed) | One paraphrase variant + one adversarial variant per seed. |
| **Held-out judge set** | 40 | Authored last, by people who do not touch training data; never enter the training JSONL. |

Adversarial variants must include at least one of: paraphrased phrasing, named
entity swap (e.g. different school / company), code-switched / non-English
fragments, deliberate misspellings, sarcasm or negation, distractor posts on
unrelated topics, contradictory sources requiring abstention.

Each scenario ships as a folder under `data/scenarios/<id>/` with:
`observations.jsonl`, `gold_report.json`, `metadata.yaml`
(author, date, guideline version, adversarial tags).

### 5.6 PII review

- All raw scenario text is first run through `DataResidencyGuard.redact()`
  before it reaches an annotator.
- The PII Reviewer spot-audits 20 % of every batch and 100 % of any batch
  flagged by `verify_clean()`.
- Any leak found triggers: (a) immediate quarantine of the batch, (b) a
  post-mortem within 48 h, (c) addition of the missed pattern to the
  `_EXTENDED_PII` registry in `app/core/data_residency.py`.
- The reviewer's sign-off (`pii_review_passed: true`, reviewer ID, timestamp)
  is appended to each scenario's `metadata.yaml`; missing sign-off blocks the
  scenario from entering the training set.

### 5.7 Throughput targets

| Activity | Rate per annotator | Notes |
|---|---|---|
| Primary labelling of a scenario | ~6 / day | ~60 min average including reading sources, drafting gold report, citing spans. |
| Double-labelling (IAA items) | counted within the 6 / day | — |
| Adjudication | ~15 disputes / day | Lower cognitive load — comparing two existing labels. |
| Scenario authoring | ~2 / day | Includes drafting, gold answer, metadata, peer review. |
| PII audit | ~40 items / day | Spot-audit + full audit on flagged batches. |

**Aggregate plan for Phase 1**: 3 annotators × 6 items/day × 20 working days =
360 primary labels in 4 weeks, comfortably covering the 180 scenarios with
double-labelling overhead and re-work budget.

---

## 6. Definition of Done for the Migration

The migration is complete when **all** of the following hold:

1. No file under `app/` contains a keyword registry, phrase list, or report
   template used in the answer path. (Enforced by a CI grep check added in
   Phase 3.)
2. Every public entry point that previously returned a regex-built report now
   returns a `SituationReport` produced by `situation_engine.py`.
3. Nightly LLM-judge run on the 40-scenario held-out set reports **≥ 85 / 100**
   for 7 consecutive days.
4. Expected Calibration Error on the held-out set ≤ 0.08.
5. Zero `DataResidencyViolationError` events in the prior 7 days of runtime
   logs.
6. `Industrial_Deployment_Strict_Recommendations.md` items tagged
   "blocking for broad deployment" are all resolved or have an explicit,
   time-boxed exception signed off by the owner.

Until all six conditions hold, the system ships only to internal users and
supervised pilots, per the standard in
`Industrial_Deployment_Strict_Recommendations.md`.
