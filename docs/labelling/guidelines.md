# Labelling Guidelines — v1.0.0

**Status**: initial release · supersedes nothing
**Owner**: Labelling Lead
**Authority**: this document is the only valid source of truth for what
constitutes a correct label. Code never encodes labelling rules.

These guidelines pair with `docs/intelligence_migration.md` §5 (team scope)
and the schemas in `app/intelligence/situation_report.py` /
`app/evals/scenario_loader.py`. Read both before labelling anything.

---

## 1. Scope

Annotators produce, for each scenario:

1. a primary `SituationReport` (the gold answer), and
2. metadata recording authorship, split, guideline version, and PII sign-off.

Annotators do **not** write production rules, prompt templates, or scoring
code. The model learns from these gold answers; nothing in this corpus
becomes an `if`/`else` branch in the engine.

---

## 2. SignalType definitions

Use the values from `app.domain.inference_models.SignalType`. For each
scenario, choose the **single best** primary type. The full enum is the
binding source; the definitions below summarise the intent.

| SignalType | Definition (intent) |
|---|---|
| `security_concern` | Confirmed or credible report of unauthorised access, data exposure, credential leak, or active exploit. |
| `legal_risk` | Statements suggesting regulatory violation, contractual breach, defamation exposure, or active litigation. |
| `reputation_risk` | Public narratives that, if amplified, materially damage brand trust. |
| `churn_risk` | Statements from identifiable users/customers indicating imminent departure. |
| `complaint` | Specific dissatisfaction tied to a product, service, or interaction. |
| `bug_report` | Reproducible technical defect with enough detail to triage. |
| `feature_request` | Specific capability request. |
| `praise` | Endorsement, recommendation, or positive testimonial. |
| `competitor_mention` | Substantive comparison or migration story involving a competitor. |
| `expansion_opportunity` | Signal of upsell, cross-sell, or partnership potential. |
| `unclear` | Evidence insufficient for any of the above; use with `abstain=True`. |
| `not_actionable` | Off-topic, spam, or content where no operator action is appropriate. |

When two types apply, prefer the one with the highest operational stakes
(security > legal > reputation > churn > complaint > everything else) and
record the rejected alternative in the dispute log.

---

## 3. Severity rubric (5 levels)

Severity is judged on the **demonstrated impact in the evidence**, not on
worst-case extrapolation.

| Severity | When to use |
|---|---|
| **SEV-1** | Imminent or in-progress harm at scale (active breach, public safety, mass-affecting outage). Operator must act now. |
| **SEV-2** | Confirmed material exposure with bounded scope (e.g., confirmed PII exposure for a defined population; named-party legal action). |
| **SEV-3** | Credible risk with partial evidence; warrants investigation within 24 h. |
| **SEV-4** | Localised or recoverable issue; routine handling. |
| **SEV-5** | Informational; no operator action required, but record for trend tracking. |

If the gold severity differs by more than one level between the two
primary annotators, route to the Adjudicator.

---

## 4. Citation policy

- Every `Claim.text` must cite at least one `Citation` whose
  `char_start`/`char_end` span actually contains the supporting evidence
  in the referenced observation.
- A citation may span up to one sentence. For multi-sentence evidence,
  emit multiple `Citation` entries.
- Do not cite spans that contain only paraphrase of the claim — the span
  must contain the **evidence**, not the conclusion.
- Spans must be exact character offsets into the observation's `text`
  field. Off-by-one is a defect.

---

## 5. Abstention policy

Abstain (`abstain=True`) **must** be used when any of the following hold:

- the observations contain only rumour, a single unverifiable account, or
  pure speculation;
- material evidence is missing (e.g., no named affected party, no time
  reference, no source corroboration);
- the observations contradict each other and there is no basis to prefer
  one over the other.

An abstaining report must:

- set `signal_type` to `unclear`,
- set `severity` to `SEV-5`,
- provide a specific `abstention_reason` (not "insufficient evidence" —
  state *which* evidence is missing),
- contain no claims and no suggested actions.

Annotators are explicitly forbidden from speculating to fill in gaps.

---

## 6. Style guide for gold reports

- Write `summary` as factual narration in the past or present tense; no
  marketing language, no second-person imperatives, no editorial
  adjectives.
- Hedge appropriately when evidence is partial ("reportedly", "according
  to a single source") and reflect the hedge in `calibrated_confidence`.
- Suggested actions are concrete (a single verb, an owner, a timeframe)
  and proportionate to severity.
- Do not reproduce raw PII in the report even if it slipped past
  redaction; flag the leak to the PII Reviewer instead.

---

## 7. Versioning and dispute workflow

- Every change to this file bumps a semver tag in §0 (`v1.0.0` → ...).
- All prior labels are re-checked against the new version on the next
  weekly calibration. Re-checked labels keep their original
  `guideline_version` in metadata; if relabelled, a new scenario folder
  is created.
- Disputes (κ-failure, severity ≥2 levels apart, conflicting SignalType)
  go to the Adjudicator. The Adjudicator's decision is final and is
  recorded with both annotators' original labels in the dispute log.

---

## 8. Targets (see also `docs/intelligence_migration.md` §5.4 / §5.7)

| Metric | Target |
|---|---|
| Cohen's κ on `signal_type` | ≥ 0.80 |
| Weighted κ on `severity` | ≥ 0.75 |
| Exact-match agreement on `abstain` | ≥ 0.85 |
| Primary labels per annotator per day | ~6 |
| Adjudications per day | ~15 |
| PII audit coverage | 20 % spot + 100 % flagged batches |
