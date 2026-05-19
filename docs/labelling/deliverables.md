# Labelling Deliverables — Phase 1 Work Brief

**Owner**: Labelling Lead
**Audience**: Domain Annotators, Adjudicator, PII Reviewer, Scenario Authors
**Companion documents**:
- `docs/intelligence_migration.md` — strategy & phase plan
- `docs/labelling/guidelines.md` — what counts as a correct label
- `data/scenarios/README.md` — folder layout contract

This brief enumerates the **four concrete deliverables** that gate the
transition from Phase 1 (data foundation) to Phase 2 (fine-tuning). Until all
four are met, no checkpoint may be promoted and no production rollout may
proceed.

| # | Deliverable | Status gate |
|---|---|---|
| 1 | Train/Val corpus: 60 seed + 120 adversarial = **180 scenarios** | All in `data/scenarios/` with `split ∈ {train, val}`. |
| 2 | Held-out judge set: **40 scenarios** | All in `data/scenarios/` with `split = heldout`. |
| 3 | Inter-Annotator Agreement: **κ ≥ 0.80** on `signal_type`, **weighted κ ≥ 0.75** on `severity` | Computed weekly per §3.4 below; logged in `data/labelling/iaa_log.jsonl`. |
| 4 | **100 %** PII sign-off | Every scenario's `metadata.yaml` carries `pii_review_passed: true` and a `pii_reviewer_id`. |

---

## 1. Deliverable 1 — Train/Val Corpus (180 scenarios)

### 1.1 Reasoning

The local Llama 3.1 8B model will be QLoRA-fine-tuned on these scenarios. We
need both **breadth** (every `SignalType × Severity` cell covered at least
twice) and **robustness** (the model must not collapse on paraphrases or
adversarial surface forms). 60 seeds give breadth; 120 adversarial variants
(2 per seed) give robustness. Splitting train/val 80/20 yields **144 train +
36 val**.

### 1.2 Specific tasks

1. **Author the coverage matrix** (Lead, before authoring starts).
   Spreadsheet with rows = `SignalType` (12 values from
   `app.domain.inference_models.SignalType`), columns = `Severity` (SEV-1 …
   SEV-5). Each non-empty cell must list **≥ 2** seed scenario IDs by end of
   Phase 1. Cells marked "N/A" (e.g. `praise × SEV-1`) require Lead sign-off.

2. **Write 60 seed scenarios** (Scenario Authors, rotating). Each seed is a
   realistic, well-formed scenario drawn from public sources or synthetic but
   plausible content. Targets per author: ~2 per day (see §5.7 of the
   migration doc).

3. **Produce 2 adversarial variants per seed** (120 total). Each variant must
   apply **at least one** of the following transformations, recorded in
   `metadata.yaml.adversarial_tags`:
   - `paraphrase` — same facts, different wording
   - `entity_swap` — different named entities (school, company, person)
   - `code_switch` — partial non-English content
   - `misspelling` — deliberate typos / leetspeak
   - `sarcasm_or_negation` — surface form contradicts intent
   - `distractor_posts` — unrelated noise mixed into the observations
   - `contradictory_sources` — sources disagree; gold label is often
     `abstain=True`

4. **Double-label every scenario** (two Domain Annotators independently).
   Disagreements feed §3 (IAA).

5. **Assign splits**. 80 % `train` / 20 % `val`, **stratified by SignalType
   and adversarial_tag** so val isn't biased toward easy cases. Lead owns
   the split assignment; record the seed for the stratifier in
   `data/labelling/split_seed.txt`.

### 1.3 Format requirements

Each scenario folder under `data/scenarios/<scenario_id>/`:

- `observations.jsonl` — one row per observation. Schema:
  `app.evals.scenario_loader.Observation`. Fields:
  `observation_id` (str), `source` (str), `timestamp` (ISO-8601 or null),
  `text` (str, ≤ 20 000 chars, PII-scrubbed).
- `gold_report.json` — single object. Schema:
  `app.intelligence.situation_report.SituationReport`. Every claim must cite
  ≥ 1 citation; every citation's `post_id` must reference an
  `observation_id` that exists in `observations.jsonl`; every
  `char_start`/`char_end` must be exact offsets into the observation `text`.
- `metadata.yaml` — schema:
  `app.evals.scenario_loader.ScenarioMetadata`. Required fields include
  `split: train | val`, `guideline_version` (semver), `annotator_ids`
  (≥ 1), `created_on` (date), `adversarial_tags` (list), and the two PII
  fields (§4).

Scenario IDs follow `<domain>_<NNN>` (e.g. `security_001`, `legal_037`).
**Never reuse an ID** — if a gold answer needs to change after PII sign-off,
create a new folder.

### 1.4 Integration

- Loader: `app.evals.scenario_loader.ScenarioLoader.discover(split="train"|"val")`
  validates every folder and refuses to yield malformed scenarios.
- Training-set builder: `python scripts/build_training_set.py`
  - Reads only `split ∈ {train, val}` (heldout is excluded by construction).
  - Emits `data/training/train.jsonl` and `data/training/val.jsonl` in
    `TrainingExample` format (`app.llm.training.data_pipeline.TrainingExample`).
  - Embeds the frozen system prompt from
    `app/llm/prompts/situation_engine_system.txt` (Phase 2 will populate
    this; a `[PLACEHOLDER]` warning is emitted until then).
- CI hook (Phase 2): a smoke job calls `build_training_set.py --dry-run` on
  every PR that touches `data/scenarios/**` to fail fast on schema breaks.

### 1.5 Definition of done

- 60 seed scenarios authored, double-labelled, adjudicated where needed.
- 120 adversarial variants produced (2 per seed, ≥ 1 transformation tag each).
- All 180 validate under `ScenarioLoader.discover()` with no exceptions.
- Stratified 80/20 split assigned and recorded.
- Coverage matrix has ≥ 2 entries in every non-N/A cell.

---

## 2. Deliverable 2 — Held-out Judge Set (40 scenarios)

### 2.1 Reasoning

The honest 85/100 floor (per `Industrial_Deployment_Strict_Recommendations.md`
and the migration plan) is only meaningful if the judge set was **never
seen during training**. A leak — even partial — invalidates every score.
40 scenarios is the minimum that gives a tight enough confidence interval
on the mean total (≈ ±3 pts at 95 %).

### 2.2 Specific tasks

1. **Hard authorship firewall.** The held-out set is authored by people who
   **do not** label or author any `train`/`val` scenarios in the same
   phase. Rotate authorship between phases if the team is small.

2. **Coverage**. Mirror the same `SignalType × Severity` matrix as the
   train set, but at lower density: ≥ 1 entry per non-N/A cell, with a
   minimum of 8 abstention cases (`abstain=True`) to keep the Abstention
   axis of the rubric exercised.

3. **Adversarial pressure**. At least **50 %** of held-out scenarios must
   carry an `adversarial_tag` and at least **5** must combine two or more
   tags (e.g. `paraphrase + contradictory_sources`).

4. **Single-pass labelling, then adjudication review**. Held-out gold
   answers are labelled once by a senior Annotator, then reviewed by the
   Adjudicator. No double-labelling is needed because held-out items are
   **not** counted toward IAA (they would skew it).

5. **Quarantine after sign-off**. Once a held-out scenario is PII-signed,
   the Lead tags the folder read-only (`chmod -R a-w`) and records its
   SHA-256 in `data/labelling/heldout_manifest.json`. Any later
   modification breaks the manifest hash and fails the nightly judge run.

### 2.3 Format requirements

- Identical to §1.3 (`observations.jsonl`, `gold_report.json`,
  `metadata.yaml`), except `metadata.yaml.split` must be `heldout`.
- `metadata.yaml.adversarial_tags` is mandatory when the scenario uses any
  transformation (do not leave it empty for adversarial cases — the judge
  drift report uses these tags).
- Scenario IDs prefixed `heldout_<NNN>` for grep-ability.

### 2.4 Integration

- Loader: `ScenarioLoader.discover(split="heldout")`.
- Judge driver: `python scripts/run_scenario_eval.py --predictions <path>
  --floor 85`. Exits non-zero when mean total < 85; suitable for nightly CI.
- Manifest check (to be added in Phase 2 alongside the nightly job):
  `python scripts/verify_heldout_manifest.py` recomputes each folder's
  hash and compares to `heldout_manifest.json`; any mismatch aborts the
  judge run.

### 2.5 Definition of done

- 40 scenarios in `data/scenarios/heldout_*/` with `split: heldout`.
- All 40 validate under `ScenarioLoader.discover(split="heldout")`.
- `heldout_manifest.json` written and the folders are read-only.
- Authorship firewall documented in `data/labelling/heldout_attestation.md`
  (signed by the Lead).


---

## 3. Deliverable 3 — Inter-Annotator Agreement

**Targets**: Cohen's κ ≥ **0.80** on `signal_type`, weighted κ ≥ **0.75** on
`severity`, exact-match agreement ≥ **0.85** on `abstain`.

### 3.1 Reasoning

A model can only learn what its labellers agree on. Cohen's κ on the
discrete `signal_type` label and weighted κ (linear or quadratic) on the
ordinal `severity` label quantify that agreement above chance. Below the
thresholds, the dataset is too noisy to support the 85/100 floor; the
model would memorise contradictions and the judge run would punish it.

The `abstain` exact-match target is separate because abstention is a binary
safety signal: confusing "act" with "abstain" is a high-cost error that κ
on `signal_type` alone wouldn't surface.

### 3.2 Specific tasks

1. **Double-labelling injection** (Lead). Every batch of 50 scenarios
   contains **10 items independently labelled by two annotators**. The
   double-labelled items are picked at random and are not flagged to the
   annotators (so they cannot game the agreement).

2. **Weekly κ computation** (Lead). At the end of every working week,
   compute and log:
   - `kappa_signal_type` — Cohen's κ across all double-labelled items
     accumulated this week.
   - `kappa_severity_weighted` — weighted κ with linear weights across the
     5-level ordinal scale.
   - `agreement_abstain` — exact-match proportion on the boolean.

3. **Failure response**. If any metric is below target:
   - Pause labelling on the affected dimension(s).
   - Lead runs a **60-minute calibration session** using the disagreeing
     items as worked examples.
   - The 50 items in the failing batch are **re-labelled** under the
     refreshed shared understanding; the originals are archived but not
     discarded (they feed the disagreement corpus used to train CoT).
   - Bump `guideline_version` in `docs/labelling/guidelines.md` if the
     calibration produced new clarifications.

4. **Adjudication routing**. Any double-labelled item with disagreement on
   `signal_type` or with `severity` differing by ≥ 2 levels is routed to
   the Adjudicator. The Adjudicator's decision is the gold; the original
   labels are preserved in the dispute log.

### 3.3 Format requirements

All IAA artefacts live under `data/labelling/`:

- `iaa_log.jsonl` — one row per weekly computation:
  ```jsonc
  {
    "week_ending": "2026-06-05",
    "n_double_labelled": 47,
    "kappa_signal_type": 0.83,
    "kappa_severity_weighted": 0.78,
    "agreement_abstain": 0.91,
    "below_target": [],
    "guideline_version_at_time": "1.0.0"
  }
  ```
- `disputes/<scenario_id>.json` — one file per adjudicated dispute:
  ```jsonc
  {
    "scenario_id": "security_042",
    "annotators": [
      {"annotator_id": "A03", "signal_type": "security_concern",
       "severity": "SEV-2", "abstain": false},
      {"annotator_id": "A07", "signal_type": "legal_risk",
       "severity": "SEV-3", "abstain": false}
    ],
    "adjudicator_id": "ADJ01",
    "decision": {"signal_type": "security_concern", "severity": "SEV-2",
                 "abstain": false},
    "decision_rationale": "evidence of unauthorised access is the dominant signal; legal exposure is a downstream consequence",
    "decided_on": "2026-06-04"
  }
  ```
- `calibration_sessions/<date>.md` — minutes of each calibration session,
  including which guideline clauses were refined.

### 3.4 Integration

- Computation script (to be added in Phase 1.5): `python
  scripts/compute_iaa.py --since <date>` reads the double-labelled items
  from `data/labelling/double_labels/` and writes a new row to
  `iaa_log.jsonl`. Uses `sklearn.metrics.cohen_kappa_score` with
  `weights="linear"` for severity.
- Dashboard hook: the latest `iaa_log.jsonl` row is surfaced in the
  weekly labelling review; if `below_target` is non-empty, the row is
  highlighted and `build_training_set.py` refuses to include items
  labelled in that batch until re-labelling is complete (a `quarantined:
  true` flag in their metadata is set by the Lead).

### 3.5 Definition of done

- ≥ 4 consecutive weekly `iaa_log.jsonl` rows with all three metrics
  meeting target.
- All disputes resolved (no open files under `disputes/`).
- No `metadata.yaml` carries a `quarantined: true` flag at Phase-1 close.

---

## 4. Deliverable 4 — 100 % PII Sign-off

### 4.1 Reasoning

The system's competitive promise is **zero egress** of personally
identifiable information (see `docs/intelligence_migration.md` §0 and
`Industrial_Deployment_Strict_Recommendations.md`). Every observation
that enters the training set or the judge set must be reviewed and
attested by a named human reviewer; one missed SSN or email becomes a
training-time leak that the model will happily reproduce at inference.

### 4.2 Specific tasks

1. **Automated first pass**. Every raw observation is processed through
   `app.core.data_residency.DataResidencyGuard.redact()` before it
   reaches an annotator. The redaction log is appended to
   `data/labelling/pii_audit/<scenario_id>.json`.

2. **Human review**. The PII Reviewer reads every scenario before it is
   eligible for training or judging. The reviewer:
   - re-runs `DataResidencyGuard.verify_clean()` on each
     `observations.jsonl` row and on `gold_report.json`;
   - reads each observation by eye to catch context-dependent PII the
     regex didn't catch (e.g., "the principal of Northview High" is a
     contextual identifier even when the name is generic);
   - signs off in `metadata.yaml`.

3. **Spot audit + flagged audit**. The Reviewer **spot-audits 20 %** of
   every batch by re-redacting from the original source and diffing. Any
   batch where `verify_clean()` raised during the automated pass gets a
   **100 % audit**.

4. **Leak procedure**. If a leak is found at any later stage:
   - Immediate quarantine of the batch (`quarantined: true` in every
     affected `metadata.yaml`).
   - Post-mortem within **48 h**, recorded in
     `data/labelling/pii_incidents/<incident_id>.md`.
   - Missing pattern added to `_EXTENDED_PII` in
     `app/core/data_residency.py` with a unit test in
     `tests/core/test_data_residency.py` covering the pattern.
   - Affected scenarios re-redacted and re-reviewed.

### 4.3 Format requirements

PII sign-off lives in `metadata.yaml` (validated by
`app.evals.scenario_loader.ScenarioMetadata`):

```yaml
pii_review_passed: true        # boolean; required
pii_reviewer_id: P03           # required when pii_review_passed: true
```

Supporting audit records live under `data/labelling/`:

- `pii_audit/<scenario_id>.json` — output of the automated redaction pass:
  ```jsonc
  {
    "scenario_id": "security_042",
    "guard_version": "data_residency@<git-sha>",
    "redactions": [
      {"observation_id": "obs_3", "label": "email",
       "char_start": 142, "char_end": 162, "token": "<EMAIL>"},
      {"observation_id": "obs_5", "label": "ssn",
       "char_start": 88,  "char_end": 99,  "token": "<SSN>"}
    ],
    "verify_clean_passed": true,
    "reviewed_on": "2026-06-03",
    "reviewer_id": "P03"
  }
  ```
- `pii_incidents/<incident_id>.md` — post-mortem template:
  date, discovered-by, affected scenarios, missed pattern, regex added,
  test added, time-to-fix.

### 4.4 Integration

- Default-deny in the loader: `ScenarioLoader.discover(require_pii_signoff=True)`
  silently skips any scenario whose `pii_review_passed` is `False`. The
  flag defaults to `True`; passing `False` is only legal in local dev.
- Build-set gate: `scripts/build_training_set.py` calls the loader with
  the default, so unsigned scenarios cannot enter `train.jsonl` /
  `val.jsonl` even by accident.
- Judge gate: `scripts/run_scenario_eval.py` likewise calls the loader
  with the default; unsigned held-out scenarios are silently excluded.
- Runtime gate (Phase 3): `DataResidencyGuard.verify_clean()` is invoked
  at the LLM boundary; a `DataResidencyViolationError` aborts the request
  before any text reaches the model.

### 4.5 Definition of done

- Every scenario in `data/scenarios/` has `pii_review_passed: true` and a
  non-null `pii_reviewer_id`.
- Every scenario has a matching `pii_audit/<scenario_id>.json` record.
- Zero open `pii_incidents/` files.
- Zero `DataResidencyViolationError` raised when running
  `ScenarioLoader.discover()` over the full corpus in CI.

---

## 5. Phase-1 Exit Checklist

Phase 1 closes — and Phase 2 (fine-tuning) may begin — only when **all
five** boxes are checked by the Labelling Lead in writing
(`data/labelling/phase1_exit.md`):

- [ ] **Corpus**: 180 scenarios in `train` + `val`, 40 in `heldout`,
      all validate under `ScenarioLoader.discover()`.
- [ ] **Coverage**: `SignalType × Severity` matrix filled per §1.2 and §2.2;
      adversarial-tag distribution per §1.2.3 and §2.2.3.
- [ ] **IAA**: ≥ 4 consecutive weekly `iaa_log.jsonl` rows at target on all
      three metrics; no open disputes; no `quarantined: true` scenarios.
- [ ] **PII**: 100 % `pii_review_passed: true` with a named reviewer; every
      scenario has a `pii_audit/<scenario_id>.json` record; zero open
      `pii_incidents/`.
- [ ] **Quarantine**: held-out manifest written, folders read-only,
      authorship firewall attestation signed.

Once the checklist is signed, running

```
python scripts/build_training_set.py
python scripts/run_scenario_eval.py --predictions <path> --floor 85
python scripts/check_no_keyword_rules.py
```

should all exit `0` against the live tree. That is the green-light signal
for Phase 2.

---

## 6. Directory Summary

After Phase 1, the labelling outputs live as:

```
data/
  scenarios/
    <scenario_id>/
      observations.jsonl
      gold_report.json
      metadata.yaml
    heldout_<NNN>/
      ...
    _template/                       # contract example, loader ignores it
  labelling/
    split_seed.txt                   # stratification seed (§1.2.5)
    iaa_log.jsonl                    # weekly κ rows (§3.3)
    disputes/<scenario_id>.json      # adjudication records (§3.3)
    calibration_sessions/<date>.md   # session minutes (§3.3)
    pii_audit/<scenario_id>.json     # redaction records (§4.3)
    pii_incidents/<incident_id>.md   # post-mortems (§4.4)
    heldout_manifest.json            # SHA-256 of every heldout folder (§2.2.5)
    heldout_attestation.md           # authorship-firewall sign-off (§2.5)
    phase1_exit.md                   # final checklist (§5)
docs/
  labelling/
    guidelines.md                    # what counts as correct
    deliverables.md                  # this file
```
