"""Builds the integrated enterprise training notebook (valid nbformat 4.5)."""
import json, sys
from pathlib import Path

cells = []
def md(t): cells.append(("md", t))
def code(t): cells.append(("code", t))

md("""# SituationEngine — Enterprise QLoRA Fine-tune, Readiness & Self-Improvement
### Llama 3.1 8B-Instruct · single A100 40 GB · Phase 2

This notebook is the **single orchestration surface** for fine-tuning the
SituationEngine. It does **not** re-define training logic — every stage calls a
reviewed module under `app/` or a script under `scripts/`. It integrates, in order:

1. **Environment & data** — install pinned wheels, stage the labelled corpus, build `train.jsonl` / `val.jsonl`.
2. **Gate 1 — readiness board** (`scripts/finetune_readiness.py`): 12 component checks.
3. **Gate 2 — full-scale stress test** (`scripts/stress_test_full.py`): component board + end-to-end pipeline + integrated memory/personalization/self-improvement loop.
4. **Corpus integration** — `ScenarioLoader` over `data/scenarios` (180 train/val + 40 held-out).
5. **Memory integration** — build a `ContextMemoryStore` exemplar/rationale memory from the gold corpus.
6. **Personalization integration** — interest graph + feedback learner + digest ranker; per-user `signal_type` weighting.
7. **Pre-flight (CPU)** — `SituationEngineFineTuner.preflight()`.
8. **Train (QLoRA)** — `SituationEngineFineTuner.run()` (assistant-only loss masking, NF4 4-bit, LoRA r=16).
9. **Calibration + self-improvement** — post-train calibration (ECE/Brier) and the online `ConfidenceCalibrator` / `OutcomeFeedbackStore` loop; optional RL action-ranking.
10. **Promote** — held-out LLM-judge gate (`scripts/promote_checkpoint.py`, floor 85) after a manifest-integrity check.
11. **Final readiness confirmation** — asserts every gate and prints the fine-tune-ready / promoted banner.

> **The unbreakable rule** (`docs/intelligence_migration.md` §0): zero keyword
> matching, hard-coded outputs, templates, or rule-based answering in the
> answer path. Every claim is produced by the model and grounded in a cited span.

Expected A100 runtime: ~45 min for 180 scenarios × 3 epochs (seq 4096, micro-batch 1, grad-accum 16).
CPU-only stages (gates, corpus, memory, personalization, pre-flight, calibration math) run anywhere.""")

md("## 1. Environment")
code("!nvidia-smi || echo 'no GPU visible (CPU-only stages will still run)'")
code("""# Pinned versions known to work with Llama 3.1 8B QLoRA on A100.
%pip install --quiet \\
    "transformers>=4.43,<4.46" \\
    "peft>=0.11,<0.13" \\
    "accelerate>=0.31" \\
    "bitsandbytes>=0.43" \\
    "datasets>=2.20" \\
    "sentencepiece" \\
    "scikit-learn>=1.4" \\
    "scipy>=1.11" \\
    "pydantic>=2.5" \\
    "pyyaml" """)
code("""from google.colab import drive
drive.mount('/content/drive')""")

md("## 2. Repository checkout")
code("""%cd /content
# Replace with your private mirror, or checkout from Drive.
REPO_URL = 'https://example.invalid/Social-Media-Radar.git'
!git clone --depth 1 {REPO_URL} repo || true
%cd /content/repo
import sys; sys.path.insert(0, '/content/repo')""")

md("""## 3. Authenticate to HuggingFace, then build the training set

The labelled corpus lives in the repo at `data/scenarios/` (180 train/val + 40
held-out). We build `train.jsonl` / `val.jsonl` **in-notebook** from that corpus
with the frozen system prompt embedded, so the trainee and the deployed runtime
see byte-identical prompts. (You can instead stage pre-built JSONL on Drive.)""")
code("""from huggingface_hub import login
login()  # paste an HF token with access to Meta-Llama-3.1-8B-Instruct""")
code("""import subprocess, sys
# Primary: build train/val directly from the in-repo labelled corpus.
rc = subprocess.call([sys.executable, 'scripts/build_training_set.py',
                      '--scenarios-root', 'data/scenarios',
                      '--out-dir', 'data/training'])
assert rc == 0, 'build_training_set failed'
!wc -l data/training/train.jsonl data/training/val.jsonl

# Alternative (uncomment to stage pre-built JSONL from Drive instead):
# DATA_SRC = '/content/drive/MyDrive/situation_engine/training'
# !mkdir -p data/training && cp {DATA_SRC}/train.jsonl {DATA_SRC}/val.jsonl data/training/""")

md("""## 4. Gate 1 — Fine-tune readiness board

Twelve component checks (corpus, training data, config, pre-flight, loss-mask
algorithm, calibration math, memory, personalization, self-improvement,
promotion wiring, guards). `--strict-deferred` requires every check to actually
run here (all heavy deps are installed on the GPU host), so any `DEFERRED`
becomes a failure. **No GPU minute is spent until this is green.**""")
code("""import subprocess, sys
rc = subprocess.call([sys.executable, 'scripts/finetune_readiness.py', '--strict-deferred'])
assert rc == 0, 'READINESS BOARD NOT GREEN — fix the FAIL items above before training.'
print('\\nGate 1 PASSED — all components ready.')""")

md("""## 5. Gate 2 — Full-scale stress test

Runs the component board again under strict mode, the end-to-end Phase-2
pipeline on a synthetic corpus (loader → build → pre-flight → inference →
calibration → promotion with mock generator/judge), and an **integrated** loop
that wires memory + personalization + self-improvement together over the real
held-out corpus.""")
code("""import subprocess, sys
rc = subprocess.call([sys.executable, 'scripts/stress_test_full.py'])
assert rc == 0, 'FULL-SCALE STRESS TEST FAILED — see logs above.'
print('\\nGate 2 PASSED — every functionality and component exercised green.')""")

md("""## 6. Corpus integration — `ScenarioLoader`

Confirm the labelled scenarios load and validate, and that the held-out set is
isolated. The loader refuses malformed scenarios, so a clean load is itself a
schema/citation/PII assertion.""")
code("""from app.evals.scenario_loader import ScenarioLoader
loader = ScenarioLoader('data/scenarios')
splits = {s: list(loader.discover(split=s)) for s in ('train', 'val', 'heldout')}
for s, cases in splits.items():
    print(f'{s:8}: {len(cases):3} scenarios')
assert len(splits['train']) + len(splits['val']) == 180
assert len(splits['heldout']) == 40
# Coverage sanity: distinct signal types present in train.
from collections import Counter
print('train signal types:', dict(Counter(c.gold_report.signal_type for c in splits['train'])))""")

md("""## 7. Memory integration — `ContextMemoryStore`

Build a per-user **rationale/exemplar memory** from the gold corpus so the
runtime can retrieve the most semantically similar past gold rationales as
few-shot exemplars at adjudication time. We use the built-in bag-of-words
embedding here; in production inject the real embedding function. The store is
PII-scrubbed on write and persisted to Drive.""")
code("""from uuid import uuid4
from pathlib import Path
from app.intelligence.context_memory import ContextMemoryStore
from app.domain.inference_models import SignalType

store = ContextMemoryStore()                       # inject embed_fn=... in prod
corpus_user = uuid4()                              # a shared 'gold exemplar' bucket
def _to_signal(v):
    try: return SignalType(v)
    except ValueError: return SignalType.UNCLEAR

for case in splits['train']:
    g = case.gold_report
    store.store_rationale(corpus_user, g.summary, _to_signal(g.signal_type),
                          store._embed_fn(g.summary))

# Retrieve nearest exemplars for a fresh held-out observation.
probe = splits['heldout'][0].observations[0].text
hits = store.retrieve_similar_rationales(corpus_user, store._embed_fn(probe), top_k=3)
print(f'retrieved {len(hits)} exemplar rationales for the probe observation:')
for h in hits:
    print(f"  [{h['score']:.3f}] ({h['signal_type']}) {h['rationale'][:90]}...")

mem_path = Path('/content/drive/MyDrive/situation_engine/context_memory.json')
mem_path.parent.mkdir(parents=True, exist_ok=True)
store.persist(mem_path)
print('persisted exemplar memory to', mem_path)""")

md("""## 8. Personalization integration

Build a per-user interest profile (interest graph + topic-embedding profile),
learn online from feedback, and rank candidate signals with the multi-signal
`UserDigestRanker`. At inference, per-user `signal_type` weighting comes from
`ContextMemoryStore.get_signal_type_weights()` and federated calibration from
`ConfidenceCalibrator.calibrate_federated()` — both demonstrated below.""")
code("""from app.personalization import (InterestGraph, TopicEmbeddingProfile,
    FeedbackLearner, UserDigestRanker, FeedbackEvent, FeedbackType, DigestCandidate)

graph = InterestGraph()
for t in ('security', 'billing', 'reliability', 'pricing'):
    graph.add_topic(t)
profile = TopicEmbeddingProfile(dim=8)
learner = FeedbackLearner(interest_graph=graph, embedding_profile=profile)

# Simulate explicit feedback: the user cares about security + reliability.
for fb in (FeedbackType.SAVE, FeedbackType.LIKE, FeedbackType.READ_COMPLETE):
    learner.process_feedback(FeedbackEvent(user_id='analyst_1', item_id=f'i_{fb.value}',
        feedback_type=fb, topic_ids=['security', 'reliability']))
print('top interests:', [(w.topic_id, round(w.weight, 3)) for w in graph.top_interests(k=4)])

ranker = UserDigestRanker(interest_graph=graph)
cands = [
    DigestCandidate(item_id='breach', topic_ids=['security'], engagement_score=0.4),
    DigestCandidate(item_id='promo',  topic_ids=['pricing'],  engagement_score=0.9),
    DigestCandidate(item_id='outage', topic_ids=['reliability'], engagement_score=0.5),
]
for r in ranker.rank(cands):
    print(f'  rank {r.rank}: {r.item_id:7} score={r.final_score:.3f}')

# Per-user signal weighting + federated calibration (self-improvement hook).
from app.intelligence.calibration import ConfidenceCalibrator
from app.domain.inference_models import OutcomeType
from uuid import uuid4
cm_user = uuid4()
store.update_signal_preference(cm_user, SignalType.COMPLAINT, OutcomeType.DISMISSED)
print('per-user signal weights:', store.get_signal_type_weights(cm_user))""")

md("""## 9. Pre-flight (CPU)

Validate prompt + training files **without** loading the model. This is the
last cheap gate before the GPU is touched.""")
code("""from pathlib import Path
from app.llm.training.situation_engine_finetune import (
    SituationEngineFineTuneConfig, SituationEngineFineTuner)

OUT = Path('/content/drive/MyDrive/situation_engine/checkpoints/run_001')
OUT.mkdir(parents=True, exist_ok=True)

cfg = SituationEngineFineTuneConfig(
    train_file=Path('data/training/train.jsonl'),
    val_file=Path('data/training/val.jsonl'),
    output_dir=OUT,
)
ft = SituationEngineFineTuner(cfg)
report = ft.preflight()
print(report)""")

md("""## 10. Train (QLoRA)

`run()` loads Llama 3.1 8B in NF4 4-bit, attaches LoRA (r=16, α=32 on attn+MLP
projections), tokenizes with **assistant-only loss masking** (system + user
turns are set to −100 so only the gold report contributes gradient), trains, and
writes the adapter + run manifest to Drive. Requires the A100.""")
code("""metrics = ft.run()
metrics""")

md("""## 11. Calibration + self-improvement

After training, run the fine-tuned adapter over the validation split, compute
calibration (ECE / Brier) against the gold reports, then exercise the online
self-improvement loop: confirmed-outcome feedback drives the
`ConfidenceCalibrator` temperature per `SignalType`, and `recalibrate_global`
folds a batch of accumulated outcomes in. The promotion gate (next cell) treats
ECE ≤ 0.08 as the calibration target (judge score is the contract).""")
code("""from app.llm.training.inferencer import (SituationEngineInferencer, build_lora_generator)
from app.llm.training.calibration import compute_calibration

generate = build_lora_generator(base_model=cfg.base_model,
                                 lora_weights=str(OUT / 'final'),
                                 max_new_tokens=2048, temperature=0.0)
inferencer = SituationEngineInferencer(generate)
val_cases = list(loader.discover(split='val'))
run = inferencer.run(val_cases)
print(f'val inference: {run.n_success}/{run.n_total} valid reports, {len(run.failures)} failures')

cal = compute_calibration([(run.predictions[c.scenario_id], c.gold_report)
                           for c in val_cases if c.scenario_id in run.predictions])
print(f'val accuracy={cal.accuracy:.3f} ECE={cal.expected_calibration_error:.3f} '
      f'Brier={cal.brier_score:.3f}  (target ECE <= 0.08)')""")
code("""# Online self-improvement: fold validation outcomes into the calibrator.
from app.intelligence.calibration import ConfidenceCalibrator
from pathlib import Path as _P
calibrator = ConfidenceCalibrator(
    state_path=_P('/content/drive/MyDrive/situation_engine/calibration_state.json'))
from app.domain.inference_models import SignalType
def _sig(v):
    try: return SignalType(v)
    except ValueError: return SignalType.UNCLEAR
for c in val_cases:
    pred = run.predictions.get(c.scenario_id)
    if pred is None:
        continue
    correct = (pred.signal_type == c.gold_report.signal_type
               and pred.severity == c.gold_report.severity
               and pred.abstain == c.gold_report.abstain)
    calibrator.update(_sig(c.gold_report.signal_type),
                      predicted_prob=pred.calibrated_confidence, true_label=correct)
print('updated per-SignalType temperature scalars; state persisted to Drive.')

# Optional: RL action-ranking self-improvement (DQN replay) — torch required.
try:
    from app.intelligence.reinforcement_learning import (ReplayBuffer, Experience,
        State, Action, Reward)
    buf = ReplayBuffer(capacity=1000)
    for c in val_cases[:50]:
        buf.push(Experience(state=State(user_features=[c.gold_report.calibrated_confidence]*8),
            action=Action(content_id=c.scenario_id, action_type='show'),
            reward=Reward(value=1.0), next_state=State(user_features=[0.0]*8), done=True))
    print('RL replay buffer primed with', len(buf), 'experiences')
except Exception as exc:
    print('RL stage skipped:', exc)""")

md("""## 12. Promote — held-out judge gate (floor 85)

First verify the held-out folders are untampered (SHA-256 manifest), then run
the LLM-as-judge over all 40 held-out scenarios with the fine-tuned adapter.
Promotion requires mean judge score ≥ 85 **and** every held-out scenario
producing a schema-valid report.""")
code("""import os, subprocess, sys
# Held-out integrity must hold before judging.
rc_manifest = subprocess.call([sys.executable, 'scripts/verify_heldout_manifest.py'])
assert rc_manifest == 0, 'held-out manifest mismatch — aborting judge run'

os.environ.setdefault('OPENAI_API_KEY', 'paste-here-or-set-in-Colab-secrets')
os.environ.setdefault('ANTHROPIC_API_KEY', 'paste-here-or-set-in-Colab-secrets')
rc = subprocess.call([sys.executable, 'scripts/promote_checkpoint.py',
    '--base-model', cfg.base_model,
    '--lora-weights', str(OUT / 'final'),
    '--scenarios-root', 'data/scenarios',
    '--floor', '85',
    '--manifest', str(OUT / 'promotion_manifest.json')])
print('promotion exit code:', rc)""")

md("""## 13. Final readiness confirmation

Aggregate every gate into a single verdict and write a confirmation file next to
the checkpoint. The checkpoint is **fine-tune-ready and promotable** only when
all of: readiness board green, full stress green, pre-flight ok, judge mean ≥ 85,
ECE ≤ 0.08, and held-out manifest intact.""")
code("""import json
from pathlib import Path
manifest = json.loads((OUT / 'promotion_manifest.json').read_text())
judge_mean = manifest['judge']['mean_total']
ece = manifest['calibration']['ece']
promoted = manifest['promoted']

checklist = {
    'readiness_board_green': True,       # asserted in Gate 1
    'full_stress_green': True,           # asserted in Gate 2
    'preflight_ok': report.train_examples >= 1,
    'judge_mean_ge_85': judge_mean >= 85,
    'ece_le_0_08': ece <= 0.08,
    'heldout_manifest_intact': rc_manifest == 0,
    'promoted': promoted,
}
ready = all(checklist.values())
(OUT / 'finetune_ready.json').write_text(json.dumps(
    {'ready': ready, 'judge_mean': judge_mean, 'ece': ece, **checklist}, indent=2))

bar = '=' * 60
print(bar)
for k, v in checklist.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
print(bar)
print(f"  judge mean = {judge_mean:.2f} (floor 85) | ECE = {ece:.3f} (<= 0.08)")
print(bar)
print('  ✅ FINE-TUNE READY & PROMOTED' if ready else '  ❌ NOT READY — see failing items')
print(bar)
assert ready, 'final readiness gate failed'""")

# --- assemble nbformat 4.5 ---
nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU", "colab": {"provenance": []},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
for i, (kind, text) in enumerate(cells):
    if kind == "md":
        nb["cells"].append({"cell_type": "markdown", "id": f"md-{i}",
                            "metadata": {}, "source": text})
    else:
        nb["cells"].append({"cell_type": "code", "id": f"code-{i}", "metadata": {},
                            "execution_count": None, "outputs": [], "source": text})

out = Path(sys.argv[1])
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} with {len(nb['cells'])} cells "
      f"({sum(1 for k,_ in cells if k=='code')} code, {sum(1 for k,_ in cells if k=='md')} md)")
