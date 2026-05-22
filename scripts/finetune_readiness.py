"""Fine-tune readiness gate + full-component smoke test for the SituationEngine.

This is the single command the training notebook runs **before spending any GPU
minute**. It exercises every component the Phase-2 fine-tune depends on and
prints a readiness verdict.

Each check returns one of:
  PASS      — exercised successfully here.
  FAIL      — a real defect; blocks fine-tuning (exit code != 0).
  DEFERRED  — could not run here because an optional heavy dependency
              (``pydantic``/``torch``/``sklearn``/``transformers``) is absent;
              it will run on the GPU host where those wheels are installed.
              DEFERRED never fails the gate, but the banner lists them so the
              operator knows what still needs to execute on the host.

Components covered:
  1.  Labelled corpus (180 train/val + 40 held-out), schema + offset + PII.
  2.  Training data (train.jsonl / val.jsonl): 3-message structure, frozen
      system prompt, valid SituationReport assistant JSON  [pydantic-free].
  3.  Frozen system prompt present and non-empty.
  4.  QLoRA fine-tune config + trainer config (r/alpha/nf4/4-bit/target mods).
  5.  CPU pre-flight (SituationEngineFineTuner.preflight).
  6.  Assistant-only loss masking algorithm (fake-tokenizer logic test) [free].
  7.  Calibration math (ECE/Brier) + ConfidenceCalibrator online update.
  8.  Context memory store: signal-type weights + exemplar retrieval.
  9.  Personalization: interest graph + feedback learner + digest ranker.
  10. Self-improvement: OutcomeFeedbackStore -> adaptive threshold; RL replay.
  11. Judge + promotion wiring (mock generator + mock judge over held-out).
  12. No rule-based answer path + held-out manifest integrity.

Run:  python scripts/finetune_readiness.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

PASS, FAIL, DEFER = "PASS", "FAIL", "DEFERRED"


@dataclass
class Result:
    name: str
    status: str
    detail: str


def _ok(name: str, detail: str) -> Result:
    return Result(name, PASS, detail)


def _fail(name: str, detail: str) -> Result:
    return Result(name, FAIL, detail)


def _defer(name: str, detail: str) -> Result:
    return Result(name, DEFER, detail)


def _missing_dep(exc: Exception) -> bool:
    return isinstance(exc, (ImportError, ModuleNotFoundError))


# --------------------------------------------------------------------------- #
# 1. Corpus
# --------------------------------------------------------------------------- #
def check_corpus(repo: Path) -> Result:
    try:
        from verify_phase1_corpus import Checker
    except Exception as exc:  # pragma: no cover
        return _fail("corpus", f"could not import verifier: {exc}")
    chk = Checker(repo)
    chk.load_and_validate()
    fails = [m for ok, m in chk.results if not ok]
    if fails:
        return _fail("corpus", f"{len(fails)} invalid: {fails[:3]}")
    n = len(chk.scenarios)
    splits = {s: sum(1 for r in chk.scenarios.values() if r["split"] == s)
              for s in ("train", "val", "heldout")}
    if splits != {"train": 144, "val": 36, "heldout": 40}:
        return _fail("corpus", f"unexpected split counts {splits}")
    return _ok("corpus", f"{n} scenarios valid; splits={splits}")


# --------------------------------------------------------------------------- #
# 2. Training data (pydantic-free structural mirror of preflight)
# --------------------------------------------------------------------------- #
_REQUIRED_REPORT_KEYS = {"signal_type", "severity", "calibrated_confidence",
                         "summary", "claims", "citations", "suggested_actions",
                         "abstain", "abstention_reason", "schema_version"}
_VALID_SEV = {"SEV-1", "SEV-2", "SEV-3", "SEV-4", "SEV-5"}


def _check_jsonl(path: Path, frozen_prompt: str) -> Tuple[int, List[str]]:
    errs: List[str] = []
    n = 0
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        n += 1
        row = json.loads(raw)
        msgs = row.get("messages")
        if not isinstance(msgs, list) or len(msgs) != 3:
            errs.append(f"L{lineno}: messages must be 3")
            continue
        if [m.get("role") for m in msgs] != ["system", "user", "assistant"]:
            errs.append(f"L{lineno}: bad roles")
            continue
        if msgs[0]["content"].strip() != frozen_prompt.strip():
            errs.append(f"L{lineno}: system prompt != frozen prompt")
        up = json.loads(msgs[1]["content"])
        if "observations" not in up:
            errs.append(f"L{lineno}: user missing observations")
        rep = json.loads(msgs[2]["content"])
        missing = _REQUIRED_REPORT_KEYS - set(rep)
        if missing:
            errs.append(f"L{lineno}: report missing {missing}")
        if rep.get("severity") not in _VALID_SEV:
            errs.append(f"L{lineno}: bad severity")
        if not (0.0 <= rep.get("calibrated_confidence", -1) <= 1.0):
            errs.append(f"L{lineno}: bad confidence")
        if not rep.get("abstain") and not rep.get("claims"):
            errs.append(f"L{lineno}: non-abstain without claims")
    return n, errs


def check_training_data(repo: Path) -> Result:
    train = repo / "data/training/train.jsonl"
    val = repo / "data/training/val.jsonl"
    prompt = (repo / "app/llm/prompts/situation_engine_system.txt").read_text(encoding="utf-8")
    if not train.exists():
        return _defer("training_data",
                      "data/training/train.jsonl absent — run build_training_set.py")
    nt, et = _check_jsonl(train, prompt)
    nv, ev = (0, [])
    if val.exists():
        nv, ev = _check_jsonl(val, prompt)
    errs = et + ev
    if errs:
        return _fail("training_data", f"{len(errs)} issue(s): {errs[:3]}")
    return _ok("training_data", f"train={nt} val={nv}; 3-msg + frozen prompt + valid report JSON")


# --------------------------------------------------------------------------- #
# 3. Frozen prompt
# --------------------------------------------------------------------------- #
def check_frozen_prompt(repo: Path) -> Result:
    p = repo / "app/llm/prompts/situation_engine_system.txt"
    if not p.exists():
        return _fail("frozen_prompt", "missing situation_engine_system.txt")
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        return _fail("frozen_prompt", "system prompt is empty")
    return _ok("frozen_prompt", f"{len(txt)} chars present")


# --------------------------------------------------------------------------- #
# 4. Fine-tune + trainer config
# --------------------------------------------------------------------------- #
def check_finetune_config(repo: Path) -> Result:
    try:
        from app.llm.training.situation_engine_finetune import (
            SituationEngineFineTuneConfig, SituationEngineFineTuner,
        )
    except Exception as exc:
        if _missing_dep(exc):
            return _defer("finetune_config", f"deps absent: {exc}")
        return _fail("finetune_config", f"import error: {exc}")
    cfg = SituationEngineFineTuneConfig(output_dir=repo / "checkpoints/_ready_probe")
    if cfg.lora_r != 16 or cfg.lora_alpha != 32:
        return _fail("finetune_config", f"unexpected LoRA r/alpha {cfg.lora_r}/{cfg.lora_alpha}")
    for m in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if m not in cfg.target_modules:
            return _fail("finetune_config", f"missing target module {m}")
    try:
        tcfg = SituationEngineFineTuner(cfg).build_trainer_config()
    except Exception as exc:
        if _missing_dep(exc):
            return _defer("finetune_config", f"trainer cfg deps absent: {exc}")
        return _fail("finetune_config", f"build_trainer_config: {exc}")
    if not (getattr(tcfg, "use_4bit", False) and getattr(tcfg, "bnb_4bit_quant_type", "") == "nf4"):
        return _fail("finetune_config", "QLoRA 4-bit/nf4 not configured")
    return _ok("finetune_config",
               f"QLoRA r={cfg.lora_r} α={cfg.lora_alpha} nf4 4-bit, seq={cfg.max_seq_length}, "
               f"epochs={cfg.num_train_epochs}, ga={cfg.gradient_accumulation_steps}")


# --------------------------------------------------------------------------- #
# 5. CPU preflight
# --------------------------------------------------------------------------- #
def check_preflight(repo: Path) -> Result:
    try:
        from app.llm.training.situation_engine_finetune import (
            SituationEngineFineTuneConfig, SituationEngineFineTuner,
        )
    except Exception as exc:
        if _missing_dep(exc):
            return _defer("preflight", f"deps absent: {exc}")
        return _fail("preflight", f"import error: {exc}")
    train = repo / "data/training/train.jsonl"
    if not train.exists():
        return _defer("preflight", "train.jsonl absent — build it then re-run")
    cfg = SituationEngineFineTuneConfig(
        train_file=train, val_file=repo / "data/training/val.jsonl",
        output_dir=repo / "checkpoints/_ready_probe",
    )
    try:
        rep = SituationEngineFineTuner(cfg).preflight()
    except Exception as exc:
        return _fail("preflight", f"{type(exc).__name__}: {exc}")
    return _ok("preflight", f"train={rep.train_examples} val={rep.val_examples} "
                            f"assistant_chars=[{rep.assistant_min_chars},{rep.assistant_max_chars}]")


# --------------------------------------------------------------------------- #
# 6. Assistant-only loss masking algorithm (fake tokenizer; pydantic-free)
# --------------------------------------------------------------------------- #
class _FakeTokenizer:
    """Minimal stand-in modelling Llama-style chat templates for the mask test.

    Token ids are word indices; apply_chat_template emits header sentinels so
    the user-prefix is a strict prefix of the full sequence (the contract the
    real masking relies on).
    """
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False,
                            truncation=False):
        toks: List[str] = ["<bos>"]
        for m in messages:
            toks += [f"<{m['role']}>"] + m["content"].split() + [f"</{m['role']}>"]
        if add_generation_prompt:
            toks += ["<assistant>"]
        if not tokenize:
            return " ".join(toks)
        return [hash(t) % 100000 for t in toks]


def check_mask_algorithm(repo: Path) -> Result:
    # Re-implement the exact prefix-diff masking used in
    # situation_engine_finetune._encode_with_assistant_mask, then assert that
    # the system+user span is masked (-100) and the assistant span is learned.
    tok = _FakeTokenizer()
    messages = [
        {"role": "system", "content": "frozen system prompt here"},
        {"role": "user", "content": '{"observations": []}'},
        {"role": "assistant", "content": '{"signal_type": "praise"}'},
    ]
    user_only = tok.apply_chat_template(messages[:2], tokenize=True, add_generation_prompt=True)
    full = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if full[:len(user_only)] != user_only:
        return _fail("mask_algorithm", "user prefix is not a prefix of full sequence")
    labels = [-100] * len(user_only) + list(full[len(user_only):])
    n_masked = sum(1 for x in labels if x == -100)
    n_learned = sum(1 for x in labels if x != -100)
    if n_masked != len(user_only) or n_learned == 0:
        return _fail("mask_algorithm", f"masked={n_masked} learned={n_learned}")
    # Also confirm the real module's function agrees (when importable).
    try:
        from app.llm.training.situation_engine_finetune import _encode_with_assistant_mask
        ids, lab = _encode_with_assistant_mask(tok, messages, max_len=4096)
        if lab[:len(user_only)] != [-100] * len(user_only):
            return _fail("mask_algorithm", "real masker did not mask the prompt span")
    except Exception as exc:
        if not _missing_dep(exc):
            return _fail("mask_algorithm", f"real masker error: {exc}")
    return _ok("mask_algorithm",
               f"prompt span masked ({n_masked} toks), assistant learned ({n_learned} toks)")


# --------------------------------------------------------------------------- #
# 7. Calibration math + online calibrator
# --------------------------------------------------------------------------- #
def check_calibration(repo: Path) -> Result:
    # Pure cross-check of ECE on a fixed example (always runs).
    # Two preds: conf 0.9 correct, conf 0.4 wrong  → 10 buckets.
    # bucket[0.9,1.0]: conf .9 acc 1 -> |.9-1|=.1 ; bucket[0.4,0.5): conf .4 acc 0 -> .4
    # ECE = .5*.1 + .5*.4 = .25 ; Brier = ((.9-1)^2+(.4-0)^2)/2 = (.01+.16)/2 = .085
    ece_ref, brier_ref = 0.25, 0.085
    try:
        from app.intelligence.situation_report import SituationReport, Severity, Citation, Claim
        from app.llm.training.calibration import compute_calibration
        from app.intelligence.calibration import ConfidenceCalibrator
        from app.domain.inference_models import SignalType
    except Exception as exc:
        if _missing_dep(exc):
            return _defer("calibration", f"deps absent: {exc}; pure ECE ref={ece_ref}")
        return _fail("calibration", f"import error: {exc}")

    def _rep(sig, sev, conf, abstain=False):
        if abstain:
            return SituationReport(signal_type=sig, severity=sev, calibrated_confidence=conf,
                                   summary="s", abstain=True, abstention_reason="r")
        return SituationReport(signal_type=sig, severity=sev, calibrated_confidence=conf,
                               summary="s",
                               citations=[Citation(post_id="o", char_start=0, char_end=1)],
                               claims=[Claim(text="c", citation_ids=[0], confidence=conf)])
    gold1 = _rep("praise", Severity.SEV_5, 0.9)
    cand1 = _rep("praise", Severity.SEV_5, 0.9)             # correct, conf .9
    gold2 = _rep("bug_report", Severity.SEV_3, 0.4)
    cand2 = _rep("complaint", Severity.SEV_3, 0.4)          # wrong signal, conf .4
    cal = compute_calibration([(cand1, gold1), (cand2, gold2)])
    if abs(cal.expected_calibration_error - ece_ref) > 1e-6 or abs(cal.brier_score - brier_ref) > 1e-6:
        return _fail("calibration",
                     f"ECE/Brier mismatch: got ece={cal.expected_calibration_error:.4f} "
                     f"brier={cal.brier_score:.4f}, expected {ece_ref}/{brier_ref}")
    # Online calibrator: a false-positive update must raise T (cool confidence).
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        c = ConfidenceCalibrator(state_path=Path(td) / "cal.json")
        t0 = c.calibrate(2.0, SignalType.SECURITY_CONCERN)
        for _ in range(20):
            c.update(SignalType.SECURITY_CONCERN, predicted_prob=0.95, true_label=False)
        t1 = c.calibrate(2.0, SignalType.SECURITY_CONCERN)
        if not (t1 < t0):
            return _fail("calibration", f"FP updates did not reduce calibrated prob ({t0:.3f}->{t1:.3f})")
    return _ok("calibration",
               f"ECE={cal.expected_calibration_error:.3f} Brier={cal.brier_score:.3f}; "
               f"online FP update cools {t0:.3f}->{t1:.3f}")


# --------------------------------------------------------------------------- #
# 8. Context memory
# --------------------------------------------------------------------------- #
def check_memory(repo: Path) -> Result:
    # Pure cross-check of the signal-weight formula (always runs).
    # 4 dismissed of 5 -> dismissal_frac .8 -> weight max(.05, 1-.8*.95)=.24
    ref_weight = round(max(0.05, 1.0 - 0.8 * 0.95), 4)
    try:
        from app.intelligence.context_memory import ContextMemoryStore
        from app.domain.inference_models import SignalType, OutcomeType
    except Exception as exc:
        if _missing_dep(exc):
            return _defer("memory", f"deps absent: {exc}; pure weight ref={ref_weight}")
        return _fail("memory", f"import error: {exc}")
    from uuid import uuid4
    store = ContextMemoryStore()  # bag-of-words fallback embed
    uid = uuid4()
    for _ in range(4):
        store.update_signal_preference(uid, SignalType.COMPLAINT, OutcomeType.DISMISSED)
    store.update_signal_preference(uid, SignalType.COMPLAINT, OutcomeType.ACTED_ON)
    w = store.get_signal_type_weights(uid)["complaint"]
    if abs(round(w, 4) - ref_weight) > 1e-6:
        return _fail("memory", f"weight formula mismatch got {w:.4f} want {ref_weight}")
    return _ok("memory", f"signal-type weight (4 dismiss/1 act)={w:.3f}; bow-embed store ok")


# --------------------------------------------------------------------------- #
# 9. Personalization
# --------------------------------------------------------------------------- #
def check_personalization(repo: Path) -> Result:
    try:
        from app.personalization import (
            InterestGraph, FeedbackLearner, UserDigestRanker,
            FeedbackEvent, FeedbackType, DigestCandidate,
        )
    except Exception as exc:
        if _missing_dep(exc):
            return _defer("personalization", f"deps absent: {exc}")
        return _fail("personalization", f"import error: {exc}")
    g = InterestGraph()
    g.add_topic("security")
    learner = FeedbackLearner(interest_graph=g)
    before = g.get_weight("security") or 0.0
    learner.process_feedback(FeedbackEvent(
        user_id="u1", item_id="i1", feedback_type=FeedbackType.LIKE, topic_ids=["security"]))
    after = g.get_weight("security") or 0.0
    if not (after > before):
        return _fail("personalization", f"LIKE did not raise topic weight ({before}->{after})")
    ranker = UserDigestRanker(interest_graph=g)
    cands = [
        DigestCandidate(item_id="a", topic_ids=["security"], engagement_score=0.9),
        DigestCandidate(item_id="b", topic_ids=["unrelated"], engagement_score=0.1),
    ]
    ranked = ranker.rank(cands)
    if not ranked or ranked[0].item_id != "a":
        return _fail("personalization", f"ranker did not prioritise on-interest item: {[r.item_id for r in ranked]}")
    return _ok("personalization",
               f"LIKE raised weight {before:.2f}->{after:.2f}; ranker top={ranked[0].item_id}")


# --------------------------------------------------------------------------- #
# 10. Self-improvement (feedback -> adaptive threshold; RL replay)
# --------------------------------------------------------------------------- #
def check_self_improvement(repo: Path) -> Result:
    try:
        from app.intelligence.context_memory import ContextMemoryStore, OutcomeFeedbackStore
        from app.domain.inference_models import OutcomeType
    except Exception as exc:
        if _missing_dep(exc):
            return _defer("self_improvement", f"deps absent: {exc}")
        return _fail("self_improvement", f"import error: {exc}")
    from uuid import uuid4
    cm = ContextMemoryStore()
    fb = OutcomeFeedbackStore(batch_size=5)
    uid = uuid4()
    base = cm.get_noise_threshold(uid)
    for _ in range(5):  # all false positives -> tighten threshold
        fb.record_outcome(uid, uuid4(), OutcomeType.FALSE_POSITIVE, context_memory=cm)
    tightened = cm.get_noise_threshold(uid)
    if not (tightened > base):
        return _fail("self_improvement", f"high FP rate did not tighten threshold ({base}->{tightened})")
    rl_detail = "RL: deferred (torch absent)"
    try:
        from app.intelligence.reinforcement_learning import ReplayBuffer, Experience, State, Action, Reward
        buf = ReplayBuffer(capacity=10)
        for i in range(3):
            buf.push(Experience(state=State(user_features=[0.1] * 4),
                                action=Action(content_id=f"c{i}", action_type="show"),
                                reward=Reward(value=1.0), next_state=State(user_features=[0.2] * 4),
                                done=False))
        if len(buf) != 3:
            return _fail("self_improvement", f"replay buffer size {len(buf)} != 3")
        rl_detail = "RL replay buffer ok (3 experiences)"
    except Exception as exc:
        if not _missing_dep(exc):
            return _fail("self_improvement", f"RL error: {exc}")
    return _ok("self_improvement", f"FP rate tightened threshold {base:.2f}->{tightened:.2f}; {rl_detail}")


# --------------------------------------------------------------------------- #
# 11. Judge + promotion wiring (mock generator + mock judge)
# --------------------------------------------------------------------------- #
def check_promotion_wiring(repo: Path) -> Result:
    try:
        import asyncio
        from unittest.mock import AsyncMock
        from app.evals.scenario_loader import ScenarioLoader
        from app.evals.scenario_eval import ScenarioJudge
        from app.llm.training.inferencer import SituationEngineInferencer
        from app.llm.training.promotion import evaluate_and_promote
    except Exception as exc:
        if _missing_dep(exc):
            return _defer("promotion_wiring", f"deps absent: {exc}")
        return _fail("promotion_wiring", f"import error: {exc}")
    loader = ScenarioLoader(repo / "data/scenarios")
    heldout = sorted(loader.discover(split="heldout"), key=lambda c: c.scenario_id)[:5]
    if not heldout:
        return _fail("promotion_wiring", "no held-out scenarios")
    by_text = {tuple(o.text for o in c.observations): c.gold_report for c in heldout}

    def echo(messages):
        up = json.loads(messages[1]["content"])
        return by_text[tuple(o["text"] for o in up["observations"])].model_dump_json()

    inferencer = SituationEngineInferencer(echo)
    fake_router = AsyncMock()
    fake_router.complete = AsyncMock(return_value=json.dumps({
        "accuracy": 19, "groundedness": 19, "calibration": 18, "abstention": 20,
        "actionability": 19, "total": 95, "rationale": "smoke"}))
    judge = ScenarioJudge(router=fake_router, passing_floor=85)
    decision = asyncio.run(evaluate_and_promote(
        cases=heldout, inferencer=inferencer, judge=judge, floor=85))
    if not decision.promoted:
        return _fail("promotion_wiring", f"echo-gold not promoted (mean={decision.judge_report.mean_total})")
    return _ok("promotion_wiring",
               f"inference {decision.inference.n_success}/{decision.inference.n_total} ok; "
               f"judge mean={decision.judge_report.mean_total:.0f} -> promoted")


# --------------------------------------------------------------------------- #
# 12. Guards
# --------------------------------------------------------------------------- #
def check_guards(repo: Path) -> Result:
    import subprocess
    rc1 = subprocess.run([sys.executable, str(repo / "scripts/check_no_keyword_rules.py")],
                         capture_output=True, text=True)
    rc2 = subprocess.run([sys.executable, str(repo / "scripts/verify_heldout_manifest.py")],
                         capture_output=True, text=True, cwd=str(repo))
    if rc1.returncode != 0:
        return _fail("guards", "keyword-rule guard failed")
    if rc2.returncode != 0:
        return _fail("guards", f"held-out manifest integrity failed: {rc2.stderr.strip()[:120]}")
    return _ok("guards", "no rule-based answer path; held-out manifest intact")


CHECKS: List[Callable[[Path], Result]] = [
    check_corpus, check_training_data, check_frozen_prompt, check_finetune_config,
    check_preflight, check_mask_algorithm, check_calibration, check_memory,
    check_personalization, check_self_improvement, check_promotion_wiring, check_guards,
]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=_REPO)
    ap.add_argument("--strict-deferred", action="store_true",
                    help="Treat DEFERRED as failure (use on the GPU host where all deps exist).")
    args = ap.parse_args(argv)

    results: List[Result] = []
    for chk in CHECKS:
        try:
            results.append(chk(args.repo_root))
        except Exception as exc:  # a check itself crashed
            results.append(_fail(chk.__name__, f"{type(exc).__name__}: {exc}"))

    print("=" * 76)
    print("SITUATIONENGINE FINE-TUNE READINESS")
    print("=" * 76)
    for r in results:
        print(f"  [{r.status:8}] {r.name:20} {r.detail}")
    n_fail = sum(1 for r in results if r.status == FAIL)
    n_defer = sum(1 for r in results if r.status == DEFER)
    n_pass = sum(1 for r in results if r.status == PASS)
    print("-" * 76)
    print(f"  {n_pass} PASS · {n_defer} DEFERRED · {n_fail} FAIL")
    blocking = n_fail + (n_defer if args.strict_deferred else 0)
    if blocking == 0:
        if n_defer:
            print("  READY (component smoke tests with heavy deps are DEFERRED to the GPU host;")
            print("        re-run there with --strict-deferred for a full green board).")
        else:
            print("  READY FOR FINE-TUNE — all components green.")
        return 0
    print("  NOT READY — resolve FAIL items above before fine-tuning.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
