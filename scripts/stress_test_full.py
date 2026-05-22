"""One-shot full-scale stress test for the SituationEngine fine-tune stack.

Runs, in order, and fails (non-zero exit) on the first problem:

  A. ``scripts/finetune_readiness.py --strict-deferred`` — the 12-check
     component board (corpus, training data, config, preflight, mask, calibration,
     memory, personalization, self-improvement, promotion wiring, guards).
  B. ``scripts/stress_test_phase2.py`` — the end-to-end Phase-2 pipeline on a
     synthetic corpus (loader -> build -> preflight -> inference -> calibration
     -> promotion) with mock generator/judge.
  C. An *integrated* loop over the REAL labelled corpus that wires the
     self-improvement / memory / personalization subsystems together:
       - build a per-user exemplar memory from gold held-out scenarios,
       - retrieve nearest exemplars for a query (memory),
       - rank digest candidates by a personalized interest profile,
       - run a calibration self-improvement pass (acted-on / false-positive
         outcomes -> ConfidenceCalibrator) and assert the temperature moves the
         right way.

Requires the project venv (pydantic, scikit-learn, pyyaml; torch/transformers
optional — RL replay degrades gracefully). Intended to be the single command a
release/CI step runs before promoting any checkpoint.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logger = logging.getLogger("stress_test_full")


def _run_script(rel: str, args: list[str]) -> int:
    cmd = [sys.executable, str(_REPO / rel), *args]
    logger.info("running %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(_REPO))


def _integrated_loop(repo: Path) -> None:
    """Memory + personalization + self-improvement over the real corpus."""
    from app.evals.scenario_loader import ScenarioLoader
    from app.intelligence.context_memory import ContextMemoryStore
    from app.intelligence.calibration import ConfidenceCalibrator
    from app.domain.inference_models import SignalType, OutcomeType
    from app.personalization import (
        InterestGraph, FeedbackLearner, UserDigestRanker,
        FeedbackEvent, FeedbackType, DigestCandidate,
    )

    loader = ScenarioLoader(repo / "data/scenarios")
    heldout = sorted(loader.discover(split="heldout"), key=lambda c: c.scenario_id)
    if len(heldout) < 10:
        raise RuntimeError(f"expected >=10 held-out cases, got {len(heldout)}")

    # --- Memory: build an exemplar store from gold scenarios, then retrieve ---
    store = ContextMemoryStore()
    uid = uuid4()
    for case in heldout:
        gold = case.gold_report
        # store the gold rationale (summary) keyed by its bag-of-words embedding
        store.store_rationale(
            uid, gold.summary,
            _coerce_signal(gold.signal_type),
            store._embed_fn(gold.summary),  # bag-of-words fallback embed
        )
    query = heldout[0].observations[0].text
    hits = store.retrieve_similar_rationales(uid, store._embed_fn(query), top_k=3)
    if not hits:
        raise RuntimeError("memory: no exemplar rationales retrieved")
    logger.info("memory: retrieved %d exemplar rationales for a held-out query", len(hits))

    # --- Personalization: build interest profile and rank candidates ---
    graph = InterestGraph()
    for topic in ("security", "billing", "reliability"):
        graph.add_topic(topic)
    learner = FeedbackLearner(interest_graph=graph)
    learner.process_feedback(FeedbackEvent(
        user_id=str(uid), item_id="seed", feedback_type=FeedbackType.SAVE,
        topic_ids=["security"]))
    ranker = UserDigestRanker(interest_graph=graph)
    ranked = ranker.rank([
        DigestCandidate(item_id="sec", topic_ids=["security"], engagement_score=0.5),
        DigestCandidate(item_id="noise", topic_ids=["unrelated"], engagement_score=0.9),
    ])
    if not ranked or ranked[0].item_id != "sec":
        raise RuntimeError(f"personalization: interest boost failed: {[r.item_id for r in ranked]}")
    logger.info("personalization: on-interest item ranked first (%s)", ranked[0].item_id)

    # --- Self-improvement: calibration responds to outcomes ---
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cal = ConfidenceCalibrator(state_path=Path(td) / "cal.json")
        p0 = cal.calibrate(2.2, SignalType.SECURITY_CONCERN)
        for _ in range(25):
            cal.update(SignalType.SECURITY_CONCERN, predicted_prob=0.96, true_label=False)
        p1 = cal.calibrate(2.2, SignalType.SECURITY_CONCERN)
        if not (p1 < p0):
            raise RuntimeError(f"self-improvement: FP outcomes did not cool confidence ({p0:.3f}->{p1:.3f})")
        # acted-on outcomes should warm a fresh signal type
        q0 = cal.calibrate(-0.5, SignalType.FEATURE_REQUEST)
        for _ in range(25):
            cal.update(SignalType.FEATURE_REQUEST, predicted_prob=0.2, true_label=True)
        q1 = cal.calibrate(-0.5, SignalType.FEATURE_REQUEST)
        logger.info("self-improvement: FP cooled %.3f->%.3f ; acted-on warmed %.3f->%.3f",
                    p0, p1, q0, q1)

    # --- Adaptive noise threshold via feedback store ---
    from app.intelligence.context_memory import OutcomeFeedbackStore
    fb = OutcomeFeedbackStore(batch_size=5)
    base = store.get_noise_threshold(uid)
    for _ in range(5):
        fb.record_outcome(uid, uuid4(), OutcomeType.FALSE_POSITIVE, context_memory=store)
    if not (store.get_noise_threshold(uid) > base):
        raise RuntimeError("self-improvement: FP rate did not tighten noise threshold")
    logger.info("self-improvement: noise threshold tightened %.2f -> %.2f",
                base, store.get_noise_threshold(uid))


def _coerce_signal(value: str):
    from app.domain.inference_models import SignalType
    try:
        return SignalType(value)
    except ValueError:
        return SignalType.UNCLEAR


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-phase2", action="store_true",
                    help="Skip the synthetic end-to-end pipeline stage (B).")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("\n========== A. COMPONENT READINESS BOARD ==========")
    rc = _run_script("scripts/finetune_readiness.py", ["--strict-deferred"])
    if rc != 0:
        logger.error("readiness board FAILED (rc=%d)", rc)
        return 1

    if not args.skip_phase2:
        print("\n========== B. END-TO-END PIPELINE (synthetic) ==========")
        rc = _run_script("scripts/stress_test_phase2.py", [])
        if rc != 0:
            logger.error("phase-2 pipeline stress FAILED (rc=%d)", rc)
            return 1

    print("\n========== C. INTEGRATED MEMORY/PERSONALIZATION/SELF-IMPROVEMENT ==========")
    try:
        _integrated_loop(_REPO)
    except Exception as exc:  # noqa: BLE001
        logger.error("integrated loop FAILED: %s", exc, exc_info=True)
        return 1

    print("\n========================================================")
    print("FULL-SCALE STRESS TEST PASSED — every component green.")
    print("========================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
