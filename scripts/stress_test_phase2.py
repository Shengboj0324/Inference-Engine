"""End-to-end stress test for the Phase 2 plumbing.

Generates a synthetic scenario corpus (see
``app.evals.synthetic_scenarios``) and exercises every Phase 2 code
path that does **not** require a GPU:

1. ``ScenarioLoader.discover()`` for train, val, and heldout splits.
2. ``scripts/build_training_set.py`` (the real CLI, via subprocess).
3. ``SituationEngineFineTuner.preflight()`` against the produced JSONL.
4. ``SituationEngineInferencer`` with a mock generator that echoes the
   gold report \u2014 to validate prompt construction, parsing, and the
   judge integration without an LLM call.
5. ``compute_calibration`` over the same predictions.
6. ``evaluate_and_promote`` with a mock judge that scores everything 95.

The exit code is non-zero on any failure. Run before every release.

Synthetic data lives in a temporary directory; it is never written into
the real ``data/scenarios/`` tree.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock

# Allow running this script directly without `python -m`.
_REPO_ROOT_FOR_PATH = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT_FOR_PATH) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_PATH))

from app.evals.scenario_eval import ScenarioJudge
from app.evals.scenario_loader import ScenarioLoader
from app.evals.synthetic_scenarios import generate_corpus
from app.llm.training.calibration import compute_calibration
from app.llm.training.inferencer import SituationEngineInferencer
from app.llm.training.promotion import evaluate_and_promote
from app.llm.training.situation_engine_finetune import (
    SituationEngineFineTuneConfig,
    SituationEngineFineTuner,
)


logger = logging.getLogger("stress_test_phase2")


REPO_ROOT = Path(__file__).resolve().parent.parent


def _stage_loader(corpus_root: Path) -> None:
    loader = ScenarioLoader(corpus_root)
    splits = {
        s: list(loader.discover(split=s))
        for s in ("train", "val", "heldout")
    }
    for split, cases in splits.items():
        if not cases:
            raise RuntimeError(f"loader: no {split} cases discovered")
        logger.info("loader: %s split = %d cases", split, len(cases))


def _stage_build(corpus_root: Path, out_dir: Path) -> None:
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "build_training_set.py"),
        "--scenarios-root", str(corpus_root),
        "--out-dir", str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        logger.error(
            "build_training_set failed (rc=%d)\nstdout=%s\nstderr=%s",
            proc.returncode, proc.stdout, proc.stderr,
        )
        raise RuntimeError("build_training_set failed")
    if not (out_dir / "train.jsonl").exists():
        raise RuntimeError("build_training_set did not write train.jsonl")
    logger.info("build: train.jsonl and val.jsonl written under %s", out_dir)


def _stage_preflight(out_dir: Path) -> None:
    cfg = SituationEngineFineTuneConfig(
        train_file=out_dir / "train.jsonl",
        val_file=out_dir / "val.jsonl",
        output_dir=out_dir / "ckpts",
    )
    report = SituationEngineFineTuner(cfg).preflight()
    logger.info("preflight: %s", report)
    if report.train_examples < 1:
        raise RuntimeError("preflight: zero training examples")


def _make_echo_gold(heldout):
    """Mock generator that returns the gold report for the matching case.

    Matches on the tuple of observation ``text`` fields, which is unique
    per synthetic scenario by construction.
    """
    by_text = {
        tuple(o.text for o in case.observations): case.gold_report
        for case in heldout
    }

    def _echo(messages):
        user_payload = json.loads(messages[1]["content"])
        key = tuple(o["text"] for o in user_payload["observations"])
        gold = by_text.get(key)
        if gold is None:
            raise RuntimeError("echo_gold: no matching case")
        return gold.model_dump_json()

    return _echo


def _stage_inference_and_calibration(corpus_root: Path) -> None:
    loader = ScenarioLoader(corpus_root)
    heldout = list(loader.discover(split="heldout"))
    inferencer = SituationEngineInferencer(_make_echo_gold(heldout))
    run = inferencer.run(heldout)
    if run.failures:
        raise RuntimeError(f"inference failures: {run.failures}")
    if run.n_success != len(heldout):
        raise RuntimeError(f"missing predictions: {run.n_success}/{len(heldout)}")
    cal = compute_calibration(
        [(run.predictions[c.scenario_id], c.gold_report) for c in heldout]
    )
    if cal.n != len(heldout):
        raise RuntimeError("calibration: wrong number of pairs")
    if cal.accuracy != 1.0:
        raise RuntimeError(
            f"calibration: echo-gold should yield 100% accuracy, got {cal.accuracy}"
        )
    logger.info(
        "inference+calibration: n=%d accuracy=%.2f ece=%.3f brier=%.3f",
        cal.n, cal.accuracy, cal.expected_calibration_error, cal.brier_score,
    )


async def _stage_promotion(corpus_root: Path, manifest_path: Path) -> None:
    loader = ScenarioLoader(corpus_root)
    heldout = list(loader.discover(split="heldout"))
    inferencer = SituationEngineInferencer(_make_echo_gold(heldout))

    # Mock judge: 95/100 across all axes.
    fake_router = AsyncMock()
    fake_router.complete = AsyncMock(return_value=json.dumps({
        "accuracy": 19, "groundedness": 19, "calibration": 19,
        "abstention": 19, "actionability": 19, "total": 95,
        "rationale": "synthetic",
    }))
    judge = ScenarioJudge(router=fake_router, passing_floor=85)

    decision = await evaluate_and_promote(
        cases=heldout,
        inferencer=inferencer,
        judge=judge,
        floor=85,
        checkpoint_path=corpus_root,  # any path, hashed for the manifest
        manifest_path=manifest_path,
    )
    if not decision.promoted:
        raise RuntimeError(
            f"promotion stage: not promoted (mean={decision.judge_report.mean_total})"
        )
    if not manifest_path.exists():
        raise RuntimeError("promotion manifest was not written")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not payload["promoted"]:
        raise RuntimeError("manifest does not record promoted=true")
    logger.info(
        "promotion: PASSED (mean=%.2f, manifest=%s)",
        decision.judge_report.mean_total, manifest_path,
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-tmp", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    tmp = Path(tempfile.mkdtemp(prefix="phase2_stress_"))
    logger.info("staging synthetic corpus under %s", tmp)
    try:
        corpus = generate_corpus(tmp / "scenarios", n_train=4, n_val=2, n_heldout=3)
        out_dir = tmp / "training"
        _stage_loader(corpus)
        _stage_build(corpus, out_dir)
        _stage_preflight(out_dir)
        _stage_inference_and_calibration(corpus)
        asyncio.run(_stage_promotion(corpus, tmp / "promotion_manifest.json"))
        logger.info("ALL STAGES PASSED")
        return 0
    except Exception as exc:
        logger.error("STRESS TEST FAILED: %s", exc, exc_info=True)
        return 1
    finally:
        if args.keep_tmp:
            logger.info("keeping tmp dir: %s", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
