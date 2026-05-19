"""Promote a fine-tuned checkpoint if it clears the held-out judge floor.

Runs the LoRA-adapted base model over every scenario in
``data/scenarios/`` with ``split=heldout``, scores each output against
the gold report with the :class:`ScenarioJudge`, computes calibration
metrics, and writes a release manifest.

Exits 0 iff the mean judge score is at or above ``--floor`` and every
held-out scenario produced a schema-valid SituationReport. Anything else
exits non-zero so this script can be wired directly into CI as the
release gate.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

_REPO_ROOT_FOR_PATH = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT_FOR_PATH) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_PATH))

from app.evals.scenario_eval import ScenarioJudge
from app.llm.training.inferencer import (
    SituationEngineInferencer,
    build_lora_generator,
    load_scenarios_for_inference,
)
from app.llm.training.promotion import evaluate_and_promote


logger = logging.getLogger("promote_checkpoint")


async def _run(args: argparse.Namespace) -> int:
    if not args.lora_weights.exists():
        logger.error("lora weights path does not exist: %s", args.lora_weights)
        return 2

    cases = load_scenarios_for_inference(
        args.scenarios_root, split="heldout",
    )
    if not cases:
        logger.error(
            "no heldout scenarios under %s", args.scenarios_root,
        )
        return 2

    generate = build_lora_generator(
        base_model=args.base_model,
        lora_weights=args.lora_weights,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )
    inferencer = SituationEngineInferencer(generate)
    judge = ScenarioJudge(
        judge_model=args.judge_model,
        passing_floor=args.floor,
    )

    decision = await evaluate_and_promote(
        cases=cases,
        inferencer=inferencer,
        judge=judge,
        floor=args.floor,
        checkpoint_path=args.lora_weights,
        manifest_path=args.manifest,
    )

    logger.info(
        "judge mean_total=%.2f (floor=%d) inference success=%d/%d failures=%d",
        decision.judge_report.mean_total, args.floor,
        decision.inference.n_success, decision.inference.n_total,
        len(decision.inference.failures),
    )
    if decision.promoted:
        logger.info("PROMOTED: checkpoint cleared the floor")
        return 0
    logger.error("REJECTED: checkpoint did not clear the floor")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-weights", type=Path, required=True)
    parser.add_argument(
        "--scenarios-root", type=Path, default=Path("data/scenarios"),
    )
    parser.add_argument(
        "--judge-model", default="claude-3-5-sonnet-20241022",
    )
    parser.add_argument("--floor", type=int, default=85)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("checkpoints/promotion_manifest.json"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
