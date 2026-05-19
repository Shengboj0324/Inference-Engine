"""Run a fine-tuned SituationEngine model over a scenarios split.

Loads the LoRA adapter at ``--lora-weights`` on top of ``--base-model``
and emits a predictions JSON suitable for ``run_scenario_eval.py``::

    python scripts/run_situation_inference.py \\
        --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --lora-weights checkpoints/situation_engine/final \\
        --split heldout \\
        --out predictions/heldout.json

Schema-failing completions are recorded in a sibling
``<out>.failures.json`` file. The script exits non-zero if any scenario
fails (so the promotion gate notices), unless ``--allow-failures`` is
passed for local debugging.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT_FOR_PATH = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT_FOR_PATH) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_PATH))

from app.llm.training.inferencer import (
    SituationEngineInferencer,
    build_lora_generator,
    load_scenarios_for_inference,
    write_predictions_file,
)


logger = logging.getLogger("run_situation_inference")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-weights", type=Path, required=True)
    parser.add_argument(
        "--scenarios-root", type=Path, default=Path("data/scenarios"),
    )
    parser.add_argument(
        "--split", choices=["train", "val", "heldout"], default="heldout",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--allow-failures", action="store_true",
        help="Local debugging only; do not use for promotion runs.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.lora_weights.exists():
        logger.error("lora weights path does not exist: %s", args.lora_weights)
        return 2

    cases = load_scenarios_for_inference(args.scenarios_root, args.split)
    if not cases:
        logger.error(
            "no scenarios with split=%s under %s", args.split, args.scenarios_root,
        )
        return 2

    generate = build_lora_generator(
        base_model=args.base_model,
        lora_weights=args.lora_weights,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    inferencer = SituationEngineInferencer(generate)
    run_report = inferencer.run(cases)

    write_predictions_file(run_report, args.out)
    failures_path = args.out.with_suffix(args.out.suffix + ".failures.json")
    failures_path.write_text(
        json.dumps(
            [
                {
                    "scenario_id": f.scenario_id,
                    "error_kind": f.error_kind,
                    "detail": f.detail,
                }
                for f in run_report.failures
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "wrote %d predictions and %d failures to %s",
        run_report.n_success, len(run_report.failures), args.out,
    )

    if run_report.failures and not args.allow_failures:
        logger.error(
            "%d scenario(s) failed schema validation; aborting (use "
            "--allow-failures to override for local debugging)",
            len(run_report.failures),
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
