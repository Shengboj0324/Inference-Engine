"""Convert labelled scenarios into TrainingExample JSONL for fine-tuning.

Reads every scenario under ``data/scenarios/`` whose metadata.split is
``train`` or ``val``, and emits two JSONL files at the requested output
directory:

    train.jsonl   — all scenarios with split == "train"
    val.jsonl     — all scenarios with split == "val"

Scenarios in the ``heldout`` split are deliberately excluded; they exist
only for the LLM-as-judge evaluation and must never leak into training.

Prompt construction is delegated to ``app.llm.training.situation_prompt``
so the trainee and the deployed runtime see byte-identical prompts. A
missing or empty frozen system prompt is a hard error: the build is
aborted rather than papered over with a placeholder.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

_REPO_ROOT_FOR_PATH = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT_FOR_PATH) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_PATH))

from app.evals.scenario_loader import ScenarioCase, ScenarioLoader
from app.llm.training.data_pipeline import TrainingExample
from app.llm.training.situation_prompt import (
    MissingSystemPromptError,
    build_training_messages,
    load_system_prompt,
)


logger = logging.getLogger("build_training_set")


def _to_example(case: ScenarioCase, system_prompt: str) -> TrainingExample:
    return TrainingExample(
        messages=build_training_messages(
            case.observations, case.gold_report, system_prompt=system_prompt,
        ),
        source=f"internal_label/{case.metadata.guideline_version}/"
               f"{case.scenario_id}",
        quality_score=None,
        contains_pii=False,
        anonymized=True,
    )


def _write_jsonl(path: Path, examples: Iterable[TrainingExample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(ex.model_dump_json())
            fh.write("\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios-root", type=Path,
        default=Path("data/scenarios"),
        help="Directory containing labelled scenario folders.",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/training"),
        help="Output directory for train.jsonl and val.jsonl.",
    )
    parser.add_argument(
        "--system-prompt-path", type=Path, default=None,
        help="Override path to the frozen system prompt; defaults to "
             "app/llm/prompts/situation_engine_system.txt.",
    )
    parser.add_argument(
        "--allow-unsigned-pii", action="store_true",
        help="Local-dev only: include scenarios without pii_review_passed.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        system_prompt = load_system_prompt(args.system_prompt_path)
    except MissingSystemPromptError as exc:
        logger.error("%s", exc)
        return 2

    loader = ScenarioLoader(args.scenarios_root)
    require_pii = not args.allow_unsigned_pii

    def _eligible(it):
        # Quarantined scenarios (IAA-failed batches per §3.4, or PII leaks per
        # §4.4) must never enter train.jsonl / val.jsonl, even when signed.
        for c in it:
            if getattr(c.metadata, "quarantined", False):
                logger.warning("skipping quarantined scenario %s", c.scenario_id)
                continue
            yield c

    # Sort cases by scenario_id for deterministic JSONL output across runs.
    train_cases = sorted(
        _eligible(loader.discover(split="train", require_pii_signoff=require_pii)),
        key=lambda c: c.scenario_id,
    )
    val_cases = sorted(
        _eligible(loader.discover(split="val", require_pii_signoff=require_pii)),
        key=lambda c: c.scenario_id,
    )

    if not train_cases and not val_cases:
        logger.error(
            "no scenarios with split in {train, val} found under %s",
            args.scenarios_root,
        )
        return 1

    n_train = _write_jsonl(
        args.out_dir / "train.jsonl",
        (_to_example(c, system_prompt) for c in train_cases),
    )
    n_val = _write_jsonl(
        args.out_dir / "val.jsonl",
        (_to_example(c, system_prompt) for c in val_cases),
    )
    logger.info(
        "wrote %d train examples and %d val examples to %s",
        n_train, n_val, args.out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
