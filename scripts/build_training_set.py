"""Convert labelled scenarios into TrainingExample JSONL for fine-tuning.

Reads every scenario under ``data/scenarios/`` whose metadata.split is
``train`` or ``val``, and emits two JSONL files at the requested output
directory:

    train.jsonl   — all scenarios with split == "train"
    val.jsonl     — all scenarios with split == "val"

Scenarios in the ``heldout`` split are deliberately excluded; they exist
only for the LLM-as-judge evaluation and must never leak into training.

The frozen system prompt for the trainee model is loaded from
``app/llm/prompts/situation_engine_system.txt`` if present, otherwise a
placeholder is written so a labelling/ML reviewer notices.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

from app.evals.scenario_loader import ScenarioCase, ScenarioLoader
from app.llm.training.data_pipeline import TrainingExample


logger = logging.getLogger("build_training_set")


def _load_system_prompt(repo_root: Path) -> str:
    candidate = repo_root / "app" / "llm" / "prompts" / "situation_engine_system.txt"
    if candidate.exists():
        return candidate.read_text(encoding="utf-8").strip()
    return (
        "[PLACEHOLDER] The frozen situation-engine system prompt has not "
        "been authored yet. Replace this string by populating "
        "app/llm/prompts/situation_engine_system.txt before training."
    )


def _user_content(case: ScenarioCase) -> str:
    observations_payload = [
        {
            "observation_id": o.observation_id,
            "source": o.source,
            "timestamp": o.timestamp,
            "text": o.text,
        }
        for o in case.observations
    ]
    return json.dumps(
        {"observations": observations_payload},
        ensure_ascii=False,
        indent=2,
    )


def _assistant_content(case: ScenarioCase) -> str:
    return json.dumps(
        case.gold_report.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    )


def _to_example(case: ScenarioCase, system_prompt: str) -> TrainingExample:
    return TrainingExample(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _user_content(case)},
            {"role": "assistant", "content": _assistant_content(case)},
        ],
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
        "--repo-root", type=Path, default=Path("."),
        help="Repository root (used to locate the frozen system prompt).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    loader = ScenarioLoader(args.scenarios_root)
    system_prompt = _load_system_prompt(args.repo_root)

    train_cases = list(loader.discover(split="train"))
    val_cases = list(loader.discover(split="val"))

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
    if "[PLACEHOLDER]" in system_prompt:
        logger.warning(
            "frozen system prompt is a placeholder; populate "
            "app/llm/prompts/situation_engine_system.txt before training"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
