#!/usr/bin/env python3
"""CLI: fit per-SignalType temperature scalars from the labelled JSONL dataset.

Usage::

    python training/calibrate.py [--dataset PATH] [--state PATH] [--epochs N]

For each example in the dataset the script uses the stored ``confidence`` as
the predicted probability and treats ``signal_type`` as the ground-truth label.
It calls ``ConfidenceCalibrator.update()`` once per example (one gradient step).
After all examples have been processed the updated scalars are persisted to the
state file and a summary is printed to stdout.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow the script to be run from the repository root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.domain.inference_models import SignalType
from app.intelligence.calibration import ConfidenceCalibrator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_DATASET: Path = Path("training/signal_classification_dataset.jsonl")
_DEFAULT_STATE: Path = Path("training/calibration_state.json")
_DEFAULT_EPOCHS: int = 1


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace with ``dataset``, ``state``, and ``epochs`` attributes.
    """
    parser = argparse.ArgumentParser(
        description="Fit ConfidenceCalibrator temperature scalars from JSONL dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_DEFAULT_DATASET,
        help="Path to the labelled JSONL dataset (default: training/signal_classification_dataset.jsonl).",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=_DEFAULT_STATE,
        help="Path to calibration_state.json (default: training/calibration_state.json).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=_DEFAULT_EPOCHS,
        help="Number of passes over the dataset (default: 1).",
    )
    return parser.parse_args()


def _load_dataset(path: Path) -> list:
    """Load and return all records from a JSONL file.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        List of dicts, one per non-empty line.

    Raises:
        SystemExit: If the file is missing or malformed.
    """
    if not path.exists():
        logger.error("Dataset not found: %s", path)
        sys.exit(1)
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d: %s", lineno, exc)
    return records


def main() -> None:
    """Entry point: load dataset, run calibration updates, print summary."""
    args = _parse_args()
    calibrator = ConfidenceCalibrator(state_path=args.state)
    records = _load_dataset(args.dataset)
    logger.info("Loaded %d examples from %s", len(records), args.dataset)

    n_updated = 0
    n_skipped = 0

    for epoch in range(args.epochs):
        logger.info("Epoch %d/%d", epoch + 1, args.epochs)
        for record in records:
            try:
                signal_type = SignalType(record["signal_type"])
            except (KeyError, ValueError):
                n_skipped += 1
                continue

            # Use the dataset confidence as the predicted probability.
            # The dataset label IS the ground truth, so true_label=True for
            # correctly labelled examples; hard-negatives have is_hard_negative=True.
            predicted_prob: float = float(record.get("confidence", 0.7))
            is_hard_negative: bool = bool(record.get("is_hard_negative", False))
            true_label: bool = not is_hard_negative

            calibrator.update(signal_type, predicted_prob, true_label)
            n_updated += 1

    logger.info("Calibration complete: %d updates, %d skipped", n_updated, n_skipped)
    logger.info("Scalars written to: %s", args.state)

    # Print final scalars for inspection
    print("\nFinal temperature scalars:")
    for st_val, t in sorted(calibrator._scalars.items()):
        print(f"  {st_val:30s}  T={t:.4f}")


if __name__ == "__main__":
    main()

