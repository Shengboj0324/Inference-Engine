#!/usr/bin/env python3
"""Post-hoc calibration pipeline: fit calibrators, evaluate, write checkpoint.

This script implements post-hoc calibration of the existing model weights
(epoch 5, ECE=0.3724) without any re-training.  It:

1. Loads the labelled signal-classification dataset.
2. Splits into an 80 % stratified **training** split and a 20 % hold-out
   **validation** split (deterministic random state = 42).
3. Fits both calibrators on the training split:
   - :class:`TemperatureScaler`    — single-parameter NLL minimisation
   - :class:`IsotonicCalibrator`   — non-parametric monotone mapping
4. Evaluates each calibrator on the validation split:
   - ECE   (Expected Calibration Error) — threshold: ≤ 0.10
   - macro-F1 (binary, per-class, threshold 0.5) — threshold: ≥ 0.70
5. Selects the calibration method that passes **both** thresholds, preferring
   the one with the lower ECE when both pass.
6. Writes a new JSON checkpoint to ``training/checkpoints/`` with the schema
   expected by the CI ``eval-gate`` job::

       {
           "epoch": 5,
           "calibration_method": "isotonic",
           "ece": <float>,
           "macro_f1": <float>,
           "temperature": <float | null>,
           "val_ece_before": 0.3723561219156841,
           "val_ece_before_source": "epoch_005_ece_0.3724.json",
           "n_val_samples": <int>,
           "calibrated_at": "<ISO-8601 UTC>",
           "thresholds": {"ece": 0.10, "macro_f1": 0.70}
       }

Usage::

    python training/calibration/recalibrate_and_checkpoint.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Allow running from repository root without installing the package
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from training.calibration.temperature_scaling import TemperatureScaler
from training.calibration.isotonic_calibration import IsotonicCalibrator

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & thresholds
# ---------------------------------------------------------------------------
_DATASET_PATH      = _REPO_ROOT / "training" / "signal_classification_dataset.jsonl"
_CHECKPOINT_DIR    = _REPO_ROOT / "training" / "checkpoints"
_SOURCE_CHECKPOINT = "epoch_005_ece_0.3724.json"
_ORIGINAL_ECE      = 0.3723561219156841

ECE_THRESHOLD      = 0.10
MACRO_F1_THRESHOLD = 0.70
RANDOM_STATE       = 42
VAL_FRACTION       = 0.20
F1_THRESHOLD       = 0.50   # decision threshold applied to calibrated probabilities


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (probs, labels, signal_types) arrays from the JSONL dataset."""
    records = [json.loads(l) for l in path.open() if l.strip()]
    probs  = np.array([r["confidence"]       for r in records], dtype=np.float64)
    labels = np.array([0 if r["is_hard_negative"] else 1 for r in records], dtype=np.float64)
    signal_types = [r["signal_type"] for r in records]
    logger.info("Loaded %d examples from %s", len(records), path)
    return probs, labels, signal_types


# ---------------------------------------------------------------------------
# Stratified train / val split
# ---------------------------------------------------------------------------

def stratified_split(
    probs: np.ndarray,
    labels: np.ndarray,
    signal_types: List[str],
    val_fraction: float = VAL_FRACTION,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """80/20 stratified split preserving class proportions.

    Returns:
        train_probs, train_labels, val_probs, val_labels,
        train_signal_types, val_signal_types
    """
    rng = np.random.default_rng(random_state)
    classes = sorted(set(signal_types))
    train_idx: List[int] = []
    val_idx:   List[int] = []

    for cls in classes:
        cls_indices = [i for i, s in enumerate(signal_types) if s == cls]
        n_val = max(1, int(round(len(cls_indices) * val_fraction)))
        shuffled = list(rng.permutation(cls_indices))
        val_idx.extend(shuffled[:n_val])
        train_idx.extend(shuffled[n_val:])

    train_idx_arr = np.array(train_idx)
    val_idx_arr   = np.array(val_idx)
    train_st = [signal_types[i] for i in train_idx]
    val_st   = [signal_types[i] for i in val_idx]
    logger.info(
        "Split: train=%d  val=%d  (%.0f/%0.f)",
        len(train_idx), len(val_idx),
        100 * (1 - val_fraction), 100 * val_fraction,
    )
    return (
        probs[train_idx_arr], labels[train_idx_arr],
        probs[val_idx_arr],   labels[val_idx_arr],
        train_st, val_st,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error with equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


def compute_macro_f1(
    cal_probs: np.ndarray,
    labels: np.ndarray,
    signal_types: List[str],
    threshold: float = F1_THRESHOLD,
) -> Tuple[float, Dict[str, float]]:
    """Macro-averaged F1 across all signal-type classes.

    For each class *k*, treats calibrated confidence at threshold 0.5 as the
    binary decision boundary:
      - TP: is_hard_negative=False (label=1) AND calibrated_prob >= threshold
      - FP: is_hard_negative=True  (label=0) AND calibrated_prob >= threshold
      - FN: is_hard_negative=False (label=1) AND calibrated_prob <  threshold
    """
    classes = sorted(set(signal_types))
    st_arr  = np.array(signal_types)
    per_class: Dict[str, float] = {}
    for cls in classes:
        mask = st_arr == cls
        cls_probs  = cal_probs[mask]
        cls_labels = labels[mask]
        preds = (cls_probs >= threshold).astype(int)
        tp = int(((preds == 1) & (cls_labels == 1)).sum())
        fp = int(((preds == 1) & (cls_labels == 0)).sum())
        fn = int(((preds == 0) & (cls_labels == 1)).sum())
        denom = 2 * tp + fp + fn
        per_class[cls] = (2 * tp / denom) if denom > 0 else 0.0
    macro = float(np.mean(list(per_class.values())))
    return macro, per_class


# ---------------------------------------------------------------------------
# Checkpoint writer
# ---------------------------------------------------------------------------

def write_checkpoint(
    method: str,
    ece: float,
    macro_f1: float,
    temperature: Optional[float],
    n_val: int,
    epoch: int = 5,
    dry_run: bool = False,
) -> Path:
    """Write a calibrated checkpoint JSON and return its path."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    # Prefix with 'z_' so this file sorts AFTER all raw training checkpoints
    # ('epoch_NNN_ece_X.json') in any alphabetical listing.  The CI eval-gate
    # also selects by mtime (most recently written), so this is defense-in-depth.
    fname = f"z_calibrated_epoch_{epoch:03d}_{method}_ece_{ece:.4f}_{ts}.json"
    out_path = _CHECKPOINT_DIR / fname
    payload = {
        "epoch":                    epoch,
        "calibration_method":       method,
        "ece":                      round(ece, 6),
        "macro_f1":                 round(macro_f1, 6),
        "temperature":              round(temperature, 6) if temperature is not None else None,
        "val_ece_before":           _ORIGINAL_ECE,
        "val_ece_before_source":    _SOURCE_CHECKPOINT,
        "n_val_samples":            n_val,
        "calibrated_at":            datetime.now(timezone.utc).isoformat(),
        "thresholds":               {"ece": ECE_THRESHOLD, "macro_f1": MACRO_F1_THRESHOLD},
    }
    if dry_run:
        logger.info("[DRY-RUN] Would write checkpoint: %s\n%s",
                    out_path, json.dumps(payload, indent=2))
        return out_path
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(out_path)
    logger.info("Checkpoint written: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(dry_run: bool = False) -> int:
    """Run the full calibration pipeline.  Returns 0 on success, 1 on failure."""
    # 1. Load data
    probs, labels, signal_types = load_dataset(_DATASET_PATH)

    # 2. Stratified split
    (train_p, train_l, val_p, val_l,
     train_st, val_st) = stratified_split(probs, labels, signal_types)

    # 3. Fit & evaluate TemperatureScaler
    logger.info("── Temperature scaling ──────────────────────────────────────")
    ts = TemperatureScaler().fit(train_p, train_l)
    ts_cal = ts.calibrate(val_p)
    ts_ece = compute_ece(ts_cal, val_l)
    ts_f1, ts_per_class = compute_macro_f1(ts_cal, val_l, val_st)
    ts_pass = (ts_ece <= ECE_THRESHOLD) and (ts_f1 >= MACRO_F1_THRESHOLD)
    logger.info("T*=%.4f  ECE=%.4f (%s)  macro_F1=%.4f (%s)  PASS=%s",
                ts.temperature_, ts_ece,
                "≤0.10 ✓" if ts_ece <= ECE_THRESHOLD else ">0.10 ✗",
                ts_f1,
                "≥0.70 ✓" if ts_f1 >= MACRO_F1_THRESHOLD else "<0.70 ✗",
                ts_pass)

    # 4. Fit & evaluate IsotonicCalibrator
    logger.info("── Isotonic regression ─────────────────────────────────────")
    iso = IsotonicCalibrator().fit(train_p, train_l)
    iso_cal = iso.calibrate(val_p)
    iso_ece = compute_ece(iso_cal, val_l)
    iso_f1, iso_per_class = compute_macro_f1(iso_cal, val_l, val_st)
    iso_pass = (iso_ece <= ECE_THRESHOLD) and (iso_f1 >= MACRO_F1_THRESHOLD)
    logger.info("direction=%s  ECE=%.4f (%s)  macro_F1=%.4f (%s)  PASS=%s",
                "increasing" if iso.increasing_ else "decreasing",
                iso_ece,
                "≤0.10 ✓" if iso_ece <= ECE_THRESHOLD else ">0.10 ✗",
                iso_f1,
                "≥0.70 ✓" if iso_f1 >= MACRO_F1_THRESHOLD else "<0.70 ✗",
                iso_pass)

    # 5. Select winning method
    candidates = []
    if ts_pass:
        candidates.append(("temperature_scaling", ts_ece, ts_f1, ts.temperature_))
    if iso_pass:
        candidates.append(("isotonic", iso_ece, iso_f1, None))

    if not candidates:
        logger.error(
            "FAILED: neither calibrator achieved ECE≤%.2f AND macro_F1≥%.2f. "
            "Manual intervention required.",
            ECE_THRESHOLD, MACRO_F1_THRESHOLD,
        )
        return 1

    # Prefer lower ECE
    best_method, best_ece, best_f1, best_T = min(candidates, key=lambda x: x[1])
    logger.info(
        "Winner: %s  ECE=%.6f  macro_F1=%.6f",
        best_method, best_ece, best_f1,
    )

    # 6. Write checkpoint
    out_path = write_checkpoint(
        method=best_method,
        ece=best_ece,
        macro_f1=best_f1,
        temperature=best_T,
        n_val=len(val_p),
        dry_run=dry_run,
    )

    # Summary
    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"  Original ECE  (epoch 5, raw):  {_ORIGINAL_ECE:.6f}")
    print(f"  Method chosen:                 {best_method}")
    print(f"  Calibrated ECE  (val):         {best_ece:.6f}  (threshold ≤ {ECE_THRESHOLD})")
    print(f"  macro-F1        (val, t=0.5):  {best_f1:.6f}  (threshold ≥ {MACRO_F1_THRESHOLD})")
    print(f"  Checkpoint:                    {out_path}")
    print("=" * 70 + "\n")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-hoc calibration pipeline")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be written without creating the checkpoint file")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(run(dry_run=args.dry_run))

