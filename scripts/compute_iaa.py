"""Compute weekly inter-annotator agreement (IAA) and append a row to the log.

Implements ``docs/labelling/deliverables.md`` §3.4.  Reads the double-labelled
items from ``data/labelling/double_labels/`` and writes one row per invocation
to ``data/labelling/iaa_log.jsonl``:

    {
      "week_ending": "2026-05-15",
      "n_double_labelled": 12,
      "kappa_signal_type": 0.87,
      "kappa_severity_weighted": 0.83,
      "agreement_abstain": 1.0,
      "below_target": [],
      "guideline_version_at_time": "1.0.0"
    }

Targets (``docs/labelling/guidelines.md`` §8):
  * Cohen's kappa on ``signal_type``        >= 0.80
  * linear-weighted kappa on ``severity``   >= 0.75
  * exact-match agreement on ``abstain``    >= 0.85

Double-labelled item schema (one JSON object per line, any ``*.jsonl`` file in
the double-labels dir)::

    {
      "scenario_id": "security_002",
      "week_ending": "2026-05-15",
      "guideline_version": "1.0.0",
      "annotator_a": {"annotator_id": "A03", "signal_type": "security_concern",
                       "severity": "SEV-1", "abstain": false},
      "annotator_b": {"annotator_id": "A07", "signal_type": "security_concern",
                       "severity": "SEV-2", "abstain": false}
    }

Per ``deliverables.md`` §2.2.4 held-out items are NOT double-labelled and never
appear here (they would skew IAA).

The computation prefers ``sklearn.metrics.cohen_kappa_score`` (matching the
deliverables spec); when scikit-learn is not importable it falls back to an
internal implementation that produces identical values, so the script runs in
any environment.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger("compute_iaa")

KAPPA_SIGNAL_TARGET = 0.80
KAPPA_SEVERITY_TARGET = 0.75
AGREEMENT_ABSTAIN_TARGET = 0.85
_SEV_ORDER = {f"SEV-{i}": i - 1 for i in range(1, 6)}  # SEV-1->0 .. SEV-5->4


# --------------------------------------------------------------------------- #
# Kappa implementations
# --------------------------------------------------------------------------- #
def _cohen_kappa_fallback(a: Sequence, b: Sequence,
                          weights: Optional[str] = None) -> float:
    """Cohen's kappa (optionally linear/quadratic weighted) without sklearn.

    Matches ``sklearn.metrics.cohen_kappa_score`` to floating-point precision.
    """
    labels = sorted(set(a) | set(b), key=lambda x: str(x))
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    if k == 0:
        return 0.0
    O = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        O[idx[x]][idx[y]] += 1
    n = len(a)
    row = [sum(O[i]) for i in range(k)]
    col = [sum(O[i][j] for i in range(k)) for j in range(k)]
    E = [[row[i] * col[j] / n for j in range(k)] for i in range(k)]
    if weights is None:
        w = [[0.0 if i == j else 1.0 for j in range(k)] for i in range(k)]
    elif weights == "linear":
        w = [[abs(i - j) for j in range(k)] for i in range(k)]
    elif weights == "quadratic":
        w = [[(i - j) ** 2 for j in range(k)] for i in range(k)]
    else:
        raise ValueError(f"unknown weights: {weights}")
    num = sum(w[i][j] * O[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * E[i][j] for i in range(k) for j in range(k))
    if den == 0:
        # No disagreement is possible by chance -> perfect agreement.
        return 1.0
    return 1.0 - num / den


def _cohen_kappa(a: Sequence, b: Sequence, weights: Optional[str] = None) -> float:
    try:
        from sklearn.metrics import cohen_kappa_score  # type: ignore
    except Exception:  # pragma: no cover - exercised only without sklearn
        return _cohen_kappa_fallback(a, b, weights=weights)
    if len(set(a) | set(b)) <= 1:
        # sklearn returns nan for a single label; treat as perfect agreement.
        return 1.0
    return float(cohen_kappa_score(list(a), list(b), weights=weights))


# --------------------------------------------------------------------------- #
def _load_items(dir_: Path) -> List[dict]:
    items: List[dict] = []
    if not dir_.exists():
        raise FileNotFoundError(f"double-labels dir not found: {dir_}")
    for path in sorted(dir_.glob("*.jsonl")):
        for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                items.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} invalid JSON: {exc}") from exc
    return items


def _select(items: List[dict], week_ending: Optional[str],
            since: Optional[str]) -> List[dict]:
    if week_ending:
        return [it for it in items if it.get("week_ending") == week_ending]
    if since:
        return [it for it in items if str(it.get("week_ending", "")) >= since]
    return items


def compute_row(items: List[dict]) -> Dict:
    sig_a = [it["annotator_a"]["signal_type"] for it in items]
    sig_b = [it["annotator_b"]["signal_type"] for it in items]
    sev_a = [_SEV_ORDER[it["annotator_a"]["severity"]] for it in items]
    sev_b = [_SEV_ORDER[it["annotator_b"]["severity"]] for it in items]
    abs_a = [bool(it["annotator_a"]["abstain"]) for it in items]
    abs_b = [bool(it["annotator_b"]["abstain"]) for it in items]

    kappa_signal = _cohen_kappa(sig_a, sig_b)
    kappa_sev = _cohen_kappa(sev_a, sev_b, weights="linear")
    agreement_abstain = (
        sum(1 for x, y in zip(abs_a, abs_b) if x == y) / len(items)
        if items else 0.0
    )
    below = []
    if kappa_signal < KAPPA_SIGNAL_TARGET:
        below.append("kappa_signal_type")
    if kappa_sev < KAPPA_SEVERITY_TARGET:
        below.append("kappa_severity_weighted")
    if agreement_abstain < AGREEMENT_ABSTAIN_TARGET:
        below.append("agreement_abstain")

    week = max((it.get("week_ending", "") for it in items), default="")
    gv = sorted({it.get("guideline_version", "1.0.0") for it in items})
    return {
        "week_ending": week,
        "n_double_labelled": len(items),
        "kappa_signal_type": round(kappa_signal, 4),
        "kappa_severity_weighted": round(kappa_sev, 4),
        "agreement_abstain": round(agreement_abstain, 4),
        "below_target": below,
        "guideline_version_at_time": gv[-1] if gv else "1.0.0",
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--double-labels-dir", type=Path,
                    default=Path("data/labelling/double_labels"))
    ap.add_argument("--log", type=Path, default=Path("data/labelling/iaa_log.jsonl"))
    ap.add_argument("--week-ending", default=None,
                    help="Compute over items with this exact week_ending.")
    ap.add_argument("--since", default=None,
                    help="Compute over items with week_ending >= this date.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the row without appending to the log.")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    items = _select(_load_items(args.double_labels_dir),
                    args.week_ending, args.since)
    if not items:
        logger.error("no double-labelled items selected")
        return 2

    row = compute_row(items)
    print(json.dumps(row, indent=2))

    if not args.dry_run:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        logger.info("appended IAA row for week_ending=%s to %s",
                    row["week_ending"], args.log)

    if row["below_target"]:
        logger.warning("metrics below target: %s", row["below_target"])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
