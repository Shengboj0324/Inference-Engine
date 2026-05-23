"""P1 — backfill ``quality_score`` into existing train/val JSONL in place.

The fine-tune set was emitted with ``quality_score: null`` for every record.
This rewrites ``data/training/{train,val}.jsonl`` populating the field from the
governance-derived score in :mod:`app.evals.data_quality`, matching each record
to its scenario via the ``source`` field (``internal_label/<ver>/<scenario_id>``).

Pure standard library: imports ``data_quality`` by file path so it runs without
the pydantic app stack.  Idempotent; prints the resulting score distribution.

Usage::

    python scripts/backfill_quality_scores.py
    python scripts/backfill_quality_scores.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _load_data_quality():
    spec = importlib.util.spec_from_file_location(
        "data_quality", _REPO / "app" / "evals" / "data_quality.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scenario_id_from_source(source: str) -> str:
    # "internal_label/1.0.0/bug_001" -> "bug_001"
    return source.rsplit("/", 1)[-1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios-root", default=str(_REPO / "data" / "scenarios"))
    ap.add_argument("--labelling-root", default=str(_REPO / "data" / "labelling"))
    ap.add_argument("--training-dir", default=str(_REPO / "data" / "training"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    dq = _load_data_quality()
    dist = {"min": 1.0, "max": 0.0, "n": 0, "sum": 0.0, "below_1": 0, "buckets": {}}

    for name in ("train.jsonl", "val.jsonl"):
        path = os.path.join(args.training_dir, name)
        if not os.path.exists(path):
            continue
        out_lines = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sid = _scenario_id_from_source(rec.get("source", ""))
                score = dq.score_from_disk(sid, args.scenarios_root, args.labelling_root)
                rec["quality_score"] = score
                out_lines.append(json.dumps(rec))
                dist["n"] += 1
                dist["sum"] += score
                dist["min"] = min(dist["min"], score)
                dist["max"] = max(dist["max"], score)
                dist["below_1"] += int(score < 1.0)
                b = round(score, 1)
                dist["buckets"][b] = dist["buckets"].get(b, 0) + 1
        if not args.dry_run:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(out_lines) + "\n")

    if dist["n"]:
        mean = dist["sum"] / dist["n"]
        print(f"quality_score backfilled: n={dist['n']} mean={mean:.4f} "
              f"min={dist['min']:.4f} max={dist['max']:.4f} below_1.0={dist['below_1']}")
        print("distribution by 0.1 bucket:",
              {k: dist["buckets"][k] for k in sorted(dist["buckets"])})
        print("DRY RUN (no files written)" if args.dry_run else "files updated.")
    else:
        print("no training records found.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
