"""P2 — repair the train/val split so every signal class is represented in val.

The original split left ``val.jsonl`` with **no** praise / unclear /
notactionable scenarios, so the promotion gate was blind to those classes.
This re-partitions the existing 180 train/val scenarios (held-out is never
touched) so that:

  * every class present in train+val also appears in ``val``;
  * the total ``val`` count is preserved (default 36) by moving an equal number
    of scenarios out of the most over-represented val classes;
  * message content is preserved — records are only re-labelled, not rebuilt.

It updates each moved scenario's ``metadata.yaml`` ``split:`` line *and*
re-partitions ``data/training/{train,val}.jsonl`` so the on-disk training set
and the scenario metadata stay consistent.  Deterministic (id-sorted) and
idempotent.  Pure standard library.

Usage::

    python scripts/restratify_split.py --dry-run
    python scripts/restratify_split.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SPLIT_RE = re.compile(r"^(split:\s*)(\w+)\s*$", re.MULTILINE)


def _class_of(scenario_id: str) -> str:
    return scenario_id.rsplit("_", 1)[0]


def _read_split(meta_path: Path) -> str:
    m = _SPLIT_RE.search(meta_path.read_text(encoding="utf-8"))
    return m.group(2) if m else ""


def _set_split(meta_path: Path, new_split: str) -> None:
    text = meta_path.read_text(encoding="utf-8")
    text = _SPLIT_RE.sub(lambda mm: f"{mm.group(1)}{new_split}", text, count=1)
    meta_path.write_text(text, encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios-root", default=str(_REPO / "data" / "scenarios"))
    ap.add_argument("--training-dir", default=str(_REPO / "data" / "training"))
    ap.add_argument("--target-val", type=int, default=36)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    scen_root = Path(args.scenarios_root)
    # Map scenario_id -> split for train/val scenarios.
    splits = {}
    for d in sorted(scen_root.iterdir()):
        if not d.is_dir() or d.name == "_template":
            continue
        meta = d / "metadata.yaml"
        if not meta.exists():
            continue
        sp = _read_split(meta)
        if sp in ("train", "val"):
            splits[d.name] = sp

    by_class_val = defaultdict(list)
    by_class_train = defaultdict(list)
    for sid, sp in splits.items():
        (by_class_val if sp == "val" else by_class_train)[_class_of(sid)].append(sid)
    for c in by_class_val:
        by_class_val[c].sort()
    for c in by_class_train:
        by_class_train[c].sort()

    all_classes = set(by_class_val) | set(by_class_train)
    missing = sorted(c for c in all_classes if not by_class_val.get(c))

    to_val, to_train = [], []
    # 1) For each class missing from val, promote its lowest-id train scenario.
    for c in missing:
        if by_class_train.get(c):
            to_val.append(by_class_train[c][0])
    # 2) Demote the same count from the most over-represented val classes
    #    (never dropping a class to zero), highest-id first.
    n_demote = len(to_val)
    pool = []
    for c, ids in by_class_val.items():
        # keep at least 1 per class
        for sid in sorted(ids, reverse=True)[:max(0, len(ids) - 1)]:
            pool.append((len(ids), sid))
    pool.sort(key=lambda t: (-t[0], t[1]))  # biggest classes first, then id desc
    to_train = [sid for _, sid in pool[:n_demote]]

    print(f"val classes missing: {missing}")
    print(f"promote train->val: {to_val}")
    print(f"demote val->train: {to_train}")

    if args.dry_run:
        print("DRY RUN (no changes written)")
        return 0

    new_split = dict(splits)
    for sid in to_val:
        _set_split(scen_root / sid / "metadata.yaml", "val")
        new_split[sid] = "val"
    for sid in to_train:
        _set_split(scen_root / sid / "metadata.yaml", "train")
        new_split[sid] = "train"

    # Re-partition the JSONL records to match the new split.
    def _sid(rec):
        return rec.get("source", "").rsplit("/", 1)[-1]

    records = []
    for name in ("train.jsonl", "val.jsonl"):
        p = Path(args.training_dir) / name
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
    train_recs = [r for r in records if new_split.get(_sid(r)) == "train"]
    val_recs = [r for r in records if new_split.get(_sid(r)) == "val"]
    (Path(args.training_dir) / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train_recs) + "\n", encoding="utf-8")
    (Path(args.training_dir) / "val.jsonl").write_text(
        "\n".join(json.dumps(r) for r in val_recs) + "\n", encoding="utf-8")

    vc = defaultdict(int)
    for r in val_recs:
        vc[_class_of(_sid(r))] += 1
    print(f"new counts: train={len(train_recs)} val={len(val_recs)} (target {args.target_val})")
    print(f"new val class coverage ({len(vc)} classes): {dict(sorted(vc.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
