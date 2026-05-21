"""Build or verify the SHA-256 manifest of the held-out scenario folders.

Implements ``docs/labelling/deliverables.md`` §2.2.5 / §2.4.

After a held-out scenario is PII-signed, the Lead records its SHA-256 in
``data/labelling/heldout_manifest.json`` and tags the folder read-only.  Any
later modification changes the folder hash and must fail the nightly judge run.

Folder hash definition (stable, order-independent): for each regular file in
the folder, hash ``"<relpath>\\n" + bytes`` and fold the per-file SHA-256
digests together in sorted-relpath order; the folder digest is the SHA-256 of
the concatenated per-file digests.  This makes the hash independent of
filesystem iteration order and sensitive to any content or filename change.

Usage::

    # First time, after held-out PII sign-off:
    python scripts/verify_heldout_manifest.py --write

    # Nightly / pre-judge integrity check (exits non-zero on any mismatch):
    python scripts/verify_heldout_manifest.py

    # Optionally make the held-out folders read-only (chmod a-w):
    python scripts/verify_heldout_manifest.py --write --lock
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import stat
import sys
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger("verify_heldout_manifest")

_HELDOUT_PREFIX = "heldout_"


def _folder_hash(folder: Path) -> Dict:
    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )
    per_file = []
    rel_names = []
    for p in files:
        rel = p.relative_to(folder).as_posix()
        rel_names.append(rel)
        h = hashlib.sha256()
        h.update((rel + "\n").encode("utf-8"))
        h.update(p.read_bytes())
        per_file.append(h.hexdigest())
    folder_h = hashlib.sha256("".join(per_file).encode("ascii")).hexdigest()
    return {"sha256": folder_h, "files": rel_names}


def _heldout_folders(root: Path) -> List[Path]:
    return sorted(
        c for c in root.iterdir()
        if c.is_dir() and c.name.startswith(_HELDOUT_PREFIX)
    )


def build_manifest(root: Path) -> Dict:
    entries = {}
    for folder in _heldout_folders(root):
        entries[folder.name] = _folder_hash(folder)
    return {
        "version": "1.0.0",
        "scenarios_root": str(root),
        "n_scenarios": len(entries),
        "entries": entries,
    }


def lock_folders(root: Path) -> None:
    """chmod a-w on every file/dir in each held-out folder (best effort)."""
    ro = ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    for folder in _heldout_folders(root):
        for p in [folder, *folder.rglob("*")]:
            try:
                os.chmod(p, os.stat(p).st_mode & ro)
            except OSError as exc:  # pragma: no cover
                logger.warning("could not chmod %s: %s", p, exc)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios-root", type=Path, default=Path("data/scenarios"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/labelling/heldout_manifest.json"))
    ap.add_argument("--write", action="store_true",
                    help="(Re)build the manifest and write it to disk.")
    ap.add_argument("--lock", action="store_true",
                    help="Also chmod the held-out folders read-only.")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    current = build_manifest(args.scenarios_root)

    if args.write:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(current, indent=2) + "\n", encoding="utf-8")
        logger.info("wrote manifest for %d held-out folders to %s",
                    current["n_scenarios"], args.manifest)
        if args.lock:
            lock_folders(args.scenarios_root)
            logger.info("locked held-out folders read-only")
        return 0

    # Verify mode
    if not args.manifest.exists():
        logger.error("manifest not found: %s (run with --write first)", args.manifest)
        return 2
    saved = json.loads(args.manifest.read_text(encoding="utf-8"))
    saved_entries = saved.get("entries", {})
    cur_entries = current["entries"]

    mismatches = []
    for sid, info in saved_entries.items():
        if sid not in cur_entries:
            mismatches.append(f"{sid}: folder missing")
        elif cur_entries[sid]["sha256"] != info["sha256"]:
            mismatches.append(f"{sid}: hash changed")
    for sid in cur_entries:
        if sid not in saved_entries:
            mismatches.append(f"{sid}: new folder not in manifest")

    if mismatches:
        logger.error("held-out manifest verification FAILED (%d issue(s)):",
                     len(mismatches))
        for m in mismatches:
            logger.error("  - %s", m)
        return 1
    logger.info("held-out manifest OK: %d folders match", len(saved_entries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
