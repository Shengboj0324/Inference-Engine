"""Persistent BloomFilter-backed dedup store for the desktop ingestion runtime.

Wraps :class:`app.scraping.probabilistic_structures.BloomFilter` with a
small disk snapshot so the same bit array survives sidecar restarts —
without this, every restart would re-pay the SQL UNIQUE-violation
round-trip on each duplicate fetch and the per-platform "duplicate"
counters in :class:`IngestStats` would reset.

Semantics
---------
* ``seen(key)`` — probabilistic *might-have-seen* check.  False positives
  are rare (default 1%); false negatives are impossible.  Callers use this
  as a fast-path: when it returns ``False`` the item is definitely new
  and can bypass the SQL existence check.
* ``mark(key)`` — record a key as seen.  Increments the snapshot-dirty
  counter; the file is rewritten when ``flush()`` is called explicitly
  or when ``mark_count`` since the last flush exceeds ``autosave_every``.
* ``flush()`` — write the current bit array to disk atomically.  Safe to
  call from a signal handler / shutdown hook.

The on-disk format is intentionally minimal pickle of the bit array plus
the metadata needed to rebuild the filter at the original capacity, so
we never have to migrate a half-full file on a parameter change.
"""

from __future__ import annotations

import logging
import os
import pickle
import threading
from pathlib import Path
from typing import Iterable, Optional

from app.scraping.probabilistic_structures import BloomFilter
from app.local.user_data_dir import get_user_data_dir

logger = logging.getLogger(__name__)

_DEFAULT_FILE_NAME = "ingest_dedup.bloom"
_DEFAULT_CAPACITY = 200_000
_DEFAULT_FPR = 0.005
_DEFAULT_AUTOSAVE_EVERY = 250
_SNAPSHOT_VERSION = 1


def _make_key(platform: str, source_id: str) -> str:
    """Stable composite key matching the SQL UNIQUE constraint shape."""
    return f"{platform}::{source_id}"


class DedupStore:
    """Thread-safe persistent wrapper around :class:`BloomFilter`."""

    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        expected_elements: int = _DEFAULT_CAPACITY,
        false_positive_rate: float = _DEFAULT_FPR,
        autosave_every: int = _DEFAULT_AUTOSAVE_EVERY,
    ) -> None:
        self._path = path or (get_user_data_dir() / _DEFAULT_FILE_NAME)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._expected = int(expected_elements)
        self._fpr = float(false_positive_rate)
        self._autosave_every = max(1, int(autosave_every))
        self._lock = threading.RLock()
        self._dirty_since_save = 0
        self._filter = self._load() or BloomFilter(
            expected_elements=self._expected,
            false_positive_rate=self._fpr,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> Optional[BloomFilter]:
        if not self._path.exists():
            return None
        try:
            with self._path.open("rb") as fh:
                blob = pickle.load(fh)
        except (OSError, pickle.UnpicklingError, EOFError) as exc:
            logger.warning("dedup snapshot unreadable (%s); starting empty", exc)
            return None
        if not isinstance(blob, dict) or blob.get("v") != _SNAPSHOT_VERSION:
            logger.warning("dedup snapshot version mismatch; starting empty")
            return None
        # If the user has changed expected capacity / FPR since the snapshot
        # was written, prefer the new config — bit-array rebuild is cheap
        # next time items are marked and we never want stale tuning to
        # silently degrade the FPR.
        if blob.get("expected") != self._expected or blob.get("fpr") != self._fpr:
            logger.info(
                "dedup config changed (expected=%s→%s fpr=%s→%s); "
                "discarding old snapshot",
                blob.get("expected"), self._expected,
                blob.get("fpr"), self._fpr,
            )
            return None
        bf = BloomFilter(
            expected_elements=self._expected,
            false_positive_rate=self._fpr,
        )
        bf.bit_array = list(blob["bits"])
        bf.elements_added = int(blob.get("added", 0))
        return bf

    def flush(self) -> None:
        with self._lock:
            payload = {
                "v": _SNAPSHOT_VERSION,
                "expected": self._expected,
                "fpr": self._fpr,
                "added": self._filter.elements_added,
                "bits": list(self._filter.bit_array),
            }
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            with tmp.open("wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, self._path)
            self._dirty_since_save = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def seen(self, platform: str, source_id: str) -> bool:
        with self._lock:
            return self._filter.contains(_make_key(platform, source_id))

    def mark(self, platform: str, source_id: str) -> None:
        key = _make_key(platform, source_id)
        with self._lock:
            self._filter.add(key)
            self._dirty_since_save += 1
            if self._dirty_since_save >= self._autosave_every:
                self.flush()

    def mark_many(self, pairs: Iterable[tuple[str, str]]) -> int:
        added = 0
        with self._lock:
            for platform, source_id in pairs:
                self._filter.add(_make_key(platform, source_id))
                added += 1
            self._dirty_since_save += added
            if self._dirty_since_save >= self._autosave_every:
                self.flush()
        return added

    def stats(self) -> dict:
        with self._lock:
            return self._filter.get_statistics()
