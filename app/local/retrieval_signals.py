"""Local-only retrieval personalization signals for the desktop sidecar.

Persists two per-``content_id`` counters used by
:class:`LocalRAGRetriever` to re-rank candidates returned from the
vector search:

* ``cited_count``   — incremented every time the snippet was actually
  surfaced as a citation in a chat reply.  Implicit positive signal.
* ``feedback_score`` — sum of explicit ``+1`` / ``-1`` user thumbs from
  the chat UI.  Clamped at write-time to prevent runaway weights.

The store lives entirely in the user-data SQLite file; signal payloads
never leave the machine and never enter the LLM prompt body.  The
retriever applies them as a *small* boost on top of cosine similarity
so personalization can nudge the ranking but never override semantics.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from app.local.user_data_dir import get_user_data_dir


_DB_FILENAME = "retrieval_signals.sqlite3"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS retrieval_signals (
    content_id TEXT PRIMARY KEY,
    cited_count INTEGER NOT NULL DEFAULT 0,
    feedback_score REAL NOT NULL DEFAULT 0.0,
    last_seen_ts REAL NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_signals_last_seen
    ON retrieval_signals(last_seen_ts DESC);
"""

# Clamp explicit feedback so a single mis-click cannot dominate ranking.
_FEEDBACK_MIN = -10.0
_FEEDBACK_MAX = 10.0


class RetrievalSignalsStore:
    """Thread-safe SQLite-backed personalization signals."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = db_path or (get_user_data_dir() / _DB_FILENAME)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self._path), check_same_thread=False, isolation_level=None,
        )
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def record_citation(self, content_id: str) -> None:
        """Increment the implicit-positive counter for ``content_id``."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO retrieval_signals(content_id, cited_count, last_seen_ts) "
                "VALUES (?, 1, ?) "
                "ON CONFLICT(content_id) DO UPDATE SET "
                "  cited_count = cited_count + 1, last_seen_ts = excluded.last_seen_ts",
                (str(content_id), now),
            )

    def record_feedback(self, content_id: str, score: float) -> float:
        """Add ``score`` to the feedback accumulator.  Returns the new total.

        ``score`` should be ``+1.0`` for thumbs-up and ``-1.0`` for
        thumbs-down; arbitrary floats are accepted to keep the schema
        forward-compatible with a numeric rating widget.
        """
        delta = max(-1.0, min(1.0, float(score)))
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO retrieval_signals(content_id, feedback_score, last_seen_ts) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(content_id) DO UPDATE SET "
                "  feedback_score = MAX(?, MIN(?, feedback_score + excluded.feedback_score)), "
                "  last_seen_ts = excluded.last_seen_ts",
                (str(content_id), delta, now, _FEEDBACK_MIN, _FEEDBACK_MAX),
            )
            row = self._conn.execute(
                "SELECT feedback_score FROM retrieval_signals WHERE content_id = ?",
                (str(content_id),),
            ).fetchone()
        return float(row["feedback_score"]) if row else 0.0

    def clear(self) -> int:
        """Drop every signal row.  Returns the number deleted."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM retrieval_signals")
            return cur.rowcount

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------

    def get_signals(self, content_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """Bulk-load signal rows keyed by content id.

        Returns a dict mapping ``content_id`` -> ``{cited, feedback,
        last_seen}``.  Missing ids are simply absent from the map (the
        re-rank treats them as zero, identical to a never-seen item).
        """
        if not content_ids:
            return {}
        placeholders = ",".join("?" * len(content_ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT content_id, cited_count, feedback_score, last_seen_ts "
                f"FROM retrieval_signals WHERE content_id IN ({placeholders})",
                tuple(str(c) for c in content_ids),
            ).fetchall()
        return {
            r["content_id"]: {
                "cited": int(r["cited_count"]),
                "feedback": float(r["feedback_score"]),
                "last_seen": float(r["last_seen_ts"]),
            }
            for r in rows
        }

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM retrieval_signals"
            ).fetchone()
        return int(row["n"])

    def list_top(self, *, limit: int = 50) -> List[Dict[str, float]]:
        """Return the most-recently-touched signal rows for the UI."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT content_id, cited_count, feedback_score, last_seen_ts "
                "FROM retrieval_signals "
                "ORDER BY last_seen_ts DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [
            {
                "content_id": r["content_id"],
                "cited": int(r["cited_count"]),
                "feedback": float(r["feedback_score"]),
                "last_seen": float(r["last_seen_ts"]),
            }
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_global_store: Optional[RetrievalSignalsStore] = None
_global_store_lock = threading.Lock()


def get_signals_store() -> RetrievalSignalsStore:
    """Return the process-wide :class:`RetrievalSignalsStore` singleton."""
    global _global_store
    with _global_store_lock:
        if _global_store is None:
            _global_store = RetrievalSignalsStore()
        return _global_store


def reset_signals_store() -> None:
    """Drop the cached singleton (test-only)."""
    global _global_store
    with _global_store_lock:
        if _global_store is not None:
            try:
                _global_store.close()
            except Exception:
                pass
        _global_store = None
