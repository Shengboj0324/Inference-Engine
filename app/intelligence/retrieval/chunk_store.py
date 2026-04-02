"""Retrieval asset lifecycle — SQLite-backed chunk corpus.

``ChunkStore`` is the persistent repository for text chunks derived from
ingested content.  It now uses a ``sqlite3`` database as its backing store
so that chunks survive process restarts.

Key design decisions
--------------------
- **SQLite single-connection, lock-protected** — one ``sqlite3.Connection``
  with ``check_same_thread=False`` protected by a ``threading.Lock`` gives
  full thread safety at low cost.  All writes go through a single code path.
- **``:memory:`` default** — ``db_path=":memory:"`` (default) preserves
  backward compatibility: in-memory stores are created per-instance as before,
  all existing tests continue to work unchanged.
- **Embeddings as JSON** — embedding vectors are stored as JSON arrays so no
  extension is needed.  Cosine similarity is computed in Python.
- **FIFO eviction** — when ``max_size`` is set, the oldest row (lowest rowid)
  is evicted on each ``ingest()`` call that would push the count over the limit.
- **Age-based retention** — when ``max_age_hours`` is set, stale rows are
  evicted lazily on ``ingest()`` and explicitly via ``evict_stale()``.
- **Full public API preserved** — ``ingest``, ``ingest_batch``, ``chunk_text``,
  ``get``, ``get_by_observation``, ``search``, ``keyword_search``, ``count``,
  ``observation_ids``, ``clear``.

New public surface
------------------
- ``ChunkRecord.embedding_version`` — optional string identifying the embedding
  model version; enables stale-embedding detection after model upgrades.
- ``ChunkStore(db_path=...)`` — path to the SQLite file (``":memory:"``
  = in-process, ``"/path/to/chunks.db"`` = persistent).
- ``ChunkStore(max_age_hours=...)`` — evict chunks older than this many hours.
- ``ChunkStore.evict_stale()`` — explicit retention sweep; returns evicted count.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class ChunkRecord(BaseModel):
    """One indexable chunk derived from a ``ContentItem``.

    Attributes
    ----------
    chunk_id:          Auto-generated UUID for this chunk.
    observation_id:    UUID or string ID of the originating ``ContentItem``.
    source_family:     String value of ``SourceFamily``.
    text:              Raw chunk text (≤ ``ChunkStore.max_chunk_chars`` chars).
    chunk_index:       Zero-based position within the originating content item.
    embedding:         Optional dense embedding vector for cosine-similarity
                       search.  Length must match across all records.
    embedding_version: Optional string identifying the embedding model version
                       (e.g. ``"text-embedding-3-small"``).  Useful for
                       invalidating stale embeddings after a model upgrade.
    metadata:          Arbitrary key-value pairs (signal_type, published_at…).
    created_at:        UTC timestamp of ingestion (stored as Unix epoch float).
    """

    chunk_id:          str              = Field(default_factory=lambda: str(uuid4()))
    observation_id:    str
    source_family:     str              = "unknown"
    text:              str
    chunk_index:       int              = 0
    embedding:         Optional[List[float]] = None
    embedding_version: Optional[str]   = None
    metadata:          Dict[str, Any]  = Field(default_factory=dict)
    created_at:        datetime        = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id          TEXT    PRIMARY KEY,
    observation_id    TEXT    NOT NULL,
    source_family     TEXT    NOT NULL DEFAULT 'unknown',
    text              TEXT    NOT NULL,
    chunk_index       INTEGER NOT NULL DEFAULT 0,
    embedding         TEXT,
    embedding_version TEXT,
    metadata          TEXT    NOT NULL DEFAULT '{}',
    created_at        REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obs       ON chunks(observation_id);
CREATE INDEX IF NOT EXISTS idx_created   ON chunks(created_at);
"""


class ChunkStore:
    """SQLite-backed, thread-safe chunk corpus.

    Args:
        db_path:         SQLite database path.  ``":memory:"`` (default) creates
                         an in-process store compatible with all existing tests.
        max_chunk_chars: Hard cap on ``ChunkRecord.text`` length (chars).
                         Exceeding records are silently truncated on ingestion.
        max_size:        Maximum number of records before FIFO eviction kicks in
                         (``None`` = unlimited).
        max_age_hours:   Evict chunks older than this many hours on every
                         ``ingest()`` call and on explicit ``evict_stale()``
                         calls.  ``None`` = no age-based retention.
    """

    def __init__(
        self,
        db_path:         str            = ":memory:",
        max_chunk_chars: int            = 4_000,
        max_size:        Optional[int]  = None,
        max_age_hours:   Optional[float] = None,
    ) -> None:
        if max_chunk_chars <= 0:
            raise ValueError(f"max_chunk_chars must be > 0; got {max_chunk_chars!r}")
        if max_size is not None and max_size <= 0:
            raise ValueError(f"max_size must be > 0; got {max_size!r}")
        if max_age_hours is not None and max_age_hours <= 0:
            raise ValueError(f"max_age_hours must be > 0; got {max_age_hours!r}")

        self.max_chunk_chars = max_chunk_chars
        self.max_size        = max_size
        self.max_age_hours   = max_age_hours
        self._db_path        = db_path
        self._lock           = threading.Lock()
        self._conn           = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_CREATE_TABLE_SQL)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def ingest(self, record: ChunkRecord) -> str:
        """Add a ``ChunkRecord`` to the store.

        Truncates ``record.text`` to ``max_chunk_chars`` before storing.

        Returns:
            The ``chunk_id`` of the stored record.

        Raises:
            ValueError: If ``record.text`` is empty after truncation.
        """
        text = record.text[:self.max_chunk_chars].strip()
        if not text:
            raise ValueError(
                f"ChunkRecord for observation {record.observation_id!r} "
                f"has empty text after truncation to {self.max_chunk_chars} chars"
            )
        if len(record.text) > self.max_chunk_chars:
            record = record.model_copy(update={"text": text})

        emb_json  = json.dumps(record.embedding) if record.embedding is not None else None
        meta_json = json.dumps(record.metadata)
        created_ts = record.created_at.timestamp()

        with self._lock:
            self._conn.execute(
                """INSERT OR IGNORE INTO chunks
                   (chunk_id, observation_id, source_family, text, chunk_index,
                    embedding, embedding_version, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.chunk_id,
                    record.observation_id,
                    record.source_family,
                    text,
                    record.chunk_index,
                    emb_json,
                    record.embedding_version,
                    meta_json,
                    created_ts,
                ),
            )
            self._conn.commit()

            # ── FIFO eviction ──────────────────────────────────────────
            if self.max_size:
                n = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                while n > self.max_size:
                    self._conn.execute(
                        "DELETE FROM chunks WHERE rowid = "
                        "(SELECT MIN(rowid) FROM chunks)"
                    )
                    n -= 1
                self._conn.commit()

            # ── Age-based eviction ─────────────────────────────────────
            if self.max_age_hours is not None:
                self._evict_stale_unlocked()

        return record.chunk_id

    def ingest_batch(self, records: List[ChunkRecord]) -> List[str]:
        """Ingest multiple records; returns list of ``chunk_id``s (same order)."""
        return [self.ingest(r) for r in records]

    def chunk_text(
        self,
        observation_id: str,
        text:           str,
        source_family:  str                   = "unknown",
        chunk_size:     int                   = 800,
        overlap:        int                   = 100,
        metadata:       Optional[Dict[str, Any]] = None,
        embedding_version: Optional[str]      = None,
    ) -> List[str]:
        """Split *text* into overlapping chunks, ingest each, return their IDs.

        Args:
            observation_id:    Originating content item identifier.
            text:              Full text to chunk.
            source_family:     SourceFamily string for every chunk.
            chunk_size:        Target character length per chunk.
            overlap:           Character overlap between consecutive chunks.
            metadata:          Optional metadata applied to every chunk.
            embedding_version: Embedding model version tag for every chunk.

        Returns:
            List of ``chunk_id`` strings in chunk order.

        Raises:
            ValueError: If ``chunk_size`` ≤ ``overlap``.
        """
        if chunk_size <= overlap:
            raise ValueError(f"chunk_size ({chunk_size}) must be > overlap ({overlap})")
        effective = min(chunk_size, self.max_chunk_chars)
        ids: List[str] = []
        idx = 0
        chunk_index = 0
        while idx < len(text):
            chunk_slice = text[idx: idx + effective]
            record = ChunkRecord(
                observation_id=observation_id,
                source_family=source_family,
                text=chunk_slice,
                chunk_index=chunk_index,
                embedding_version=embedding_version,
                metadata=dict(metadata or {}),
            )
            ids.append(self.ingest(record))
            idx += effective - overlap
            chunk_index += 1
        return ids

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get(self, chunk_id: str) -> Optional[ChunkRecord]:
        """Retrieve a single record by ``chunk_id``.  Returns ``None`` if absent."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def get_by_observation(self, observation_id: str) -> List[ChunkRecord]:
        """Return all chunks for a given ``observation_id``, ordered by chunk_index."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE observation_id = ? ORDER BY chunk_index",
                (observation_id,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def search(
        self,
        query_embedding: List[float],
        top_k:           int = 10,
        metadata_filter: Optional[Callable[[ChunkRecord], bool]] = None,
    ) -> List["SearchHit"]:
        """Cosine-similarity search over embedded chunks.

        Chunks without an embedding are skipped.  Raises ``RuntimeError``
        when no embedded chunks exist.

        Args:
            query_embedding:  Query vector.
            top_k:            Number of results to return.
            metadata_filter:  Optional predicate; only matching chunks are scored.

        Returns:
            List of :class:`SearchHit` sorted by descending similarity.

        Raises:
            ValueError:  If ``query_embedding`` is the zero vector.
            RuntimeError: If no embedded chunks are available.
        """
        q_norm = _l2_norm(query_embedding)
        if q_norm == 0.0:
            raise ValueError("query_embedding must not be the zero vector")

        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE embedding IS NOT NULL"
            ).fetchall()

        if not rows:
            raise RuntimeError(
                "ChunkStore.search() requires records with pre-computed embeddings; "
                "none found.  Ingest records with ChunkRecord.embedding set."
            )

        hits: List[SearchHit] = []
        for row in rows:
            record = self._row_to_record(row)
            if metadata_filter and not metadata_filter(record):
                continue
            sim = _cosine_similarity(query_embedding, q_norm, record.embedding)  # type: ignore[arg-type]
            hits.append(SearchHit(record=record, score=sim))

        hits.sort(key=lambda h: -h.score)
        return hits[:top_k]

    def keyword_search(
        self,
        query:           str,
        top_k:           int = 10,
        metadata_filter: Optional[Callable[[ChunkRecord], bool]] = None,
    ) -> List["SearchHit"]:
        """TF-style keyword search (no embeddings required).

        Scores each chunk by term-frequency of query tokens in chunk text
        (case-insensitive).  Suitable as a fallback when embeddings are absent.
        """
        tokens = set(query.lower().split())
        if not tokens:
            return []

        with self._lock:
            rows = self._conn.execute("SELECT * FROM chunks").fetchall()

        hits: List[SearchHit] = []
        for row in rows:
            record = self._row_to_record(row)
            if metadata_filter and not metadata_filter(record):
                continue
            lower = record.text.lower()
            score = sum(lower.count(tok) for tok in tokens) / max(1, len(lower.split()))
            if score > 0:
                hits.append(SearchHit(record=record, score=score))

        hits.sort(key=lambda h: -h.score)
        return hits[:top_k]

    # ------------------------------------------------------------------
    # Stats / management
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Total number of chunks in the store."""
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def observation_ids(self) -> List[str]:
        """Sorted list of unique observation IDs with ingested chunks."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT observation_id FROM chunks ORDER BY observation_id"
            ).fetchall()
        return [r[0] for r in rows]

    def clear(self) -> None:
        """Remove all records (irreversible)."""
        with self._lock:
            self._conn.execute("DELETE FROM chunks")
            self._conn.commit()

    def evict_stale(self) -> int:
        """Explicitly remove all chunks older than ``max_age_hours``.

        Returns:
            Number of chunks evicted.  0 when ``max_age_hours`` is not set.

        Raises:
            RuntimeError: If ``max_age_hours`` is not configured.
        """
        if self.max_age_hours is None:
            raise RuntimeError(
                "evict_stale() requires max_age_hours to be configured; "
                "pass max_age_hours=<hours> to ChunkStore()"
            )
        with self._lock:
            return self._evict_stale_unlocked()

    @property
    def db_path(self) -> str:
        """Path to the underlying SQLite database."""
        return self._db_path

    # ------------------------------------------------------------------
    # Internal helpers (must be called under self._lock where noted)
    # ------------------------------------------------------------------

    def _evict_stale_unlocked(self) -> int:
        """Evict records older than ``max_age_hours``.  Caller must hold lock."""
        cutoff_ts = (
            datetime.now(timezone.utc) - timedelta(hours=self.max_age_hours)  # type: ignore[arg-type]
        ).timestamp()
        cur = self._conn.execute(
            "DELETE FROM chunks WHERE created_at < ?", (cutoff_ts,)
        )
        self._conn.commit()
        return cur.rowcount

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ChunkRecord:
        """Deserialise a SQLite row into a ``ChunkRecord``."""
        return ChunkRecord(
            chunk_id=row["chunk_id"],
            observation_id=row["observation_id"],
            source_family=row["source_family"],
            text=row["text"],
            chunk_index=row["chunk_index"],
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            embedding_version=row["embedding_version"],
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromtimestamp(row["created_at"], tz=timezone.utc),
        )


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------

class SearchHit(BaseModel):
    """A single ranked retrieval result."""

    model_config = {"arbitrary_types_allowed": True}

    record: ChunkRecord
    score:  float = Field(description="Cosine similarity (0–1) or keyword score")


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _l2_norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine_similarity(q: List[float], q_norm: float, d: List[float]) -> float:
    d_norm = _l2_norm(d)
    if d_norm == 0.0:
        return 0.0
    dot = sum(qi * di for qi, di in zip(q, d))
    return dot / (q_norm * d_norm)

