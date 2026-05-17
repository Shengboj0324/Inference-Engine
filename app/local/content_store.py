"""SQLite-backed ContentItem store for the desktop sidecar.

Local-first replacement for ``ContentItemDB`` (Postgres + pgvector).  The
embedding column is preserved as a packed float32 BLOB — the same wire
format the optional ``sqlite_vec`` extension uses — so we keep the door
open to ANN search later without a data migration.

The store enforces the same uniqueness invariant as the production
schema: ``(user_id, source_platform, source_id)``.  ``upsert()`` returns
``True`` when the row is newly inserted, ``False`` when it collided
with an existing record (i.e. a duplicate fetch).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
from uuid import UUID

from app.core.models import ContentItem, MediaType, SourcePlatform
from app.local.sqlite_store import cosine_top_k, decode_vector, encode_vector, load_sqlite_vec
from app.local.user_data_dir import get_user_data_dir


_DB_FILENAME = "content.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS content_items (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    source_platform TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_url TEXT NOT NULL,
    author TEXT,
    channel TEXT,
    title TEXT NOT NULL,
    raw_text TEXT,
    media_type TEXT NOT NULL,
    media_urls_json TEXT NOT NULL DEFAULT '[]',
    published_at REAL NOT NULL,
    fetched_at REAL NOT NULL,
    topics_json TEXT NOT NULL DEFAULT '[]',
    lang TEXT,
    embedding BLOB,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(user_id, source_platform, source_id)
);

CREATE INDEX IF NOT EXISTS idx_content_published
    ON content_items(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_content_fetched
    ON content_items(fetched_at);
CREATE INDEX IF NOT EXISTS idx_content_platform
    ON content_items(source_platform, published_at DESC);
"""


def _to_epoch(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _from_epoch(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


class ContentStore:
    """Thread-safe SQLite store for normalised :class:`ContentItem` rows."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = db_path or (get_user_data_dir() / _DB_FILENAME)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self._path), check_same_thread=False, isolation_level=None,
        )
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.row_factory = sqlite3.Row
        # Best-effort load of sqlite-vec; pure-Python cosine is used either
        # way today, but loading keeps the door open for a future ANN swap
        # without a data migration (the embedding column is already the
        # packed-float32 BLOB layout sqlite-vec expects).
        self._vec_available = load_sqlite_vec(self._conn)
        self.ensure_schema()

    def ensure_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)

    def upsert(self, item: ContentItem) -> bool:
        """Insert ``item``; return True if new, False if duplicate."""
        emb_blob = encode_vector(item.embedding) if item.embedding else None
        params = (
            str(item.id), str(item.user_id), item.source_platform.value,
            item.source_id, item.source_url, item.author, item.channel,
            item.title, item.raw_text, item.media_type.value,
            json.dumps(list(item.media_urls)),
            _to_epoch(item.published_at), _to_epoch(item.fetched_at),
            json.dumps(list(item.topics)), item.lang, emb_blob,
            json.dumps(item.metadata, default=str),
        )
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO content_items(id, user_id, source_platform, "
                    "source_id, source_url, author, channel, title, raw_text, "
                    "media_type, media_urls_json, published_at, fetched_at, "
                    "topics_json, lang, embedding, metadata_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    params,
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def exists(self, user_id: UUID, platform: SourcePlatform, source_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM content_items WHERE user_id = ? AND "
                "source_platform = ? AND source_id = ? LIMIT 1",
                (str(user_id), platform.value, source_id),
            ).fetchone()
        return row is not None

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM content_items"
            ).fetchone()
        return int(row["n"])

    def count_with_embedding(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM content_items WHERE embedding IS NOT NULL"
            ).fetchone()
        return int(row["n"])

    def count_missing_embedding(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM content_items WHERE embedding IS NULL"
            ).fetchone()
        return int(row["n"])

    def list_missing_embeddings(
        self, *, limit: int = 500,
    ) -> List[Tuple[str, str, str]]:
        """Return ``(id, title, raw_text)`` triples for rows lacking an embedding.

        Lightweight projection — full row hydration would be wasteful
        when the caller only needs the input text for re-embedding.
        Ordered by ``published_at DESC`` so recent items embed first
        when a backfill run is bounded by ``limit``.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, title, raw_text FROM content_items "
                "WHERE embedding IS NULL "
                "ORDER BY published_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [(r["id"], r["title"] or "", r["raw_text"] or "") for r in rows]

    def set_embedding(self, content_id: str, vector: Sequence[float]) -> bool:
        """Persist ``vector`` on the row keyed by ``content_id``.

        Returns ``True`` when a row was updated, ``False`` when the id is
        unknown or ``vector`` is empty (no-op).  Used by the RAG backfill
        path so historical items become searchable without re-ingestion.
        """
        if not vector:
            return False
        blob = encode_vector(vector)
        with self._lock:
            cur = self._conn.execute(
                "UPDATE content_items SET embedding = ? WHERE id = ?",
                (blob, str(content_id)),
            )
            return cur.rowcount > 0

    def search_similar(
        self,
        query_embedding: Sequence[float],
        *,
        k: int = 8,
        since: Optional[datetime] = None,
        platforms: Optional[Iterable[SourcePlatform]] = None,
        min_score: float = 0.0,
    ) -> List[Tuple[ContentItem, float]]:
        """Return up to ``k`` items most similar to ``query_embedding``.

        Pure-Python cosine over rows whose ``embedding`` is non-null.
        Rows with mismatched embedding dimensions are silently skipped so
        a model swap mid-life does not poison the search path.
        """
        if not query_embedding:
            return []
        sql = "SELECT * FROM content_items WHERE embedding IS NOT NULL"
        params: list = []
        if since is not None:
            sql += " AND published_at >= ?"
            params.append(_to_epoch(since))
        platform_values = (
            [p.value for p in platforms] if platforms is not None else None
        )
        if platform_values:
            placeholders = ",".join("?" * len(platform_values))
            sql += f" AND source_platform IN ({placeholders})"
            params.extend(platform_values)
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        if not rows:
            return []
        query_len = len(query_embedding)
        candidates: List[Tuple[str, List[float]]] = []
        row_by_id: dict = {}
        for row in rows:
            try:
                vec = decode_vector(row["embedding"])
            except Exception:
                continue
            if len(vec) != query_len:
                continue
            candidates.append((row["id"], vec))
            row_by_id[row["id"]] = row
        if not candidates:
            return []
        ranked = cosine_top_k(list(query_embedding), candidates, k=max(1, int(k)))
        results: List[Tuple[ContentItem, float]] = []
        for cand_id, score in ranked:
            if score < min_score:
                continue
            row = row_by_id.get(cand_id)
            if row is None:
                continue
            results.append((self._row_to_item(row), float(score)))
        return results

    def delete_older_than(self, cutoff: datetime) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM content_items WHERE fetched_at < ?",
                (_to_epoch(cutoff),),
            )
            return cur.rowcount

    def list_recent(self, *, limit: int = 50) -> List[ContentItem]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM content_items "
                "ORDER BY published_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def list_by_platform(
        self, platform: SourcePlatform, *, limit: int = 50
    ) -> List[ContentItem]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM content_items WHERE source_platform = ? "
                "ORDER BY published_at DESC LIMIT ?",
                (platform.value, int(limit)),
            ).fetchall()
        return [self._row_to_item(r) for r in rows]

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> ContentItem:
        emb = decode_vector(row["embedding"]) if row["embedding"] else None
        return ContentItem(
            id=UUID(row["id"]),
            user_id=UUID(row["user_id"]),
            source_platform=SourcePlatform(row["source_platform"]),
            source_id=row["source_id"],
            source_url=row["source_url"],
            author=row["author"],
            channel=row["channel"],
            title=row["title"],
            raw_text=row["raw_text"],
            media_type=MediaType(row["media_type"]),
            media_urls=json.loads(row["media_urls_json"] or "[]"),
            published_at=_from_epoch(row["published_at"]),
            fetched_at=_from_epoch(row["fetched_at"]),
            topics=json.loads(row["topics_json"] or "[]"),
            lang=row["lang"],
            embedding=emb,
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_global_store: Optional[ContentStore] = None
_global_store_lock = threading.Lock()


def get_content_store() -> ContentStore:
    """Return the process-wide :class:`ContentStore` singleton."""
    global _global_store
    with _global_store_lock:
        if _global_store is None:
            _global_store = ContentStore()
        return _global_store


def reset_content_store() -> None:
    """Drop the cached singleton (test-only)."""
    global _global_store
    with _global_store_lock:
        if _global_store is not None:
            try:
                _global_store.close()
            except Exception:
                pass
        _global_store = None
