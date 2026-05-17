"""SQLite-backed chat session store for the desktop sidecar.

This is a deliberately thin wrapper around ``sqlite3``: chat traffic on
a single-user desktop is low-volume, the calls are short, and async
aiosqlite would buy us nothing while adding loop-binding fragility.
Schema migrations are handled by ``ChatStore.ensure_schema()`` which
runs idempotently on every construction.

The store is **not** shared with the legacy server-mode Postgres
schema.  It lives at ``<user_data_dir>/chat.sqlite3`` and is owned
entirely by the desktop sidecar.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from app.local.user_data_dir import get_user_data_dir


_DB_FILENAME = "chat.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    model TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
    content TEXT NOT NULL,
    tokens INTEGER,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session_created
    ON messages(session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_sessions_updated
    ON sessions(updated_at);
"""


@dataclass(frozen=True)
class ChatSession:
    id: str
    title: str
    model: Optional[str]
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class ChatMessage:
    id: str
    session_id: str
    role: str
    content: str
    tokens: Optional[int]
    created_at: float


class ChatStore:
    """Thread-safe SQLite chat store keyed by opaque session ids."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = db_path or (get_user_data_dir() / _DB_FILENAME)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # sqlite3 connections are single-thread by default; serialise our
        # own access with a re-entrant lock and ``check_same_thread=False``
        # so the FastAPI thread pool can drive the same connection.
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; we manage transactions explicitly
        )
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.row_factory = sqlite3.Row
        self.ensure_schema()

    def ensure_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(
        self, *, title: str = "New chat", model: Optional[str] = None
    ) -> ChatSession:
        now = time.time()
        sid = uuid.uuid4().hex
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions(id, title, model, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (sid, title, model, now, now),
            )
        return ChatSession(id=sid, title=title, model=model,
                           created_at=now, updated_at=now)

    def list_sessions(self, *, limit: int = 100) -> List[ChatSession]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, title, model, created_at, updated_at FROM sessions "
                "ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [ChatSession(**dict(r)) for r in rows]

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, title, model, created_at, updated_at FROM sessions "
                "WHERE id = ?",
                (session_id,),
            ).fetchone()
        return ChatSession(**dict(row)) if row else None

    def rename_session(self, session_id: str, title: str) -> bool:
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, session_id),
            )
            return cur.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM sessions WHERE id = ?", (session_id,)
            )
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def append_message(
        self,
        session_id: str,
        *,
        role: str,
        content: str,
        tokens: Optional[int] = None,
    ) -> ChatMessage:
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"invalid role: {role!r}")
        if not isinstance(content, str) or not content:
            raise ValueError("content must be a non-empty string")
        now = time.time()
        mid = uuid.uuid4().hex
        with self._lock:
            if self.get_session(session_id) is None:
                raise KeyError(f"unknown session: {session_id!r}")
            self._conn.execute(
                "INSERT INTO messages(id, session_id, role, content, tokens, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (mid, session_id, role, content, tokens, now),
            )
            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
        return ChatMessage(id=mid, session_id=session_id, role=role,
                           content=content, tokens=tokens, created_at=now)

    def list_messages(self, session_id: str) -> List[ChatMessage]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, session_id, role, content, tokens, created_at "
                "FROM messages WHERE session_id = ? ORDER BY created_at ASC, id ASC",
                (session_id,),
            ).fetchall()
        return [ChatMessage(**dict(r)) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_global_store: Optional[ChatStore] = None
_global_store_lock = threading.Lock()


def get_chat_store() -> ChatStore:
    """Return the process-wide :class:`ChatStore` singleton."""
    global _global_store
    with _global_store_lock:
        if _global_store is None:
            _global_store = ChatStore()
        return _global_store


def reset_chat_store() -> None:
    """Drop the cached singleton (test-only)."""
    global _global_store
    with _global_store_lock:
        if _global_store is not None:
            try:
                _global_store.close()
            except Exception:
                pass
        _global_store = None
