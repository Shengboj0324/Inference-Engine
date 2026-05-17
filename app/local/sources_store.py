"""SQLite-backed source configuration store for the desktop sidecar.

Local-first parity for ``PlatformConfigDB`` from the legacy Postgres
schema, scoped to a single desktop user.  Credentials are stored as
opaque JSON blobs; the Phase 2 :mod:`app.local.key_vault` remains the
canonical store for LLM provider keys, while connector credentials
(Reddit OAuth, RSS URLs, etc.) live here because each platform's shape
is different and connector-specific.

The store is a thin sqlite3 wrapper following the same pattern as
:mod:`app.local.chat_store` and :mod:`app.local.permissions`.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.local.user_data_dir import get_user_data_dir


_DB_FILENAME = "sources.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    platform TEXT NOT NULL UNIQUE,
    enabled INTEGER NOT NULL DEFAULT 1,
    credentials_json TEXT NOT NULL DEFAULT '{}',
    settings_json TEXT NOT NULL DEFAULT '{}',
    last_fetch_at REAL,
    last_status TEXT,
    last_error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sources_enabled ON sources(enabled);
"""


@dataclass(frozen=True)
class SourceConfig:
    id: str
    platform: str
    enabled: bool
    credentials: Dict[str, Any]
    settings: Dict[str, Any]
    last_fetch_at: Optional[float]
    last_status: Optional[str]
    last_error: Optional[str]
    created_at: float
    updated_at: float

    @property
    def last_fetch_dt(self) -> Optional[datetime]:
        return datetime.utcfromtimestamp(self.last_fetch_at) if self.last_fetch_at else None


class SourcesStore:
    """Thread-safe local store of per-platform connector configurations."""

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
        self.ensure_schema()

    def ensure_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Mutating API
    # ------------------------------------------------------------------

    def upsert(
        self,
        platform: str,
        *,
        credentials: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ) -> SourceConfig:
        if not isinstance(platform, str) or not platform.strip():
            raise ValueError("platform must be a non-empty string")
        platform = platform.strip().lower()
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT id, created_at FROM sources WHERE platform = ?",
                (platform,),
            ).fetchone()
            sid = row["id"] if row else uuid.uuid4().hex
            created_at = row["created_at"] if row else now
            self._conn.execute(
                "INSERT INTO sources(id, platform, enabled, credentials_json, "
                "settings_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(platform) DO UPDATE SET "
                "  enabled = excluded.enabled, "
                "  credentials_json = excluded.credentials_json, "
                "  settings_json = excluded.settings_json, "
                "  updated_at = excluded.updated_at",
                (
                    sid, platform, 1 if enabled else 0,
                    json.dumps(credentials or {}, sort_keys=True),
                    json.dumps(settings or {}, sort_keys=True),
                    created_at, now,
                ),
            )
        return self.get(platform)  # type: ignore[return-value]

    def set_enabled(self, platform: str, enabled: bool) -> bool:
        platform = platform.strip().lower()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE sources SET enabled = ?, updated_at = ? WHERE platform = ?",
                (1 if enabled else 0, time.time(), platform),
            )
            return cur.rowcount > 0

    def record_run(
        self,
        platform: str,
        *,
        status: str,
        error: Optional[str] = None,
        when: Optional[float] = None,
    ) -> bool:
        platform = platform.strip().lower()
        ts = when if when is not None else time.time()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE sources SET last_fetch_at = ?, last_status = ?, "
                "last_error = ?, updated_at = ? WHERE platform = ?",
                (ts, status, error, ts, platform),
            )
            return cur.rowcount > 0

    def delete(self, platform: str) -> bool:
        platform = platform.strip().lower()
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM sources WHERE platform = ?", (platform,)
            )
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get(self, platform: str) -> Optional[SourceConfig]:
        platform = platform.strip().lower()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sources WHERE platform = ?", (platform,)
            ).fetchone()
        return self._row_to_config(row) if row else None

    def list_all(self, *, enabled_only: bool = False) -> List[SourceConfig]:
        with self._lock:
            if enabled_only:
                rows = self._conn.execute(
                    "SELECT * FROM sources WHERE enabled = 1 ORDER BY platform"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM sources ORDER BY platform"
                ).fetchall()
        return [self._row_to_config(r) for r in rows]

    @staticmethod
    def _row_to_config(row: sqlite3.Row) -> SourceConfig:
        return SourceConfig(
            id=row["id"],
            platform=row["platform"],
            enabled=bool(row["enabled"]),
            credentials=json.loads(row["credentials_json"] or "{}"),
            settings=json.loads(row["settings_json"] or "{}"),
            last_fetch_at=row["last_fetch_at"],
            last_status=row["last_status"],
            last_error=row["last_error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_global_store: Optional[SourcesStore] = None
_global_store_lock = threading.Lock()


def get_sources_store() -> SourcesStore:
    """Return the process-wide :class:`SourcesStore` singleton."""
    global _global_store
    with _global_store_lock:
        if _global_store is None:
            _global_store = SourcesStore()
        return _global_store


def reset_sources_store() -> None:
    """Drop the cached singleton (test-only)."""
    global _global_store
    with _global_store_lock:
        if _global_store is not None:
            try:
                _global_store.close()
            except Exception:
                pass
        _global_store = None
