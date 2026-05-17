"""SQLite + (optional) sqlite-vec engine factory for the desktop sidecar.

Replaces ``app/core/db.py``'s Postgres engine in desktop mode.  Public surface:

* :func:`create_local_engine` — return a SQLAlchemy async engine bound to
  the user-data-dir SQLite file (``:memory:`` accepted for tests).
* :func:`make_session_factory` — async sessionmaker built on top.
* :func:`cosine_top_k`        — pure-Python fallback for ANN search when
  the ``sqlite_vec`` extension is unavailable.
* :func:`load_sqlite_vec`     — best-effort extension loader; returns
  ``True`` if the extension is now available on ``conn``.

The vector helpers use a portable BLOB-of-float32 representation so an
operator can later swap in the native extension without a data migration.
"""

from __future__ import annotations

import array
import logging
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.local.user_data_dir import get_user_data_dir

logger = logging.getLogger(__name__)

_DEFAULT_DB_FILENAME = "social_radar.sqlite3"


def _resolve_sqlite_url(path: Optional[str]) -> str:
    """Return a SQLAlchemy-ready ``sqlite+aiosqlite://`` URL."""
    if path == ":memory:" or path == "":
        return "sqlite+aiosqlite:///:memory:"
    if path is None:
        target = get_user_data_dir(ensure=True) / _DEFAULT_DB_FILENAME
    else:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{target}"


def create_local_engine(
    path: Optional[str] = None,
    *,
    echo: bool = False,
) -> AsyncEngine:
    """Create an async SQLite engine sized for a desktop single-user load."""
    url = _resolve_sqlite_url(path)
    # SQLite has no real pool; ``pool_pre_ping`` is still useful for
    # detecting stale connections after suspend/resume on laptops.
    return create_async_engine(
        url,
        echo=echo,
        pool_pre_ping=True,
        connect_args={"timeout": 30},
    )


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Return a sessionmaker mirroring ``app.core.db.AsyncSessionLocal``."""
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------


def encode_vector(values: Sequence[float]) -> bytes:
    """Encode a vector as a packed little-endian float32 blob."""
    return array.array("f", [float(v) for v in values]).tobytes()


def decode_vector(blob: bytes) -> List[float]:
    """Inverse of :func:`encode_vector`."""
    return list(array.array("f").frombytes(blob) or array.array("f", blob))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def cosine_top_k(
    query: Sequence[float],
    candidates: Iterable[Tuple[str, Sequence[float]]],
    k: int = 10,
) -> List[Tuple[str, float]]:
    """Brute-force fallback for ANN search.

    Args:
        query: Query embedding.
        candidates: Iterable of ``(id, embedding)`` rows.
        k: Number of top hits to return.

    Returns:
        List of ``(id, score)`` sorted by descending cosine similarity.
    """
    scored: List[Tuple[str, float]] = []
    for cand_id, emb in candidates:
        scored.append((cand_id, _cosine(query, emb)))
    scored.sort(key=lambda r: r[1], reverse=True)
    return scored[:k]


def load_sqlite_vec(conn) -> bool:
    """Best-effort loader for the optional ``sqlite_vec`` extension."""
    try:
        import sqlite_vec  # type: ignore[import-not-found]
    except ImportError:
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except Exception:  # noqa: BLE001 - extension loading is OS-specific
        logger.warning("sqlite_vec is installed but failed to load; "
                       "falling back to Python cosine search.")
        return False
