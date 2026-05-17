"""In-process TTL cache used by the local pub/sub and ``CacheManager``.

Two layers live here:

* :class:`LocalTTLStore` — a low-level async-safe dict with monotonic TTLs.
  Used directly by :mod:`app.local.local_pubsub` to back the
  ``set`` / ``setex`` / ``get`` / ``delete`` / ``exists`` operations of the
  ``LocalRedis`` shim.

* :class:`LocalCacheManager` — a higher-level adapter exposing the same
  surface as :class:`app.core.cache.CacheManager` so call sites can be
  swapped at the factory level without code changes.

Expiry is opportunistic (lazy on read) plus a single sweeper task that
runs only when the store is non-empty, keeping resource use proportional
to actual load.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

_BytesOrStr = Union[bytes, str]


class LocalTTLStore:
    """Thread/async-safe dict with per-entry TTL in seconds."""

    def __init__(self) -> None:
        self._entries: Dict[str, Tuple[Any, Optional[float]]] = {}
        # Lazy lock — instantiated on first await to bind to the running loop.
        self._lock_obj: Optional[asyncio.Lock] = None

    @property
    def _lock(self) -> asyncio.Lock:
        if self._lock_obj is None:
            self._lock_obj = asyncio.Lock()
        return self._lock_obj

    async def get(self, key: str, *, decode: bool = False) -> Optional[_BytesOrStr]:
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if expires_at is not None and expires_at <= time.monotonic():
                self._entries.pop(key, None)
                return None
            if decode and isinstance(value, bytes):
                return value.decode()
            return value

    async def set(self, key: str, value: _BytesOrStr, ttl: Optional[int]) -> None:
        expires_at = time.monotonic() + ttl if ttl else None
        async with self._lock:
            self._entries[key] = (value, expires_at)

    async def delete(self, *keys: str) -> int:
        removed = 0
        async with self._lock:
            for k in keys:
                if self._entries.pop(k, None) is not None:
                    removed += 1
        return removed

    async def exists(self, *keys: str) -> int:
        present = 0
        async with self._lock:
            now = time.monotonic()
            for k in keys:
                entry = self._entries.get(k)
                if entry is None:
                    continue
                _, expires_at = entry
                if expires_at is not None and expires_at <= now:
                    self._entries.pop(k, None)
                    continue
                present += 1
        return present

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()

    def clear_sync(self) -> None:
        """Test-only synchronous reset; the loop may not be running."""
        self._entries.clear()

    async def size(self) -> int:
        async with self._lock:
            return len(self._entries)


class LocalCacheManager:
    """``CacheManager``-compatible adapter wrapping :class:`LocalTTLStore`."""

    def __init__(self, *, key_prefix: str = "smr", default_ttl: int = 3600) -> None:
        self._store = LocalTTLStore()
        self._prefix = key_prefix
        self._default_ttl = default_ttl

    def _make_key(self, namespace: str, key: str) -> str:
        return f"{self._prefix}:{namespace}:{key}"

    async def get(
        self,
        namespace: str,
        key: str,
        deserializer: Optional[Callable[[str], Any]] = None,
    ) -> Optional[Any]:
        raw = await self._store.get(self._make_key(namespace, key), decode=True)
        if raw is None:
            return None
        if deserializer is not None:
            return deserializer(raw)
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return raw

    async def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serializer: Optional[Callable[[Any], str]] = None,
    ) -> bool:
        if serializer is not None:
            serialized = serializer(value)
        elif isinstance(value, (dict, list)):
            serialized = json.dumps(value)
        else:
            serialized = str(value)
        await self._store.set(
            self._make_key(namespace, key),
            serialized,
            ttl or self._default_ttl,
        )
        return True

    async def delete(self, namespace: str, key: str) -> bool:
        removed = await self._store.delete(self._make_key(namespace, key))
        return removed > 0

    async def exists(self, namespace: str, key: str) -> bool:
        return await self._store.exists(self._make_key(namespace, key)) > 0

    async def clear(self) -> None:
        await self._store.clear()
