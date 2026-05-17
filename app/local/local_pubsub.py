"""In-process pub/sub backend matching the ``redis.asyncio`` surface.

The desktop sidecar is a single-process FastAPI app: there is no second
process to broker messages between.  This module therefore implements a
hub-and-spoke ``asyncio.Queue`` fan-out that exposes the exact subset of
the ``redis.asyncio`` API that the codebase uses today:

* ``from_url(url, *, decode_responses=False, **_)`` → ``LocalRedis``
* ``LocalRedis.ping()`` / ``.publish(channel, message)`` /
  ``.get/set/setex/delete/exists`` (delegated to the TTL store) /
  ``.aclose()``
* ``LocalRedis.pubsub()`` → ``LocalPubSub`` with ``subscribe(*channels)``,
  ``unsubscribe(*channels)``, ``listen()`` (async iterator), ``aclose()``.

Message shape mirrors redis-py:

    {"type": "message", "channel": <str>, "data": <str|bytes>, "pattern": None}

The hub state is module-level so producers and consumers in the same
process see each other without any coordination.  Tests can reset it
via :func:`reset_hub`.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Union

from app.local.local_cache import LocalTTLStore

_BytesOrStr = Union[bytes, str]


class _Hub:
    """Module-level message broker.  All ``LocalRedis`` instances share it."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        # Lazy lock — instantiated on first use to honour whichever loop is
        # actually running.  Eager construction breaks Python-3.9 test
        # suites that swap loops between cases.
        self._lock_obj: Optional[asyncio.Lock] = None

    @property
    def _lock(self) -> asyncio.Lock:
        if self._lock_obj is None:
            self._lock_obj = asyncio.Lock()
        return self._lock_obj

    async def subscribe(self, channel: str, queue: asyncio.Queue) -> None:
        async with self._lock:
            self._subscribers[channel].add(queue)

    async def unsubscribe(self, channel: str, queue: asyncio.Queue) -> None:
        async with self._lock:
            self._subscribers.get(channel, set()).discard(queue)
            if channel in self._subscribers and not self._subscribers[channel]:
                del self._subscribers[channel]

    async def publish(self, channel: str, message: _BytesOrStr) -> int:
        async with self._lock:
            queues = list(self._subscribers.get(channel, ()))
        delivered = 0
        for q in queues:
            try:
                q.put_nowait({"type": "message", "channel": channel,
                              "data": message, "pattern": None})
                delivered += 1
            except asyncio.QueueFull:
                continue
        return delivered

    async def reset(self) -> None:
        async with self._lock:
            self._subscribers.clear()


_hub = _Hub()
_ttl_store = LocalTTLStore()


def reset_hub() -> None:
    """Test-only: drop every subscription and clear the shared TTL store."""
    _hub._subscribers.clear()
    _ttl_store.clear_sync()


class LocalPubSub:
    """``redis.asyncio.client.PubSub`` lookalike backed by ``asyncio.Queue``."""

    def __init__(self, *, decode_responses: bool, queue_max: int = 10_000) -> None:
        self._decode = decode_responses
        # Queue is bound to the running loop on first access.
        self._queue_max = queue_max
        self._queue_obj: Optional[asyncio.Queue] = None
        self._channels: Set[str] = set()
        self._closed = False

    @property
    def _queue(self) -> asyncio.Queue:
        if self._queue_obj is None:
            self._queue_obj = asyncio.Queue(maxsize=self._queue_max)
        return self._queue_obj

    async def subscribe(self, *channels: str) -> None:
        for ch in channels:
            self._channels.add(ch)
            await _hub.subscribe(ch, self._queue)

    async def unsubscribe(self, *channels: str) -> None:
        targets = list(channels) if channels else list(self._channels)
        for ch in targets:
            await _hub.unsubscribe(ch, self._queue)
            self._channels.discard(ch)

    async def listen(self) -> AsyncIterator[Dict[str, Any]]:
        """Yield messages forever until ``aclose`` is called."""
        while not self._closed:
            message = await self._queue.get()
            if message.get("type") == "__sentinel__":
                break
            if self._decode and isinstance(message.get("data"), bytes):
                message = {**message, "data": message["data"].decode()}
            yield message

    async def aclose(self) -> None:
        self._closed = True
        await self.unsubscribe()
        try:
            self._queue.put_nowait({"type": "__sentinel__"})
        except asyncio.QueueFull:
            pass


class LocalRedis:
    """Subset of ``redis.asyncio.Redis`` sufficient for the codebase today."""

    def __init__(self, *, decode_responses: bool = False, **_: Any) -> None:
        self._decode = decode_responses

    async def ping(self) -> bool:
        return True

    async def publish(self, channel: str, message: _BytesOrStr) -> int:
        return await _hub.publish(channel, message)

    def pubsub(self, **kwargs: Any) -> LocalPubSub:
        return LocalPubSub(decode_responses=self._decode)

    async def get(self, key: str) -> Optional[_BytesOrStr]:
        return await _ttl_store.get(key, decode=self._decode)

    async def set(self, key: str, value: _BytesOrStr) -> bool:
        await _ttl_store.set(key, value, ttl=None)
        return True

    async def setex(self, key: str, ttl: int, value: _BytesOrStr) -> bool:
        await _ttl_store.set(key, value, ttl=int(ttl))
        return True

    async def delete(self, *keys: str) -> int:
        return await _ttl_store.delete(*keys)

    async def exists(self, *keys: str) -> int:
        return await _ttl_store.exists(*keys)

    async def aclose(self) -> None:
        return None


def from_url(url: str, *, decode_responses: bool = False, **kwargs: Any) -> LocalRedis:
    """Mirror of ``redis.asyncio.from_url``.  ``url`` is accepted but ignored."""
    return LocalRedis(decode_responses=decode_responses, **kwargs)
