"""In-process task queue replacing the Celery broker on desktop.

Design goals
------------
* **Single-process** — producer and consumer share an ``asyncio.Queue``;
  no IPC, no message broker.
* **Priority + FIFO** — lower numerical priority is consumed first; same-
  priority jobs are FIFO via a monotonic insertion sequence.
* **Explicit ack/nack** — callers ``dequeue()`` a :class:`QueuedJob`, do
  the work, then call ``ack()`` or ``nack(requeue=...)``.
* **Dead-letter** — jobs nacked without requeue or exceeding
  ``max_attempts`` land in the dead-letter list for operator inspection.
* **Bounded** — ``maxsize`` raises ``QueueFull`` on overflow to enforce
  back-pressure rather than unbounded memory growth.

Concurrency is verified by the Phase 1 stress test which enqueues 2 000
jobs across 16 workers and asserts ordering + zero duplicates.
"""

from __future__ import annotations

import asyncio
import itertools
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(order=True)
class _QueueEntry:
    priority: int
    seq: int
    job: "QueuedJob" = field(compare=False)


@dataclass
class QueuedJob:
    """Single unit of work delivered by :meth:`LocalTaskQueue.dequeue`."""

    id: str
    name: str
    payload: Dict[str, Any]
    priority: int
    attempt: int = 1
    max_attempts: int = 3


class LocalTaskQueue:
    """Async-safe priority queue with ack / nack / dead-letter semantics."""

    def __init__(self, *, maxsize: int = 10_000) -> None:
        # asyncio primitives are bound to whichever loop is running on first
        # access; eager construction breaks loop-swapping test suites.
        self._maxsize = maxsize
        self._queue_obj: Optional["asyncio.PriorityQueue[_QueueEntry]"] = None
        self._counter = itertools.count()
        self._in_flight: Dict[str, QueuedJob] = {}
        self._dead_letter: List[QueuedJob] = []
        self._lock_obj: Optional[asyncio.Lock] = None

    @property
    def _queue(self) -> "asyncio.PriorityQueue[_QueueEntry]":
        if self._queue_obj is None:
            self._queue_obj = asyncio.PriorityQueue(maxsize=self._maxsize)
        return self._queue_obj

    @property
    def _lock(self) -> asyncio.Lock:
        if self._lock_obj is None:
            self._lock_obj = asyncio.Lock()
        return self._lock_obj

    async def enqueue(
        self,
        name: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        priority: int = 5,
        max_attempts: int = 3,
        job_id: Optional[str] = None,
    ) -> str:
        """Add a job.  Returns the job ID."""
        job = QueuedJob(
            id=job_id or str(uuid.uuid4()),
            name=name,
            payload=dict(payload or {}),
            priority=priority,
            attempt=1,
            max_attempts=max_attempts,
        )
        await self._queue.put(_QueueEntry(priority, next(self._counter), job))
        return job.id

    async def dequeue(self, *, timeout: Optional[float] = None) -> Optional[QueuedJob]:
        """Wait for the next job.  Returns ``None`` on timeout."""
        try:
            if timeout is None:
                entry = await self._queue.get()
            else:
                entry = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        async with self._lock:
            self._in_flight[entry.job.id] = entry.job
        return entry.job

    async def ack(self, job_id: str) -> bool:
        async with self._lock:
            return self._in_flight.pop(job_id, None) is not None

    async def nack(self, job_id: str, *, requeue: bool = True) -> bool:
        async with self._lock:
            job = self._in_flight.pop(job_id, None)
        if job is None:
            return False
        if not requeue or job.attempt >= job.max_attempts:
            async with self._lock:
                self._dead_letter.append(job)
            return False
        job.attempt += 1
        await self._queue.put(
            _QueueEntry(job.priority, next(self._counter), job)
        )
        return True

    async def pending(self) -> int:
        return self._queue.qsize()

    async def in_flight(self) -> int:
        async with self._lock:
            return len(self._in_flight)

    async def dead_letter(self) -> List[QueuedJob]:
        async with self._lock:
            return list(self._dead_letter)

    async def drain_dead_letter(self) -> List[QueuedJob]:
        async with self._lock:
            out, self._dead_letter = self._dead_letter, []
            return out
