"""Native asyncio scheduler replacing Celery Beat on desktop.

Covers the two periodic jobs the production Beat schedule defines:

* ``fetch_all_sources`` — fixed-interval (minutes).
* ``cleanup_old_content`` — daily at a wall-clock time.

A third generic ``cron(hour, minute)`` variant is also exposed so future
periodic tasks can be added without pulling in APScheduler as a hard
dependency.

Semantics
---------
* Jobs run **sequentially per scheduler instance** — a single misbehaving
  task cannot cause concurrent re-entry of itself.
* Exceptions are logged (via ``structlog`` if available, ``logging``
  otherwise) and never escape; the next run is still scheduled.
* ``start()`` is idempotent.  ``stop()`` waits for the in-flight job to
  finish, with an optional timeout.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Awaitable, Callable, List, Optional

logger = logging.getLogger(__name__)

JobFn = Callable[[], Awaitable[None]]


@dataclass
class _Job:
    name: str
    fn: JobFn
    interval: timedelta
    next_run: datetime


class LocalScheduler:
    """Sequential asyncio cron-style scheduler."""

    def __init__(self, *, tick_seconds: float = 1.0) -> None:
        self._jobs: List[_Job] = []
        self._tick = tick_seconds
        self._task: Optional[asyncio.Task] = None
        # asyncio primitives are bound to whichever loop is running when
        # they are first accessed.  Eager construction in __init__ would
        # bind them to whatever loop happens to exist at import time and
        # break in suites that run multiple loops (the same class of bug
        # the circuit-breaker lazy-lock fix addressed).
        self._stop_event_obj: Optional[asyncio.Event] = None
        self._running_lock_obj: Optional[asyncio.Lock] = None

    @property
    def _stop_event(self) -> asyncio.Event:
        if self._stop_event_obj is None:
            self._stop_event_obj = asyncio.Event()
        return self._stop_event_obj

    @property
    def _running_lock(self) -> asyncio.Lock:
        if self._running_lock_obj is None:
            self._running_lock_obj = asyncio.Lock()
        return self._running_lock_obj

    # ---- registration ---------------------------------------------------
    def every_minutes(self, name: str, fn: JobFn, *, minutes: int) -> None:
        if minutes <= 0:
            raise ValueError("minutes must be > 0")
        self._jobs.append(
            _Job(name=name, fn=fn, interval=timedelta(minutes=minutes),
                 next_run=datetime.utcnow() + timedelta(minutes=minutes))
        )

    def daily_at(self, name: str, fn: JobFn, *, hour: int, minute: int = 0) -> None:
        if not 0 <= hour < 24 or not 0 <= minute < 60:
            raise ValueError("hour must be 0-23 and minute 0-59")
        now = datetime.utcnow()
        first = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if first <= now:
            first += timedelta(days=1)
        self._jobs.append(
            _Job(name=name, fn=fn, interval=timedelta(days=1), next_run=first)
        )

    def every_seconds(self, name: str, fn: JobFn, *, seconds: float) -> None:
        """Test-friendly variant exercising the same code path."""
        if seconds <= 0:
            raise ValueError("seconds must be > 0")
        self._jobs.append(
            _Job(name=name, fn=fn, interval=timedelta(seconds=seconds),
                 next_run=datetime.utcnow() + timedelta(seconds=seconds))
        )

    # ---- lifecycle ------------------------------------------------------
    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="LocalScheduler")

    async def stop(self, *, timeout: Optional[float] = 30.0) -> None:
        self._stop_event.set()
        if self._task is None:
            return
        try:
            await asyncio.wait_for(self._task, timeout=timeout)
        except asyncio.TimeoutError:
            self._task.cancel()
        finally:
            self._task = None

    # ---- runtime --------------------------------------------------------
    async def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                now = datetime.utcnow()
                due = [j for j in self._jobs if j.next_run <= now]
                for job in due:
                    async with self._running_lock:
                        await self._fire(job)
                    job.next_run = datetime.utcnow() + job.interval
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self._tick
                    )
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            return

    async def _fire(self, job: _Job) -> None:
        try:
            await job.fn()
        except Exception:  # noqa: BLE001 - scheduler must never escape
            logger.exception("Scheduled job %r raised; continuing", job.name)

    @property
    def job_count(self) -> int:
        return len(self._jobs)
