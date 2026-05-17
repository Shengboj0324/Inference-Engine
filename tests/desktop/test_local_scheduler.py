"""Tests for ``app.local.local_scheduler``."""

from __future__ import annotations

import asyncio
import time

import pytest

from app.local.local_scheduler import LocalScheduler


class TestRegistration:
    def test_every_seconds_validates(self) -> None:
        s = LocalScheduler()
        with pytest.raises(ValueError):
            s.every_seconds("bad", lambda: asyncio.sleep(0), seconds=0)

    def test_every_minutes_validates(self) -> None:
        s = LocalScheduler()
        with pytest.raises(ValueError):
            s.every_minutes("bad", lambda: asyncio.sleep(0), minutes=0)

    def test_daily_at_validates(self) -> None:
        s = LocalScheduler()
        with pytest.raises(ValueError):
            s.daily_at("bad", lambda: asyncio.sleep(0), hour=24)
        with pytest.raises(ValueError):
            s.daily_at("bad", lambda: asyncio.sleep(0), hour=0, minute=60)

    def test_job_count(self) -> None:
        s = LocalScheduler()
        s.every_seconds("a", lambda: asyncio.sleep(0), seconds=1)
        s.every_minutes("b", lambda: asyncio.sleep(0), minutes=1)
        s.daily_at("c", lambda: asyncio.sleep(0), hour=2)
        assert s.job_count == 3


class TestExecution:
    @pytest.mark.asyncio
    async def test_runs_at_expected_cadence(self) -> None:
        counter = {"n": 0}

        async def fn() -> None:
            counter["n"] += 1

        s = LocalScheduler(tick_seconds=0.05)
        s.every_seconds("tick", fn, seconds=0.2)
        s.start()
        await asyncio.sleep(0.75)
        await s.stop(timeout=2.0)
        # Expect ~3 invocations across 0.75s with a 0.2s cadence.
        assert 2 <= counter["n"] <= 5, counter

    @pytest.mark.asyncio
    async def test_exception_in_job_does_not_kill_scheduler(self) -> None:
        good = {"n": 0}

        async def boom() -> None:
            raise RuntimeError("planned failure")

        async def good_fn() -> None:
            good["n"] += 1

        s = LocalScheduler(tick_seconds=0.05)
        s.every_seconds("boom", boom, seconds=0.15)
        s.every_seconds("good", good_fn, seconds=0.15)
        s.start()
        await asyncio.sleep(0.6)
        await s.stop(timeout=2.0)
        assert good["n"] >= 2

    @pytest.mark.asyncio
    async def test_stop_is_graceful(self) -> None:
        held = asyncio.Event()
        released = asyncio.Event()

        async def long_job() -> None:
            held.set()
            await asyncio.sleep(0.2)
            released.set()

        s = LocalScheduler(tick_seconds=0.05)
        s.every_seconds("long", long_job, seconds=0.1)
        s.start()
        await asyncio.wait_for(held.wait(), timeout=2.0)
        await s.stop(timeout=2.0)
        assert released.is_set()

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self) -> None:
        s = LocalScheduler(tick_seconds=0.05)
        s.every_seconds("x", lambda: asyncio.sleep(0), seconds=10)
        s.start()
        task1 = s._task
        s.start()
        assert s._task is task1
        await s.stop(timeout=1.0)

    @pytest.mark.asyncio
    async def test_celery_beat_equivalent_registration(self) -> None:
        """Mirrors the two production Beat tasks defined in celery_app.py."""
        s = LocalScheduler()
        s.every_minutes("fetch_all_sources", lambda: asyncio.sleep(0), minutes=15)
        s.daily_at("cleanup_old_content", lambda: asyncio.sleep(0), hour=2)
        assert s.job_count == 2
