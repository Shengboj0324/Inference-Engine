"""Tests for ``app.local.local_queue``."""

from __future__ import annotations

import asyncio

import pytest

from app.local.local_queue import LocalTaskQueue


class TestEnqueueDequeue:
    @pytest.mark.asyncio
    async def test_simple_round_trip(self) -> None:
        q = LocalTaskQueue()
        job_id = await q.enqueue("fetch", {"url": "https://x"})
        job = await q.dequeue(timeout=1.0)
        assert job is not None
        assert job.id == job_id
        assert job.name == "fetch"
        assert job.payload == {"url": "https://x"}
        await q.ack(job.id)
        assert await q.in_flight() == 0

    @pytest.mark.asyncio
    async def test_priority_ordering(self) -> None:
        q = LocalTaskQueue()
        await q.enqueue("low", priority=10, job_id="L")
        await q.enqueue("high", priority=1, job_id="H")
        await q.enqueue("mid", priority=5, job_id="M")
        ids = [(await q.dequeue(timeout=1.0)).id for _ in range(3)]
        assert ids == ["H", "M", "L"]

    @pytest.mark.asyncio
    async def test_fifo_within_same_priority(self) -> None:
        q = LocalTaskQueue()
        for i in range(5):
            await q.enqueue("t", {"n": i}, priority=5, job_id=f"j{i}")
        seen = [(await q.dequeue(timeout=1.0)).id for _ in range(5)]
        assert seen == [f"j{i}" for i in range(5)]

    @pytest.mark.asyncio
    async def test_dequeue_timeout_returns_none(self) -> None:
        q = LocalTaskQueue()
        assert await q.dequeue(timeout=0.05) is None


class TestAckNackDeadLetter:
    @pytest.mark.asyncio
    async def test_nack_requeues_and_increments_attempt(self) -> None:
        q = LocalTaskQueue()
        await q.enqueue("retry", {}, max_attempts=3, job_id="r1")
        job = await q.dequeue(timeout=1.0)
        assert job.attempt == 1
        assert await q.nack(job.id, requeue=True) is True
        job2 = await q.dequeue(timeout=1.0)
        assert job2.id == "r1"
        assert job2.attempt == 2

    @pytest.mark.asyncio
    async def test_max_attempts_routes_to_dead_letter(self) -> None:
        q = LocalTaskQueue()
        await q.enqueue("doomed", max_attempts=2, job_id="d1")
        # Attempt 1
        j = await q.dequeue(timeout=1.0)
        await q.nack(j.id, requeue=True)
        # Attempt 2 — at the cap; nack should send to DLQ
        j = await q.dequeue(timeout=1.0)
        assert await q.nack(j.id, requeue=True) is False
        dlq = await q.dead_letter()
        assert [d.id for d in dlq] == ["d1"]

    @pytest.mark.asyncio
    async def test_nack_without_requeue_sends_to_dlq_immediately(self) -> None:
        q = LocalTaskQueue()
        await q.enqueue("once", max_attempts=5, job_id="o1")
        j = await q.dequeue(timeout=1.0)
        assert await q.nack(j.id, requeue=False) is False
        dlq = await q.dead_letter()
        assert [d.id for d in dlq] == ["o1"]

    @pytest.mark.asyncio
    async def test_ack_unknown_job_returns_false(self) -> None:
        q = LocalTaskQueue()
        assert await q.ack("missing") is False

    @pytest.mark.asyncio
    async def test_drain_dead_letter_empties_it(self) -> None:
        q = LocalTaskQueue()
        await q.enqueue("x", max_attempts=1, job_id="x1")
        j = await q.dequeue(timeout=1.0)
        await q.nack(j.id, requeue=False)
        drained = await q.drain_dead_letter()
        assert [d.id for d in drained] == ["x1"]
        assert await q.dead_letter() == []


class TestBackPressure:
    @pytest.mark.asyncio
    async def test_maxsize_blocks_enqueue(self) -> None:
        q = LocalTaskQueue(maxsize=2)
        await q.enqueue("a")
        await q.enqueue("b")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(q.enqueue("c"), timeout=0.1)
