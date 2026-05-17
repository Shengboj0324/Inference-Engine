"""Phase 1 concurrency stress tests for the local-first backends.

These tests deliberately use high fan-out to surface race conditions in
the shared module-level state (pub/sub hub, queue, TTL store).  They
must complete in under a few seconds on a developer laptop to remain
viable in the regression suite.
"""

from __future__ import annotations

import asyncio

import pytest

from app.local import local_pubsub as lps
from app.local.local_cache import LocalCacheManager
from app.local.local_queue import LocalTaskQueue


class TestPubSubStress:
    @pytest.mark.asyncio
    async def test_100_subscribers_each_receive_50_messages(self) -> None:
        n_subs, n_msgs = 100, 50
        publisher = lps.from_url("redis://x", decode_responses=True)
        subs = [lps.from_url("redis://x", decode_responses=True).pubsub()
                for _ in range(n_subs)]
        for ps in subs:
            await ps.subscribe("stress")

        results = [0] * n_subs

        async def listener(idx: int, ps) -> None:
            async for _ in ps.listen():
                results[idx] += 1
                if results[idx] == n_msgs:
                    await ps.aclose()
                    return

        tasks = [asyncio.create_task(listener(i, ps)) for i, ps in enumerate(subs)]
        await asyncio.sleep(0)
        for i in range(n_msgs):
            delivered = await publisher.publish("stress", f"msg-{i}")
            assert delivered == n_subs

        await asyncio.wait_for(asyncio.gather(*tasks), timeout=10.0)
        assert results == [n_msgs] * n_subs


class TestQueueStress:
    @pytest.mark.asyncio
    async def test_2000_jobs_16_workers_no_duplicates(self) -> None:
        q = LocalTaskQueue(maxsize=4096)
        n_jobs, n_workers = 2000, 16

        producer_done = asyncio.Event()
        processed: list = []
        processed_lock = asyncio.Lock()

        async def producer() -> None:
            for i in range(n_jobs):
                await q.enqueue("work", {"i": i}, job_id=f"j{i}")
            producer_done.set()

        async def worker() -> None:
            while True:
                if producer_done.is_set() and await q.pending() == 0:
                    return
                job = await q.dequeue(timeout=0.1)
                if job is None:
                    continue
                async with processed_lock:
                    processed.append(job.id)
                await q.ack(job.id)

        await asyncio.gather(
            producer(),
            *[worker() for _ in range(n_workers)],
        )
        assert len(processed) == n_jobs
        assert len(set(processed)) == n_jobs, "duplicate delivery detected"
        assert await q.in_flight() == 0


class TestCacheStress:
    @pytest.mark.asyncio
    async def test_concurrent_set_get_no_corruption(self) -> None:
        m = LocalCacheManager()
        n_writers, n_keys, iterations = 32, 64, 100

        async def writer(wid: int) -> None:
            for it in range(iterations):
                key = f"k-{(wid + it) % n_keys}"
                await m.set("ns", key, {"w": wid, "i": it})

        async def reader(_: int) -> None:
            for _ in range(iterations):
                await m.get("ns", "k-0")

        await asyncio.gather(
            *[writer(i) for i in range(n_writers)],
            *[reader(i) for i in range(n_writers)],
        )
        # Final invariant: every key still parses as a dict with the writer
        # tag — proves no half-written serializations slipped through.
        for k_idx in range(n_keys):
            val = await m.get("ns", f"k-{k_idx}")
            assert val is None or isinstance(val, dict)
            if isinstance(val, dict):
                assert "w" in val and "i" in val
