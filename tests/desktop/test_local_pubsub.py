"""Tests for ``app.local.local_pubsub``."""

from __future__ import annotations

import asyncio

import pytest

from app.local import local_pubsub as lps


class TestLocalRedisShim:
    @pytest.mark.asyncio
    async def test_ping(self) -> None:
        client = lps.from_url("redis://ignored")
        assert await client.ping() is True
        await client.aclose()

    @pytest.mark.asyncio
    async def test_set_get_setex_delete_exists(self) -> None:
        c = lps.from_url("redis://ignored", decode_responses=True)
        await c.setex("k1", 60, "v1")
        assert await c.get("k1") == "v1"
        assert await c.exists("k1") == 1
        assert await c.delete("k1") == 1
        assert await c.get("k1") is None
        assert await c.exists("k1") == 0

    @pytest.mark.asyncio
    async def test_setex_expires(self) -> None:
        c = lps.from_url("redis://ignored", decode_responses=True)
        await c.setex("ephemeral", 1, "bye")
        # Fast-forward by manipulating monotonic via the store's internal API
        from app.local.local_cache import LocalTTLStore
        # Force expiry by calling sleep(2)
        await asyncio.sleep(1.1)
        assert await c.get("ephemeral") is None


class TestPublishSubscribeFanout:
    @pytest.mark.asyncio
    async def test_single_subscriber_receives_published_message(self) -> None:
        publisher = lps.from_url("redis://ignored", decode_responses=True)
        subscriber = lps.from_url("redis://ignored", decode_responses=True)
        ps = subscriber.pubsub()
        await ps.subscribe("ch1")

        async def consume():
            async for msg in ps.listen():
                return msg

        consumer_task = asyncio.create_task(consume())
        await asyncio.sleep(0)  # let the subscriber register
        delivered = await publisher.publish("ch1", "hello")
        assert delivered == 1
        msg = await asyncio.wait_for(consumer_task, timeout=2.0)
        assert msg["type"] == "message"
        assert msg["channel"] == "ch1"
        assert msg["data"] == "hello"
        await ps.aclose()

    @pytest.mark.asyncio
    async def test_fanout_to_multiple_subscribers(self) -> None:
        publisher = lps.from_url("redis://ignored", decode_responses=True)
        n_subscribers = 25
        subs = [lps.from_url("redis://ignored", decode_responses=True).pubsub()
                for _ in range(n_subscribers)]
        for ps in subs:
            await ps.subscribe("fanout")

        async def one(ps):
            async for msg in ps.listen():
                return msg["data"]

        tasks = [asyncio.create_task(one(ps)) for ps in subs]
        await asyncio.sleep(0)
        delivered = await publisher.publish("fanout", "broadcast")
        assert delivered == n_subscribers
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)
        assert results == ["broadcast"] * n_subscribers
        for ps in subs:
            await ps.aclose()

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_delivery(self) -> None:
        publisher = lps.from_url("redis://ignored", decode_responses=True)
        ps = lps.from_url("redis://ignored", decode_responses=True).pubsub()
        await ps.subscribe("ch")
        assert await publisher.publish("ch", "first") == 1
        await ps.unsubscribe("ch")
        assert await publisher.publish("ch", "second") == 0
        await ps.aclose()

    @pytest.mark.asyncio
    async def test_publish_to_no_subscribers_returns_zero(self) -> None:
        publisher = lps.from_url("redis://ignored")
        assert await publisher.publish("nobody", "x") == 0

    @pytest.mark.asyncio
    async def test_listen_terminates_on_aclose(self) -> None:
        ps = lps.from_url("redis://ignored", decode_responses=True).pubsub()
        await ps.subscribe("c")

        async def consume():
            received = []
            async for msg in ps.listen():
                received.append(msg)
            return received

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await ps.aclose()
        out = await asyncio.wait_for(task, timeout=2.0)
        assert out == []  # sentinel terminates before yielding


class TestRedisDropInParity:
    """Confirm the surface used by app/api/routes/signals.py works verbatim."""

    @pytest.mark.asyncio
    async def test_signals_ws_style_loop(self) -> None:
        publisher = lps.from_url("redis://ignored", decode_responses=True)
        client = lps.from_url("redis://ignored", decode_responses=True)
        ps = client.pubsub()
        channel = "signals:test-user"
        await ps.subscribe(channel)

        async def listener(out):
            async for msg in ps.listen():
                if msg["type"] == "message":
                    out.append(msg["data"])
                    if len(out) == 3:
                        await ps.aclose()
                        return

        sink: list = []
        t = asyncio.create_task(listener(sink))
        await asyncio.sleep(0)
        for body in ("a", "b", "c"):
            await publisher.publish(channel, body)
        await asyncio.wait_for(t, timeout=2.0)
        assert sink == ["a", "b", "c"]
