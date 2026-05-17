"""Tests for ``app.local.local_cache``."""

from __future__ import annotations

import asyncio
import time

import pytest

from app.local.local_cache import LocalCacheManager, LocalTTLStore


class TestLocalTTLStore:
    @pytest.mark.asyncio
    async def test_set_get_round_trip(self) -> None:
        s = LocalTTLStore()
        await s.set("k", "v", ttl=60)
        assert await s.get("k", decode=True) == "v"

    @pytest.mark.asyncio
    async def test_expiry_returns_none_after_ttl(self) -> None:
        s = LocalTTLStore()
        await s.set("ephemeral", "x", ttl=0)  # treated as no-ttl
        await s.set("short", "y", ttl=1)
        await asyncio.sleep(1.05)
        assert await s.get("short") is None

    @pytest.mark.asyncio
    async def test_delete_returns_count(self) -> None:
        s = LocalTTLStore()
        await s.set("a", "1", ttl=60)
        await s.set("b", "2", ttl=60)
        assert await s.delete("a", "b", "missing") == 2

    @pytest.mark.asyncio
    async def test_exists_skips_expired(self) -> None:
        s = LocalTTLStore()
        await s.set("alive", "v", ttl=60)
        await s.set("dead", "v", ttl=1)
        await asyncio.sleep(1.05)
        assert await s.exists("alive", "dead", "ghost") == 1


class TestLocalCacheManager:
    @pytest.mark.asyncio
    async def test_namespaces_isolated(self) -> None:
        m = LocalCacheManager()
        await m.set("ns1", "k", {"x": 1})
        await m.set("ns2", "k", {"x": 2})
        assert await m.get("ns1", "k") == {"x": 1}
        assert await m.get("ns2", "k") == {"x": 2}

    @pytest.mark.asyncio
    async def test_default_json_serialization(self) -> None:
        m = LocalCacheManager()
        await m.set("ns", "k", [1, 2, 3])
        assert await m.get("ns", "k") == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_custom_serializer_round_trip(self) -> None:
        m = LocalCacheManager()
        await m.set("ns", "k", {"a": 1}, serializer=lambda v: f"X{v['a']}")
        assert await m.get("ns", "k", deserializer=lambda s: int(s[1:])) == 1

    @pytest.mark.asyncio
    async def test_delete_and_exists(self) -> None:
        m = LocalCacheManager()
        await m.set("ns", "k", "v")
        assert await m.exists("ns", "k") is True
        assert await m.delete("ns", "k") is True
        assert await m.exists("ns", "k") is False
        assert await m.delete("ns", "k") is False

    @pytest.mark.asyncio
    async def test_clear_drops_everything(self) -> None:
        m = LocalCacheManager()
        for i in range(10):
            await m.set("ns", f"k{i}", i)
        await m.clear()
        for i in range(10):
            assert await m.get("ns", f"k{i}") is None

    @pytest.mark.asyncio
    async def test_returns_none_on_miss(self) -> None:
        m = LocalCacheManager()
        assert await m.get("ns", "never") is None

    @pytest.mark.asyncio
    async def test_string_value_round_trip(self) -> None:
        m = LocalCacheManager()
        await m.set("ns", "k", "raw-string")
        assert await m.get("ns", "k") == "raw-string"
