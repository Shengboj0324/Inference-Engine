"""Tests for ``app.local.sqlite_store``."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from sqlalchemy import text

from app.local import sqlite_store as ss


class TestEngineFactory:
    @pytest.mark.asyncio
    async def test_memory_engine_runs_query(self) -> None:
        engine = ss.create_local_engine(":memory:")
        try:
            async with engine.begin() as conn:
                result = await conn.execute(text("SELECT 1 AS n"))
                row = result.first()
                assert row.n == 1
        finally:
            await engine.dispose()

    @pytest.mark.asyncio
    async def test_file_engine_creates_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "sub" / "data.sqlite3"
        engine = ss.create_local_engine(str(db_path))
        try:
            async with engine.begin() as conn:
                await conn.execute(text("CREATE TABLE t (id INTEGER)"))
                await conn.execute(text("INSERT INTO t VALUES (1), (2)"))
            assert db_path.is_file()
        finally:
            await engine.dispose()

    @pytest.mark.asyncio
    async def test_default_path_uses_user_data_dir(
        self, tmp_data_dir: Path
    ) -> None:
        engine = ss.create_local_engine()
        try:
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
        finally:
            await engine.dispose()
        assert (tmp_data_dir / "data" / "social_radar.sqlite3").is_file()

    @pytest.mark.asyncio
    async def test_session_factory_round_trip(self) -> None:
        engine = ss.create_local_engine(":memory:")
        Session = ss.make_session_factory(engine)
        try:
            async with engine.begin() as conn:
                await conn.execute(text("CREATE TABLE k (v TEXT)"))
                await conn.execute(text("INSERT INTO k VALUES ('hi')"))
            async with Session() as session:
                result = await session.execute(text("SELECT v FROM k"))
                assert result.scalar_one() == "hi"
        finally:
            await engine.dispose()


class TestVectorHelpers:
    def test_encode_decode_round_trip(self) -> None:
        v = [0.1, -0.5, 2.0, 3.14]
        blob = ss.encode_vector(v)
        out = ss.decode_vector(blob)
        for a, b in zip(v, out):
            assert math.isclose(a, b, rel_tol=1e-4)

    def test_cosine_top_k_correctness(self) -> None:
        query = [1.0, 0.0, 0.0]
        candidates = [
            ("perfect", [1.0, 0.0, 0.0]),
            ("orthogonal", [0.0, 1.0, 0.0]),
            ("opposite", [-1.0, 0.0, 0.0]),
            ("close", [0.9, 0.1, 0.0]),
        ]
        ranked = ss.cosine_top_k(query, candidates, k=3)
        assert ranked[0][0] == "perfect"
        assert ranked[1][0] == "close"
        assert ranked[2][0] == "orthogonal"

    def test_cosine_zero_vector_returns_zero(self) -> None:
        ranked = ss.cosine_top_k([1.0, 0.0], [("z", [0.0, 0.0])], k=1)
        assert ranked == [("z", 0.0)]

    def test_cosine_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            ss.cosine_top_k([1.0, 0.0], [("bad", [1.0])], k=1)

    def test_load_sqlite_vec_returns_bool_and_never_raises(self) -> None:
        # The loader must degrade gracefully whether or not ``sqlite_vec``
        # is installed in the current environment.  Phase 6 ships the
        # extension as an optional dependency, so we only assert the
        # contract (bool return, no raise) rather than a specific value.
        import sqlite3
        conn = sqlite3.connect(":memory:")
        try:
            result = ss.load_sqlite_vec(conn)
            assert isinstance(result, bool)
        finally:
            conn.close()
