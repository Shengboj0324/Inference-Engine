"""Phase 5/6 contract tests for the local RAG layer.

Locks the behaviours that downstream consumers (chat path, desktop UI,
deployment runbook) implicitly depend on:

* soft-fail when the embedder is not configured / permission denied
* PII scrubbing on every snippet body
* personalization re-rank is bounded and never overrides ``min_score``
* explicit ``personalize=False`` returns baseline ordering
* ``embedding_version`` selector surfaces rows produced by prior models
* background reindex job lifecycle (start \u2192 poll \u2192 done; cancel)

Kept narrow on purpose: the per-route shape is already pinned by
``tests/contract/test_public_api_surface.py``; this file is the
behaviour ledger.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routes import rag as rag_routes
from app.core.config import settings
from app.core.models import ContentItem, MediaType, SourcePlatform
from app.local import rag_retriever as rr
from app.local.content_store import ContentStore, get_content_store, reset_content_store
from app.local.rag_jobs import reset_reindex_jobs
from app.local.rag_retriever import LocalRAGRetriever
from app.local.retrieval_signals import RetrievalSignalsStore, reset_signals_store


# ---------------------------------------------------------------------------
# Fakes + fixtures
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """In-memory embedder; deterministic so ranking assertions are stable."""

    model = "fake:v1"

    def __init__(self, *, configured: bool = True, permitted: bool = True,
                 vector: List[float] | None = None) -> None:
        self._configured = configured
        self._permitted = permitted
        self._vector = vector or [1.0, 0.0, 0.0]

    def is_configured(self) -> bool: return self._configured
    def is_permitted(self) -> bool: return self._permitted

    async def embed_text(self, text: str) -> List[float]:  # noqa: ARG002
        return list(self._vector)

    async def embed_batch(self, texts):  # noqa: ANN001
        return [list(self._vector) for _ in texts]


def _mk(title: str, emb, src: str, *, version: str | None = "fake:v1",
        raw: str | None = None) -> ContentItem:
    return ContentItem(
        id=uuid4(), user_id=uuid4(),
        source_platform=SourcePlatform.REDDIT, source_id=src,
        source_url="https://example.test/" + src,
        author="a", channel=None, title=title, raw_text=raw or title,
        media_type=MediaType.TEXT, media_urls=[],
        published_at=datetime.now(timezone.utc),
        fetched_at=datetime.now(timezone.utc),
        topics=[], lang="en", embedding=emb,
        metadata={"embedding_version": version} if version else {},
    )


@pytest.fixture(autouse=True)
def _reset_local_singletons() -> Iterator[None]:
    """Every test gets a fresh ContentStore / signals store / job manager.

    The local-first modules cache their handles in process globals; without
    this reset, rows from earlier tests bleed into later ones via the
    user-data SQLite file referenced by ``tmp_data_dir``.
    """
    reset_content_store(); reset_signals_store(); reset_reindex_jobs()
    rr.reset_rag_retriever()
    yield
    reset_content_store(); reset_signals_store(); reset_reindex_jobs()
    rr.reset_rag_retriever()


@pytest.fixture
def desktop_client(
    tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    """A desktop-mode TestClient with isolated stores + fake embedder."""
    monkeypatch.setattr(settings, "deployment_mode", "desktop")
    reset_content_store(); reset_signals_store(); reset_reindex_jobs()
    rr.reset_rag_retriever()
    # Force the retriever singleton to use our fake embedder + the fresh
    # content store so the request path matches what the contract probe sees.
    store = get_content_store()
    fake = _FakeEmbedder()
    rr._global_retriever = LocalRAGRetriever(content=store, embedder=fake)
    monkeypatch.setattr(rag_routes, "_embedder", lambda: fake)
    with TestClient(app) as client:
        yield client
    reset_content_store(); reset_signals_store(); reset_reindex_jobs()
    rr.reset_rag_retriever()


# ---------------------------------------------------------------------------
# Soft-fail + scrubbing
# ---------------------------------------------------------------------------


class TestRetrieverSoftFail:
    def test_returns_empty_when_query_blank(self, tmp_data_dir: Path) -> None:
        r = LocalRAGRetriever(embedder=_FakeEmbedder())
        out = asyncio.run(r.retrieve("   "))
        assert out == []

    def test_returns_empty_when_embedder_unconfigured(self, tmp_data_dir: Path) -> None:
        # An embedder that raises must not propagate; the contract is "[]".
        class _Raises(_FakeEmbedder):
            async def embed_text(self, text):  # noqa: ARG002
                raise RuntimeError("no key")
        r = LocalRAGRetriever(embedder=_Raises())
        assert asyncio.run(r.retrieve("hi")) == []

    def test_pii_scrubbed_in_snippets(self, tmp_data_dir: Path) -> None:
        store = get_content_store()
        store.upsert(_mk("Hello", [1.0, 0.0, 0.0], "s1",
                        raw="Contact me at foo@example.com please"))
        r = LocalRAGRetriever(content=store, embedder=_FakeEmbedder())
        out = asyncio.run(r.retrieve("hi", k=1))
        assert out, "expected at least one hit"
        # The DataResidencyGuard replaces emails with a redaction marker;
        # the exact token is the guard's contract, but the raw address
        # must NOT appear in the snippet body.
        assert "foo@example.com" not in out[0].text


# ---------------------------------------------------------------------------
# Personalization re-rank
# ---------------------------------------------------------------------------


class TestPersonalizationRerank:
    def _seed(self, store: ContentStore) -> tuple[str, str]:
        a = _mk("A-near", [1.0, 0.0, 0.0], "s1")
        b = _mk("B-near", [0.99, 0.10, 0.0], "s2")
        store.upsert(a); store.upsert(b)
        return str(a.id), str(b.id)


    def test_citation_signal_promotes_lower_baseline(self, tmp_data_dir: Path) -> None:
        store = get_content_store()
        a_id, b_id = self._seed(store)
        sigs = RetrievalSignalsStore(db_path=tmp_data_dir / "sig.sqlite3")
        for _ in range(10): sigs.record_citation(b_id)
        sigs.record_feedback(b_id, 1.0); sigs.record_feedback(b_id, 1.0)
        r = LocalRAGRetriever(content=store, embedder=_FakeEmbedder(), signals=sigs)
        out = asyncio.run(r.retrieve("hi", k=2))
        assert [s.title for s in out] == ["B-near", "A-near"]
        assert out[0].personalization_bonus > 0.0
        assert out[0].base_score < out[1].base_score
        # Bonus is bounded; documented worst-case is ~0.10.
        assert out[0].personalization_bonus <= 0.15

    def test_personalize_false_returns_baseline(self, tmp_data_dir: Path) -> None:
        store = get_content_store()
        _, b_id = self._seed(store)
        sigs = RetrievalSignalsStore(db_path=tmp_data_dir / "sig2.sqlite3")
        for _ in range(20): sigs.record_citation(b_id)
        r = LocalRAGRetriever(content=store, embedder=_FakeEmbedder(), signals=sigs)
        out = asyncio.run(r.retrieve("hi", k=2, personalize=False))
        assert [s.title for s in out] == ["A-near", "B-near"]
        assert all(s.personalization_bonus == 0.0 for s in out)

    def test_min_score_floors_against_base_not_boosted(self, tmp_data_dir: Path) -> None:
        store = get_content_store()
        store.upsert(_mk("C-far", [0.0, 1.0, 0.0], "s3"))
        sigs = RetrievalSignalsStore(db_path=tmp_data_dir / "sig3.sqlite3")
        # Discover the row id by issuing a query that matches it.
        r0 = LocalRAGRetriever(content=store, embedder=_FakeEmbedder(vector=[0.0, 1.0, 0.0]), signals=sigs)
        hit = asyncio.run(r0.retrieve("any", k=1))
        far_id = hit[0].content_id
        for _ in range(50): sigs.record_citation(far_id)
        r = LocalRAGRetriever(content=store, embedder=_FakeEmbedder(), signals=sigs)
        out = asyncio.run(r.retrieve("hi", k=3, min_score=0.5))
        assert all(s.title != "C-far" for s in out)


class TestStaleEmbeddingSelector:
    def test_model_swap_surfaces_existing_rows(self, tmp_data_dir: Path) -> None:
        store = get_content_store()
        store.upsert(_mk("X", [1.0, 0.0], "x", version="old:v1"))
        store.upsert(_mk("Y", [0.0, 1.0], "y", version="old:v1"))
        store.upsert(_mk("Z", None, "z", version=None))
        stale = store.list_stale_embeddings("new:v2", limit=10)
        ids = {row[0] for row in stale}
        assert len(ids) == 3
        assert store.count_stale_embeddings("new:v2") == 3
        # Only the missing row is stale relative to the prior model.
        assert store.count_stale_embeddings("old:v1") == 1


class TestRAGRoutes:
    def test_feedback_persists_and_signals_reflect_it(
        self, desktop_client: TestClient,
    ) -> None:
        store = get_content_store()
        store.upsert(_mk("A", [1.0, 0.0, 0.0], "s1"))
        store.upsert(_mk("B", [0.99, 0.1, 0.0], "s2"))
        hits = desktop_client.post("/api/v1/rag/search", json={"query": "hi", "k": 2})
        assert hits.status_code == 200
        cid = hits.json()["snippets"][0]["content_id"]
        fb = desktop_client.post(
            "/api/v1/rag/feedback", json={"content_id": cid, "score": 1.0},
        )
        assert fb.status_code == 200
        assert fb.json()["feedback_total"] == pytest.approx(1.0)
        sig = desktop_client.get("/api/v1/rag/signals?limit=10")
        assert sig.status_code == 200
        assert sig.json()["count"] == 1
        cleared = desktop_client.delete("/api/v1/rag/signals")
        assert cleared.status_code == 200 and cleared.json()["cleared"] == 1

    def test_background_job_lifecycle(self, desktop_client: TestClient) -> None:
        store = get_content_store()
        store.upsert(_mk("S", None, "stale1", version=None))
        start = desktop_client.post(
            "/api/v1/rag/reindex/jobs", json={"batch_size": 4},
        )
        assert start.status_code == 202
        jid = start.json()["id"]
        body = None
        for _ in range(40):
            time.sleep(0.05)
            got = desktop_client.get(f"/api/v1/rag/reindex/jobs/{jid}")
            body = got.json()
            if body["status"] in ("done", "error", "cancelled"):
                break
        assert body is not None and body["status"] == "done", body
        assert body["embedded"] >= 1
        assert desktop_client.get(
            "/api/v1/rag/reindex/jobs/no-such-id"
        ).status_code == 404

    def test_reindex_job_returns_409_without_embedder(
        self, desktop_client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        unavail = _FakeEmbedder(configured=False, permitted=False)
        rr._global_retriever = LocalRAGRetriever(
            content=get_content_store(), embedder=unavail,
        )
        monkeypatch.setattr(rag_routes, "_embedder", lambda: unavail)
        resp = desktop_client.post(
            "/api/v1/rag/reindex/jobs", json={"batch_size": 4},
        )
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Phase 6 follow-up: time decay + telemetry
# ---------------------------------------------------------------------------


class TestBonusTimeDecay:
    # Use a realistic ``now`` so ``now - 5*365d`` stays positive and does not
    # trip the "no last_seen" short-circuit branch in the bonus calculator.
    _NOW = 1_700_000_000.0  # 2023-11-14, well beyond 5 years past epoch.

    def test_one_halflife_halves_the_bonus(self) -> None:
        now = self._NOW
        sig_fresh = {"cited": 10, "feedback": 2.0, "last_seen": now}
        sig_old = {"cited": 10, "feedback": 2.0, "last_seen": now - 30 * 24 * 3600}
        fresh = LocalRAGRetriever._personalization_bonus(sig_fresh, now=now)
        old = LocalRAGRetriever._personalization_bonus(sig_old, now=now)
        assert fresh > 0.0
        # Halflife is exactly 30 days; allow 1% tolerance for float drift.
        assert abs(old / fresh - 0.5) < 0.01, (fresh, old)

    def test_ancient_signal_still_contributes_floor(self) -> None:
        now = self._NOW
        sig = {"cited": 10, "feedback": 2.0, "last_seen": now - 5 * 365 * 24 * 3600}
        b = LocalRAGRetriever._personalization_bonus(sig, now=now)
        raw = LocalRAGRetriever._personalization_bonus(
            {"cited": 10, "feedback": 2.0, "last_seen": now}, now=now,
        )
        # Floor is 1% of the raw bonus; ancient signals stay visible but tiny.
        assert b >= raw * 0.01 - 1e-9
        assert b < raw * 0.5

    def test_missing_last_seen_skips_decay(self) -> None:
        now = self._NOW
        sig = {"cited": 10, "feedback": 2.0, "last_seen": 0.0}
        b = LocalRAGRetriever._personalization_bonus(sig, now=now)
        raw = LocalRAGRetriever._personalization_bonus(
            {"cited": 10, "feedback": 2.0, "last_seen": now}, now=now,
        )
        # last_seen of 0 = no time information; do not penalise.
        assert b == pytest.approx(raw)


class TestTelemetryRoute:
    def test_counters_track_personalization_effect(
        self, desktop_client: TestClient,
    ) -> None:
        store = get_content_store()
        a = _mk("A", [1.0, 0.0, 0.0], "ta")
        b = _mk("B", [0.99, 0.1, 0.0], "tb")
        store.upsert(a); store.upsert(b)

        desktop_client.post("/api/v1/rag/search", json={"query": "x", "k": 2})
        for _ in range(8):
            desktop_client.post(
                "/api/v1/rag/feedback",
                json={"content_id": str(b.id), "score": 1.0},
            )
        desktop_client.post("/api/v1/rag/search", json={"query": "x", "k": 2})
        desktop_client.post("/api/v1/rag/search", json={"query": "x", "k": 2})

        body = desktop_client.get("/api/v1/rag/telemetry").json()
        assert body["queries_total"] >= 3
        assert body["queries_personalized"] >= 1
        assert body["queries_promoted_to_top"] >= 1
        assert body["queries_reordered"] >= 1
        assert body["signal_lookups_failed"] == 0

        r = desktop_client.delete("/api/v1/rag/telemetry")
        assert r.status_code == 200 and r.json()["reset"] is True
        body2 = desktop_client.get("/api/v1/rag/telemetry").json()
        assert body2["queries_total"] == 0
        assert body2["queries_personalized"] == 0
        assert body2["queries_promoted_to_top"] == 0
