"""Tests for Tier 3.2 (consolidation) + 3.3 (MMR) + 3.1 version stamping
integrated into ContextMemoryStore."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.core.models import MediaType, SourcePlatform
from app.domain.inference_models import (
    SignalInference,
    SignalPrediction,
    SignalType,
)
from app.domain.normalized_models import (
    ContentQuality,
    NormalizedObservation,
    SentimentPolarity,
)
from app.domain.raw_models import RawObservation
from app.intelligence.context_memory import ContextMemoryStore
from app.intelligence.memory_consolidation import MemoryConsolidator


def _norm(text, uid):
    now = datetime.now(timezone.utc)
    raw = RawObservation(
        user_id=uid, source_platform=SourcePlatform.REDDIT,
        source_id=f"s_{uuid4().hex[:6]}", source_url="https://x", author="a",
        title="t", raw_text=text, media_type=MediaType.TEXT, published_at=now,
    )
    return NormalizedObservation(
        raw_observation_id=raw.id, user_id=uid, source_platform=raw.source_platform,
        source_id=raw.source_id, source_url=raw.source_url, author=raw.author,
        title=raw.title, normalized_text=text, original_language="en",
        sentiment_polarity=SentimentPolarity.NEUTRAL, content_quality=ContentQuality.HIGH,
        pii_scrubbed=False, pii_entity_count=0, audit_trail={},
        media_type=raw.media_type, published_at=now, fetched_at=now,
    )


def _inf(uid, norm):
    pred = SignalPrediction(signal_type=SignalType.PRAISE, probability=0.8,
                            evidence_spans=[], rationale="m")
    return SignalInference(
        normalized_observation_id=norm.id, user_id=uid, predictions=[pred],
        top_prediction=pred, abstained=False, abstention_reason=None,
        model_name="m", model_version="0", inference_method="single_call",
    )


async def _store(s, uid, text):
    n = _norm(text, uid)
    await s.store(uid, n, _inf(uid, n))


class TestConsolidation:
    def test_consolidator_picks_one_rep_per_cluster(self):
        # 3 well-separated clusters in 3-d space.
        embs = [[1, 0, 0]] * 5 + [[0, 1, 0]] * 5 + [[0, 0, 1]] * 5
        reps, labels = MemoryConsolidator(seed=0).consolidate(embs, target_k=3)
        assert len(reps) == 3
        assert len(set(labels)) == 3

    def test_consolidate_keeps_all_when_below_target(self):
        reps, labels = MemoryConsolidator().consolidate([[1, 0], [0, 1]], target_k=5)
        assert reps == [0, 1]

    @pytest.mark.asyncio
    async def test_consolidate_user_retains_topic_recall(self):
        store = ContextMemoryStore()
        uid = uuid4()
        topics = ["alpha", "beta", "gamma", "delta", "epsilon"]
        for t in topics:
            for _ in range(20):
                await _store(store, uid, f"{t} {t} {t}")
        assert len(store._records[str(uid)]) == 100
        out = store.consolidate_user(uid, target_k=5)
        assert out["after"] == 5 and out["removed"] == 95
        hits = 0
        for t in topics:
            r = await store.retrieve(uid, f"{t} {t}", top_k=1)
            if r and t in r[0].normalized_text:
                hits += 1
        assert hits == 5  # every topic still retrievable

    @pytest.mark.asyncio
    async def test_consolidate_summarize_fn_applied(self):
        store = ContextMemoryStore()
        uid = uuid4()
        for _ in range(10):
            await _store(store, uid, "alpha alpha alpha")
        for _ in range(10):
            await _store(store, uid, "beta beta beta")
        store.consolidate_user(uid, target_k=2, summarize_fn=lambda texts: "SUMMARY")
        assert all(r.normalized_text == "SUMMARY" for r in store._records[str(uid)])


class TestMMR:
    @pytest.mark.asyncio
    async def test_mmr_surfaces_diverse_relevant_item(self):
        store = ContextMemoryStore()
        uid = uuid4()
        for _ in range(4):
            await _store(store, uid, "alpha alpha alpha")
        await _store(store, uid, "beta beta beta")
        plain = await store.retrieve(uid, "alpha alpha beta", top_k=3)
        mmr = await store.retrieve(uid, "alpha alpha beta", top_k=3, mmr_lambda=0.7)
        assert "alpha" in mmr[0].normalized_text          # top relevant kept
        assert any("beta" in r.normalized_text for r in mmr)      # diversified
        assert not any("beta" in r.normalized_text for r in plain)  # default unchanged

    @pytest.mark.asyncio
    async def test_default_retrieve_unchanged(self):
        store = ContextMemoryStore()
        uid = uuid4()
        await _store(store, uid, "alpha beta gamma")
        out = await store.retrieve(uid, "alpha beta gamma", top_k=3)
        assert len(out) == 1 and 0.0 <= out[0].score <= 1.0 + 1e-6


class TestEmbeddingVersionStamping:
    @pytest.mark.asyncio
    async def test_version_persists_and_needs_reembed(self, tmp_path):
        store = ContextMemoryStore()
        uid = uuid4()
        await _store(store, uid, "hello world")
        store.set_embedding_version("bow:512")
        p = tmp_path / "cm.json"
        store.persist(p)
        store2 = ContextMemoryStore()
        store2.load_from_disk(p)
        assert store2._embedding_version == "bow:512"
        assert store2.needs_reembed("bow:512") is False
        assert store2.needs_reembed("st:new-model") is True
