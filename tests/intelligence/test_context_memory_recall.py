"""Regression tests for ContextMemoryStore vector-memory recall.

These tests cover the v1.1 memory hardening:

1. Deterministic fallback embedding (cross-process stable hashing trick).
2. Persistence + restore of the core per-user vector memory
   (``records`` + ``embeddings``) so semantic recall survives a restart.
3. True global-oldest eviction under capacity pressure.
4. Optional recency-weighted retrieval.
5. Optional ``signal_types`` / ``min_score`` retrieval filters.
6. Backward compatibility with legacy ("1.0") snapshots.

They use the real pydantic domain models (no mocks) and the built-in
bag-of-words fallback embedder, so they run anywhere numpy + pydantic are
installed — no torch / transformers / network required.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
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
from app.intelligence.context_memory import (
    ContextMemoryStore,
    _bow_embed,
    _stable_token_bucket,
)


# ---------------------------------------------------------------------------
# Factory helpers (real domain models)
# ---------------------------------------------------------------------------

def _raw(text: str, user_id=None) -> RawObservation:
    return RawObservation(
        user_id=user_id or uuid4(),
        source_platform=SourcePlatform.REDDIT,
        source_id=f"sid_{uuid4().hex[:8]}",
        source_url="https://reddit.com/r/test",
        author="u_test",
        title="t",
        raw_text=text,
        media_type=MediaType.TEXT,
        published_at=datetime.now(timezone.utc),
    )


def _norm(raw: RawObservation, text: str) -> NormalizedObservation:
    return NormalizedObservation(
        raw_observation_id=raw.id,
        user_id=raw.user_id,
        source_platform=raw.source_platform,
        source_id=raw.source_id,
        source_url=raw.source_url,
        author=raw.author,
        title=raw.title,
        normalized_text=text,
        original_language="en",
        sentiment_polarity=SentimentPolarity.NEUTRAL,
        content_quality=ContentQuality.HIGH,
        pii_scrubbed=False,
        pii_entity_count=0,
        audit_trail={},
        media_type=raw.media_type,
        published_at=raw.published_at,
        fetched_at=raw.published_at,
    )


def _inf(norm: NormalizedObservation, sig: SignalType, prob: float = 0.8) -> SignalInference:
    pred = SignalPrediction(
        signal_type=sig, probability=prob, evidence_spans=[], rationale="mock",
    )
    return SignalInference(
        normalized_observation_id=norm.id,
        user_id=norm.user_id,
        predictions=[pred],
        top_prediction=pred,
        abstained=False,
        abstention_reason=None,
        model_name="mock",
        model_version="0",
        inference_method="single_call",
    )


async def _store(store: ContextMemoryStore, uid, text: str, sig: SignalType) -> NormalizedObservation:
    raw = _raw(text, user_id=uid)
    norm = _norm(raw, text)
    await store.store(uid, norm, _inf(norm, sig))
    return norm


# ---------------------------------------------------------------------------
# 1. Deterministic embedding
# ---------------------------------------------------------------------------

class TestDeterministicEmbedding:
    def test_token_bucket_is_stable(self):
        # crc32-based bucketing is independent of PYTHONHASHSEED.
        assert _stable_token_bucket("crash") == _stable_token_bucket("crash")
        assert 0 <= _stable_token_bucket("anything") < 512

    def test_bow_embed_repeatable(self):
        a = _bow_embed("the app keeps crashing on login")
        b = _bow_embed("the app keeps crashing on login")
        assert a == b
        assert len(a) == 512
        # L2-normalised
        assert abs(sum(x * x for x in a) - 1.0) < 1e-6

    def test_bow_embed_matches_known_buckets(self):
        # A token must land in its crc32 bucket — pins determinism to a value
        # that does not depend on process hash seed.
        v = _bow_embed("crash")
        assert v[_stable_token_bucket("crash")] > 0.0


# ---------------------------------------------------------------------------
# 2. Persistence of the core vector memory
# ---------------------------------------------------------------------------

class TestVectorMemoryPersistence:
    @pytest.mark.asyncio
    async def test_records_survive_persist_reload(self, tmp_path: Path):
        store = ContextMemoryStore()
        uid = uuid4()
        await _store(store, uid, "the app keeps crashing on login", SignalType.BUG_REPORT)

        p = tmp_path / "cm.json"
        store.persist(p)

        payload = json.loads(p.read_text())
        assert "records" in payload and "embeddings" in payload
        assert payload["version"] == "1.1"

        store2 = ContextMemoryStore()
        store2.load_from_disk(p)
        results = await store2.retrieve(uid, "crash on login", top_k=3)
        assert len(results) == 1
        assert "crash" in results[0].normalized_text
        assert results[0].signal_type is SignalType.BUG_REPORT
        assert abs(results[0].confidence - 0.8) < 1e-6

    @pytest.mark.asyncio
    async def test_cross_process_embedding_recall(self, tmp_path: Path):
        # Stored embeddings (persisted) must remain comparable to a query
        # embedded "later" — guaranteed by deterministic bucketing.
        store = ContextMemoryStore()
        uid = uuid4()
        await _store(store, uid, "billing invoice overcharge refund", SignalType.COMPLAINT)
        p = tmp_path / "cm.json"
        store.persist(p)

        store2 = ContextMemoryStore()
        store2.load_from_disk(p)
        hit = await store2.retrieve(uid, "overcharge refund billing", top_k=1)
        assert hit and hit[0].score > 0.0

    @pytest.mark.asyncio
    async def test_user_profile_persists(self, tmp_path: Path):
        store = ContextMemoryStore()
        uid = uuid4()
        store.update_user_profile(uid, {"display_name": "Acme PM", "interests": ["uptime"]})
        p = tmp_path / "cm.json"
        store.persist(p)
        store2 = ContextMemoryStore()
        store2.load_from_disk(p)
        assert store2.get_user_profile(uid).get("display_name") == "Acme PM"


# ---------------------------------------------------------------------------
# 3. Global-oldest eviction
# ---------------------------------------------------------------------------

class TestGlobalOldestEviction:
    @pytest.mark.asyncio
    async def test_evicts_globally_oldest_not_first_bucket(self):
        store = ContextMemoryStore(max_records=4)
        a, b = uuid4(), uuid4()
        now = datetime.now(timezone.utc)

        # B's bucket is created first but holds a RECENT record.
        await _store(store, b, "b recent observation", SignalType.PRAISE)
        # A's bucket created later; backdate A's first record to be oldest.
        for i in range(4):
            await _store(store, a, f"a observation {i}", SignalType.PRAISE)
            if i == 0:
                store._records[str(a)][0].created_at = now - timedelta(hours=10)

        a_oldest_present = any(
            r.created_at < now - timedelta(hours=5)
            for r in store._records.get(str(a), [])
        )
        b_recent_present = any(
            "recent" in r.normalized_text for r in store._records.get(str(b), [])
        )
        assert not a_oldest_present, "globally-oldest record should have been evicted"
        assert b_recent_present, "a recent record in another bucket must be kept"
        assert store._total <= 4


# ---------------------------------------------------------------------------
# 4. Recency-weighted retrieval
# ---------------------------------------------------------------------------

class TestRecencyWeighting:
    @pytest.mark.asyncio
    async def test_recency_promotes_newer_among_equal_similarity(self):
        store = ContextMemoryStore()
        uid = uuid4()
        await _store(store, uid, "login crash bug", SignalType.BUG_REPORT)
        await _store(store, uid, "login crash bug", SignalType.BUG_REPORT)
        store._records[str(uid)][0].created_at = datetime.now(timezone.utc) - timedelta(days=60)
        newer_id = store._records[str(uid)][1].observation_id

        ranked = await store.retrieve(uid, "login crash bug", top_k=2, recency_half_life_days=14)
        assert ranked[0].observation_id == newer_id

    @pytest.mark.asyncio
    async def test_default_retrieve_ignores_recency(self):
        store = ContextMemoryStore()
        uid = uuid4()
        await _store(store, uid, "alpha beta", SignalType.SUPPORT_REQUEST)
        out = await store.retrieve(uid, "alpha beta", top_k=5)
        assert len(out) == 1
        # score is plain cosine similarity in the default path
        assert 0.0 <= out[0].score <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# 5. Retrieval filters
# ---------------------------------------------------------------------------

class TestRetrievalFilters:
    @pytest.mark.asyncio
    async def test_signal_type_filter(self):
        store = ContextMemoryStore()
        uid = uuid4()
        await _store(store, uid, "great product love it", SignalType.PRAISE)
        await _store(store, uid, "it is broken and crashes", SignalType.BUG_REPORT)
        bugs = await store.retrieve(
            uid, "broken crash", top_k=5, signal_types={SignalType.BUG_REPORT}
        )
        assert len(bugs) == 1
        assert bugs[0].signal_type is SignalType.BUG_REPORT

    @pytest.mark.asyncio
    async def test_min_score_filter(self):
        store = ContextMemoryStore()
        uid = uuid4()
        await _store(store, uid, "completely unrelated topic xyz", SignalType.SUPPORT_REQUEST)
        out = await store.retrieve(uid, "nothing in common here", top_k=5, min_score=0.99)
        assert out == []


# ---------------------------------------------------------------------------
# 6. Backward compatibility
# ---------------------------------------------------------------------------

class TestLegacySnapshotCompat:
    def test_v1_0_snapshot_loads(self, tmp_path: Path):
        uid = str(uuid4())
        legacy = {
            "version": "1.0",
            "preferences": {
                uid: {"complaint": {"acted": 0, "dismissed": 2,
                                    "snoozed": 0, "false_positive": 0}}
            },
            "inference_history": {},
            "history_seen": {},
            "rationale_memory": {},
            "noise_thresholds": {uid: 0.45},
            "competitor_aliases": {},
            "channel_prefs": {},
            "source_embeddings": {},
        }
        p = tmp_path / "legacy.json"
        p.write_text(json.dumps(legacy), encoding="utf-8")

        store = ContextMemoryStore()
        store.load_from_disk(p)  # must not raise on missing records/embeddings
        from uuid import UUID
        weights = store.get_signal_type_weights(UUID(uid))
        assert weights.get("complaint", 1.0) < 1.0
        assert store._total == 0
