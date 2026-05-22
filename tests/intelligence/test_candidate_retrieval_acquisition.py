"""Regression tests for information-acquisition hardening.

Covers:
1. The pure-numpy ``_BruteForceBackend`` fallback for HNSW (exactness, distance
   semantics for cosine/l2/ip, batch add, persistence) — so dense retrieval
   degrades gracefully when the optional ``hnswlib`` wheel is absent.
2. ``HNSWIndex`` transparently using that fallback when ``hnswlib`` is missing
   (forced via monkeypatch so the test is environment-independent).
3. ``CandidateRetriever`` score fusion: the dense+sparse semantic signal is now
   normalised onto [0, 1] so it competes fairly with the fixed-magnitude entity
   regex / platform-prior sources, instead of being structurally drowned out
   (the "information acquisition mismatch").

All tests use the real domain models and need only numpy + pydantic (no
``hnswlib``, torch, transformers, or network).
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

import app.intelligence.hnsw_search as hsmod
from app.core.models import MediaType, SourcePlatform
from app.domain.inference_models import SignalType
from app.domain.normalized_models import EntityMention, NormalizedObservation
from app.intelligence.candidate_retrieval import (
    CandidateRetriever,
    ExemplarSignal,
    _rrf_merge,
)
from app.intelligence.hnsw_search import HNSWConfig, HNSWIndex, _BruteForceBackend


def _obs(text, title="t", embedding=None, entities=None, platform=SourcePlatform.RSS):
    now = datetime.now(timezone.utc)
    return NormalizedObservation(
        raw_observation_id=uuid4(),
        user_id=uuid4(),
        source_platform=platform,
        source_id="s1",
        source_url="https://example.com",
        author="a",
        title=title,
        normalized_text=text,
        media_type=MediaType.TEXT,
        published_at=now,
        fetched_at=now,
        entities=entities or [],
        embedding=embedding or [],
    )


# ---------------------------------------------------------------------------
# 1. _BruteForceBackend
# ---------------------------------------------------------------------------

class TestBruteForceBackend:
    def test_exact_nearest_neighbor_cosine(self):
        b = _BruteForceBackend(space="cosine", dim=4)
        b.init_index(max_elements=10)
        b.add_items([1.0, 0.0, 0.0, 0.0], 0)
        b.add_items([0.0, 1.0, 0.0, 0.0], 1)
        b.add_items([0.9, 0.1, 0.0, 0.0], 2)
        labels, dists = b.knn_query([1.0, 0.0, 0.0, 0.0], k=2)
        assert list(labels[0]) == [0, 2]
        # identical vector -> cosine distance ~0
        assert abs(float(dists[0][0])) < 1e-5

    def test_distance_semantics_recover_cosine(self):
        b = _BruteForceBackend(space="cosine", dim=2)
        b.add_items([1.0, 0.0], 0)
        b.add_items([0.0, 1.0], 1)  # orthogonal -> cos sim 0 -> distance 1
        labels, dists = b.knn_query([1.0, 0.0], k=2)
        d = {int(l): float(x) for l, x in zip(labels[0], dists[0])}
        assert abs((1.0 - d[0]) - 1.0) < 1e-5   # identical
        assert abs((1.0 - d[1]) - 0.0) < 1e-5   # orthogonal

    def test_l2_space(self):
        b = _BruteForceBackend(space="l2", dim=2)
        b.add_items([0.0, 0.0], 0)
        b.add_items([3.0, 4.0], 1)
        labels, dists = b.knn_query([0.0, 0.0], k=2)
        d = {int(l): float(x) for l, x in zip(labels[0], dists[0])}
        assert abs(d[0]) < 1e-6
        assert abs(d[1] - 25.0) < 1e-4   # squared L2 of (3,4)

    def test_batch_add_and_count(self):
        b = _BruteForceBackend(space="cosine", dim=3)
        b.add_items([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [0, 1])
        assert b.get_current_count() == 2

    def test_empty_index_returns_empty(self):
        b = _BruteForceBackend(space="cosine", dim=3)
        labels, dists = b.knn_query([1.0, 0.0, 0.0], k=5)
        assert labels.shape[1] == 0 and dists.shape[1] == 0

    def test_save_load_roundtrip(self, tmp_path):
        b = _BruteForceBackend(space="cosine", dim=2)
        b.init_index(max_elements=5)
        b.add_items([[1.0, 0.0], [0.0, 1.0]], [0, 1])
        p = str(tmp_path / "bf.pkl")
        b.save_index(p)
        b2 = _BruteForceBackend(space="cosine", dim=2)
        b2.load_index(p)
        assert b2.get_current_count() == 2
        labels, _ = b2.knn_query([1.0, 0.0], k=1)
        assert int(labels[0][0]) == 0


# ---------------------------------------------------------------------------
# 2. HNSWIndex falls back transparently when hnswlib is missing
# ---------------------------------------------------------------------------

class TestHNSWFallback:
    def test_index_works_without_hnswlib(self, monkeypatch):
        monkeypatch.setattr(hsmod, "hnswlib", None)
        idx = HNSWIndex(HNSWConfig(dimension=4, space="cosine"))
        idx.add_vector("a", [1.0, 0.0, 0.0, 0.0])
        idx.add_vector("b", [0.0, 1.0, 0.0, 0.0])
        idx.add_vector("c", [0.9, 0.1, 0.0, 0.0])
        res = idx.search([1.0, 0.0, 0.0, 0.0], k=2)
        assert [r.id for r in res] == ["a", "c"]
        assert abs((1.0 - res[0].distance) - 1.0) < 1e-4

    def test_statistics_without_hnswlib(self, monkeypatch):
        monkeypatch.setattr(hsmod, "hnswlib", None)
        idx = HNSWIndex(HNSWConfig(dimension=3, space="cosine"))
        idx.add_batch(["x", "y"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        stats = idx.get_statistics()
        assert stats.num_vectors == 2
        assert stats.dimension == 3


# ---------------------------------------------------------------------------
# 3. CandidateRetriever score fusion — the acquisition-mismatch fix
# ---------------------------------------------------------------------------

class TestAcquisitionScoreFusion:
    def _retriever(self, exemplars, tmp_path, top_k=5):
        # Force the brute-force fallback so the test is hnswlib-independent and
        # deterministic; isolate from any on-disk exemplar/priors files.
        return CandidateRetriever(
            exemplar_bank=exemplars,
            top_k=top_k,
            exemplar_bank_path=tmp_path / "no_bank.json",
            priors_config_path=tmp_path / "no_priors.json",
        )

    def test_semantic_match_beats_coarse_regex(self, monkeypatch, tmp_path):
        monkeypatch.setattr(hsmod, "hnswlib", None)
        ex = ExemplarSignal(
            signal_type=SignalType.COMPLAINT,
            text="software crash and error frustrating",
            embedding=[0.5] * 8,
            entities=[],
            platform="reddit",
        )
        retr = self._retriever([ex], tmp_path)
        # bug-language text triggers the BUG_REPORT regex; a PRODUCT entity makes
        # the entity/regex path run. The semantic COMPLAINT exemplar (identical
        # embedding) must still win after normalisation.
        obs = _obs(
            "the software keeps crashing with an error every time",
            title="crash",
            embedding=[0.5] * 8,
            entities=[EntityMention(entity_name="WidgetPro", entity_type="PRODUCT", confidence=0.9)],
        )
        cands = retr.retrieve_candidates(obs)
        assert cands, "expected at least one candidate"
        assert cands[0].signal_type == SignalType.COMPLAINT

    def test_embedding_scores_normalised_to_unit_top(self, monkeypatch, tmp_path):
        monkeypatch.setattr(hsmod, "hnswlib", None)
        ex = ExemplarSignal(
            signal_type=SignalType.COMPLAINT, text="crash error",
            embedding=[0.5] * 8, entities=[], platform="reddit",
        )
        retr = self._retriever([ex], tmp_path)
        obs = _obs("crash error", embedding=[0.5] * 8)
        emb = retr._retrieve_by_embedding(obs)
        assert emb, "embedding source should return candidates"
        top = max(score for _, score, _ in emb)
        assert abs(top - 1.0) < 1e-6
        assert all(0.0 <= s <= 1.0 for _, s, _ in emb)

    def test_embedding_source_dedups_signal_type(self, monkeypatch, tmp_path):
        monkeypatch.setattr(hsmod, "hnswlib", None)
        exs = [
            ExemplarSignal(signal_type=SignalType.BUG_REPORT, text=f"bug crash {i}",
                           embedding=[0.5] * 8, entities=[], platform="reddit")
            for i in range(4)
        ]
        retr = self._retriever(exs, tmp_path)
        obs = _obs("neutral text", embedding=[0.5] * 8)
        emb = retr._retrieve_by_embedding(obs)
        seen = [st for st, _, _ in emb]
        assert len(seen) == len(set(seen))  # each signal_type at most once

    def test_sparse_search_contract_preserved(self, monkeypatch, tmp_path):
        monkeypatch.setattr(hsmod, "hnswlib", None)
        exemplars = [
            ExemplarSignal(signal_type=SignalType.COMPLAINT,
                           text="broken product issue complaint", embedding=[0.1] * 8,
                           entities=[], platform="reddit"),
            ExemplarSignal(signal_type=SignalType.FEATURE_REQUEST,
                           text="please add dark mode feature", embedding=[0.2] * 8,
                           entities=[], platform="reddit"),
        ]
        retr = self._retriever(exemplars, tmp_path)
        results = retr._sparse_search("broken product issue", k=3)
        assert results and results[0] == 0


# ---------------------------------------------------------------------------
# 4. RRF merge primitive
# ---------------------------------------------------------------------------

class TestRRFMerge:
    def test_ranks_documents_in_both_lists_higher(self):
        merged = _rrf_merge([[1, 2, 3], [3, 4, 1]])
        order = [idx for idx, _ in merged]
        # docs 1 and 3 appear in both lists -> should outrank singletons
        assert order[0] in (1, 3)
        assert set([1, 3]).issubset(set(order))

    def test_handles_none_list_defensively(self):
        merged = _rrf_merge([None, [1, 2]])  # type: ignore[list-item]
        assert {idx for idx, _ in merged} == {1, 2}
