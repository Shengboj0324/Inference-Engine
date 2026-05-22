"""Tests for Tier 1.1 (learned retrieval fusion) and Tier 1.2 (reranker in the
candidate path).

Both features are opt-in: the default ``CandidateRetriever`` behaviour is
unchanged unless a trained fusion model and/or a reranker is attached.

Pure-numpy fusion + TF-IDF reranker fallback; no torch / hnswlib / network.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

import app.intelligence.hnsw_search as hsmod
from app.core.models import MediaType, SourcePlatform
from app.domain.inference_models import SignalType
from app.domain.normalized_models import NormalizedObservation
from app.intelligence.candidate_retrieval import CandidateRetriever, ExemplarSignal
from app.intelligence.reranker import Reranker
from app.intelligence.retrieval_fusion import FEATURE_NAMES, LogisticFusion


def _obs(text, embedding=None, platform=SourcePlatform.RSS, entities=None):
    now = datetime.now(timezone.utc)
    return NormalizedObservation(
        raw_observation_id=uuid4(), user_id=uuid4(), source_platform=platform,
        source_id="s", source_url="https://example.com", author="a", title="",
        normalized_text=text, media_type=MediaType.TEXT, published_at=now,
        fetched_at=now, entities=entities or [], embedding=embedding or [],
    )


# ---------------------------------------------------------------------------
# Tier 1.1 — LogisticFusion
# ---------------------------------------------------------------------------

class TestLogisticFusion:
    def test_learns_separable_data(self):
        # y depends on feature 0; model should give high prob when f0 high.
        X = [[1, 0, 0], [0.9, 0, 0], [0, 0, 1], [0, 0, 0.9]]
        y = [1, 1, 0, 0]
        m = LogisticFusion(3).fit(X, y, epochs=800)
        assert m.predict_one([1, 0, 0]) > 0.5
        assert m.predict_one([0, 0, 1]) < 0.5

    def test_beats_fixed_weights_on_adversarial(self):
        rng = np.random.default_rng(0)

        def sample():
            return {
                "right": [rng.uniform(0.40, 0.60), 0.0, 0.0],
                "distractor": [0.0, 0.0, rng.uniform(0.80, 1.00)],
            }

        def fixed(fv):
            return fv[0] * 0.4 + fv[1] * 0.3 + fv[2] * 0.3

        X, y = [], []
        for s in (sample() for _ in range(400)):
            for t, fv in s.items():
                X.append(fv)
                y.append(1.0 if t == "right" else 0.0)
        m = LogisticFusion(3).fit(X, y)

        test = [sample() for _ in range(200)]
        base = sum(max(s, key=lambda t: fixed(s[t])) == "right" for s in test) / len(test)
        learned = sum(max(s, key=lambda t: m.predict_one(s[t])) == "right" for s in test) / len(test)
        assert learned >= base + 0.3
        assert learned >= 0.9
        assert m.w[0] > m.w[2]  # embedding weighted above platform

    def test_predictions_bounded(self):
        m = LogisticFusion(3).fit([[1, 0, 0], [0, 0, 1]], [1, 0])
        assert 0.0 <= m.predict_one([1e6, -1e6, 1e6]) <= 1.0

    def test_deterministic_fit(self):
        X, y = [[1, 0, 0], [0, 0, 1]], [1, 0]
        a, b = LogisticFusion(3).fit(X, y), LogisticFusion(3).fit(X, y)
        assert np.allclose(a.w, b.w) and abs(a.b - b.b) < 1e-12

    def test_round_trip(self, tmp_path):
        m = LogisticFusion(3).fit([[1, 0, 0], [0, 0, 1]], [1, 0])
        m2 = LogisticFusion.from_dict(m.to_dict())
        assert abs(m2.predict_one([0.5, 0, 0]) - m.predict_one([0.5, 0, 0])) < 1e-12
        p = tmp_path / "f.json"
        m.save(p)
        m3 = LogisticFusion.load(p)
        assert abs(m3.predict_one([0.5, 0, 0]) - m.predict_one([0.5, 0, 0])) < 1e-12
        assert m.to_dict()["feature_names"] == FEATURE_NAMES

    def test_untrained_flag_and_bad_input(self):
        m = LogisticFusion(3)
        assert not m.is_trained()
        with pytest.raises(ValueError):
            m.fit([], [])
        with pytest.raises(ValueError):
            m.fit([[1, 0]], [1])  # wrong feature dim


# ---------------------------------------------------------------------------
# Tier 1.2 — Reranker.score_pair
# ---------------------------------------------------------------------------

class TestRerankerScorePair:
    def test_matching_scores_higher(self):
        rk = Reranker()
        assert rk.score_pair("billing refund overcharge",
                             "billing refund overcharge issue") > \
               rk.score_pair("billing refund overcharge", "weather forecast tomorrow")

    def test_never_raises(self):
        rk = Reranker()
        assert rk.score_pair("", "") == 0.0
        assert 0.0 <= rk.score_pair("a b c", "a b c") <= 1.0


# ---------------------------------------------------------------------------
# Integration into CandidateRetriever (opt-in)
# ---------------------------------------------------------------------------

class TestRetrieverIntegration:
    def _retriever(self, exemplars, tmp_path, **kw):
        return CandidateRetriever(
            exemplar_bank=exemplars, top_k=5,
            exemplar_bank_path=tmp_path / "no_bank.json",
            priors_config_path=tmp_path / "no_priors.json", **kw,
        )

    def test_fused_path_used_when_model_trained(self, monkeypatch, tmp_path):
        monkeypatch.setattr(hsmod, "hnswlib", None)
        model = LogisticFusion(3).fit([[1, 0, 0], [0, 0, 1]], [1, 0])
        exs = [ExemplarSignal(signal_type=SignalType.COMPLAINT, text="crash error",
                              embedding=[0.5] * 8, entities=[], platform="reddit")]
        retr = self._retriever(exs, tmp_path, fusion_model=model)
        obs = _obs("crash error", embedding=[0.5] * 8)
        feats = retr.extract_fusion_features(obs)
        assert all(len(v) == 3 for v in feats.values())
        cands = retr.retrieve_candidates(obs)
        assert cands and all(c.source == "learned_fusion" for c in cands)
        assert all(0.0 <= c.score <= 1.0 for c in cands)

    def test_untrained_model_uses_legacy_path(self, monkeypatch, tmp_path):
        monkeypatch.setattr(hsmod, "hnswlib", None)
        exs = [ExemplarSignal(signal_type=SignalType.COMPLAINT, text="crash error",
                              embedding=[0.5] * 8, entities=[], platform="reddit")]
        retr = self._retriever(exs, tmp_path, fusion_model=LogisticFusion(3))  # not fit
        cands = retr.retrieve_candidates(_obs("crash error", embedding=[0.5] * 8))
        assert cands and all(c.source != "learned_fusion" for c in cands)

    def test_reranker_reorders_candidate_path(self, monkeypatch, tmp_path):
        monkeypatch.setattr(hsmod, "hnswlib", None)
        exs = [
            ExemplarSignal(signal_type=SignalType.BUG_REPORT,
                           text="app crashes with a fatal error on startup",
                           embedding=[0.5] * 8, entities=[], platform="reddit"),
            ExemplarSignal(signal_type=SignalType.PRICE_SENSITIVITY,
                           text="the subscription price is too expensive",
                           embedding=[0.5] * 8, entities=[], platform="reddit"),
        ]
        retr = self._retriever(exs, tmp_path, reranker=Reranker())
        emb = retr._retrieve_by_embedding(_obs("my app crashes with a fatal error",
                                               embedding=[0.5] * 8))
        ranked = sorted(emb, key=lambda t: t[1], reverse=True)
        assert ranked[0][0] == SignalType.BUG_REPORT
        assert all(0.0 <= s <= 1.0 for _, s, _ in emb)

    def test_default_path_unchanged(self, monkeypatch, tmp_path):
        # No model, no reranker -> the original behaviour (semantic match wins).
        monkeypatch.setattr(hsmod, "hnswlib", None)
        exs = [ExemplarSignal(signal_type=SignalType.COMPLAINT,
                              text="software crash and error",
                              embedding=[0.5] * 8, entities=[], platform="reddit")]
        retr = self._retriever(exs, tmp_path)
        cands = retr.retrieve_candidates(_obs("software crash and error", embedding=[0.5] * 8))
        assert cands[0].signal_type == SignalType.COMPLAINT
