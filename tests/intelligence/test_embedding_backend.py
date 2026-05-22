"""Tests for Tier 3.1 — EmbeddingBackend, version stamping, shared bow embedder."""

from __future__ import annotations

from app.core.text_embedding import DEFAULT_DIM, bow_embed, stable_token_bucket
from app.intelligence.embedding_backend import EmbeddingBackend, detect_stale_versions


class TestSharedBowEmbedder:
    def test_deterministic_and_normalised(self):
        a = bow_embed("the app keeps crashing")
        b = bow_embed("the app keeps crashing")
        assert a == b
        assert len(a) == DEFAULT_DIM
        assert abs(sum(x * x for x in a) - 1.0) < 1e-6

    def test_token_bucket_stable(self):
        assert stable_token_bucket("crash") == stable_token_bucket("crash")
        assert 0 <= stable_token_bucket("anything") < DEFAULT_DIM


class TestEmbeddingBackend:
    def test_fallback_when_no_real_model(self):
        be = EmbeddingBackend(prefer_real=False)
        assert be.embedding_version == "bow:512"
        assert be.uses_real_model is False
        assert len(be.embed("hello")) == 512

    def test_callable_and_deterministic(self):
        be = EmbeddingBackend(prefer_real=False)
        assert be("a b c") == be.embed("a b c")
        assert be.embed("x y") == be.embed("x y")

    def test_auto_falls_back_gracefully(self):
        # sentence-transformers may be absent; must not raise either way.
        be = EmbeddingBackend()
        assert be.embedding_version.startswith(("bow:", "st:"))
        assert len(be.embed("hello world")) > 0

    def test_detect_stale_versions(self):
        stored = {"a": "bow:512", "b": "st:model", "c": None}
        assert set(detect_stale_versions(stored, "bow:512")) == {"b", "c"}
        assert detect_stale_versions({"a": "bow:512"}, "bow:512") == []
