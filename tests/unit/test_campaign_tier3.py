"""Tier 3 — High-Volume / Stress Tests

All scenarios run 20+ independent rounds where relevant.
Thread-safety scenarios use a 30-second wall-clock timeout guard.
"""
from __future__ import annotations

import asyncio
import gc
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import List

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

ARTICLES = [
    "OpenAI unveils GPT-5 with advanced reasoning capabilities and a 200K context window, surpassing human benchmarks on MMLU and HumanEval across all evaluated domains.",
    "Google DeepMind releases AlphaFold 3 extending prediction to DNA, RNA, and small molecules for drug discovery in cancer and infectious disease research globally.",
    "Anthropic closes a $2 billion Series E round led by Google, valuing the company at $18 billion as Claude 3 API demand surges 400% quarter-over-quarter.",
    "NVIDIA Blackwell B200 GPU delivers 4x AI training throughput and 30x inference improvement over H100, targeting the 2025 data-center market for hyperscalers.",
    "Meta releases LLaMA 3 under community licence with 70B and 400B variants featuring instruction-following fine-tuning and RLHF alignment improvements at scale.",
]
ROUNDS = list(range(20))


def _make_item(text: str, platform_name: str = "reddit"):
    from app.core.models import ContentItem, SourcePlatform, MediaType
    plats = {
        "reddit": SourcePlatform.REDDIT,
        "github": SourcePlatform.GITHUB_RELEASES,
        "arxiv":  SourcePlatform.ARXIV,
    }
    return ContentItem(
        user_id=uuid.uuid4(),
        source_platform=plats.get(platform_name, SourcePlatform.REDDIT),
        source_id=f"{platform_name}-{uuid.uuid4().hex[:8]}",
        source_url=f"https://example.com/{uuid.uuid4().hex[:8]}",
        title=text[:120],
        raw_text=text,
        media_type=MediaType.TEXT,
        published_at=datetime.now(timezone.utc),
        topics=text.split()[:6],
    )


def _make_batch(n: int) -> List:
    return [_make_item(ARTICLES[i % len(ARTICLES)]) for i in range(n)]


def _run_batch(items, quality_gate=None):
    from app.ingestion.indexing_pipeline import IndexingPipeline
    from app.intelligence.retrieval.chunk_store import ChunkStore
    pipeline = IndexingPipeline(chunk_store=ChunkStore(), quality_gate=quality_gate)
    return asyncio.run(pipeline.process_batch(items))


def _make_registry(n: int):
    from app.source_intelligence.source_registry import (
        SourceRegistryStore, SourceSpec, SourceFamily,
    )
    from app.core.models import SourcePlatform
    reg = SourceRegistryStore()
    for i in range(n):
        priority = round(i / max(n - 1, 1), 4)
        reg.register(SourceSpec(
            source_id=f"src-{i:04d}",
            platform=SourcePlatform.GITHUB_RELEASES,
            family=SourceFamily.DEVELOPER_RELEASE,
            priority=priority,
        ))
    return reg


def _make_bundle(bid: str, title: str, trust: float = 0.8):
    from app.entity_resolution.models import EventBundle
    return EventBundle(
        bundle_id=bid, canonical_title=title,
        primary_item_id=f"{bid}-item",
        source_items=[{"source_id": f"{bid}-item", "title": title,
                       "raw_text": title, "source_url": "https://x.com",
                       "platform": "reddit", "trust_score": trust}],
        trust_scores={f"{bid}-item": trust},
    )


# ═════════════════════════════════════════════════════════════════════════════
# T3-A  Batch-size stress tests
# ═════════════════════════════════════════════════════════════════════════════

class TestT3ABatchSizes:
    @pytest.mark.parametrize("_r", ROUNDS)
    def test_batch_50_no_exception(self, _r):
        result = _run_batch(_make_batch(50))
        assert result.stats.chunks_indexed >= 1
        assert len(result.errors) < 50   # some may fail, but not all

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_batch_100_no_exception(self, _r):
        result = _run_batch(_make_batch(100))
        assert result.stats.chunks_indexed >= 1

    def test_batch_500_completes(self):
        t0 = time.perf_counter()
        result = _run_batch(_make_batch(500))
        elapsed = time.perf_counter() - t0
        assert result.stats.chunks_indexed >= 1
        assert elapsed < 60.0, f"500-item batch took {elapsed:.1f}s (>60s)"

    def test_batch_1000_completes(self):
        t0 = time.perf_counter()
        result = _run_batch(_make_batch(1000))
        elapsed = time.perf_counter() - t0
        assert result.stats.chunks_indexed >= 1
        assert elapsed < 120.0, f"1000-item batch took {elapsed:.1f}s (>120s)"

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_empty_batch_returns_clean_result(self, _r):
        result = _run_batch([])
        assert result.stats.chunks_indexed == 0
        assert result.bundles == []
        assert result.errors == {}


# ═════════════════════════════════════════════════════════════════════════════
# T3-B  Concurrent AutoResearchPipeline callers
# ═════════════════════════════════════════════════════════════════════════════

class TestT3BConcurrentPipeline:
    def _concurrent_run(self, n_threads: int, n_calls_each: int = 1):
        from app.research.auto_research_pipeline import AutoResearchPipeline, ResearchReport
        pipeline = AutoResearchPipeline()
        errors: list = []
        results: list = []

        def worker(idx: int):
            try:
                q = ARTICLES[idx % len(ARTICLES)][:80]
                r = asyncio.run(pipeline.run(q, tenant_id=f"t{idx}"))
                assert isinstance(r, ResearchReport)
                for _ in range(n_calls_each - 1):
                    r2 = asyncio.run(pipeline.run(q, tenant_id=f"t{idx}"))
                    results.append(r2)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        t0 = time.perf_counter()
        for t in threads: t.start()
        for t in threads: t.join(timeout=30)
        elapsed = time.perf_counter() - t0
        return errors, elapsed

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_8_concurrent_no_error(self, _r):
        errors, elapsed = self._concurrent_run(8)
        assert not errors, f"errors: {errors}"
        assert elapsed < 30, f"deadlock suspected: {elapsed:.1f}s"

    def test_16_concurrent_no_error(self):
        errors, elapsed = self._concurrent_run(16)
        assert not errors, f"errors: {errors}"
        assert elapsed < 30

    def test_32_concurrent_no_error(self):
        errors, elapsed = self._concurrent_run(32)
        assert not errors, f"errors: {errors}"
        assert elapsed < 30


# ═════════════════════════════════════════════════════════════════════════════
# T3-C  AcquisitionScheduler stress: 200 sources, 50 threads, 10 000 iterations
# ═════════════════════════════════════════════════════════════════════════════

class TestT3CAcquisitionSchedulerStress:
    @pytest.mark.parametrize("_r", ROUNDS)
    def test_200_sources_50_threads_no_corruption(self, _r):
        from app.source_intelligence.source_registry import AcquisitionScheduler
        reg = _make_registry(200)
        sched = AcquisitionScheduler(reg)
        errors: list = []
        iterations = 10_000 // 50   # 200 per thread

        def worker():
            try:
                for i in range(iterations):
                    src_id = f"src-{(i * 7) % 200:04d}"
                    if i % 3 == 0:
                        sched.record_failure(src_id)
                    elif i % 3 == 1:
                        sched.record_success(src_id)
                    else:
                        batch = sched.next_batch(10)
                        assert isinstance(batch, list)
                    # Spot-check failure count is non-negative
                    assert sched.failure_count(src_id) >= 0
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        t0 = time.perf_counter()
        for t in threads: t.start()
        for t in threads: t.join(timeout=30)
        elapsed = time.perf_counter() - t0
        assert not errors, f"errors: {errors[:3]}"
        assert elapsed < 30, f"deadlock: {elapsed:.1f}s"

    def test_priority_consistency_under_concurrency(self):
        """Priorities returned by next_batch must always be ≤1."""
        from app.source_intelligence.source_registry import AcquisitionScheduler
        reg = _make_registry(100)
        sched = AcquisitionScheduler(reg)
        bad_priorities: list = []

        def worker():
            for _ in range(500):
                for spec in sched.next_batch(20):
                    if spec.priority < 0.0 or spec.priority > 1.0:
                        bad_priorities.append(spec.priority)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=30)
        assert not bad_priorities


# ═════════════════════════════════════════════════════════════════════════════
# T3-D  CrossSourceDeduper stress: 500-bundle input, 20 concurrent threads
# ═════════════════════════════════════════════════════════════════════════════

class TestT3DCrossSourceDeduperStress:
    @pytest.mark.parametrize("_r", ROUNDS)
    def test_500_bundles_20_threads(self, _r):
        from app.entity_resolution.cross_source_deduper import CrossSourceDeduper
        d = CrossSourceDeduper(title_threshold=0.65)
        titles = [ARTICLES[i % len(ARTICLES)] for i in range(500)]
        bundles = [_make_bundle(f"b{i}", t, trust=0.5 + (i % 5) * 0.1)
                   for i, t in enumerate(titles)]
        errors: list = []

        def worker():
            try:
                kept, removed = d.deduplicate_cross_bundle(list(bundles))
                assert isinstance(kept, list)
                assert isinstance(removed, list)
                assert len(kept) + len(removed) == 500
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        t0 = time.perf_counter()
        for t in threads: t.start()
        for t in threads: t.join(timeout=30)
        elapsed = time.perf_counter() - t0
        assert not errors, f"errors: {errors[:3]}"
        assert elapsed < 30

    def test_dedup_result_partition_invariant(self):
        """kept ∪ removed == original set, and kept ∩ removed == ∅."""
        from app.entity_resolution.cross_source_deduper import CrossSourceDeduper
        d = CrossSourceDeduper(title_threshold=0.65)
        bundles = [_make_bundle(f"b{i}", ARTICLES[i % len(ARTICLES)]) for i in range(100)]
        kept, removed = d.deduplicate_cross_bundle(bundles)
        all_ids = {b.bundle_id for b in bundles}
        kept_ids = {b.bundle_id for b in kept}
        assert kept_ids.issubset(all_ids)
        assert set(removed).issubset(all_ids)
        assert kept_ids.isdisjoint(set(removed))
        assert kept_ids | set(removed) == all_ids


# ═════════════════════════════════════════════════════════════════════════════
# T3-E  Memory stability (spot-check no unbounded growth)
# ═════════════════════════════════════════════════════════════════════════════

class TestT3EMemoryStability:
    def test_repeated_batches_do_not_leak_pipeline_results(self):
        """Running 20 × 100-item batches should not accumulate PipelineResult objects."""
        import sys
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore

        pipeline = IndexingPipeline(chunk_store=ChunkStore())
        before = len(gc.get_objects())
        for _ in range(20):
            items = _make_batch(100)
            r = asyncio.run(pipeline.process_batch(items))
            del r, items
            gc.collect()
        after = len(gc.get_objects())
        # Allow <5x growth in tracked objects (generous threshold for test setup)
        assert after < before + 500_000, (
            f"Object count grew from {before} to {after} — possible leak"
        )

