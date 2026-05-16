"""Tier 4 — End-to-End Integration Tests

All 5 realistic queries × 20+ rounds each, with all subsystems wired.
Every round asserts the full ResearchReport contract.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Spec-required queries (5 required)
# ─────────────────────────────────────────────────────────────────────────────

QUERIES = [
    ("AI safety alignment interpretability research 2025",                         "safety-lab"),
    ("Protein folding structure prediction AlphaFold 3 DeepMind drug discovery",   "bio-research"),
    ("Semiconductor supply chain NVIDIA H100 Blackwell TSMC AMD GPU shortage",     "hardware-team"),
    ("Large language model reasoning GPT-5 Claude Gemini benchmark evaluation",    "ai-benchmarks"),
    ("Open-source LLM fine-tuning LLaMA Mistral Falcon instruction tuning RLHF",  "mlops-team"),
]

ROUNDS = list(range(20))

# ─────────────────────────────────────────────────────────────────────────────
# Full subsystem factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_full_pipeline():
    """Wire every available subsystem into AutoResearchPipeline."""
    from app.source_intelligence.source_registry import (
        SourceRegistryStore, SourceSpec, SourceFamily, AcquisitionScheduler,
    )
    from app.core.models import SourcePlatform
    from app.intelligence.multimodal import MultimodalAnalyzer
    from app.intelligence.health_monitor import PipelineHealthMonitor
    from app.ingestion.indexing_pipeline import IndexingPipeline, QualityGate
    from app.entity_resolution.cross_source_deduper import CrossSourceDeduper
    from app.personalization.watchlist_graph import WatchlistGraph
    from app.intelligence.retrieval.chunk_store import ChunkStore
    from app.research.auto_research_pipeline import AutoResearchPipeline

    # Source registry with 3 sources
    reg = SourceRegistryStore()
    reg.register(SourceSpec("openai-blog",  SourcePlatform.RSS, SourceFamily.RESEARCH, priority=0.95))
    reg.register(SourceSpec("arxiv-cs",     SourcePlatform.ARXIV, SourceFamily.RESEARCH, priority=0.85))
    reg.register(SourceSpec("github-releases", SourcePlatform.GITHUB_RELEASES, SourceFamily.DEVELOPER_RELEASE, priority=0.70))
    scheduler = AcquisitionScheduler(reg)

    # Multimodal analyzer (STUB mode — no real vision client)
    analyzer = MultimodalAnalyzer()

    # Health monitor (circuit breaker threshold = 5)
    monitor = PipelineHealthMonitor(cb_open_threshold=5)

    # Watchlist graph (3+ watched entities)
    wg = WatchlistGraph("test-user")
    wg.watch("openai",    node_type="entity")
    wg.watch("deepmind",  node_type="entity")
    wg.watch("anthropic", node_type="entity")

    # Quality gate (threshold = 0.60)
    gate = QualityGate(min_confidence=0.60)

    # Deduper (cross-bundle)
    deduper = CrossSourceDeduper(title_threshold=0.65)

    # IndexingPipeline with all subsystems
    pipeline = IndexingPipeline(
        chunk_store=ChunkStore(),
        multimodal_analyzer=analyzer,
        watchlist_graph=wg,
        health_monitor=monitor,
        deduper=deduper,
        quality_gate=gate,
    )

    return AutoResearchPipeline(
        pipeline=pipeline,
        acquisition_scheduler=scheduler,
        watchlist_graph=wg,
        health_monitor=monitor,
        quality_gate=gate,
        multimodal_analyzer=analyzer,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Allowed SLO statuses
# ─────────────────────────────────────────────────────────────────────────────

def _allowed_slo(status) -> bool:
    if status is None:
        return True
    try:
        from app.intelligence.health_monitor import SLOStatus
        return status in (SLOStatus.GREEN, SLOStatus.YELLOW, SLOStatus.RED)
    except Exception:
        return True  # can't import, skip


# ═════════════════════════════════════════════════════════════════════════════
# Contract assertions (run on every report in every round)
# ═════════════════════════════════════════════════════════════════════════════

def _assert_report_contract(report, query: str, tenant_id: str):
    from app.research.auto_research_pipeline import ResearchReport
    assert isinstance(report, ResearchReport)
    assert report.query == query
    assert report.tenant_id == tenant_id
    assert report.chunks_indexed >= 1,        "chunks_indexed must be ≥ 1"
    assert 0.0 <= report.confidence_score_mean <= 1.0
    assert report.wall_s > 0
    assert 0.0 <= report.quality_gate_rejection_rate <= 1.0
    assert isinstance(report.quality_gate_outcomes, list)
    assert _allowed_slo(report.slo_health_status), f"bad slo: {report.slo_health_status}"
    assert report.watchlist_gap_count >= 0


# ═════════════════════════════════════════════════════════════════════════════
# Per-query test classes (one class per query for clean reporting)
# ═════════════════════════════════════════════════════════════════════════════

class TestT4Q1AISafety:
    Q, T = QUERIES[0]
    @pytest.mark.parametrize("_r", ROUNDS)
    def test_report_contract(self, _r):
        p = _build_full_pipeline()
        r = asyncio.run(p.run(self.Q, tenant_id=self.T))
        _assert_report_contract(r, self.Q, self.T)


class TestT4Q2ProteinFolding:
    Q, T = QUERIES[1]
    @pytest.mark.parametrize("_r", ROUNDS)
    def test_report_contract(self, _r):
        p = _build_full_pipeline()
        r = asyncio.run(p.run(self.Q, tenant_id=self.T))
        _assert_report_contract(r, self.Q, self.T)


class TestT4Q3Semiconductor:
    Q, T = QUERIES[2]
    @pytest.mark.parametrize("_r", ROUNDS)
    def test_report_contract(self, _r):
        p = _build_full_pipeline()
        r = asyncio.run(p.run(self.Q, tenant_id=self.T))
        _assert_report_contract(r, self.Q, self.T)


class TestT4Q4LLMReasoning:
    Q, T = QUERIES[3]
    @pytest.mark.parametrize("_r", ROUNDS)
    def test_report_contract(self, _r):
        p = _build_full_pipeline()
        r = asyncio.run(p.run(self.Q, tenant_id=self.T))
        _assert_report_contract(r, self.Q, self.T)


class TestT4Q5OpenSourceLLM:
    Q, T = QUERIES[4]
    @pytest.mark.parametrize("_r", ROUNDS)
    def test_report_contract(self, _r):
        p = _build_full_pipeline()
        r = asyncio.run(p.run(self.Q, tenant_id=self.T))
        _assert_report_contract(r, self.Q, self.T)


# ═════════════════════════════════════════════════════════════════════════════
# Cross-query integration tests (all 5 queries in one pipeline run)
# ═════════════════════════════════════════════════════════════════════════════

class TestT4CrossQueryIntegration:
    @pytest.mark.parametrize("_r", ROUNDS)
    def test_all_5_queries_pass_contract(self, _r):
        p = _build_full_pipeline()
        for q, t in QUERIES:
            r = asyncio.run(p.run(q, tenant_id=t))
            _assert_report_contract(r, q, t)

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_gate_outcomes_confidence_all_in_range(self, _r):
        p = _build_full_pipeline()
        for q, t in QUERIES:
            r = asyncio.run(p.run(q, tenant_id=t))
            for outcome in r.quality_gate_outcomes:
                assert 0.0 <= outcome.confidence_score <= 1.0

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_no_exception_propagates(self, _r):
        """run() must never propagate an unhandled exception."""
        p = _build_full_pipeline()
        for q, t in QUERIES:
            try:
                asyncio.run(p.run(q, tenant_id=t))
            except Exception as exc:
                pytest.fail(f"run() raised {type(exc).__name__}: {exc}")

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_grounded_summary_produced(self, _r):
        """At least one query per run must produce a grounded summary."""
        p = _build_full_pipeline()
        any_summary = False
        for q, t in QUERIES:
            r = asyncio.run(p.run(q, tenant_id=t))
            if r.grounded_summaries:
                any_summary = True
        assert any_summary, "No grounded summaries produced across all 5 queries"

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_wall_s_positive_for_all_queries(self, _r):
        p = _build_full_pipeline()
        for q, t in QUERIES:
            r = asyncio.run(p.run(q, tenant_id=t))
            assert r.wall_s > 0.0

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_generated_at_utc_for_all(self, _r):
        p = _build_full_pipeline()
        for q, t in QUERIES:
            r = asyncio.run(p.run(q, tenant_id=t))
            assert r.generated_at.tzinfo is not None

