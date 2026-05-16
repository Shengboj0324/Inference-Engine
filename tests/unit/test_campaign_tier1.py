"""Tier 1 — Unit / Functional Tests  (20+ rounds via @pytest.mark.parametrize)

Covers:
  C1  SourceSpec.priority + SourceRegistryStore.next_batch / update_priority
  C2  AcquisitionScheduler (back-off, trust gate, health monitor, thread safety)
  C3  MultimodalAnalyzer.to_evidence_sources
  C4  CrossSourceDeduper.deduplicate_cross_bundle
  C5  QualityGate / QualityGateResult
  C6  AutoResearchPipeline / ResearchReport
"""
from __future__ import annotations
import asyncio
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

ROUNDS = list(range(20))  # 20 parametrize IDs → 20 independent rounds


def _make_spec(sid: str, priority: float = 0.5):
    from app.source_intelligence.source_registry import SourceSpec, SourceFamily
    from app.core.models import SourcePlatform
    return SourceSpec(
        source_id=sid,
        platform=SourcePlatform.GITHUB_RELEASES,
        family=SourceFamily.DEVELOPER_RELEASE,
        priority=priority,
    )


def _filled_registry(n: int = 5):
    from app.source_intelligence.source_registry import SourceRegistryStore
    reg = SourceRegistryStore()
    priorities = [round(i / (n - 1), 3) for i in range(n)]
    for i, p in enumerate(priorities):
        reg.register(_make_spec(f"src-{i}", p))
    return reg


def _scheduler(registry=None, **kw):
    from app.source_intelligence.source_registry import AcquisitionScheduler, SourceRegistryStore
    if registry is None:
        registry = _filled_registry()
    return AcquisitionScheduler(registry, **kw)


def _obs(has_image=False, has_video=False):
    from app.domain.raw_models import RawObservation
    from app.core.models import SourcePlatform, MediaType
    meta = {}
    if has_image:
        meta["image_url"] = "https://cdn.example.com/img/announce.jpg"
    if has_video:
        meta["video_url"] = "https://cdn.example.com/vid/keynote.mp4"
    return RawObservation(
        user_id=uuid.uuid4(),
        source_id="post-001",
        source_url="https://example.com/post",
        title="AI Announcement",
        raw_text="OpenAI announces GPT-5 with breakthrough capabilities.",
        media_type=MediaType.IMAGE if has_image else (MediaType.VIDEO if has_video else MediaType.TEXT),
        source_platform=SourcePlatform.REDDIT,
        published_at=datetime.now(timezone.utc),
        platform_metadata=meta,
    )


def _bundle(bid: str, title: str, trust: float = 0.8):
    from app.entity_resolution.models import EventBundle
    return EventBundle(
        bundle_id=bid,
        canonical_title=title,
        primary_item_id=f"{bid}-item",
        source_items=[{"source_id": f"{bid}-item", "title": title,
                       "raw_text": title, "source_url": "https://x.com",
                       "platform": "reddit", "trust_score": trust}],
        trust_scores={f"{bid}-item": trust},
    )


def _gate(threshold: float = 0.60):
    from app.ingestion.indexing_pipeline import QualityGate
    return QualityGate(min_confidence=threshold)


def _mock_summary(score: float):
    s = MagicMock()
    s.confidence_score = score
    s.summary_id = f"s-{int(score * 100)}"
    return s


# ═════════════════════════════════════════════════════════════════════════════
# C1 – SourceSpec.priority + SourceRegistryStore
# ═════════════════════════════════════════════════════════════════════════════

class TestC1SourceSpecPriority:
    @pytest.mark.parametrize("_r", ROUNDS)
    def test_default_priority_is_0_5(self, _r):
        from app.source_intelligence.source_registry import SourceSpec, SourceFamily
        from app.core.models import SourcePlatform
        spec = SourceSpec(source_id="x", platform=SourcePlatform.GITHUB_RELEASES,
                          family=SourceFamily.RESEARCH)
        assert spec.priority == 0.5

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_boundary_priority_0(self, _r):
        spec = _make_spec("s", 0.0)
        assert spec.priority == 0.0

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_boundary_priority_1(self, _r):
        spec = _make_spec("s", 1.0)
        assert spec.priority == 1.0

    @pytest.mark.parametrize("val", [1.001, -0.001, 2.0, -1.0])
    def test_invalid_priority_raises(self, val):
        with pytest.raises(ValueError, match="priority"):
            _make_spec("s", val)

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_next_batch_sorted_descending(self, _r):
        reg = _filled_registry(10)
        batch = reg.next_batch(10)
        prios = [s.priority for s in batch]
        assert prios == sorted(prios, reverse=True)

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_next_batch_respects_n(self, _r):
        reg = _filled_registry(10)
        assert len(reg.next_batch(3)) == 3

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_next_batch_min_priority_exact(self, _r):
        reg = _filled_registry(10)
        batch = reg.next_batch(100, min_priority=0.5)
        assert all(s.priority >= 0.5 for s in batch)
        assert len(batch) >= 1



# ═════════════════════════════════════════════════════════════════════════════
# C2 – AcquisitionScheduler
# ═════════════════════════════════════════════════════════════════════════════

class TestC2AcquisitionScheduler:
    @pytest.mark.parametrize("_r", ROUNDS)
    def test_eligible_fresh_source(self, _r):
        sched = _scheduler()
        assert sched.is_eligible("src-0")

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_failure_blocks_source(self, _r):
        sched = _scheduler(base_backoff_s=60.0)
        sched.record_failure("src-0")
        assert not sched.is_eligible("src-0")
        assert sched.backoff_remaining_s("src-0") > 0.0

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_success_clears_backoff(self, _r):
        sched = _scheduler(base_backoff_s=60.0)
        sched.record_failure("src-0")
        sched.record_success("src-0")
        assert sched.is_eligible("src-0")
        assert sched.backoff_remaining_s("src-0") == 0.0
        assert sched.failure_count("src-0") == 0

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_max_retries_suspends_forever(self, _r):
        sched = _scheduler(base_backoff_s=0.001, max_retries=3)
        for _ in range(3):
            sched.record_failure("src-1")
        assert sched.failure_count("src-1") == 3
        assert not sched.is_eligible("src-1")

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_exponential_backoff_grows(self, _r):
        sched = _scheduler(base_backoff_s=60.0, max_retries=10)
        sched.record_failure("src-2")
        rem1 = sched.backoff_remaining_s("src-2")
        sched.record_success("src-2")
        sched.record_failure("src-2")
        sched.record_failure("src-2")
        rem2 = sched.backoff_remaining_s("src-2")
        # After 2 consecutive failures backoff = 60*2 = 120 > 60
        assert rem2 > rem1

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_health_monitor_failure_called(self, _r):
        mock_mon = MagicMock()
        sched = _scheduler(health_monitor=mock_mon)
        sched.record_failure("src-0")
        mock_mon.record_connector_failure.assert_called_once_with("src-0")

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_health_monitor_success_called(self, _r):
        mock_mon = MagicMock()
        sched = _scheduler(health_monitor=mock_mon)
        sched.record_success("src-0")
        mock_mon.record_connector_success.assert_called_once_with("src-0")

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_next_batch_excludes_blocked(self, _r):
        sched = _scheduler(base_backoff_s=60.0)
        sched.record_failure("src-4")  # highest priority = src-4 (priority=1.0)
        batch_ids = {s.source_id for s in sched.next_batch(10)}
        assert "src-4" not in batch_ids

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_broken_health_monitor_no_propagation(self, _r):
        bad = MagicMock()
        bad.record_connector_failure.side_effect = RuntimeError("boom")
        bad.record_connector_success.side_effect = RuntimeError("boom")
        sched = _scheduler(health_monitor=bad)
        sched.record_failure("src-0")  # must not raise
        sched.record_success("src-0")  # must not raise

    def test_constructor_bad_threshold_raises(self):
        from app.source_intelligence.source_registry import AcquisitionScheduler, SourceRegistryStore
        with pytest.raises(ValueError):
            AcquisitionScheduler(SourceRegistryStore(), min_authority_threshold=1.5)

    def test_constructor_bad_base_backoff_raises(self):
        from app.source_intelligence.source_registry import AcquisitionScheduler, SourceRegistryStore
        with pytest.raises(ValueError):
            AcquisitionScheduler(SourceRegistryStore(), base_backoff_s=0)

    def test_constructor_bad_max_retries_raises(self):
        from app.source_intelligence.source_registry import AcquisitionScheduler, SourceRegistryStore
        with pytest.raises(ValueError):
            AcquisitionScheduler(SourceRegistryStore(), max_retries=0)

    def test_thread_safety_200_iterations(self):
        sched = _scheduler()
        errors = []
        def worker():
            try:
                for _ in range(200):
                    sched.record_failure("src-2")
                    sched.failure_count("src-2")
                    sched.record_success("src-2")
                    sched.next_batch(5)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=30)
        assert not errors


# ═════════════════════════════════════════════════════════════════════════════
# C3 – MultimodalAnalyzer.to_evidence_sources
# ═════════════════════════════════════════════════════════════════════════════

class TestC3MultimodalEvidenceSources:
    @pytest.mark.parametrize("_r", ROUNDS)
    def test_image_returns_one_source(self, _r):
        from app.intelligence.multimodal import MultimodalAnalyzer
        sources = MultimodalAnalyzer().to_evidence_sources(_obs(has_image=True))
        assert len(sources) == 1
        assert sources[0]["modality"] == "image"

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_video_returns_one_source(self, _r):
        from app.intelligence.multimodal import MultimodalAnalyzer
        sources = MultimodalAnalyzer().to_evidence_sources(_obs(has_video=True))
        assert len(sources) == 1
        assert sources[0]["modality"] == "video"

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_no_media_returns_empty(self, _r):
        from app.intelligence.multimodal import MultimodalAnalyzer
        assert MultimodalAnalyzer().to_evidence_sources(_obs()) == []

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_required_keys_present(self, _r):
        from app.intelligence.multimodal import MultimodalAnalyzer
        src = MultimodalAnalyzer().to_evidence_sources(_obs(has_image=True))[0]
        for key in ("source_id", "title", "url", "platform", "trust_score", "content_snippet", "modality"):
            assert key in src

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_trust_score_in_range(self, _r):
        from app.intelligence.multimodal import MultimodalAnalyzer
        src = MultimodalAnalyzer().to_evidence_sources(_obs(has_image=True))[0]
        assert 0.0 <= src["trust_score"] <= 1.0

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_source_id_unique_per_url(self, _r):
        from app.intelligence.multimodal import MultimodalAnalyzer
        from app.domain.raw_models import RawObservation
        from app.core.models import SourcePlatform, MediaType
        def obs_url(url):
            return RawObservation(
                user_id=uuid.uuid4(), source_id="p", source_url="https://x.com",
                title="T", raw_text="T", media_type=MediaType.IMAGE,
                source_platform=SourcePlatform.REDDIT,
                published_at=datetime.now(timezone.utc),
                platform_metadata={"image_url": url},
            )
        a = MultimodalAnalyzer().to_evidence_sources(obs_url("https://a.com/a.jpg"))
        b = MultimodalAnalyzer().to_evidence_sources(obs_url("https://b.com/b.jpg"))
        assert a[0]["source_id"] != b[0]["source_id"]

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_disabled_mode_returns_empty(self, _r):
        from app.intelligence.multimodal import MultimodalAnalyzer, CapabilityMode
        a = MultimodalAnalyzer(execution_mode=CapabilityMode.DISABLED)
        assert a.to_evidence_sources(_obs(has_image=True)) == []


# ═════════════════════════════════════════════════════════════════════════════
# C4 – CrossSourceDeduper.deduplicate_cross_bundle
# ═════════════════════════════════════════════════════════════════════════════

class TestC4CrossBundleDedup:
    def _deduper(self, threshold=0.65):
        from app.entity_resolution.cross_source_deduper import CrossSourceDeduper
        return CrossSourceDeduper(title_threshold=threshold)

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_identical_title_collapses_to_one(self, _r):
        d = self._deduper()
        b1 = _bundle("b1", "OpenAI announces GPT-5 language model release today", trust=0.9)
        b2 = _bundle("b2", "OpenAI announces GPT-5 language model release today", trust=0.5)
        kept, removed = d.deduplicate_cross_bundle([b1, b2])
        assert len(kept) == 1
        assert kept[0].bundle_id == "b1"
        assert "b2" in removed

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_distinct_titles_both_kept(self, _r):
        d = self._deduper()
        b1 = _bundle("b1", "OpenAI GPT-5 reasoning breakthrough capability")
        b2 = _bundle("b2", "NVIDIA Blackwell H100 GPU supply chain shortage")
        kept, removed = d.deduplicate_cross_bundle([b1, b2])
        assert len(kept) == 2
        assert removed == []

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_empty_list_returns_empty(self, _r):
        kept, removed = self._deduper().deduplicate_cross_bundle([])
        assert kept == [] and removed == []

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_single_bundle_passthrough(self, _r):
        b = _bundle("only", "AlphaFold 3 protein structure drug discovery")
        kept, removed = self._deduper().deduplicate_cross_bundle([b])
        assert len(kept) == 1 and removed == []

    def test_wrong_type_raises(self):
        from app.entity_resolution.cross_source_deduper import CrossSourceDeduper
        with pytest.raises(TypeError):
            CrossSourceDeduper().deduplicate_cross_bundle("not-a-list")

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_higher_trust_wins(self, _r):
        d = self._deduper()
        lo = _bundle("lo", "Anthropic raises two billion dollar funding round today", trust=0.1)
        hi = _bundle("hi", "Anthropic raises two billion dollar funding round today", trust=0.99)
        kept, removed = d.deduplicate_cross_bundle([lo, hi])
        assert kept[0].bundle_id == "hi"

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_return_types(self, _r):
        kept, removed = self._deduper().deduplicate_cross_bundle([_bundle("a", "X"), _bundle("b", "Y")])
        assert isinstance(kept, list)
        assert isinstance(removed, list)


# ═════════════════════════════════════════════════════════════════════════════
# C5 – QualityGate / QualityGateResult
# ═════════════════════════════════════════════════════════════════════════════

class TestC5QualityGate:
    @pytest.mark.parametrize("_r", ROUNDS)
    def test_above_threshold_passes(self, _r):
        r = _gate(0.60).evaluate(_mock_summary(0.80), "AI")
        assert r.passed is True
        assert r.rejection_reason is None
        assert r.confidence_score == pytest.approx(0.80)

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_below_threshold_fails(self, _r):
        r = _gate(0.60).evaluate(_mock_summary(0.40), "AI")
        assert r.passed is False
        assert "below threshold" in r.rejection_reason
        assert r.confidence_score == pytest.approx(0.40)

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_exactly_at_threshold_passes(self, _r):
        r = _gate(0.60).evaluate(_mock_summary(0.60), "AI")
        assert r.passed is True

    @pytest.mark.parametrize("score", [0.0, 0.001, 0.999, 1.0])
    def test_boundary_scores(self, score):
        r = _gate(0.0).evaluate(_mock_summary(score), "AI")
        assert r.passed is True  # gate at 0.0 → everything passes
        assert 0.0 <= r.confidence_score <= 1.0

    def test_bad_threshold_raises(self):
        from app.ingestion.indexing_pipeline import QualityGate
        with pytest.raises(ValueError):
            QualityGate(min_confidence=1.001)
        with pytest.raises(ValueError):
            QualityGate(min_confidence=-0.001)

    def test_wrong_summary_type_raises(self):
        with pytest.raises(TypeError):
            _gate().evaluate("not-a-summary", "topic")

    def test_empty_topic_raises(self):
        with pytest.raises(TypeError):
            _gate().evaluate(_mock_summary(0.7), "")

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_evaluate_batch_order_preserved(self, _r):
        scores = [0.9, 0.3, 0.7, 0.1, 0.6]
        results = _gate(0.60).evaluate_batch([_mock_summary(s) for s in scores], "AI")
        assert len(results) == 5
        for i, (s, r) in enumerate(zip(scores, results)):
            assert r.confidence_score == pytest.approx(s), f"index {i} mismatch"

    def test_evaluate_batch_wrong_type_raises(self):
        from app.ingestion.indexing_pipeline import QualityGate
        with pytest.raises(TypeError):
            QualityGate().evaluate_batch("not-a-list", "AI")


# ═════════════════════════════════════════════════════════════════════════════
# C6 – AutoResearchPipeline / ResearchReport
# ═════════════════════════════════════════════════════════════════════════════

class TestC6AutoResearchPipeline:
    def _run(self, query="AI safety alignment", tenant_id="default", **kw):
        from app.research.auto_research_pipeline import AutoResearchPipeline
        p = AutoResearchPipeline(**kw)
        return asyncio.run(p.run(query, tenant_id=tenant_id))

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_returns_research_report(self, _r):
        from app.research.auto_research_pipeline import ResearchReport
        r = self._run()
        assert isinstance(r, ResearchReport)

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_query_stored(self, _r):
        r = self._run(query="protein folding AlphaFold")
        assert r.query == "protein folding AlphaFold"

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_tenant_stored(self, _r):
        r = self._run(tenant_id="acme-corp")
        assert r.tenant_id == "acme-corp"

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_wall_s_positive(self, _r):
        r = self._run()
        assert r.wall_s > 0.0

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_chunks_non_negative(self, _r):
        r = self._run()
        assert r.chunks_indexed >= 0

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_confidence_in_range(self, _r):
        r = self._run()
        assert 0.0 <= r.confidence_score_mean <= 1.0

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_rejection_rate_in_range(self, _r):
        r = self._run()
        assert 0.0 <= r.quality_gate_rejection_rate <= 1.0

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_gate_outcomes_is_list(self, _r):
        r = self._run()
        assert isinstance(r.quality_gate_outcomes, list)

    def test_empty_query_raises(self):
        from app.research.auto_research_pipeline import AutoResearchPipeline
        with pytest.raises(ValueError):
            asyncio.run(AutoResearchPipeline().run(""))

    def test_empty_tenant_raises(self):
        from app.research.auto_research_pipeline import AutoResearchPipeline
        with pytest.raises(ValueError):
            asyncio.run(AutoResearchPipeline().run("AI", tenant_id=""))

    def test_bad_threshold_raises(self):
        from app.research.auto_research_pipeline import AutoResearchPipeline
        with pytest.raises(ValueError):
            AutoResearchPipeline(min_confidence_threshold=2.0)

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_generated_at_is_utc(self, _r):
        r = self._run()
        assert r.generated_at.tzinfo is not None

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_broken_pipeline_no_raise(self, _r):
        from unittest.mock import AsyncMock
        from app.research.auto_research_pipeline import AutoResearchPipeline, ResearchReport
        bad = MagicMock()
        bad.process_batch = AsyncMock(side_effect=RuntimeError("down"))
        bad.build_grounded_summary = MagicMock(return_value=None)
        bad._store = MagicMock()
        bad._store.get_by_observation.return_value = []
        p = AutoResearchPipeline(pipeline=bad)
        r = asyncio.run(p.run("AI safety"))
        assert isinstance(r, ResearchReport)

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_update_priority_returns_true(self, _r):
        reg = _filled_registry(5)
        assert reg.update_priority("src-0", 0.1) is True
        assert abs(reg.get("src-0").priority - 0.1) < 1e-9

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_update_priority_missing_returns_false(self, _r):
        reg = _filled_registry(5)
        assert reg.update_priority("ghost", 0.5) is False

    @pytest.mark.parametrize("bad", [1.5, -0.1])
    def test_update_priority_bad_range_raises(self, bad):
        reg = _filled_registry(5)
        with pytest.raises(ValueError):
            reg.update_priority("src-0", bad)

    def test_next_batch_n_zero_raises(self):
        with pytest.raises(ValueError):
            _filled_registry().next_batch(0)

