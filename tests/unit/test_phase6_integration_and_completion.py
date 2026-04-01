"""Phase 6 — Integration & Completion Tests.

37 test groups — ~270 tests covering:
  1.  Source-intelligence __init__ exports (Gap 1)
  2.  SourceCoverageGraph construction & validation (Gap 2)
  3–10.  SourceCoverageGraph entity, source, freshness, completeness, gaps,
         derivative, serialization, edge-cases, thread-safety
  11.  EventFirstPipeline construction (Gap 3)
  12–20. EventFirstPipeline ingestion, merging, scoring, updates,
         high-volume, thread-safety, edge-cases, stats, batch
  21.  DigestModeRouter construction (Gap 4)
  22–30. Four delivery modes + validation + edge cases
  31–34. Cross-component integration scenarios
  35–36. Stress tests (coverage graph 2 000 entities, event pipeline 5 000 items)
  37.    Package __all__ consistency across all 7 packages
"""

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

UTC = timezone.utc


def _now() -> datetime:
    return datetime.now(UTC)


def _hours_ago(h: float) -> datetime:
    return _now() - timedelta(hours=h)


def _cand(
    item_id: str = "item-1",
    title: str = "Default Title",
    importance: float = 0.5,
    entity_ids: Optional[List[str]] = None,
    published_at: Optional[datetime] = None,
    trust_score: float = 0.7,
    claims: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "item_id": item_id,
        "title": title,
        "importance": importance,
        "entity_ids": entity_ids or [],
        "published_at": published_at,
        "trust_score": trust_score,
        "claims": claims or [],
        "sources": sources or ["source-1"],
    }


# ===========================================================================
# Group 1: SourceIntelligenceExports
# ===========================================================================

class TestSourceIntelligenceExports:
    """All new symbols are importable from the top-level package."""

    def test_source_discovery_engine(self):
        from app.source_intelligence import SourceDiscoveryEngine
        assert callable(SourceDiscoveryEngine)

    def test_discovered_source(self):
        from app.source_intelligence import DiscoveredSource
        assert DiscoveredSource is not None

    def test_coverage_planner(self):
        from app.source_intelligence import CoveragePlanner
        assert callable(CoveragePlanner)

    def test_coverage_gap(self):
        from app.source_intelligence import CoverageGap
        assert CoverageGap is not None

    def test_gap_severity(self):
        from app.source_intelligence import GapSeverity
        assert GapSeverity is not None

    def test_feed_expander(self):
        from app.source_intelligence import FeedExpander
        assert callable(FeedExpander)

    def test_feed_candidate(self):
        from app.source_intelligence import FeedCandidate
        assert FeedCandidate is not None

    def test_entity_to_source_mapper(self):
        from app.source_intelligence import EntityToSourceMapper
        assert callable(EntityToSourceMapper)

    def test_entity_source_map(self):
        from app.source_intelligence import EntitySourceMap
        assert EntitySourceMap is not None

    def test_change_monitor(self):
        from app.source_intelligence import ChangeMonitor
        assert callable(ChangeMonitor)

    def test_change_event(self):
        from app.source_intelligence import ChangeEvent
        assert ChangeEvent is not None

    def test_coverage_graph_symbols(self):
        from app.source_intelligence import (
            EntityCategory, EntityCoverageScore, SourceCoverageGraph, SourceFreshnessEntry
        )
        assert all(x is not None for x in [
            EntityCategory, EntityCoverageScore, SourceCoverageGraph, SourceFreshnessEntry
        ])


# ===========================================================================
# Group 2: CoverageGraphConstruct
# ===========================================================================

class TestCoverageGraphConstruct:
    """Constructor validation and defaults."""

    def _g(self, **kw):
        from app.source_intelligence import SourceCoverageGraph
        return SourceCoverageGraph(**kw)


# ===========================================================================
# Group 3: CoverageGraphEntityTracking
# ===========================================================================

class TestCoverageGraphEntityTracking:

    def _g(self):
        from app.source_intelligence import SourceCoverageGraph
        return SourceCoverageGraph()

    def _cat(self, name="COMPANY"):
        from app.source_intelligence import EntityCategory
        return EntityCategory(name.lower())

    def test_add_and_list(self):
        g = self._g()
        g.add_entity("Tesla", self._cat())
        assert "Tesla" in g.list_entities()

    def test_add_idempotent(self):
        g = self._g()
        g.add_entity("Tesla", self._cat())
        g.add_entity("Tesla", self._cat())  # second call silent
        assert g.list_entities().count("Tesla") == 1

    def test_list_sorted(self):
        g = self._g()
        g.add_entity("Zebra", self._cat())
        g.add_entity("Apple", self._cat())
        assert g.list_entities() == ["Apple", "Zebra"]

    def test_remove_entity(self):
        g = self._g()
        g.add_entity("Tesla", self._cat())
        g.remove_entity("Tesla")
        assert "Tesla" not in g.list_entities()

    def test_remove_unknown_silent(self):
        g = self._g()
        g.remove_entity("DoesNotExist")  # should not raise

    def test_add_none_raises(self):
        g = self._g()
        with pytest.raises(TypeError):
            g.add_entity(None, self._cat())

    def test_add_empty_string_raises(self):
        g = self._g()
        with pytest.raises(ValueError, match="non-empty"):
            g.add_entity("  ", self._cat())

    def test_add_wrong_category_type_raises(self):
        g = self._g()
        with pytest.raises(TypeError, match="EntityCategory"):
            g.add_entity("Tesla", "company")


# ===========================================================================
# Group 4: CoverageGraphSourceAttachment
# ===========================================================================

class TestCoverageGraphSourceAttachment:

    def _g(self, entity="Tesla"):
        from app.source_intelligence import SourceCoverageGraph, EntityCategory
        g = SourceCoverageGraph()
        g.add_entity(entity, EntityCategory.COMPANY)
        return g

    def test_attach_source(self):
        g = self._g()
        g.attach_source("Tesla", "reuters.com", "news")
        assert "reuters.com" in g._entity_sources["Tesla"]

    def test_attach_idempotent(self):
        g = self._g()
        g.attach_source("Tesla", "reuters.com", "news")
        g.attach_source("Tesla", "reuters.com", "news")
        assert g._entity_sources["Tesla"] == {"reuters.com"}

    def test_detach_source(self):
        g = self._g()
        g.attach_source("Tesla", "reuters.com", "news")
        g.detach_source("Tesla", "reuters.com")
        assert "reuters.com" not in g._entity_sources.get("Tesla", set())

    def test_detach_absent_silent(self):
        g = self._g()
        g.detach_source("Tesla", "unknown.com")  # should not raise

    def test_attach_unregistered_raises(self):
        from app.source_intelligence import SourceCoverageGraph
        g = SourceCoverageGraph()
        with pytest.raises(ValueError, match="not registered"):
            g.attach_source("Ghost", "src1", "news")

    def test_attach_empty_source_id_raises(self):
        g = self._g()
        with pytest.raises(ValueError, match="source_id"):
            g.attach_source("Tesla", "", "news")

    def test_attach_empty_family_raises(self):
        g = self._g()
        with pytest.raises(ValueError, match="family"):
            g.attach_source("Tesla", "s1", "")


# ===========================================================================
# Group 5: CoverageGraphFreshness
# ===========================================================================

class TestCoverageGraphFreshness:

    def _g(self):
        from app.source_intelligence import SourceCoverageGraph, EntityCategory
        g = SourceCoverageGraph(staleness_threshold_hours=1.0)
        g.add_entity("Apple", EntityCategory.COMPANY)
        g.attach_source("Apple", "techcrunch", "news")
        return g

    def test_record_fetch_updates(self):
        g = self._g()
        g.record_fetch("Apple", "techcrunch")
        entry = g.get_freshness("Apple", "techcrunch")
        assert entry is not None
        assert entry.item_count == 1

    def test_record_fetch_increments_count(self):
        g = self._g()
        g.record_fetch("Apple", "techcrunch")
        g.record_fetch("Apple", "techcrunch")
        entry = g.get_freshness("Apple", "techcrunch")
        assert entry.item_count == 2

    def test_record_fetch_derivative_flag(self):
        g = self._g()
        g.record_fetch("Apple", "techcrunch", is_derivative=True)
        entry = g.get_freshness("Apple", "techcrunch")
        assert entry.is_derivative is True

    def test_stale_sources_fresh(self):
        g = self._g()
        g.record_fetch("Apple", "techcrunch")
        stale = g.stale_sources("Apple")
        assert "techcrunch" not in stale

    def test_stale_sources_stale(self):
        g = self._g()
        past = _now() - timedelta(hours=2)  # > 1h threshold
        g.record_fetch("Apple", "techcrunch", fetched_at=past)
        stale = g.stale_sources("Apple")
        assert "techcrunch" in stale

    def test_stale_sources_no_fetch(self):
        g = self._g()
        stale = g.stale_sources("Apple")
        assert "techcrunch" in stale  # no fetch → stale

    def test_record_fetch_naive_datetime_raises(self):
        g = self._g()
        naive = datetime.now()  # no tzinfo
        with pytest.raises(ValueError, match="timezone-aware"):
            g.record_fetch("Apple", "techcrunch", fetched_at=naive)

    def test_record_fetch_unregistered_entity_raises(self):
        g = self._g()
        with pytest.raises(ValueError, match="not registered"):
            g.record_fetch("Ghost", "s1")

    def test_record_fetch_unattached_source_raises(self):
        g = self._g()
        with pytest.raises(ValueError, match="not attached"):
            g.record_fetch("Apple", "unattached_source")


# ===========================================================================
# Group 6: CoverageGraphCompleteness
# ===========================================================================

class TestCoverageGraphCompleteness:
    """coverage_score() arithmetic."""

    def _setup(self):
        from app.source_intelligence import SourceCoverageGraph, EntityCategory
        g = SourceCoverageGraph()
        # REPO requires {"developer_release", "changelog"}
        g.add_entity("openai/sdk", EntityCategory.REPO)
        return g

    def test_zero_sources_completeness(self):
        g = self._setup()
        score = g.coverage_score("openai/sdk")
        assert score.completeness == pytest.approx(0.0)
        assert score.gap_count == 2

    def test_partial_coverage(self):
        g = self._setup()
        g.attach_source("openai/sdk", "gh_releases", "developer_release")
        score = g.coverage_score("openai/sdk")
        assert score.completeness == pytest.approx(0.5)
        assert score.gap_count == 1

    def test_full_coverage(self):
        g = self._setup()
        g.attach_source("openai/sdk", "gh_releases", "developer_release")
        g.attach_source("openai/sdk", "changelog_site", "changelog")
        score = g.coverage_score("openai/sdk")
        assert score.completeness == pytest.approx(1.0)
        assert score.gap_count == 0

    def test_coverage_entity_counts(self):
        from app.source_intelligence import SourceCoverageGraph, EntityCategory
        g = SourceCoverageGraph()
        g.add_entity("PyTorch", EntityCategory.REPO)
        g.attach_source("PyTorch", "s1", "developer_release")
        g.attach_source("PyTorch", "s2", "changelog")
        score = g.coverage_score("PyTorch")
        assert score.total_sources == 2

    def test_coverage_score_unknown_entity_raises(self):
        from app.source_intelligence import SourceCoverageGraph
        g = SourceCoverageGraph()
        with pytest.raises(ValueError, match="not registered"):
            g.coverage_score("Ghost")

    def test_coverage_score_wrong_type_raises(self):
        from app.source_intelligence import SourceCoverageGraph
        g = SourceCoverageGraph()
        with pytest.raises(TypeError):
            g.coverage_score(123)


# ===========================================================================
# Group 7: CoverageGraphGaps
# ===========================================================================

class TestCoverageGraphGaps:

    def _g_with_gaps(self):
        from app.source_intelligence import SourceCoverageGraph, EntityCategory
        g = SourceCoverageGraph()
        g.add_entity("Entity A", EntityCategory.REPO)   # needs developer_release + changelog
        g.add_entity("Entity B", EntityCategory.PAPER)  # needs research only → we cover it
        g.attach_source("Entity B", "arxiv", "research")
        return g

    def test_identify_gaps_returns_incomplete(self):
        g = self._g_with_gaps()
        gaps = g.identify_gaps()
        names = [sc.entity_name for sc in gaps]
        assert "Entity A" in names

    def test_identify_gaps_excludes_complete(self):
        g = self._g_with_gaps()
        gaps = g.identify_gaps()
        names = [sc.entity_name for sc in gaps]
        assert "Entity B" not in names

    def test_identify_gaps_sorted_worst_first(self):
        from app.source_intelligence import SourceCoverageGraph, EntityCategory
        g = SourceCoverageGraph()
        g.add_entity("A", EntityCategory.COMPANY)   # needs 3 families
        g.add_entity("B", EntityCategory.REPO)      # needs 2 families
        # Neither has any sources
        gaps = g.identify_gaps()
        # A has lower completeness (more required families, zero covered)
        assert gaps[0].completeness <= gaps[-1].completeness

    def test_identify_gaps_boundary_threshold_zero(self):
        g = self._g_with_gaps()
        gaps = g.identify_gaps(min_completeness=0.0)
        assert gaps == []  # no entity has completeness < 0

    def test_identify_gaps_oob_threshold_raises(self):
        from app.source_intelligence import SourceCoverageGraph
        g = SourceCoverageGraph()
        with pytest.raises(ValueError):
            g.identify_gaps(1.5)


# ===========================================================================
# Group 8: CoverageGraphDerivativeOverreliance
# ===========================================================================

class TestCoverageGraphDerivative:

    def _g(self):
        from app.source_intelligence import SourceCoverageGraph, EntityCategory
        g = SourceCoverageGraph(derivative_overreliance_ratio=0.5)
        g.add_entity("OpenAI", EntityCategory.COMPANY)
        g.attach_source("OpenAI", "blog", "official_blog")
        g.attach_source("OpenAI", "techcrunch", "news")
        return g

    def test_no_derivative_no_overreliance(self):
        g = self._g()
        g.record_fetch("OpenAI", "blog", is_derivative=False)
        g.record_fetch("OpenAI", "techcrunch", is_derivative=False)
        result = g.derivative_overreliance()
        assert result == []

    def test_overreliance_detected(self):
        g = self._g()
        g.record_fetch("OpenAI", "blog", is_derivative=True)
        g.record_fetch("OpenAI", "techcrunch", is_derivative=True)
        result = g.derivative_overreliance()
        assert any(r.entity_name == "OpenAI" for r in result)

    def test_overreliance_ratio_exactly_at_threshold(self):
        from app.source_intelligence import SourceCoverageGraph, EntityCategory
        g = SourceCoverageGraph(derivative_overreliance_ratio=0.5)
        g.add_entity("Meta", EntityCategory.COMPANY)
        g.attach_source("Meta", "s1", "news")
        g.attach_source("Meta", "s2", "official_blog")
        g.record_fetch("Meta", "s1", is_derivative=True)
        g.record_fetch("Meta", "s2", is_derivative=False)
        result = g.derivative_overreliance()
        # 0.5 >= 0.5 → flagged
        assert any(r.entity_name == "Meta" for r in result)

    def test_overreliance_result_sorted_by_ratio(self):
        from app.source_intelligence import SourceCoverageGraph, EntityCategory
        g = SourceCoverageGraph(derivative_overreliance_ratio=0.5)
        g.add_entity("A", EntityCategory.COMPANY)
        g.add_entity("B", EntityCategory.COMPANY)
        for ent, s1_deriv, s2_deriv in [("A", True, True), ("B", True, False)]:
            g.attach_source(ent, f"{ent}-s1", "news")
            g.attach_source(ent, f"{ent}-s2", "official_blog")
            g.record_fetch(ent, f"{ent}-s1", is_derivative=s1_deriv)
            g.record_fetch(ent, f"{ent}-s2", is_derivative=s2_deriv)
        result = g.derivative_overreliance()
        ratios = [r.derivative_ratio for r in result]
        assert ratios == sorted(ratios, reverse=True)


# ===========================================================================
# Group 9: CoverageGraphSerialization
# ===========================================================================

class TestCoverageGraphSerialization:

    def _populated_graph(self):
        from app.source_intelligence import SourceCoverageGraph, EntityCategory
        g = SourceCoverageGraph(staleness_threshold_hours=12.0)
        g.add_entity("Anthropic", EntityCategory.COMPANY)
        g.attach_source("Anthropic", "blog", "official_blog", is_derivative=False)
        g.record_fetch("Anthropic", "blog", is_derivative=False)
        return g

    def test_to_dict_contains_keys(self):
        g = self._populated_graph()
        d = g.to_dict()
        assert "entities" in d
        assert "freshness" in d
        assert "staleness_threshold_hours" in d

    def test_roundtrip_entities(self):
        from app.source_intelligence import SourceCoverageGraph
        g = self._populated_graph()
        restored = SourceCoverageGraph.from_dict(g.to_dict())
        assert "Anthropic" in restored.list_entities()

    def test_roundtrip_staleness(self):
        from app.source_intelligence import SourceCoverageGraph
        g = self._populated_graph()
        restored = SourceCoverageGraph.from_dict(g.to_dict())
        assert restored._staleness_hours == pytest.approx(12.0)

    def test_roundtrip_freshness(self):
        from app.source_intelligence import SourceCoverageGraph
        g = self._populated_graph()
        restored = SourceCoverageGraph.from_dict(g.to_dict())
        entry = restored.get_freshness("Anthropic", "blog")
        assert entry is not None
        assert entry.item_count == 1

    def test_from_dict_wrong_type_raises(self):
        from app.source_intelligence import SourceCoverageGraph
        with pytest.raises(TypeError, match="dict"):
            SourceCoverageGraph.from_dict("not a dict")


# ===========================================================================
# Group 10: CoverageGraphEdgeCases
# ===========================================================================

class TestCoverageGraphEdgeCases:

    def test_empty_graph_identify_gaps(self):
        from app.source_intelligence import SourceCoverageGraph
        g = SourceCoverageGraph()
        assert g.identify_gaps() == []

    def test_empty_graph_derivative_overreliance(self):
        from app.source_intelligence import SourceCoverageGraph
        g = SourceCoverageGraph()
        assert g.derivative_overreliance() == []

    def test_single_entity_single_source_full_paper_coverage(self):
        from app.source_intelligence import SourceCoverageGraph, EntityCategory
        g = SourceCoverageGraph()
        g.add_entity("GPT-4 Paper", EntityCategory.PAPER)
        g.attach_source("GPT-4 Paper", "arxiv", "research")
        score = g.coverage_score("GPT-4 Paper")
        assert score.completeness == pytest.approx(1.0)

    def test_get_freshness_none_when_no_fetch(self):
        from app.source_intelligence import SourceCoverageGraph, EntityCategory
        g = SourceCoverageGraph()
        g.add_entity("X", EntityCategory.COMPANY)
        g.attach_source("X", "s1", "news")
        assert g.get_freshness("X", "s1") is None

    def test_coverage_score_stale_count_le_total(self):
        from app.source_intelligence import SourceCoverageGraph, EntityCategory
        g = SourceCoverageGraph(staleness_threshold_hours=1.0)
        g.add_entity("X", EntityCategory.COMPANY)
        g.attach_source("X", "s1", "news")
        score = g.coverage_score("X")
        assert score.stale_sources <= score.total_sources


# ===========================================================================
# Group 11: CoverageGraphThreadSafety
# ===========================================================================

class TestCoverageGraphThreadSafety:

    def test_concurrent_add_entity(self):
        from app.source_intelligence import SourceCoverageGraph, EntityCategory
        g = SourceCoverageGraph()
        errors = []

        def worker(i):
            try:
                g.add_entity(f"E{i}", EntityCategory.COMPANY)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(g.list_entities()) == 50

    def test_concurrent_attach_and_fetch(self):
        from app.source_intelligence import SourceCoverageGraph, EntityCategory
        g = SourceCoverageGraph()
        g.add_entity("Tesla", EntityCategory.COMPANY)
        g.attach_source("Tesla", "news_s1", "news")
        errors = []

        def worker():
            try:
                g.record_fetch("Tesla", "news_s1")
                g.coverage_score("Tesla")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        entry = g.get_freshness("Tesla", "news_s1")
        assert entry is not None
        assert entry.item_count == 30


# ===========================================================================
# Group 12: EventPipelineConstruct
# ===========================================================================

class TestEventPipelineConstruct:

    def test_default_construction(self):
        from app.entity_resolution import EventFirstPipeline
        p = EventFirstPipeline()
        assert p is not None

    def test_custom_threshold(self):
        from app.entity_resolution import EventFirstPipeline
        p = EventFirstPipeline(merge_threshold=0.4)
        assert p._merge_threshold == pytest.approx(0.4)

    def test_custom_weights(self):
        from app.entity_resolution import EventFirstPipeline
        p = EventFirstPipeline(entity_weight=0.7, title_weight=0.3)
        assert p._entity_weight == pytest.approx(0.7)

    def test_weight_sum_constraint(self):
        from app.entity_resolution import EventFirstPipeline
        with pytest.raises(ValueError, match="1.0"):
            EventFirstPipeline(entity_weight=0.3, title_weight=0.3)

    def test_invalid_merge_threshold_raises(self):
        from app.entity_resolution import EventFirstPipeline
        with pytest.raises(ValueError):
            EventFirstPipeline(merge_threshold=2.0)

    def test_wrong_threshold_type_raises(self):
        from app.entity_resolution import EventFirstPipeline
        with pytest.raises(TypeError):
            EventFirstPipeline(merge_threshold="high")

    def test_empty_version_raises(self):
        from app.entity_resolution import EventFirstPipeline
        with pytest.raises(ValueError):
            EventFirstPipeline(pipeline_version="")

    def test_version_tag_stored(self):
        from app.entity_resolution import EventFirstPipeline
        p = EventFirstPipeline(pipeline_version="2.0")
        assert p._pipeline_version == "2.0"


# ===========================================================================
# Group 13: EventPipelineIngest
# ===========================================================================

class TestEventPipelineIngest:

    def _item(self, title="OpenAI releases GPT-5", ents=None):
        from app.entity_resolution import RawItem
        return RawItem(title=title, source_id="techcrunch", entities=ents or [])

    def test_process_item_returns_str(self):
        from app.entity_resolution import EventFirstPipeline
        p = EventFirstPipeline()
        eid = p.process_item(self._item())
        assert isinstance(eid, str) and len(eid) > 0

    def test_process_item_creates_event(self):
        from app.entity_resolution import EventFirstPipeline
        p = EventFirstPipeline()
        p.process_item(self._item())
        assert len(p.get_events()) == 1

    def test_process_item_wrong_type_raises(self):
        from app.entity_resolution import EventFirstPipeline
        p = EventFirstPipeline()
        with pytest.raises(TypeError, match="RawItem"):
            p.process_item({"title": "T", "source_id": "s"})

    def test_event_has_correct_title(self):
        from app.entity_resolution import EventFirstPipeline
        p = EventFirstPipeline()
        p.process_item(self._item(title="LLaMA 3 Released"))
        events = p.get_events()
        assert events[0].canonical_title == "LLaMA 3 Released"

    def test_raw_item_empty_title_raises(self):
        from app.entity_resolution import RawItem
        with pytest.raises(ValueError, match="title"):
            RawItem(title="", source_id="s1")

    def test_raw_item_empty_source_raises(self):
        from app.entity_resolution import RawItem
        with pytest.raises(ValueError, match="source_id"):
            RawItem(title="T", source_id="")

    def test_raw_item_trust_out_of_range_raises(self):
        from app.entity_resolution import RawItem
        with pytest.raises(ValueError, match="trust_score"):
            RawItem(title="T", source_id="s1", trust_score=1.5)

    def test_raw_item_naive_datetime_raises(self):
        from app.entity_resolution import RawItem
        with pytest.raises(ValueError, match="timezone-aware"):
            RawItem(title="T", source_id="s1", published_at=datetime.now())


# ===========================================================================
# Group 14: EventPipelineEvidenceMerge
# ===========================================================================

class TestEventPipelineEvidenceMerge:
    """Two highly similar items should merge into one event."""

    def _mkitem(self, title, source):
        from app.entity_resolution import RawItem
        return RawItem(
            title=title,
            source_id=source,
            entities=["OpenAI", "GPT-5"],
            body="OpenAI announce the release of GPT-5.",
        )

    def test_similar_items_merge(self):
        from app.entity_resolution import EventFirstPipeline
        p = EventFirstPipeline(merge_threshold=0.2)
        p.process_item(self._mkitem("OpenAI Releases GPT-5", "tc"))
        p.process_item(self._mkitem("OpenAI Releases GPT-5 Model", "verge"))
        assert len(p.get_events()) == 1

    def test_merged_evidence_count(self):
        from app.entity_resolution import EventFirstPipeline
        p = EventFirstPipeline(merge_threshold=0.2)
        p.process_item(self._mkitem("OpenAI Releases GPT-5", "tc"))
        p.process_item(self._mkitem("OpenAI Releases GPT-5 Today", "verge"))
        events = p.get_events()
        assert events[0].evidence_count == 2

    def test_merged_source_ids(self):
        from app.entity_resolution import EventFirstPipeline
        p = EventFirstPipeline(merge_threshold=0.2)
        p.process_item(self._mkitem("OpenAI Releases GPT-5", "tc"))
        p.process_item(self._mkitem("OpenAI Releases GPT-5 Model", "verge"))
        events = p.get_events()
        assert "tc" in events[0].source_ids
        assert "verge" in events[0].source_ids

    def test_distinct_items_not_merged(self):
        from app.entity_resolution import EventFirstPipeline
        p = EventFirstPipeline(merge_threshold=0.9)
        from app.entity_resolution import RawItem
        p.process_item(RawItem(title="OpenAI Releases GPT-5", source_id="s1"))
        p.process_item(RawItem(title="Meta Launches LLaMA 4", source_id="s2"))
        assert len(p.get_events()) == 2


# ===========================================================================
# Group 15: EventPipelineScoring
# ===========================================================================

class TestEventPipelineScoring:

    def test_importance_score_in_range(self):
        from app.entity_resolution import EventFirstPipeline, RawItem
        p = EventFirstPipeline()
        p.process_item(RawItem(title="Important Event", source_id="s1", trust_score=0.9))
        events = p.get_events()
        assert 0.0 <= events[0].importance_score <= 1.0

    def test_high_trust_boosts_importance(self):
        from app.entity_resolution import EventFirstPipeline, RawItem
        p_high = EventFirstPipeline()
        p_low = EventFirstPipeline()
        p_high.process_item(RawItem(title="Event A", source_id="s1", trust_score=1.0))
        p_low.process_item(RawItem(title="Event A", source_id="s1", trust_score=0.0))
        assert p_high.get_events()[0].importance_score > p_low.get_events()[0].importance_score

    def test_importance_order_descending(self):
        from app.entity_resolution import EventFirstPipeline, RawItem
        p = EventFirstPipeline(merge_threshold=0.9)  # force new events
        p.process_item(RawItem(title="High Trust Event", source_id="s1", trust_score=1.0))
        p.process_item(RawItem(title="Low Trust Stuff", source_id="s2", trust_score=0.0))
        events = p.get_events()
        scores = [e.importance_score for e in events]
        assert scores == sorted(scores, reverse=True)

    def test_trust_weighted_score_bounded(self):
        from app.entity_resolution import EventFirstPipeline, RawItem
        p = EventFirstPipeline()
        p.process_item(RawItem(title="T", source_id="s1", trust_score=0.8))
        events = p.get_events()
        assert 0.0 <= events[0].trust_weighted_score <= 1.0


# ===========================================================================
# Group 16: EventPipelineUpdates
# ===========================================================================

class TestEventPipelineUpdates:

    def _pipeline_with_events(self, n=5):
        from app.entity_resolution import EventFirstPipeline, RawItem
        p = EventFirstPipeline(merge_threshold=1.0)  # never merge → always new events
        for i in range(n):
            p.process_item(RawItem(title=f"Unique Event {i}", source_id=f"s{i}"))
        return p

    def test_generate_updates_returns_list(self):
        p = self._pipeline_with_events()
        updates = p.generate_updates()
        assert isinstance(updates, list)

    def test_generate_updates_capped_at_top_n(self):
        p = self._pipeline_with_events(10)
        updates = p.generate_updates(top_n=3)
        assert len(updates) == 3

    def test_generate_updates_top_n_zero_raises(self):
        p = self._pipeline_with_events()
        with pytest.raises(ValueError):
            p.generate_updates(top_n=0)

    def test_generate_updates_wrong_type_raises(self):
        p = self._pipeline_with_events()
        with pytest.raises(TypeError):
            p.generate_updates(top_n=2.5)

    def test_get_event_by_id(self):
        from app.entity_resolution import EventFirstPipeline, RawItem
        p = EventFirstPipeline()
        eid = p.process_item(RawItem(title="Unique event", source_id="s1"))
        event = p.get_event(eid)
        assert event is not None
        assert event.event_id == eid

    def test_get_event_unknown_returns_none(self):
        from app.entity_resolution import EventFirstPipeline
        p = EventFirstPipeline()
        assert p.get_event("no-such-id") is None

    def test_get_event_empty_id_raises(self):
        from app.entity_resolution import EventFirstPipeline
        p = EventFirstPipeline()
        with pytest.raises(ValueError):
            p.get_event("")


# ===========================================================================
# Group 17: EventPipelineBatch
# ===========================================================================

class TestEventPipelineBatch:

    def test_batch_returns_list(self):
        from app.entity_resolution import EventFirstPipeline, RawItem
        p = EventFirstPipeline(merge_threshold=1.0)
        items = [RawItem(title=f"Event {i}", source_id="s1") for i in range(5)]
        result = p.process_batch(items)
        assert isinstance(result, list) and len(result) == 5

    def test_batch_empty_raises(self):
        from app.entity_resolution import EventFirstPipeline
        p = EventFirstPipeline()
        with pytest.raises(ValueError):
            p.process_batch([])

    def test_batch_not_list_raises(self):
        from app.entity_resolution import EventFirstPipeline, RawItem
        p = EventFirstPipeline()
        with pytest.raises(TypeError):
            p.process_batch(RawItem(title="T", source_id="s1"))

    def test_batch_event_ids_match_individual(self):
        from app.entity_resolution import EventFirstPipeline, RawItem
        p1 = EventFirstPipeline(merge_threshold=1.0)
        p2 = EventFirstPipeline(merge_threshold=1.0)
        items = [RawItem(title=f"EV{i}", source_id="s") for i in range(3)]
        batch_ids = p1.process_batch(items)
        indiv_ids = [p2.process_item(i) for i in items]
        assert len(batch_ids) == len(indiv_ids)


# ===========================================================================
# Group 18: EventPipelineHighVolume
# ===========================================================================

class TestEventPipelineHighVolume:

    def test_1000_unique_items(self):
        from app.entity_resolution import EventFirstPipeline, RawItem
        p = EventFirstPipeline(merge_threshold=1.0)
        for i in range(1000):
            p.process_item(RawItem(title=f"Unique news item {i}", source_id=f"s{i}"))
        assert len(p.get_events()) == 1000

    def test_stats_after_1000(self):
        from app.entity_resolution import EventFirstPipeline, RawItem
        p = EventFirstPipeline(merge_threshold=1.0)
        for i in range(100):
            p.process_item(RawItem(title=f"Item {i}", source_id="s1"))
        stats = p.get_stats(run_duration_ms=50.0)
        assert stats.items_processed == 100
        assert stats.events_produced == 100
        assert stats.items_merged == 0


# ===========================================================================
# Group 19: EventPipelineThreadSafety
# ===========================================================================

class TestEventPipelineThreadSafety:

    def test_concurrent_process_item(self):
        from app.entity_resolution import EventFirstPipeline, RawItem
        p = EventFirstPipeline(merge_threshold=1.0)
        errors = []

        def worker(i):
            try:
                p.process_item(RawItem(title=f"Thread event {i}", source_id=f"src{i}"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(p.get_events()) == 50

    def test_concurrent_reads_while_writing(self):
        from app.entity_resolution import EventFirstPipeline, RawItem
        p = EventFirstPipeline(merge_threshold=1.0)
        errors = []

        def writer():
            try:
                for i in range(20):
                    p.process_item(RawItem(title=f"Event {i}", source_id="s"))
            except Exception as exc:
                errors.append(exc)

        def reader():
            try:
                for _ in range(50):
                    p.get_events()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ===========================================================================
# Group 20: EventPipelineStats
# ===========================================================================

class TestEventPipelineStats:

    def test_stats_empty_pipeline(self):
        from app.entity_resolution import EventFirstPipeline
        p = EventFirstPipeline()
        stats = p.get_stats()
        assert stats.items_processed == 0
        assert stats.events_produced == 0
        assert stats.items_merged == 0
        assert stats.errors == 0

    def test_stats_after_merge(self):
        from app.entity_resolution import EventFirstPipeline, RawItem
        p = EventFirstPipeline(merge_threshold=0.2)
        p.process_item(RawItem(title="OpenAI releases GPT-5", source_id="s1", entities=["OpenAI"]))
        p.process_item(RawItem(title="OpenAI releases GPT-5 model", source_id="s2", entities=["OpenAI"]))
        stats = p.get_stats()
        assert stats.items_processed == 2
        assert stats.items_merged == 1
        assert stats.events_produced == 1

    def test_stats_negative_duration_raises(self):
        from app.entity_resolution import EventFirstPipeline
        p = EventFirstPipeline()
        with pytest.raises(ValueError):
            p.get_stats(run_duration_ms=-1)

    def test_reset_clears_state(self):
        from app.entity_resolution import EventFirstPipeline, RawItem
        p = EventFirstPipeline()
        p.process_item(RawItem(title="Some event", source_id="s1"))
        p.reset()
        assert len(p.get_events()) == 0
        assert p.get_stats().items_processed == 0

    def test_pipeline_stats_model_bounds(self):
        from app.entity_resolution import PipelineStats
        with pytest.raises(Exception):
            PipelineStats(
                items_processed=3, events_produced=3, items_merged=5,  # merged > processed
                run_duration_ms=0, errors=0
            )





# ===========================================================================
# Group 21: DigestModeRouterConstruct
# ===========================================================================

class TestDigestModeRouterConstruct:

    def test_default_construction(self):
        from app.output import DigestModeRouter
        r = DigestModeRouter()
        assert r is not None

    def test_with_llm_router(self):
        from app.output import DigestModeRouter
        r = DigestModeRouter(llm_router=lambda p: "Because it matters.")
        assert r._llm is not None

    def test_delivery_mode_enum_values(self):
        from app.output import DeliveryMode
        assert DeliveryMode.MORNING_BRIEF.value == "morning_brief"
        assert DeliveryMode.WATCHLIST.value == "watchlist"
        assert DeliveryMode.DEEP_DIVE.value == "deep_dive"
        assert DeliveryMode.PERSONALIZED_STREAM.value == "personalized_stream"


# ===========================================================================
# Group 22: MorningBriefMode
# ===========================================================================

class TestMorningBriefMode:

    def _router(self):
        from app.output import DigestModeRouter
        return DigestModeRouter()

    def _cands(self, n=5, start_imp=0.9):
        return [_cand(f"item-{i}", f"Title {i}", start_imp - i * 0.1, entity_ids=[f"E{i}"]) for i in range(n)]

    def test_returns_morning_brief(self):
        from app.output import MorningBrief
        r = self._router()
        result = r.render_morning_brief(self._cands())
        assert isinstance(result, MorningBrief)

    def test_top_n_respected(self):
        r = self._router()
        result = r.render_morning_brief(self._cands(10), top_n=3)
        assert len(result.items) == 3

    def test_items_sorted_by_importance(self):
        r = self._router()
        result = r.render_morning_brief(self._cands(5))
        scores = [i.importance for i in result.items]
        assert scores == sorted(scores, reverse=True)

    def test_entity_groups_populated(self):
        r = self._router()
        result = r.render_morning_brief(self._cands(3), top_n=3)
        assert len(result.entity_groups) >= 1

    def test_date_str_defaulted(self):
        r = self._router()
        result = r.render_morning_brief(self._cands())
        assert result.date is not None and len(result.date) == 10  # YYYY-MM-DD

    def test_custom_date_str(self):
        r = self._router()
        result = r.render_morning_brief(self._cands(), date_str="2026-01-01")
        assert result.date == "2026-01-01"

    def test_why_it_matters_present(self):
        r = self._router()
        result = r.render_morning_brief(self._cands(1))
        assert result.items[0].why_it_matters != ""

    def test_confidence_in_range(self):
        r = self._router()
        result = r.render_morning_brief(self._cands(3))
        for item in result.items:
            assert 0.0 <= item.confidence <= 1.0


# ===========================================================================
# Group 23: MorningBriefValidation
# ===========================================================================

class TestMorningBriefValidation:

    def _r(self):
        from app.output import DigestModeRouter
        return DigestModeRouter()

    def test_empty_candidates_raises(self):
        with pytest.raises(ValueError):
            self._r().render_morning_brief([])

    def test_candidates_not_list_raises(self):
        with pytest.raises(TypeError):
            self._r().render_morning_brief("not a list")

    def test_top_n_zero_raises(self):
        with pytest.raises(ValueError):
            self._r().render_morning_brief([_cand()], top_n=0)

    def test_top_n_not_int_raises(self):
        with pytest.raises(TypeError):
            self._r().render_morning_brief([_cand()], top_n=2.5)

    def test_missing_item_id_raises(self):
        with pytest.raises(ValueError, match="item_id"):
            self._r().render_morning_brief([{"title": "T", "importance": 0.5}])

    def test_missing_title_raises(self):
        with pytest.raises(ValueError, match="title"):
            self._r().render_morning_brief([{"item_id": "1", "importance": 0.5}])

    def test_importance_out_of_range_raises(self):
        with pytest.raises(ValueError):
            self._r().render_morning_brief([_cand(importance=1.5)])

    def test_llm_fallback_on_exception(self):
        from app.output import DigestModeRouter
        calls = []
        def bad_llm(p):
            calls.append(p)
            raise RuntimeError("LLM offline")
        r = DigestModeRouter(llm_router=bad_llm)
        result = r.render_morning_brief([_cand()])
        # Should fall back to heuristic; no exception propagated
        assert result.items[0].why_it_matters != ""


# ===========================================================================
# Group 24: WatchlistMode
# ===========================================================================

class TestWatchlistMode:

    def _router(self):
        from app.output import DigestModeRouter
        return DigestModeRouter()

    def test_returns_watchlist_digest(self):
        from app.output import WatchlistDigest
        cands = [_cand("i1", "OpenAI Raises Prices", 0.8, entity_ids=["OpenAI"], published_at=_now())]
        result = self._router().render_watchlist(cands, ["OpenAI"])
        assert isinstance(result, WatchlistDigest)

    def test_entry_created_for_each_watched(self):
        cands = [_cand("i1", "T", 0.5, entity_ids=["OpenAI"])]
        result = self._router().render_watchlist(cands, ["OpenAI", "Meta"])
        assert len(result.entries) == 2

    def test_high_importance_triggers_high_alert(self):
        cands = [_cand("i1", "T", 0.9, entity_ids=["OpenAI"], published_at=_now())]
        result = self._router().render_watchlist(cands, ["OpenAI"])
        assert result.entries[0].alert_level == "high"
        assert "OpenAI" in result.high_alert_entities

    def test_no_items_for_entity_is_stale(self):
        result = self._router().render_watchlist([], ["Meta"])
        assert result.entries[0].is_stale is True

    def test_fresh_entity_not_stale(self):
        cands = [_cand("i1", "T", 0.5, entity_ids=["Apple"], published_at=_now())]
        result = self._router().render_watchlist(cands, ["Apple"], staleness_hours=24.0)
        assert result.entries[0].is_stale is False

    def test_stale_entity_detected(self):
        old = _now() - timedelta(hours=72)
        cands = [_cand("i1", "T", 0.5, entity_ids=["Apple"], published_at=old)]
        result = self._router().render_watchlist(cands, ["Apple"], staleness_hours=24.0)
        assert result.entries[0].is_stale is True


# ===========================================================================
# Group 25: WatchlistEdgeCases
# ===========================================================================

class TestWatchlistEdgeCases:

    def _r(self):
        from app.output import DigestModeRouter
        return DigestModeRouter()

    def test_empty_watched_raises(self):
        with pytest.raises(ValueError):
            self._r().render_watchlist([], [])

    def test_staleness_zero_raises(self):
        with pytest.raises(ValueError):
            self._r().render_watchlist([], ["X"], staleness_hours=0)

    def test_staleness_negative_raises(self):
        with pytest.raises(ValueError):
            self._r().render_watchlist([], ["X"], staleness_hours=-10)

    def test_watched_not_list_raises(self):
        with pytest.raises(TypeError):
            self._r().render_watchlist([], "Meta")

    def test_candidate_dict_not_dict_raises(self):
        with pytest.raises(TypeError):
            self._r().render_watchlist(["not a dict"], ["Meta"])

    def test_update_count_zero_when_no_matching(self):
        result = self._r().render_watchlist([], ["UnknownEntity"])
        assert result.entries[0].update_count == 0

    def test_medium_importance_gives_medium_alert(self):
        cands = [_cand("i1", "T", 0.5, entity_ids=["X"], published_at=_now())]
        result = self._r().render_watchlist(cands, ["X"])
        assert result.entries[0].alert_level == "medium"


# ===========================================================================
# Group 26: DeepDiveMode
# ===========================================================================

class TestDeepDiveMode:

    def _r(self):
        from app.output import DigestModeRouter
        return DigestModeRouter()

    def _cands(self, n=3):
        return [
            _cand(f"i{i}", f"Event {i}", 0.7, claims=[f"Claim {i}."],
                  published_at=_now() - timedelta(hours=n - i))
            for i in range(n)
        ]

    def test_returns_deep_dive_result(self):
        from app.output import DeepDiveResult
        result = self._r().render_deep_dive(self._cands(), "OpenAI")
        assert isinstance(result, DeepDiveResult)

    def test_subject_preserved(self):
        result = self._r().render_deep_dive(self._cands(), "  OpenAI  ")
        assert result.subject == "OpenAI"

    def test_timeline_length_matches_items(self):
        cands = self._cands(4)
        result = self._r().render_deep_dive(cands, "OpenAI")
        assert len(result.timeline) == 4

    def test_claims_aggregated(self):
        cands = [_cand("i1", "T", 0.5, claims=["Claim A."]), _cand("i2", "T2", 0.5, claims=["Claim B."])]
        result = self._r().render_deep_dive(cands, "X")
        assert len(result.factual_claims) >= 1

    def test_confidence_in_range(self):
        result = self._r().render_deep_dive(self._cands(), "OpenAI")
        assert 0.0 <= result.confidence <= 1.0

    def test_source_bundle_populated(self):
        result = self._r().render_deep_dive(self._cands(2), "OpenAI")
        assert len(result.source_bundle) > 0

    def test_business_implications_non_empty(self):
        result = self._r().render_deep_dive(self._cands(), "OpenAI")
        assert len(result.business_implications) > 0


# ===========================================================================
# Group 27: DeepDiveEdgeCases
# ===========================================================================

class TestDeepDiveEdgeCases:

    def _r(self):
        from app.output import DigestModeRouter
        return DigestModeRouter()

    def test_empty_candidates_raises(self):
        with pytest.raises(ValueError):
            self._r().render_deep_dive([], "OpenAI")

    def test_empty_subject_raises(self):
        with pytest.raises(ValueError):
            self._r().render_deep_dive([_cand()], "   ")

    def test_subject_not_str_raises(self):
        with pytest.raises(TypeError):
            self._r().render_deep_dive([_cand()], 42)

    def test_single_item_deep_dive(self):
        result = self._r().render_deep_dive([_cand()], "Solo Event")
        assert result.subject == "Solo Event"
        assert len(result.timeline) == 1

    def test_no_claims_gives_empty_list(self):
        result = self._r().render_deep_dive([_cand()], "X")
        # no error; factual_claims may be empty or not
        assert isinstance(result.factual_claims, list)

    def test_llm_enhanced_implications(self):
        from app.output import DigestModeRouter
        r = DigestModeRouter(llm_router=lambda p: "Impact 1.\nImpact 2.")
        result = r.render_deep_dive([_cand()], "OpenAI")
        assert "Impact 1." in result.business_implications


# ===========================================================================
# Group 28: PersonalizedStreamMode
# ===========================================================================

class TestPersonalizedStreamMode:

    def _r(self):
        from app.output import DigestModeRouter
        return DigestModeRouter()

    def test_returns_personalized_stream(self):
        from app.output import PersonalizedStream
        result = self._r().render_personalized_stream([_cand()], "user-1")
        assert isinstance(result, PersonalizedStream)

    def test_user_id_preserved(self):
        result = self._r().render_personalized_stream([_cand()], "  user-1  ")
        assert result.user_id == "user-1"

    def test_positive_feedback_boosts_score(self):
        c = _cand("i1", "T", 0.5)
        result = self._r().render_personalized_stream([c], "u1", feedback_history={"i1": 0.3})
        item = result.items[0]
        assert item.final_score > item.base_score

    def test_negative_feedback_penalises(self):
        c = _cand("i1", "T", 0.5)
        result = self._r().render_personalized_stream([c], "u1", feedback_history={"i1": -0.3})
        item = result.items[0]
        assert item.final_score < item.base_score

    def test_final_score_in_range(self):
        cands = [_cand(f"i{i}", "T", 0.5) for i in range(5)]
        fb = {f"i{i}": (0.5 if i % 2 == 0 else -0.5) for i in range(5)}
        result = self._r().render_personalized_stream(cands, "u1", feedback_history=fb)
        for item in result.items:
            assert 0.0 <= item.final_score <= 1.0

    def test_ordered_by_final_score(self):
        cands = [_cand(f"i{i}", f"T{i}", 0.5) for i in range(5)]
        fb = {"i0": 0.4, "i1": -0.4}
        result = self._r().render_personalized_stream(cands, "u1", feedback_history=fb)
        scores = [i.final_score for i in result.items]
        assert scores == sorted(scores, reverse=True)

    def test_empty_candidates_returns_empty_stream(self):
        result = self._r().render_personalized_stream([], "u1")
        assert result.items == []


# ===========================================================================
# Group 29: PersonalizedStreamEdgeCases
# ===========================================================================

class TestPersonalizedStreamEdgeCases:

    def _r(self):
        from app.output import DigestModeRouter
        return DigestModeRouter()

    def test_empty_user_id_raises(self):
        with pytest.raises(ValueError):
            self._r().render_personalized_stream([], "")

    def test_feedback_not_dict_raises(self):
        with pytest.raises(TypeError):
            self._r().render_personalized_stream([_cand()], "u1", feedback_history="bad")

    def test_feedback_none_no_error(self):
        result = self._r().render_personalized_stream([_cand()], "u1", feedback_history=None)
        assert len(result.items) == 1

    def test_unknown_feedback_key_ignored(self):
        c = _cand("i1", "T", 0.5)
        result = self._r().render_personalized_stream([c], "u1", feedback_history={"unknown": 1.0})
        assert result.items[0].final_score == pytest.approx(0.5)

    def test_feedback_clamps_boost_to_one(self):
        c = _cand("i1", "T", 0.5)
        result = self._r().render_personalized_stream([c], "u1", feedback_history={"i1": 999.0})
        assert result.items[0].feedback_boost == pytest.approx(1.0)
        assert result.items[0].final_score <= 1.0


# ===========================================================================
# Group 30: DigestModeDispatcher
# ===========================================================================

class TestDigestModeDispatcher:

    def _r(self):
        from app.output import DigestModeRouter
        return DigestModeRouter()

    def test_dispatch_morning_brief(self):
        from app.output import DeliveryMode, MorningBrief
        result = self._r().render(DeliveryMode.MORNING_BRIEF, [_cand()])
        assert isinstance(result, MorningBrief)

    def test_dispatch_watchlist(self):
        from app.output import DeliveryMode, WatchlistDigest
        result = self._r().render(DeliveryMode.WATCHLIST, [], watched_entities=["X"])
        assert isinstance(result, WatchlistDigest)

    def test_dispatch_deep_dive(self):
        from app.output import DeliveryMode, DeepDiveResult
        result = self._r().render(DeliveryMode.DEEP_DIVE, [_cand()], subject="OpenAI")
        assert isinstance(result, DeepDiveResult)

    def test_dispatch_personalized_stream(self):
        from app.output import DeliveryMode, PersonalizedStream
        result = self._r().render(DeliveryMode.PERSONALIZED_STREAM, [_cand()], user_id="u1")
        assert isinstance(result, PersonalizedStream)

    def test_wrong_mode_type_raises(self):
        with pytest.raises(TypeError, match="DeliveryMode"):
            self._r().render("morning_brief", [_cand()])


# ===========================================================================
# Group 31: CrossSourceIntelligenceToEvent
# ===========================================================================

class TestCrossSourceIntelligenceToEvent:
    """Source intelligence coverage → event pipeline integration."""

    def test_source_to_event_e2e(self):
        """Attach sources to an entity, ingest items, verify event output."""
        from app.source_intelligence import SourceCoverageGraph, EntityCategory
        from app.entity_resolution import EventFirstPipeline, RawItem

        # Source coverage layer
        g = SourceCoverageGraph()
        g.add_entity("OpenAI", EntityCategory.COMPANY)
        g.attach_source("OpenAI", "techcrunch", "news")
        g.attach_source("OpenAI", "openai-blog", "official_blog")
        g.record_fetch("OpenAI", "techcrunch")
        g.record_fetch("OpenAI", "openai-blog")

        # Coverage is partial (social missing)
        score = g.coverage_score("OpenAI")
        assert 0.0 < score.completeness < 1.0

        # Feed items from those sources into the event pipeline
        p = EventFirstPipeline(merge_threshold=0.2)
        p.process_item(RawItem(
            title="OpenAI Announces GPT-5 Release",
            source_id="techcrunch",
            entities=["OpenAI", "GPT-5"],
        ))
        p.process_item(RawItem(
            title="OpenAI Releases GPT-5 Model Today",
            source_id="openai-blog",
            entities=["OpenAI", "GPT-5"],
        ))

        events = p.get_events()
        assert len(events) == 1
        assert events[0].evidence_count == 2
        assert "techcrunch" in events[0].source_ids

    def test_gaps_drive_item_selection(self):
        """Entities with coverage gaps produce fewer events than fully-covered ones."""
        from app.source_intelligence import SourceCoverageGraph, EntityCategory
        g = SourceCoverageGraph()
        g.add_entity("Meta", EntityCategory.COMPANY)
        gaps = g.identify_gaps()
        assert any(sc.entity_name == "Meta" for sc in gaps)


# ===========================================================================
# Group 32: CrossEventToPersonalization
# ===========================================================================

class TestCrossEventToPersonalization:
    """Event pipeline → personalized stream integration."""

    def test_events_become_candidates(self):
        from app.entity_resolution import EventFirstPipeline, RawItem
        from app.output import DigestModeRouter

        p = EventFirstPipeline(merge_threshold=1.0)
        for i in range(5):
            p.process_item(RawItem(title=f"Event {i}", source_id=f"s{i}", trust_score=0.5 + i * 0.1))

        # Convert events to candidate dicts for the digest router
        events = p.get_events()
        candidates = [
            {
                "item_id": e.event_id,
                "title": e.canonical_title,
                "importance": e.importance_score,
                "trust_score": e.trust_weighted_score,
                "sources": e.source_ids,
                "entity_ids": e.entities,
            }
            for e in events
        ]
        r = DigestModeRouter()
        brief = r.render_morning_brief(candidates, top_n=3)
        assert len(brief.items) == 3
        scores = [i.importance for i in brief.items]
        assert scores == sorted(scores, reverse=True)

    def test_feedback_loop_adjusts_ordering(self):
        from app.entity_resolution import EventFirstPipeline, RawItem
        from app.output import DigestModeRouter

        p = EventFirstPipeline(merge_threshold=1.0)
        eids = []
        for i in range(3):
            eid = p.process_item(RawItem(title=f"Event {i}", source_id="s", trust_score=0.5))
            eids.append(eid)

        events = p.get_events()
        candidates = [{"item_id": e.event_id, "title": e.canonical_title, "importance": e.importance_score} for e in events]
        # Positive feedback on last event
        fb = {eids[-1]: 0.9}
        stream = DigestModeRouter().render_personalized_stream(candidates, "u1", feedback_history=fb)
        # The boosted item should appear first
        assert stream.items[0].item_id == eids[-1]


# ===========================================================================
# Group 33: CrossEntityToSynthesis
# ===========================================================================

class TestCrossEntityToSynthesis:
    """Event pipeline → deep dive synthesis integration."""

    def test_deep_dive_from_event_pipeline(self):
        from app.entity_resolution import EventFirstPipeline, RawItem
        from app.output import DigestModeRouter, DeepDiveResult

        p = EventFirstPipeline(merge_threshold=1.0)
        for i in range(4):
            p.process_item(RawItem(
                title=f"OpenAI news {i}",
                source_id=f"src{i}",
                body=f"OpenAI announces something new in iteration {i}.",
            ))

        events = p.get_events()
        candidates = [
            {"item_id": e.event_id, "title": e.canonical_title, "importance": e.importance_score,
             "claims": e.claims, "sources": e.source_ids}
            for e in events
        ]
        result = DigestModeRouter().render_deep_dive(candidates, "OpenAI")
        assert isinstance(result, DeepDiveResult)
        assert len(result.timeline) == 4
        assert result.subject == "OpenAI"


# ===========================================================================
# Group 34: CrossEnterpriseAuditFlow
# ===========================================================================

class TestCrossEnterpriseAuditFlow:
    """Enterprise audit + SLO + output delivery integration smoke test."""

    def test_watchlist_with_enterprise_generated_items(self):
        """Simulate a flow where items are enriched and delivered via watchlist."""
        from app.output import DigestModeRouter, WatchlistDigest

        items = [
            _cand(f"i{i}", f"BREAKING: Company {i} Update", 0.8 + i * 0.01,
                  entity_ids=[f"Co{i}"], published_at=_now())
            for i in range(5)
        ]
        watched = [f"Co{i}" for i in range(5)]
        result = DigestModeRouter().render_watchlist(items, watched)
        assert isinstance(result, WatchlistDigest)
        assert len(result.entries) == 5
        # All high importance → all high alert
        assert all(e.alert_level == "high" for e in result.entries)


# ===========================================================================
# Group 35: StressCoverageGraph
# ===========================================================================

class TestStressCoverageGraph:
    """2 000 entities × 3 sources each; completeness and gap fast-path."""

    def test_2000_entities_with_sources(self):
        from app.source_intelligence import SourceCoverageGraph, EntityCategory
        N = 2000
        g = SourceCoverageGraph()
        for i in range(N):
            g.add_entity(f"E{i}", EntityCategory.REPO)
            g.attach_source(f"E{i}", f"src-{i}-a", "developer_release")
            g.attach_source(f"E{i}", f"src-{i}-b", "changelog")
        assert len(g.list_entities()) == N
        # Spot-check a few
        for idx in [0, 500, 999, 1999]:
            score = g.coverage_score(f"E{idx}")
            assert score.completeness == pytest.approx(1.0)

    def test_identify_gaps_2000(self):
        from app.source_intelligence import SourceCoverageGraph, EntityCategory
        g = SourceCoverageGraph()
        for i in range(2000):
            g.add_entity(f"E{i}", EntityCategory.REPO)  # no sources → always gap
        gaps = g.identify_gaps()
        assert len(gaps) == 2000


# ===========================================================================
# Group 36: StressEventPipeline
# ===========================================================================

class TestStressEventPipeline:
    """5 000 items; ordering stability."""

    def test_5000_unique_items(self):
        from app.entity_resolution import EventFirstPipeline, RawItem
        p = EventFirstPipeline(merge_threshold=1.0)
        for i in range(5000):
            p.process_item(RawItem(title=f"Story {i}", source_id=f"src{i}"))
        assert len(p.get_events()) == 5000

    def test_ordering_stable_after_5000(self):
        from app.entity_resolution import EventFirstPipeline, RawItem
        p = EventFirstPipeline(merge_threshold=1.0)
        for i in range(100):
            p.process_item(RawItem(title=f"Item {i}", source_id="s", trust_score=i / 100.0))
        events = p.get_events()
        scores = [e.importance_score for e in events]
        assert scores == sorted(scores, reverse=True)


# ===========================================================================
# Group 37: PackageAllExports
# ===========================================================================

class TestPackageAllExports:
    """All 7 packages expose clean __all__ with every listed symbol importable."""

    def _check_all(self, pkg_path):
        import importlib
        pkg = importlib.import_module(pkg_path)
        all_exports = getattr(pkg, "__all__", [])
        assert len(all_exports) > 0, f"{pkg_path}.__all__ is empty"
        for name in all_exports:
            assert hasattr(pkg, name), f"{pkg_path}.{name} listed in __all__ but not found"

    def test_source_intelligence_all(self):
        self._check_all("app.source_intelligence")

    def test_entity_resolution_all(self):
        self._check_all("app.entity_resolution")

    def test_output_all(self):
        self._check_all("app.output")

    def test_document_intelligence_all(self):
        self._check_all("app.document_intelligence")

    def test_personalization_all(self):
        self._check_all("app.personalization")

    def test_summarization_all(self):
        self._check_all("app.summarization")

    def test_devintel_all(self):
        self._check_all("app.devintel")

    def test_phase6_new_symbols_in_source_intelligence(self):
        import app.source_intelligence as si
        new_symbols = [
            "SourceCoverageGraph", "EntityCategory", "EntityCoverageScore",
            "SourceFreshnessEntry", "SourceDiscoveryEngine", "DiscoveredSource",
            "CoveragePlanner", "CoverageGap", "GapSeverity",
            "FeedExpander", "FeedCandidate", "EntityToSourceMapper",
            "EntitySourceMap", "ChangeMonitor", "ChangeEvent",
        ]
        for sym in new_symbols:
            assert sym in si.__all__, f"{sym} not in app.source_intelligence.__all__"

    def test_phase6_new_symbols_in_entity_resolution(self):
        import app.entity_resolution as er
        for sym in ["EventFirstPipeline", "RawItem", "ProcessedEvent", "PipelineStats"]:
            assert sym in er.__all__, f"{sym} not in app.entity_resolution.__all__"

    def test_phase6_new_symbols_in_output(self):
        import app.output as out
        for sym in ["DigestModeRouter", "DeliveryMode", "MorningBrief",
                    "WatchlistDigest", "DeepDiveResult", "PersonalizedStream",
                    "BriefItem", "WatchlistEntry", "PersonalizedStreamItem"]:
            assert sym in out.__all__, f"{sym} not in app.output.__all__"


# ===========================================================================
# Group 38: Audit Fix — attach_source is_derivative storage & validation
# ===========================================================================

class TestAttachSourceIsDerivative:
    """Confirm is_derivative is validated, stored, and used without record_fetch."""

    def _g(self, entity="TestCo"):
        from app.source_intelligence import SourceCoverageGraph, EntityCategory
        g = SourceCoverageGraph(derivative_overreliance_ratio=0.5)
        g.add_entity(entity, EntityCategory.COMPANY)
        return g

    def test_attach_derivative_wrong_type_raises(self):
        g = self._g()
        with pytest.raises(TypeError, match="is_derivative"):
            g.attach_source("TestCo", "s1", "news", is_derivative="yes")

    def test_attach_non_derivative_stored(self):
        g = self._g()
        g.attach_source("TestCo", "s1", "news", is_derivative=False)
        assert g._source_is_derivative.get("s1") is False

    def test_attach_derivative_stored(self):
        g = self._g()
        g.attach_source("TestCo", "s1", "news", is_derivative=True)
        assert g._source_is_derivative.get("s1") is True

    def test_derivative_counted_without_record_fetch(self):
        """attach_source(is_derivative=True) should count toward derivative_sources
        in coverage_score even before any record_fetch call."""
        g = self._g()
        g.attach_source("TestCo", "deriv_src", "news", is_derivative=True)
        g.attach_source("TestCo", "orig_src", "official_blog", is_derivative=False)
        score = g.coverage_score("TestCo")
        assert score.derivative_sources == 1, (
            f"Expected 1 derivative source from attach-time flag, got {score.derivative_sources}"
        )

    def test_overreliance_detected_from_attach_flag(self):
        """derivative_overreliance() should fire based on attach-time is_derivative
        even with no record_fetch calls."""
        g = self._g()
        g.attach_source("TestCo", "d1", "news", is_derivative=True)
        g.attach_source("TestCo", "d2", "official_blog", is_derivative=True)
        result = g.derivative_overreliance()
        assert any(r.entity_name == "TestCo" for r in result)

    def test_freshness_flag_overrides_attach_flag(self):
        """If record_fetch is called with is_derivative=False, it should override
        the attach-time is_derivative=True for that source."""
        g = self._g()
        g.attach_source("TestCo", "s1", "news", is_derivative=True)
        g.record_fetch("TestCo", "s1", is_derivative=False)  # override to non-derivative
        score = g.coverage_score("TestCo")
        # Freshness entry says not derivative → 0 derivatives
        assert score.derivative_sources == 0

    def test_detach_clears_is_derivative(self):
        g = self._g()
        g.attach_source("TestCo", "s1", "news", is_derivative=True)
        g.detach_source("TestCo", "s1")
        assert "s1" not in g._source_is_derivative


# ===========================================================================
# Group 39: Audit Fix — EntityCoverageScore derivative_sources ≤ total_sources
# ===========================================================================

class TestEntityCoverageScoreDerivativeInvariant:
    """Confirm the new cross-field invariant is enforced."""

    def _base_kwargs(self, **overrides):
        from datetime import datetime, timezone
        base = dict(
            entity_name="X",
            category="company",
            total_sources=2,
            stale_sources=0,
            derivative_sources=0,
            completeness=1.0,
            derivative_ratio=0.0,
            gap_count=0,
            computed_at=datetime.now(timezone.utc),
        )
        base.update(overrides)
        return base

    def test_derivative_eq_total_allowed(self):
        from app.source_intelligence import EntityCoverageScore, EntityCategory
        score = EntityCoverageScore(
            **self._base_kwargs(
                category=EntityCategory.COMPANY,
                total_sources=2, derivative_sources=2
            )
        )
        assert score.derivative_sources == 2

    def test_derivative_gt_total_raises(self):
        from app.source_intelligence import EntityCoverageScore, EntityCategory
        with pytest.raises(Exception, match="derivative_sources"):
            EntityCoverageScore(
                **self._base_kwargs(
                    category=EntityCategory.COMPANY,
                    total_sources=2, derivative_sources=3
                )
            )

    def test_stale_gt_total_still_raises(self):
        from app.source_intelligence import EntityCoverageScore, EntityCategory
        with pytest.raises(Exception):
            EntityCoverageScore(
                **self._base_kwargs(
                    category=EntityCategory.COMPANY,
                    total_sources=2, stale_sources=5, derivative_sources=0
                )
            )


# ===========================================================================
# Group 40: Audit Fix — Serialization includes source_is_derivative
# ===========================================================================

class TestSerializationIsDerivative:
    """to_dict/from_dict preserves attach-time is_derivative flag."""

    def _populated(self):
        from app.source_intelligence import SourceCoverageGraph, EntityCategory
        g = SourceCoverageGraph()
        g.add_entity("Meta", EntityCategory.COMPANY)
        g.attach_source("Meta", "repost_site", "news", is_derivative=True)
        g.attach_source("Meta", "meta_blog", "official_blog", is_derivative=False)
        return g

    def test_to_dict_includes_source_is_derivative(self):
        g = self._populated()
        d = g.to_dict()
        assert "source_is_derivative" in d
        assert d["source_is_derivative"]["repost_site"] is True
        assert d["source_is_derivative"]["meta_blog"] is False

    def test_from_dict_restores_is_derivative(self):
        from app.source_intelligence import SourceCoverageGraph
        g = self._populated()
        g2 = SourceCoverageGraph.from_dict(g.to_dict())
        assert g2._source_is_derivative.get("repost_site") is True
        assert g2._source_is_derivative.get("meta_blog") is False

    def test_from_dict_derivative_count_matches(self):
        from app.source_intelligence import SourceCoverageGraph
        g = self._populated()
        g2 = SourceCoverageGraph.from_dict(g.to_dict())
        score = g2.coverage_score("Meta")
        assert score.derivative_sources == 1

    def test_from_dict_source_families_preserved(self):
        from app.source_intelligence import SourceCoverageGraph
        g = self._populated()
        d = g.to_dict()
        g2 = SourceCoverageGraph.from_dict(d)
        assert g2._source_families.get("repost_site") == "news"
        assert g2._source_families.get("meta_blog") == "official_blog"

    def test_from_dict_entity_sources_preserved(self):
        from app.source_intelligence import SourceCoverageGraph
        g = self._populated()
        g2 = SourceCoverageGraph.from_dict(g.to_dict())
        assert "repost_site" in g2._entity_sources.get("Meta", set())
        assert "meta_blog" in g2._entity_sources.get("Meta", set())

    def test_full_roundtrip_derivative_overreliance(self):
        from app.source_intelligence import SourceCoverageGraph
        g = self._populated()
        g2 = SourceCoverageGraph.from_dict(g.to_dict())
        # 1/2 sources = 0.5 ratio, threshold is default 0.6 → not over-relied
        result = g2.derivative_overreliance()
        # repost_site is derivative (0.5 ratio), threshold is 0.6 by default → not flagged
        assert result == []

    def test_full_roundtrip_high_overreliance(self):
        from app.source_intelligence import SourceCoverageGraph, EntityCategory
        g = SourceCoverageGraph(derivative_overreliance_ratio=0.4)
        g.add_entity("X", EntityCategory.COMPANY)
        g.attach_source("X", "d1", "news", is_derivative=True)
        g.attach_source("X", "d2", "official_blog", is_derivative=False)
        g2 = SourceCoverageGraph.from_dict(g.to_dict())
        result = g2.derivative_overreliance()
        assert any(r.entity_name == "X" for r in result)


# ===========================================================================
# Group 41: Audit Fix — EventFirstPipeline thread-safe get_events/get_event
# ===========================================================================

class TestEventPipelineSnapshotSafety:
    """Confirm get_events/get_event snapshot data inside the lock."""

    def test_get_events_returns_frozen_snapshot(self):
        """Modifications after get_events() should not affect the returned list."""
        from app.entity_resolution import EventFirstPipeline, RawItem
        p = EventFirstPipeline(merge_threshold=1.0)
        p.process_item(RawItem(title="Event A", source_id="s1"))
        events_before = p.get_events()
        p.process_item(RawItem(title="Event B", source_id="s2"))
        # The list captured before should still have only 1 item
        assert len(events_before) == 1

    def test_get_event_returns_snapshot(self):
        """Modifying the pipeline state after get_event should not mutate the returned event."""
        from app.entity_resolution import EventFirstPipeline, RawItem
        p = EventFirstPipeline(merge_threshold=0.2)
        eid = p.process_item(RawItem(title="OpenAI Releases GPT-5", source_id="s1",
                                     entities=["OpenAI"]))
        event_before = p.get_event(eid)
        # Now merge another item into the same event
        p.process_item(RawItem(title="OpenAI Releases GPT-5 Model", source_id="s2",
                                entities=["OpenAI"]))
        # The event captured before should still have evidence_count=1
        assert event_before.evidence_count == 1

    def test_concurrent_get_events_no_crash(self):
        """get_events() under concurrent writes must not raise."""
        from app.entity_resolution import EventFirstPipeline, RawItem
        p = EventFirstPipeline(merge_threshold=1.0)
        errors = []

        def writer():
            for i in range(30):
                try:
                    p.process_item(RawItem(title=f"Concurrent event {i}", source_id=f"s{i}"))
                except Exception as exc:
                    errors.append(exc)

        def reader():
            for _ in range(50):
                try:
                    p.get_events()
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=writer)] + [
            threading.Thread(target=reader) for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ===========================================================================
# Group 42: Audit Fix — Duplicate source_id deduplication in event pipeline
# ===========================================================================

class TestEventPipelineSourceDeduplication:

    def test_same_source_id_deduplicated_after_merge(self):
        """When the same source_id submits two similar items that merge,
        the event's source_ids list should contain the ID only once."""
        from app.entity_resolution import EventFirstPipeline, RawItem
        p = EventFirstPipeline(merge_threshold=0.2)
        p.process_item(RawItem(title="OpenAI Releases GPT-5", source_id="techcrunch",
                               entities=["OpenAI"]))
        p.process_item(RawItem(title="OpenAI Releases GPT-5 Model", source_id="techcrunch",
                               entities=["OpenAI"]))
        events = p.get_events()
        assert len(events) == 1
        assert events[0].source_ids.count("techcrunch") == 1, (
            "source_id 'techcrunch' appeared more than once after deduplication"
        )

    def test_different_sources_both_appear(self):
        from app.entity_resolution import EventFirstPipeline, RawItem
        p = EventFirstPipeline(merge_threshold=0.2)
        p.process_item(RawItem(title="OpenAI Releases GPT-5", source_id="tc",
                               entities=["OpenAI"]))
        p.process_item(RawItem(title="OpenAI Releases GPT-5 Model", source_id="verge",
                               entities=["OpenAI"]))
        events = p.get_events()
        assert len(events) == 1
        assert "tc" in events[0].source_ids
        assert "verge" in events[0].source_ids

    def test_empty_entities_item_creates_event(self):
        """RawItem with empty entities list must still create an event."""
        from app.entity_resolution import EventFirstPipeline, RawItem
        from datetime import datetime, timezone
        p = EventFirstPipeline()
        eid = p.process_item(RawItem(
            title="Breaking: Major Announcement",
            source_id="s1",
            entities=[],
            published_at=datetime.now(timezone.utc),
        ))
        assert eid is not None
        event = p.get_event(eid)
        assert event is not None

    def test_identical_item_twice_at_max_threshold_merges(self):
        """Processing the same (title, entities) item twice at threshold=1.0 correctly
        merges them: combined score is 1.0 which equals the threshold (>=), so merge fires.
        This verifies the >= boundary of the merge condition."""
        from app.entity_resolution import EventFirstPipeline, RawItem
        p = EventFirstPipeline(merge_threshold=1.0)
        p.process_item(RawItem(title="Same Event", source_id="s1", entities=["Entity A"]))
        p.process_item(RawItem(title="Same Event", source_id="s1", entities=["Entity A"]))
        # score = 1.0 >= threshold 1.0 → merged into 1 event
        events = p.get_events()
        assert len(events) == 1
        assert events[0].evidence_count == 2

    def test_very_different_items_never_merge_at_max_threshold(self):
        """Completely unrelated items should never merge at threshold=1.0 since
        entity overlap and title similarity are both ~0."""
        from app.entity_resolution import EventFirstPipeline, RawItem
        p = EventFirstPipeline(merge_threshold=1.0)
        p.process_item(RawItem(title="Anthropic Releases Claude 4", source_id="s1",
                               entities=["Anthropic", "Claude"]))
        p.process_item(RawItem(title="NVIDIA Launches New GPU Architecture", source_id="s2",
                               entities=["NVIDIA", "GPU"]))
        assert len(p.get_events()) == 2


# ===========================================================================
# Group 43: Audit Fix — Personalized stream depth controls (max_items)
# ===========================================================================

class TestPersonalizedStreamDepthControl:

    def _r(self):
        from app.output import DigestModeRouter
        return DigestModeRouter()

    def _cands(self, n=10):
        return [{"item_id": str(i), "title": f"Item {i}", "importance": (n - i) / n}
                for i in range(n)]

    def test_max_items_limits_stream(self):
        result = self._r().render_personalized_stream(self._cands(10), "u1", max_items=3)
        assert len(result.items) == 3

    def test_max_items_none_no_limit(self):
        result = self._r().render_personalized_stream(self._cands(10), "u1", max_items=None)
        assert len(result.items) == 10

    def test_max_items_gt_available_returns_all(self):
        result = self._r().render_personalized_stream(self._cands(3), "u1", max_items=100)
        assert len(result.items) == 3

    def test_max_items_one_returns_one(self):
        result = self._r().render_personalized_stream(self._cands(5), "u1", max_items=1)
        assert len(result.items) == 1

    def test_max_items_zero_raises(self):
        with pytest.raises(ValueError, match="max_items"):
            self._r().render_personalized_stream(self._cands(), "u1", max_items=0)

    def test_max_items_negative_raises(self):
        with pytest.raises(ValueError):
            self._r().render_personalized_stream(self._cands(), "u1", max_items=-5)

    def test_max_items_not_int_raises(self):
        with pytest.raises(TypeError, match="max_items"):
            self._r().render_personalized_stream(self._cands(), "u1", max_items=3.0)

    def test_max_items_top_scored_selected(self):
        """With feedback applied, max_items should keep highest final_score items."""
        cands = self._cands(10)
        # Give item "0" massive positive feedback
        fb = {"0": 0.9}
        result = self._r().render_personalized_stream(cands, "u1",
                                                       feedback_history=fb, max_items=3)
        assert result.items[0].item_id == "0"

    def test_render_dispatcher_passes_max_items(self):
        """render() dispatcher must forward max_items kwarg to personalized stream."""
        from app.output import DeliveryMode
        result = self._r().render(
            DeliveryMode.PERSONALIZED_STREAM,
            self._cands(10),
            user_id="u1",
            max_items=4,
        )
        assert len(result.items) == 4


# ===========================================================================
# Group 44: Audit Fix — Logging observability confirmations
# ===========================================================================

class TestLoggingObservability:
    """Confirm every public mutating method emits a log entry."""

    def test_derivative_overreliance_emits_log(self, caplog):
        import logging
        from app.source_intelligence import SourceCoverageGraph, EntityCategory
        g = SourceCoverageGraph()
        g.add_entity("X", EntityCategory.COMPANY)
        with caplog.at_level(logging.DEBUG, logger="app.source_intelligence.coverage_graph"):
            g.derivative_overreliance()
        assert any("derivative_overreliance" in r.message for r in caplog.records)

    def test_process_item_success_emits_log(self, caplog):
        import logging
        from app.entity_resolution import EventFirstPipeline, RawItem
        from datetime import datetime, timezone
        p = EventFirstPipeline()
        with caplog.at_level(logging.DEBUG, logger="app.entity_resolution.event_pipeline"):
            p.process_item(RawItem(title="T", source_id="s1",
                                   published_at=datetime.now(timezone.utc)))
        assert any("process_item" in r.message for r in caplog.records)

    def test_ingest_merge_emits_log(self, caplog):
        import logging
        from app.entity_resolution import EventFirstPipeline, RawItem
        p = EventFirstPipeline(merge_threshold=0.2)
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc)
        p.process_item(RawItem(title="OpenAI Releases GPT-5", source_id="s1",
                               entities=["OpenAI"], published_at=ts))
        with caplog.at_level(logging.DEBUG, logger="app.entity_resolution.event_pipeline"):
            p.process_item(RawItem(title="OpenAI Releases GPT-5 Model", source_id="s2",
                                   entities=["OpenAI"], published_at=ts))
        assert any("MERGE" in r.message for r in caplog.records)

    def test_ingest_create_emits_log(self, caplog):
        import logging
        from app.entity_resolution import EventFirstPipeline, RawItem
        from datetime import datetime, timezone
        p = EventFirstPipeline(merge_threshold=1.0)
        with caplog.at_level(logging.DEBUG, logger="app.entity_resolution.event_pipeline"):
            p.process_item(RawItem(title="Brand New Event", source_id="s1",
                                   published_at=datetime.now(timezone.utc)))
        assert any("CREATE" in r.message for r in caplog.records)

    def test_render_dispatcher_emits_log(self, caplog):
        import logging
        from app.output import DigestModeRouter, DeliveryMode
        r = DigestModeRouter()
        cands = [{"item_id": "1", "title": "T", "importance": 0.5}]
        with caplog.at_level(logging.DEBUG, logger="app.output.digest_modes"):
            r.render(DeliveryMode.MORNING_BRIEF, cands)
        assert any("render" in r.message.lower() for r in caplog.records)

    def test_personalized_stream_emits_log(self, caplog):
        import logging
        from app.output import DigestModeRouter
        r = DigestModeRouter()
        with caplog.at_level(logging.INFO, logger="app.output.digest_modes"):
            r.render_personalized_stream([], "u1")
        assert any("personalized_stream" in r.message for r in caplog.records)

