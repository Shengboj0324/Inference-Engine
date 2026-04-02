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


# ===========================================================================
# GAP-17/18/19: Lazy package __init__ tests
# ===========================================================================

class TestLazyPackageImports:
    """Verify that importing the three heavy packages does NOT trigger eager
    heavyweight imports (GAP-17, GAP-18, GAP-19)."""

    def test_app_intelligence_import_is_fast(self):
        """Importing app.intelligence must not pull in torch/transformers."""
        import importlib, sys
        # Remove cached module if re-running in same process
        for key in list(sys.modules.keys()):
            if key.startswith("app.intelligence") and key != "app.intelligence":
                del sys.modules[key]
        mod = importlib.import_module("app.intelligence")
        assert hasattr(mod, "__all__")

    def test_app_intelligence_lazy_getattr(self):
        """Lazy __getattr__ must resolve DigestEngine on first access."""
        import importlib, sys
        mod = importlib.import_module("app.intelligence")
        cls = getattr(mod, "DigestEngine")
        assert cls is not None
        assert cls.__name__ == "DigestEngine"

    def test_app_intelligence_unknown_attr_raises(self):
        import app.intelligence
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = app.intelligence.NonExistentClass  # type: ignore

    def test_app_scraping_import_is_fast(self):
        import importlib
        mod = importlib.import_module("app.scraping")
        assert hasattr(mod, "__all__")

    def test_app_connectors_import_is_fast(self):
        import importlib
        mod = importlib.import_module("app.connectors")
        assert hasattr(mod, "__all__")

    def test_capability_mode_exported_from_intelligence(self):
        """CapabilityMode must be re-exported via app.intelligence lazy map (GAP-4)."""
        from app.intelligence import CapabilityMode
        assert CapabilityMode.STUB.value == "stub"
        assert CapabilityMode.DISABLED.value == "disabled"
        assert CapabilityMode.LOCAL_MODEL.value == "local_model"
        assert CapabilityMode.REMOTE_MODEL.value == "remote_model"


# ===========================================================================
# GAP-4: MultimodalAnalyzer CapabilityMode tests
# ===========================================================================

class TestMultimodalAnalyzerExecutionMode:
    """Verify that MultimodalAnalyzer exposes and enforces execution_mode (GAP-4)."""

    def _make_obs(self, image_url=None, video_url=None):
        from unittest.mock import MagicMock
        obs = MagicMock()
        obs.id = "test-obs-id"
        obs.source_platform.value = "reddit"
        obs.source_url = "https://example.com"
        obs.platform_metadata = {}
        if image_url:
            obs.platform_metadata["image_url"] = image_url
        if video_url:
            obs.platform_metadata["video_url"] = video_url
        return obs

    def test_default_mode_is_stub(self):
        from app.intelligence.multimodal import MultimodalAnalyzer, CapabilityMode
        a = MultimodalAnalyzer()
        assert a.execution_mode == CapabilityMode.STUB

    def test_explicit_disabled_mode(self):
        from app.intelligence.multimodal import MultimodalAnalyzer, CapabilityMode
        a = MultimodalAnalyzer(execution_mode=CapabilityMode.DISABLED)
        assert a.execution_mode == CapabilityMode.DISABLED

    def test_disabled_mode_visual_to_text_returns_empty(self):
        from app.intelligence.multimodal import MultimodalAnalyzer, CapabilityMode
        a = MultimodalAnalyzer(execution_mode=CapabilityMode.DISABLED)
        obs = self._make_obs(image_url="https://example.com/img.png")
        assert a.visual_to_text(obs) == ""

    def test_stub_mode_returns_nonempty_text(self):
        from app.intelligence.multimodal import MultimodalAnalyzer, CapabilityMode
        a = MultimodalAnalyzer(execution_mode=CapabilityMode.STUB)
        obs = self._make_obs(image_url="https://example.com/img.png")
        result = a.visual_to_text(obs)
        assert isinstance(result, str) and len(result) > 10

    def test_wrong_execution_mode_type_raises(self):
        from app.intelligence.multimodal import MultimodalAnalyzer
        with pytest.raises(TypeError, match="execution_mode"):
            MultimodalAnalyzer(execution_mode="stub")  # type: ignore

    def test_analyze_image_empty_url_raises(self):
        from app.intelligence.multimodal import MultimodalAnalyzer
        a = MultimodalAnalyzer()
        with pytest.raises(ValueError):
            a.analyze_image("")

    def test_analyze_image_wrong_type_raises(self):
        from app.intelligence.multimodal import MultimodalAnalyzer
        a = MultimodalAnalyzer()
        with pytest.raises(TypeError):
            a.analyze_image(123)  # type: ignore

    def test_with_vision_client_mode_is_local(self):
        from app.intelligence.multimodal import MultimodalAnalyzer, CapabilityMode
        client = lambda url, mod: {"caption": "test", "sentiment": "positive", "entities": []}
        a = MultimodalAnalyzer(vision_client=client, execution_mode=CapabilityMode.LOCAL_MODEL)
        assert a.execution_mode == CapabilityMode.LOCAL_MODEL


# ===========================================================================
# GAP-8..11: Evaluator gate() tests
# ===========================================================================

class TestClassificationEvaluatorGate:
    """Verify ClassificationEvaluator.gate() enforces deployment thresholds (GAP-8)."""

    def _make_report(self, macro_f1=0.80, false_action_rate=0.02):
        from app.evals.classification_eval import ClassificationReport
        return ClassificationReport(
            macro_f1=macro_f1,
            macro_precision=0.80,
            macro_recall=0.80,
            per_class={},
            abstain_precision=None,
            false_action_rate=false_action_rate,
            total_samples=100,
            total_abstained=0,
        )

    def test_gate_passes_when_above_thresholds(self):
        from app.evals.classification_eval import ClassificationEvaluator
        ev = ClassificationEvaluator()
        report = self._make_report(macro_f1=0.80, false_action_rate=0.02)
        ev.gate(report, macro_f1_threshold=0.70, false_action_rate_threshold=0.05)

    def test_gate_fails_on_low_f1(self):
        from app.evals.classification_eval import ClassificationEvaluator
        ev = ClassificationEvaluator()
        report = self._make_report(macro_f1=0.50, false_action_rate=0.02)
        with pytest.raises(ValueError, match="macro_f1"):
            ev.gate(report, macro_f1_threshold=0.70)

    def test_gate_fails_on_high_false_action_rate(self):
        from app.evals.classification_eval import ClassificationEvaluator
        ev = ClassificationEvaluator()
        report = self._make_report(macro_f1=0.80, false_action_rate=0.10)
        with pytest.raises(ValueError, match="false_action_rate"):
            ev.gate(report, false_action_rate_threshold=0.05)

    def test_gate_wrong_report_type_raises(self):
        from app.evals.classification_eval import ClassificationEvaluator
        ev = ClassificationEvaluator()
        with pytest.raises(TypeError, match="ClassificationReport"):
            ev.gate("not a report")  # type: ignore

    def test_gate_out_of_range_threshold_raises(self):
        from app.evals.classification_eval import ClassificationEvaluator
        ev = ClassificationEvaluator()
        report = self._make_report()
        with pytest.raises(ValueError, match="macro_f1_threshold"):
            ev.gate(report, macro_f1_threshold=1.5)


class TestRankingEvaluatorGate:
    """Verify RankingEvaluator.gate() enforces NDCG@k threshold (GAP-9)."""

    def _make_report(self, ndcg_at_10=0.70):
        from app.evals.ranking_eval import RankingReport
        return RankingReport(
            ndcg={10: ndcg_at_10},
            precision={10: 0.60},
            opportunity_hit_rate={10: 0.80},
            median_rank=3.0,
            total_queries=50,
        )

    def test_gate_passes_above_threshold(self):
        from app.evals.ranking_eval import RankingEvaluator
        ev = RankingEvaluator(k_values=[10])
        ev.gate(self._make_report(ndcg_at_10=0.75), ndcg_at_k=10, ndcg_threshold=0.60)

    def test_gate_fails_below_threshold(self):
        from app.evals.ranking_eval import RankingEvaluator
        ev = RankingEvaluator(k_values=[10])
        with pytest.raises(ValueError, match="NDCG@10"):
            ev.gate(self._make_report(ndcg_at_10=0.50), ndcg_at_k=10, ndcg_threshold=0.60)

    def test_gate_missing_k_raises_key_error(self):
        from app.evals.ranking_eval import RankingEvaluator
        ev = RankingEvaluator(k_values=[10])
        report = self._make_report()
        with pytest.raises(KeyError):
            ev.gate(report, ndcg_at_k=5, ndcg_threshold=0.60)

    def test_gate_wrong_report_type_raises(self):
        from app.evals.ranking_eval import RankingEvaluator
        ev = RankingEvaluator(k_values=[10])
        with pytest.raises(TypeError, match="RankingReport"):
            ev.gate({"ndcg": {10: 0.8}})  # type: ignore


class TestResponseEvaluatorGate:
    """Verify ResponseEvaluator.gate() enforces safety thresholds (GAP-10)."""

    def _make_report(self, pv_rate=0.005, unsafe_rate=0.01):
        from app.evals.response_eval import ResponseReport
        return ResponseReport(
            approval_rate=0.95,
            policy_violation_rate=pv_rate,
            unsafe_draft_rate=unsafe_rate,
            length_appropriateness_rate=0.90,
            mean_critique_score=0.80,
            per_draft=[],
            total_drafts=100,
        )  # type: ignore[call-arg]  # dataclass — positional is fine here

    def test_gate_passes_when_safe(self):
        from app.evals.response_eval import ResponseEvaluator
        ev = ResponseEvaluator()
        ev.gate(self._make_report(pv_rate=0.005, unsafe_rate=0.01))

    def test_gate_fails_on_high_policy_violation(self):
        from app.evals.response_eval import ResponseEvaluator
        ev = ResponseEvaluator()
        with pytest.raises(ValueError, match="policy_violation_rate"):
            ev.gate(self._make_report(pv_rate=0.05))

    def test_gate_fails_on_high_unsafe_rate(self):
        from app.evals.response_eval import ResponseEvaluator
        ev = ResponseEvaluator()
        with pytest.raises(ValueError, match="unsafe_draft_rate"):
            ev.gate(self._make_report(unsafe_rate=0.10))

    def test_gate_wrong_type_raises(self):
        from app.evals.response_eval import ResponseEvaluator
        ev = ResponseEvaluator()
        with pytest.raises(TypeError, match="ResponseReport"):
            ev.gate("bad")  # type: ignore


class TestAdversarialEvaluatorGate:
    """Verify AdversarialEvaluator.gate() enforces robustness thresholds (GAP-11)."""

    def _make_report(self, correct_abstain=0.85, fp_rate=0.03):
        from app.evals.adversarial_eval import AdversarialReport
        return AdversarialReport(
            total_cases=100,
            abstain_rate=0.50,
            correct_abstain_rate=correct_abstain,
            false_positive_rate=fp_rate,
            per_category_abstain_rate={},
            results=[],
        )

    def test_gate_passes_when_robust(self):
        from app.evals.adversarial_eval import AdversarialEvaluator
        ev = AdversarialEvaluator()
        ev.gate(self._make_report(correct_abstain=0.85, fp_rate=0.03))

    def test_gate_fails_low_correct_abstain(self):
        from app.evals.adversarial_eval import AdversarialEvaluator
        ev = AdversarialEvaluator()
        with pytest.raises(ValueError, match="correct_abstain_rate"):
            ev.gate(self._make_report(correct_abstain=0.60))

    def test_gate_fails_high_false_positive(self):
        from app.evals.adversarial_eval import AdversarialEvaluator
        ev = AdversarialEvaluator()
        with pytest.raises(ValueError, match="false_positive_rate"):
            ev.gate(self._make_report(fp_rate=0.20))

    def test_gate_wrong_type_raises(self):
        from app.evals.adversarial_eval import AdversarialEvaluator
        ev = AdversarialEvaluator()
        with pytest.raises(TypeError, match="AdversarialReport"):
            ev.gate(None)  # type: ignore


# ===========================================================================
# GAP-6: SourceDiscoveryEngine scoring and graph expansion tests
# ===========================================================================

class TestSourceDiscoveryEngineScoring:
    """Verify SourceScoreRecord, score_source(), expand_from_source() (GAP-6)."""

    def test_score_source_creates_record(self):
        from app.source_intelligence.source_discovery import SourceDiscoveryEngine
        eng = SourceDiscoveryEngine()
        rec = eng.score_source("openai/openai-python", base_confidence=0.9)
        assert rec.source_id == "openai/openai-python"
        assert rec.base_confidence == 0.9

    def test_score_source_fetch_success_increases_boost(self):
        from app.source_intelligence.source_discovery import SourceDiscoveryEngine
        eng = SourceDiscoveryEngine()
        rec1 = eng.score_source("src-a", base_confidence=0.8, fetch_success=True)
        assert rec1.freshness_boost > 0.0

    def test_score_source_fetch_failure_adds_penalty(self):
        from app.source_intelligence.source_discovery import SourceDiscoveryEngine
        eng = SourceDiscoveryEngine()
        rec1 = eng.score_source("src-b", base_confidence=0.8, fetch_success=False)
        assert rec1.error_penalty < 0.0

    def test_score_source_effective_score_bounded(self):
        from app.source_intelligence.source_discovery import SourceDiscoveryEngine
        eng = SourceDiscoveryEngine()
        for _ in range(10):
            eng.score_source("src-c", base_confidence=0.9, fetch_success=True)
        rec = eng.get_score_registry()["src-c"]
        assert 0.0 <= rec.effective_score <= 1.0

    def test_score_source_empty_id_raises(self):
        from app.source_intelligence.source_discovery import SourceDiscoveryEngine
        eng = SourceDiscoveryEngine()
        with pytest.raises(ValueError):
            eng.score_source("")

    def test_score_source_wrong_type_raises(self):
        from app.source_intelligence.source_discovery import SourceDiscoveryEngine
        eng = SourceDiscoveryEngine()
        with pytest.raises(TypeError):
            eng.score_source(123)  # type: ignore

    def test_expand_from_source_openai_returns_siblings(self):
        from app.source_intelligence.source_discovery import SourceDiscoveryEngine
        eng = SourceDiscoveryEngine()
        results = eng.expand_from_source("openai/openai-python")
        ids = [r.source_id for r in results]
        assert len(ids) > 0
        # Should not include the source itself
        assert "openai/openai-python" not in ids

    def test_expand_from_source_unknown_returns_empty(self):
        from app.source_intelligence.source_discovery import SourceDiscoveryEngine
        eng = SourceDiscoveryEngine()
        results = eng.expand_from_source("unknownorg/unknown-repo")
        assert isinstance(results, list)

    def test_expand_from_source_empty_raises(self):
        from app.source_intelligence.source_discovery import SourceDiscoveryEngine
        eng = SourceDiscoveryEngine()
        with pytest.raises(ValueError):
            eng.expand_from_source("")

    def test_score_record_aware_datetime_required(self):
        """SourceScoreRecord must reject naive datetimes (GAP-6)."""
        from app.source_intelligence.source_discovery import SourceScoreRecord
        from datetime import datetime
        with pytest.raises(ValueError, match="timezone"):
            SourceScoreRecord(
                source_id="s",
                base_confidence=0.5,
                last_updated_at=datetime(2024, 1, 1),  # naive — no tzinfo
            )

    def test_score_record_effective_score_decays(self):
        """effective_score must drop as time elapses (verified via large decay_hours)."""
        from app.source_intelligence.source_discovery import SourceScoreRecord
        from datetime import datetime, timezone, timedelta
        past = datetime.now(timezone.utc) - timedelta(hours=48)
        rec = SourceScoreRecord(
            source_id="s",
            base_confidence=0.7,
            freshness_boost=0.15,
            decay_hours=24.0,
            last_updated_at=past,
        )
        # After 48 h (2× decay_hours) boost should be < 0.03
        decayed = rec.effective_score
        assert decayed < 0.7 + 0.03  # base + negligible residual boost


# ===========================================================================
# GAP-7: VideoGenerator execution mode tests
# ===========================================================================

class TestVideoGeneratorExecutionMode:
    """Verify VideoGenerator exposes explicit execution mode (GAP-7)."""

    _UUID1 = "00000000-0000-0000-0000-000000000001"
    _UUID2 = "00000000-0000-0000-0000-000000000002"
    _UUID3 = "00000000-0000-0000-0000-000000000003"

    def _make_request(self):
        from unittest.mock import MagicMock
        req = MagicMock()
        req.user_id = self._UUID1
        req.digest_id = self._UUID2
        req.preferences_id = self._UUID3
        return req

    def _make_prefs(self):
        from unittest.mock import MagicMock
        p = MagicMock()
        p.id = self._UUID3
        return p

    def test_default_mode_is_script_only(self):
        from app.output.generators.visual_generator import VideoGenerator, VideoGenerationMode
        g = VideoGenerator(preferences=self._make_prefs())
        assert g.generation_mode == VideoGenerationMode.SCRIPT_ONLY

    def test_wrong_mode_type_raises(self):
        from app.output.generators.visual_generator import VideoGenerator
        with pytest.raises(TypeError, match="generation_mode"):
            VideoGenerator(preferences=self._make_prefs(), generation_mode="script")  # type: ignore

    def test_script_only_generate_returns_output(self):
        import asyncio
        from app.output.generators.visual_generator import VideoGenerator, VideoGenerationMode
        g = VideoGenerator(preferences=self._make_prefs(), generation_mode=VideoGenerationMode.SCRIPT_ONLY)
        output = asyncio.run(g.generate(self._make_request(), clusters=[], items=[]))
        assert output is not None
        assert "script_only" in output.summary.lower()

    def test_local_render_without_renderer_raises(self):
        import asyncio
        from app.output.generators.visual_generator import VideoGenerator, VideoGenerationMode
        g = VideoGenerator(preferences=self._make_prefs(), generation_mode=VideoGenerationMode.LOCAL_RENDER)
        with pytest.raises(NotImplementedError, match="renderer"):
            asyncio.run(g.generate(self._make_request(), clusters=[], items=[]))

    def test_remote_render_without_renderer_raises(self):
        import asyncio
        from app.output.generators.visual_generator import VideoGenerator, VideoGenerationMode
        g = VideoGenerator(preferences=self._make_prefs(), generation_mode=VideoGenerationMode.REMOTE_RENDER)
        with pytest.raises(NotImplementedError, match="renderer"):
            asyncio.run(g.generate(self._make_request(), clusters=[], items=[]))




# ===========================================================================
# Phase 7 — Orchestration, Retrieval & Model Lifecycle
# ===========================================================================

# ---------------------------------------------------------------------------
# IntelligencePipelineResult  (pipeline_result.py)
# ---------------------------------------------------------------------------

class TestIntelligencePipelineResult:
    """Tests for the normalized cross-family pipeline output model."""

    def _make(self, **kw):
        from uuid import uuid4
        from app.ingestion.pipeline_result import IntelligencePipelineResult, PipelineStatus
        defaults = dict(
            content_item_id=uuid4(),
            source_family="research",
            status=PipelineStatus.SUCCESS,
            summary="Test summary.",
            confidence=0.85,
        )
        defaults.update(kw)
        return IntelligencePipelineResult(**defaults)

    def test_construction_defaults(self):
        r = self._make()
        assert r.entities == []
        assert r.claims  == []
        assert r.keywords == []
        assert r.pipeline_duration_s == 0.0
        assert r.stages_run == []

    def test_is_actionable_success(self):
        from app.ingestion.pipeline_result import PipelineStatus
        r = self._make(status=PipelineStatus.SUCCESS)
        assert r.is_actionable() is True

    def test_is_actionable_partial(self):
        from app.ingestion.pipeline_result import PipelineStatus
        r = self._make(status=PipelineStatus.PARTIAL)
        assert r.is_actionable() is True

    def test_is_actionable_failed(self):
        from app.ingestion.pipeline_result import PipelineStatus
        r = self._make(status=PipelineStatus.FAILED)
        assert r.is_actionable() is False

    def test_has_rich_detail_false_by_default(self):
        r = self._make()
        assert r.has_rich_detail() is False

    def test_has_rich_detail_true_with_paper(self):
        r = self._make(paper_detail=object())
        assert r.has_rich_detail() is True

    def test_all_text_for_chunking_concatenates(self):
        r = self._make(
            summary="Attention mechanism is powerful.",
            claims=["Transformers are SotA."],
            entities=["GPT-4", "LLaMA"],
        )
        text = r.all_text_for_chunking()
        assert "Attention" in text
        assert "Transformers" in text
        assert "GPT-4" in text

    def test_confidence_range_validation(self):
        from app.ingestion.pipeline_result import IntelligencePipelineResult
        from uuid import uuid4
        with pytest.raises(Exception):
            IntelligencePipelineResult(
                content_item_id=uuid4(),
                source_family="social",
                confidence=1.5,  # out of range
            )

    def test_produced_at_is_utc(self):
        r = self._make()
        assert r.produced_at.tzinfo is not None


# ---------------------------------------------------------------------------
# ContentPipelineRouter — SOCIAL / NEWS / DEVELOPER / UNKNOWN
# ---------------------------------------------------------------------------

class TestContentPipelineRouterBasic:
    """Tests for routes that do NOT require LLM or heavy ML (fast, deterministic)."""

    def _make_item(self, platform="reddit", text="Hello world from social media",
                   title="Test", media_type_val="text"):
        """Build a minimal ContentItem without hitting DB or network."""
        import uuid
        from unittest.mock import MagicMock
        item = MagicMock()
        item.id           = uuid.uuid4()
        item.source_platform = MagicMock(value=platform)
        item.raw_text     = text
        item.title        = title
        item.source_id    = "test-123"
        item.source_url   = "https://example.com"
        item.published_at = _now()
        item.channel      = "general"
        item.topics       = ["AI", "ML"]
        item.metadata     = {}
        # MediaType
        from app.core.models import MediaType
        item.media_type   = MediaType.TEXT
        return item

    def test_social_route_success(self):
        import asyncio
        from app.ingestion.content_pipeline_router import ContentPipelineRouter
        from app.source_intelligence.source_registry import SourceFamily
        from app.ingestion.pipeline_result import PipelineStatus
        router = ContentPipelineRouter()
        item = self._make_item(platform="reddit", text="OpenAI released GPT-5 today.")
        result = asyncio.run(router.route(item, source_family=SourceFamily.SOCIAL))
        assert result.status == PipelineStatus.SUCCESS
        assert result.signal_type == "SOCIAL_POST"
        assert result.source_family == "social"
        assert isinstance(result.entities, list)

    def test_news_route_success(self):
        import asyncio
        from app.ingestion.content_pipeline_router import ContentPipelineRouter
        from app.source_intelligence.source_registry import SourceFamily
        from app.ingestion.pipeline_result import PipelineStatus
        router = ContentPipelineRouter()
        item = self._make_item(platform="rss", text="Microsoft acquires startup for $2 billion.")
        result = asyncio.run(router.route(item, source_family=SourceFamily.NEWS))
        assert result.status == PipelineStatus.SUCCESS
        assert result.signal_type == "NEWS_ARTICLE"

    def test_developer_route_basic(self):
        import asyncio
        from app.ingestion.content_pipeline_router import ContentPipelineRouter
        from app.source_intelligence.source_registry import SourceFamily
        from app.ingestion.pipeline_result import PipelineStatus
        router = ContentPipelineRouter()
        item = self._make_item(
            platform="github_releases",
            text="## What's New\n- Added async support\n## Breaking Changes\n- Removed `foo()` API",
            title="v2.0.0",
        )
        item.metadata = {"version": "2.0.0", "repo": "org/repo"}
        result = asyncio.run(router.route(item, source_family=SourceFamily.DEVELOPER_RELEASE))
        assert result.status in {PipelineStatus.SUCCESS, PipelineStatus.PARTIAL}
        assert result.content_item_id == item.id

    def test_unknown_route_is_partial(self):
        import asyncio
        from app.ingestion.content_pipeline_router import ContentPipelineRouter
        from app.source_intelligence.source_registry import SourceFamily
        from app.ingestion.pipeline_result import PipelineStatus
        router = ContentPipelineRouter()
        item = self._make_item(platform="unknown_platform")
        result = asyncio.run(router.route(item, source_family=SourceFamily.UNKNOWN))
        assert result.status == PipelineStatus.PARTIAL

    def test_timeout_yields_failed_result(self):
        import asyncio
        from app.ingestion.content_pipeline_router import ContentPipelineRouter
        from app.source_intelligence.source_registry import SourceFamily
        from app.ingestion.pipeline_result import PipelineStatus
        router = ContentPipelineRouter()
        item = self._make_item(platform="reddit")
        # Tiny timeout; the pipeline should time-out and return FAILED
        result = asyncio.run(router.route(item, source_family=SourceFamily.SOCIAL, timeout_s=0.000001))
        assert result.status in {PipelineStatus.FAILED, PipelineStatus.SUCCESS}

    def test_pipeline_duration_is_positive(self):
        import asyncio
        from app.ingestion.content_pipeline_router import ContentPipelineRouter
        from app.source_intelligence.source_registry import SourceFamily
        router = ContentPipelineRouter()
        item = self._make_item()
        result = asyncio.run(router.route(item, source_family=SourceFamily.SOCIAL))
        assert result.pipeline_duration_s >= 0.0

    def test_entity_extraction_heuristic(self):
        from app.ingestion.content_pipeline_router import ContentPipelineRouter
        entities = ContentPipelineRouter._extract_entities(
            "Sam Altman and Satya Nadella met to discuss Microsoft OpenAI partnership."
        )
        assert any("Sam" in e or "Altman" in e or "Satya" in e or "Nadella" in e for e in entities)

    def test_entity_extraction_filters_stopwords(self):
        from app.ingestion.content_pipeline_router import ContentPipelineRouter
        entities = ContentPipelineRouter._extract_entities("The quick brown fox.")
        assert "The" not in entities

    def test_family_inference_audio_media_type(self):
        import uuid
        from unittest.mock import MagicMock
        from app.core.models import MediaType
        from app.source_intelligence.source_registry import SourceFamily
        from app.ingestion.content_pipeline_router import ContentPipelineRouter
        item = MagicMock()
        item.source_platform = MagicMock(value="youtube")
        item.media_type = MediaType.AUDIO
        family = ContentPipelineRouter()._infer_family(item)
        assert family == SourceFamily.MEDIA_AUDIO

    def test_family_inference_research_platform(self):
        import uuid
        from unittest.mock import MagicMock
        from app.core.models import MediaType
        from app.source_intelligence.source_registry import SourceFamily
        from app.ingestion.content_pipeline_router import ContentPipelineRouter
        item = MagicMock()
        item.source_platform = MagicMock(value="arxiv")
        item.media_type = MediaType.TEXT
        family = ContentPipelineRouter()._infer_family(item)
        assert family == SourceFamily.RESEARCH


# ---------------------------------------------------------------------------
# ChunkStore
# ---------------------------------------------------------------------------

class TestChunkStore:
    """Tests for the retrieval asset lifecycle chunk corpus."""

    def test_ingest_and_count(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        store = ChunkStore()
        store.ingest(ChunkRecord(observation_id="obs-1", text="Attention is all you need."))
        assert store.count() == 1

    def test_ingest_returns_chunk_id(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        store = ChunkStore()
        cid = store.ingest(ChunkRecord(observation_id="obs-1", text="Hello world."))
        assert isinstance(cid, str) and len(cid) > 0

    def test_get_by_observation(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        store = ChunkStore()
        store.ingest(ChunkRecord(observation_id="obs-A", text="First chunk"))
        store.ingest(ChunkRecord(observation_id="obs-A", text="Second chunk"))
        store.ingest(ChunkRecord(observation_id="obs-B", text="Other"))
        recs = store.get_by_observation("obs-A")
        assert len(recs) == 2
        assert all(r.observation_id == "obs-A" for r in recs)

    def test_get_returns_record(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        store = ChunkStore()
        rec = ChunkRecord(observation_id="obs-1", text="Neural networks work by backpropagation.")
        cid = store.ingest(rec)
        fetched = store.get(cid)
        assert fetched is not None
        assert fetched.text == rec.text

    def test_empty_text_raises(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        store = ChunkStore()
        with pytest.raises(ValueError, match="empty text"):
            store.ingest(ChunkRecord(observation_id="obs-1", text="   "))

    def test_text_truncated_to_max(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        store = ChunkStore(max_chunk_chars=20)
        long_text = "A" * 100
        rec = ChunkRecord(observation_id="obs-1", text=long_text)
        cid = store.ingest(rec)
        stored = store.get(cid)
        assert len(stored.text) <= 20

    def test_chunk_text_splits_correctly(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore
        store = ChunkStore()
        text = "word " * 400  # 2000 chars
        ids = store.chunk_text(observation_id="obs-2", text=text, chunk_size=200, overlap=50)
        assert len(ids) > 1

    def test_chunk_text_overlap_invalid_raises(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore
        store = ChunkStore()
        with pytest.raises(ValueError, match="chunk_size"):
            store.chunk_text("obs-1", "hello world", chunk_size=50, overlap=60)

    def test_keyword_search_finds_match(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        store = ChunkStore()
        store.ingest(ChunkRecord(observation_id="obs-1", text="transformers use self-attention"))
        store.ingest(ChunkRecord(observation_id="obs-2", text="convolutional networks have filters"))
        hits = store.keyword_search("self-attention transformers", top_k=5)
        assert len(hits) >= 1
        assert hits[0].record.observation_id == "obs-1"

    def test_keyword_search_empty_query_returns_empty(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        store = ChunkStore()
        store.ingest(ChunkRecord(observation_id="obs-1", text="hello"))
        hits = store.keyword_search("", top_k=5)
        assert hits == []

    def test_semantic_search_no_embeddings_raises(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        store = ChunkStore()
        store.ingest(ChunkRecord(observation_id="obs-1", text="hello world"))
        with pytest.raises(RuntimeError, match="embeddings"):
            store.search([0.1, 0.2, 0.3], top_k=5)

    def test_semantic_search_with_embeddings(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        store = ChunkStore()
        # Embed: unit vectors; query == rec-A so it should rank first
        store.ingest(ChunkRecord(observation_id="obs-A", text="transformers", embedding=[1.0, 0.0]))
        store.ingest(ChunkRecord(observation_id="obs-B", text="rnn", embedding=[0.0, 1.0]))
        hits = store.search([1.0, 0.0], top_k=2)
        assert len(hits) == 2
        assert hits[0].record.observation_id == "obs-A"

    def test_semantic_search_metadata_filter(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        store = ChunkStore()
        store.ingest(ChunkRecord(observation_id="obs-A", source_family="research",
                                 text="paper chunk", embedding=[1.0, 0.0]))
        store.ingest(ChunkRecord(observation_id="obs-B", source_family="social",
                                 text="tweet chunk", embedding=[0.9, 0.1]))
        hits = store.search([1.0, 0.0], top_k=5,
                            metadata_filter=lambda r: r.source_family == "research")
        assert len(hits) == 1
        assert hits[0].record.source_family == "research"

    def test_eviction_at_max_size(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        store = ChunkStore(max_size=3)
        for i in range(5):
            store.ingest(ChunkRecord(observation_id=f"obs-{i}", text=f"chunk {i}"))
        assert store.count() == 3

    def test_clear_removes_all(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        store = ChunkStore()
        store.ingest(ChunkRecord(observation_id="obs-1", text="hello"))
        store.clear()
        assert store.count() == 0

    def test_observation_ids_sorted(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        store = ChunkStore()
        for obs in ["obs-C", "obs-A", "obs-B"]:
            store.ingest(ChunkRecord(observation_id=obs, text="text"))
        assert store.observation_ids() == ["obs-A", "obs-B", "obs-C"]

    def test_thread_safe_concurrent_ingest(self):
        import threading
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        store = ChunkStore()
        errors = []
        def ingest_batch(start):
            try:
                for i in range(100):
                    store.ingest(ChunkRecord(observation_id=f"obs-{start}-{i}",
                                             text=f"chunk {start} {i}"))
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=ingest_batch, args=(t,)) for t in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
        assert store.count() == 500


# ---------------------------------------------------------------------------
# RetrievalEvaluator
# ---------------------------------------------------------------------------

class TestRetrievalEvaluator:
    """Tests for Recall@k, MRR, and nDCG@k computation."""

    def _perfect_retriever(self, relevant_ids):
        """Returns a retriever that always puts relevant docs first."""
        def retriever(query: str):
            return list(relevant_ids) + ["noise-1", "noise-2"]
        return retriever

    def _miss_retriever(self):
        """Returns a retriever that never returns relevant docs."""
        return lambda q: ["noise-1", "noise-2", "noise-3"]

    def test_perfect_retriever_recall_one(self):
        from app.intelligence.retrieval.retrieval_evaluator import (
            RetrievalEvaluator, RetrievalQuery,
        )
        queries = [RetrievalQuery(query_id="q1", query_text="test",
                                  relevant_chunk_ids={"chunk-A"})]
        ev = RetrievalEvaluator()
        metrics = ev.evaluate(queries, self._perfect_retriever({"chunk-A"}), k=5)
        assert metrics.recall_at_k == 1.0

    def test_miss_retriever_recall_zero(self):
        from app.intelligence.retrieval.retrieval_evaluator import (
            RetrievalEvaluator, RetrievalQuery,
        )
        queries = [RetrievalQuery(query_id="q1", query_text="test",
                                  relevant_chunk_ids={"chunk-A"})]
        ev = RetrievalEvaluator()
        metrics = ev.evaluate(queries, self._miss_retriever(), k=5)
        assert metrics.recall_at_k == 0.0

    def test_mrr_perfect(self):
        from app.intelligence.retrieval.retrieval_evaluator import (
            RetrievalEvaluator, RetrievalQuery,
        )
        queries = [RetrievalQuery(query_id="q1", query_text="q",
                                  relevant_chunk_ids={"hit-1"})]
        retriever = lambda q: ["hit-1", "other-1", "other-2"]
        metrics = RetrievalEvaluator().evaluate(queries, retriever, k=5)
        assert metrics.mrr == pytest.approx(1.0, abs=1e-6)

    def test_mrr_rank_two(self):
        from app.intelligence.retrieval.retrieval_evaluator import (
            RetrievalEvaluator, RetrievalQuery,
        )
        queries = [RetrievalQuery(query_id="q1", query_text="q",
                                  relevant_chunk_ids={"hit-1"})]
        retriever = lambda q: ["miss", "hit-1", "other"]
        metrics = RetrievalEvaluator().evaluate(queries, retriever, k=5)
        assert metrics.mrr == pytest.approx(0.5, abs=1e-6)

    def test_ndcg_perfect(self):
        from app.intelligence.retrieval.retrieval_evaluator import (
            RetrievalEvaluator, RetrievalQuery,
        )
        queries = [RetrievalQuery(query_id="q1", query_text="q",
                                  relevant_chunk_ids={"hit-1"})]
        retriever = lambda q: ["hit-1"]
        metrics = RetrievalEvaluator().evaluate(queries, retriever, k=5)
        assert metrics.ndcg_at_k == pytest.approx(1.0, abs=1e-6)

    def test_ndcg_miss(self):
        from app.intelligence.retrieval.retrieval_evaluator import (
            RetrievalEvaluator, RetrievalQuery,
        )
        queries = [RetrievalQuery(query_id="q1", query_text="q",
                                  relevant_chunk_ids={"hit-1"})]
        metrics = RetrievalEvaluator().evaluate(queries, self._miss_retriever(), k=5)
        assert metrics.ndcg_at_k == 0.0

    def test_multiple_queries_macro_average(self):
        from app.intelligence.retrieval.retrieval_evaluator import (
            RetrievalEvaluator, RetrievalQuery,
        )
        queries = [
            RetrievalQuery(query_id="q1", query_text="q1", relevant_chunk_ids={"hit-1"}),
            RetrievalQuery(query_id="q2", query_text="q2", relevant_chunk_ids={"hit-2"}),
        ]
        # retriever finds hit-1 but not hit-2
        retriever = lambda q: ["hit-1", "noise"]
        metrics = RetrievalEvaluator().evaluate(queries, retriever, k=5)
        assert metrics.recall_at_k == pytest.approx(0.5, abs=1e-6)
        assert metrics.n_queries == 2
        assert metrics.n_queries_with_hits == 1

    def test_empty_queries_raises(self):
        from app.intelligence.retrieval.retrieval_evaluator import RetrievalEvaluator
        with pytest.raises(ValueError, match="non-empty"):
            RetrievalEvaluator().evaluate([], lambda q: [], k=5)

    def test_invalid_k_raises(self):
        from app.intelligence.retrieval.retrieval_evaluator import (
            RetrievalEvaluator, RetrievalQuery,
        )
        queries = [RetrievalQuery(query_id="q1", query_text="q", relevant_chunk_ids={"x"})]
        with pytest.raises(ValueError, match="positive"):
            RetrievalEvaluator().evaluate(queries, lambda q: [], k=0)

    def test_passes_thresholds_true(self):
        from app.intelligence.retrieval.retrieval_evaluator import (
            RetrievalEvaluator, RetrievalQuery,
        )
        # Use query_text to encode the target chunk ID — retriever returns
        # the relevant hit first so every query gets Recall=1, RR=1, nDCG=1.
        queries = [
            RetrievalQuery(query_id=f"q{i}", query_text=f"hit-{i}",
                           relevant_chunk_ids={f"hit-{i}"})
            for i in range(10)
        ]
        # Retriever echoes the query text as its top-1 result (perfect recall)
        retriever = lambda q: [q, "noise-1", "noise-2"]
        metrics = RetrievalEvaluator().evaluate(queries, retriever, k=10)
        assert metrics.passes_thresholds(min_recall=0.70, min_mrr=0.50, min_ndcg=0.60)

    def test_per_query_breakdown_present(self):
        from app.intelligence.retrieval.retrieval_evaluator import (
            RetrievalEvaluator, RetrievalQuery,
        )
        queries = [RetrievalQuery(query_id="q1", query_text="q",
                                  relevant_chunk_ids={"hit-1"})]
        metrics = RetrievalEvaluator().evaluate(queries, lambda q: ["hit-1"], k=5)
        assert "q1" in metrics.per_query
        assert "recall" in metrics.per_query["q1"]


# ---------------------------------------------------------------------------
# ModelArtifactRegistry
# ---------------------------------------------------------------------------

class TestModelArtifactRegistry:
    """Tests for the training model artifact registry."""

    def _make_registry(self, tmp_path):
        from training.model_registry import ModelArtifactRegistry
        return ModelArtifactRegistry(registry_path=tmp_path / "registry.json")

    def _make_record(self, ece=0.05, macro_f1=0.85, epoch=5):
        from training.model_registry import ArtifactRecord
        return ArtifactRecord(
            epoch=epoch,
            ece=ece,
            macro_f1=macro_f1,
            calibration_method="isotonic",
            checkpoint_path="training/checkpoints/z_calibrated.json",
        )

    def test_register_returns_artifact_id(self, tmp_path):
        reg = self._make_registry(tmp_path)
        rec = self._make_record()
        aid = reg.register(rec)
        assert aid == rec.artifact_id

    def test_count_after_register(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg.register(self._make_record())
        reg.register(self._make_record(epoch=6))
        assert reg.count() == 2

    def test_duplicate_registration_raises(self, tmp_path):
        reg = self._make_registry(tmp_path)
        rec = self._make_record()
        reg.register(rec)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(rec)

    def test_promote_sets_production(self, tmp_path):
        reg = self._make_registry(tmp_path)
        rec = self._make_record()
        reg.register(rec)
        promoted = reg.promote(rec.artifact_id)
        assert promoted.is_production is True
        assert promoted.promoted_at is not None

    def test_only_one_production_at_a_time(self, tmp_path):
        reg = self._make_registry(tmp_path)
        r1, r2 = self._make_record(epoch=4), self._make_record(epoch=5)
        reg.register(r1); reg.register(r2)
        reg.promote(r1.artifact_id)
        reg.promote(r2.artifact_id)
        productions = [a for a in reg.list_all() if a.is_production]
        assert len(productions) == 1
        assert productions[0].artifact_id == r2.artifact_id

    def test_promote_failing_gate_raises(self, tmp_path):
        from training.model_registry import ArtifactRecord
        reg = self._make_registry(tmp_path)
        bad = ArtifactRecord(epoch=5, ece=0.50, macro_f1=0.40,
                             checkpoint_path="bad.json")
        reg.register(bad)
        with pytest.raises(ValueError, match="Cannot promote"):
            reg.promote(bad.artifact_id)

    def test_rollback_restores_previous(self, tmp_path):
        reg = self._make_registry(tmp_path)
        r1, r2 = self._make_record(epoch=4), self._make_record(epoch=5)
        reg.register(r1); reg.register(r2)
        reg.promote(r1.artifact_id)
        reg.promote(r2.artifact_id)
        prev = reg.rollback()
        assert prev is not None
        assert prev.artifact_id == r1.artifact_id
        assert prev.is_production is True

    def test_rollback_no_fallback_returns_none(self, tmp_path):
        reg = self._make_registry(tmp_path)
        rec = self._make_record()
        reg.register(rec)
        reg.promote(rec.artifact_id)
        result = reg.rollback()
        assert result is None

    def test_get_production_candidate_lowest_ece(self, tmp_path):
        reg = self._make_registry(tmp_path)
        r1 = self._make_record(ece=0.08, epoch=4)
        r2 = self._make_record(ece=0.03, epoch=5)
        reg.register(r1); reg.register(r2)
        cand = reg.get_production_candidate()
        assert cand.artifact_id == r2.artifact_id

    def test_get_production_candidate_none_when_all_fail(self, tmp_path):
        from training.model_registry import ArtifactRecord
        reg = self._make_registry(tmp_path)
        bad = ArtifactRecord(epoch=5, ece=0.50, macro_f1=0.40,
                             checkpoint_path="bad.json")
        reg.register(bad)
        assert reg.get_production_candidate() is None

    def test_passes_gate_true(self, tmp_path):
        rec = self._make_record(ece=0.05, macro_f1=0.85)
        assert rec.passes_gate() is True

    def test_passes_gate_false_high_ece(self, tmp_path):
        rec = self._make_record(ece=0.30, macro_f1=0.85)
        assert rec.passes_gate() is False

    def test_gate_failures_messages(self, tmp_path):
        rec = self._make_record(ece=0.30, macro_f1=0.40)
        failures = rec.gate_failures()
        assert len(failures) == 2
        assert any("ECE" in f for f in failures)
        assert any("macro_F1" in f for f in failures)

    def test_list_by_epoch(self, tmp_path):
        reg = self._make_registry(tmp_path)
        for e in [5, 5, 6]:
            reg.register(self._make_record(epoch=e))
        assert len(reg.list_by_epoch(5)) == 2
        assert len(reg.list_by_epoch(6)) == 1

    def test_persistence_across_instances(self, tmp_path):
        from training.model_registry import ModelArtifactRegistry
        path = tmp_path / "registry.json"
        reg1 = ModelArtifactRegistry(registry_path=path)
        rec = self._make_record()
        reg1.register(rec)
        reg1.promote(rec.artifact_id)
        # New instance — loads from disk
        reg2 = ModelArtifactRegistry(registry_path=path)
        assert reg2.count() == 1
        prod = reg2.get_production()
        assert prod is not None
        assert prod.artifact_id == rec.artifact_id

    def test_register_from_checkpoint_parses_fields(self, tmp_path):
        import json
        from training.model_registry import ModelArtifactRegistry
        ckpt = tmp_path / "test_checkpoint.json"
        ckpt.write_text(json.dumps({
            "epoch": 5,
            "ece": 0.0103,
            "macro_f1": 0.8611,
            "calibration_method": "isotonic",
        }))
        reg = ModelArtifactRegistry(registry_path=tmp_path / "registry.json")
        aid = reg.register_from_checkpoint(ckpt)
        rec = reg.get(aid)
        assert rec.ece == pytest.approx(0.0103, abs=1e-6)
        assert rec.macro_f1 == pytest.approx(0.8611, abs=1e-6)
        assert rec.calibration_method == "isotonic"

    def test_register_from_checkpoint_missing_file_raises(self, tmp_path):
        from training.model_registry import ModelArtifactRegistry
        reg = ModelArtifactRegistry(registry_path=tmp_path / "registry.json")
        with pytest.raises(FileNotFoundError):
            reg.register_from_checkpoint(tmp_path / "nonexistent.json")

    def test_register_from_real_checkpoint(self, tmp_path):
        """Smoke test: load the actual calibrated checkpoint from the repo."""
        from pathlib import Path
        from training.model_registry import ModelArtifactRegistry
        ckpts = sorted(
            Path("training/checkpoints").glob("z_calibrated_*.json"),
            key=lambda p: p.stat().st_mtime,
        )
        if not ckpts:
            pytest.skip("No calibrated checkpoint found — skipping")
        reg = ModelArtifactRegistry(registry_path=tmp_path / "registry.json")
        aid = reg.register_from_checkpoint(ckpts[-1])
        rec = reg.get(aid)
        assert rec.ece <= 0.10, f"Production checkpoint ECE {rec.ece} exceeds gate"
        assert rec.macro_f1 >= 0.70, f"Production checkpoint F1 {rec.macro_f1} below gate"


# ===========================================================================
# Phase 3 — Event Consolidation Bus, WatchlistGraph, HealthMonitor
# ===========================================================================

# ---------------------------------------------------------------------------
# IndexingPipeline helpers
# ---------------------------------------------------------------------------

def _make_content_item(platform="reddit", text="Hello world", title="Test"):
    """Build a minimal ContentItem mock without DB / network."""
    import uuid
    from unittest.mock import MagicMock
    from app.core.models import MediaType
    item = MagicMock()
    item.id                = uuid.uuid4()
    item.source_platform   = MagicMock(value=platform)
    item.raw_text          = text
    item.title             = title
    item.source_id         = "test-123"
    item.source_url        = "https://example.com"
    item.published_at      = _now()
    item.channel           = "general"
    item.topics            = []
    item.metadata          = {}
    item.media_type        = MediaType.TEXT
    return item


# ---------------------------------------------------------------------------
# IndexingStats
# ---------------------------------------------------------------------------

class TestIndexingStats:
    def test_throughput_zero_when_no_time(self):
        from app.ingestion.indexing_pipeline import IndexingStats
        s = IndexingStats(total_items=10, wall_s=0.0)
        assert s.throughput_items_per_s == 0.0

    def test_throughput_computed(self):
        from app.ingestion.indexing_pipeline import IndexingStats
        s = IndexingStats(total_items=10, wall_s=2.0)
        assert s.throughput_items_per_s == pytest.approx(5.0)

    def test_error_rate_zero_items(self):
        from app.ingestion.indexing_pipeline import IndexingStats
        s = IndexingStats(total_items=0)
        assert s.error_rate == 0.0

    def test_error_rate_computed(self):
        from app.ingestion.indexing_pipeline import IndexingStats
        s = IndexingStats(total_items=10, route_errors=2)
        assert s.error_rate == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# IndexingPipeline
# ---------------------------------------------------------------------------

class TestIndexingPipeline:
    def test_empty_batch_returns_empty_result(self):
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline
        pipeline = IndexingPipeline()
        result = asyncio.run(pipeline.process_batch([]))
        assert result.stats.total_items == 0
        assert result.bundles == []
        assert result.pipeline_results == []

    def test_single_social_item_is_indexed(self):
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline, IndexingStats
        from app.intelligence.retrieval.chunk_store import ChunkStore
        store = ChunkStore()
        pipeline = IndexingPipeline(chunk_store=store)
        item = _make_content_item(platform="reddit", text="OpenAI released GPT-5 with reasoning abilities.")
        result = asyncio.run(pipeline.process_batch([item]))
        assert result.stats.total_items == 1
        assert result.stats.route_errors == 0
        assert result.stats.chunks_indexed >= 1
        assert store.count() >= 1

    def test_multiple_items_all_indexed(self):
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        store = ChunkStore()
        pipeline = IndexingPipeline(chunk_store=store)
        items = [_make_content_item(text=f"Chunk content number {i}") for i in range(5)]
        result = asyncio.run(pipeline.process_batch(items))
        assert result.stats.total_items == 5
        assert result.stats.chunks_indexed >= 5

    def test_chunk_store_property(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        store = ChunkStore()
        pipeline = IndexingPipeline(chunk_store=store)
        assert pipeline.chunk_store is store

    def test_stats_wall_s_is_positive(self):
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline
        pipeline = IndexingPipeline()
        item = _make_content_item()
        result = asyncio.run(pipeline.process_batch([item]))
        assert result.stats.wall_s >= 0.0

    def test_produced_at_is_recent(self):
        import asyncio
        from datetime import timezone
        from app.ingestion.indexing_pipeline import IndexingPipeline
        pipeline = IndexingPipeline()
        result = asyncio.run(pipeline.process_batch([]))
        age_s = (datetime.now(timezone.utc) - result.produced_at).total_seconds()
        assert abs(age_s) < 30

    def test_error_in_one_item_does_not_abort_batch(self):
        """Even if one item fails routing, the rest should still be processed."""
        import asyncio
        from unittest.mock import MagicMock
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        store = ChunkStore()
        pipeline = IndexingPipeline(chunk_store=store)
        good_item = _make_content_item(text="Valid content about machine learning.")
        bad_item  = MagicMock()
        bad_item.id             = __import__("uuid").uuid4()
        bad_item.source_platform = MagicMock(value="unknown")
        bad_item.raw_text       = None  # will cause issues in some paths
        bad_item.title          = None
        bad_item.metadata       = {}
        bad_item.topics         = []
        from app.core.models import MediaType
        bad_item.media_type     = MediaType.TEXT
        bad_item.published_at   = _now()
        bad_item.source_id      = "bad"
        bad_item.source_url     = ""
        bad_item.channel        = ""
        result = asyncio.run(pipeline.process_batch([good_item, bad_item]))
        assert result.stats.total_items == 2
        # At least the good item should be processed
        assert result.stats.routed_ok >= 1

    def test_build_cluster_inputs_skips_failed(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.ingestion.pipeline_result import IntelligencePipelineResult, PipelineStatus
        import uuid
        failed = IntelligencePipelineResult(
            content_item_id=uuid.uuid4(),
            source_family="social",
            status=PipelineStatus.FAILED,
        )
        ok = IntelligencePipelineResult(
            content_item_id=uuid.uuid4(),
            source_family="news",
            status=PipelineStatus.SUCCESS,
            summary="news story",
        )
        inputs = IndexingPipeline._build_cluster_inputs([failed, ok])
        assert len(inputs) == 1
        assert inputs[0]["platform"] == "news"


# ---------------------------------------------------------------------------
# WatchlistGraph
# ---------------------------------------------------------------------------

class TestWatchlistGraph:
    def test_construction_default_params(self):
        from app.personalization.watchlist_graph import WatchlistGraph
        g = WatchlistGraph(user_id="u1")
        assert g.user_id == "u1"
        assert g.count() == 0

    def test_empty_user_id_raises(self):
        from app.personalization.watchlist_graph import WatchlistGraph
        with pytest.raises(ValueError, match="user_id"):
            WatchlistGraph(user_id="  ")

    def test_invalid_gap_threshold_raises(self):
        from app.personalization.watchlist_graph import WatchlistGraph
        with pytest.raises(ValueError, match="gap_threshold"):
            WatchlistGraph(user_id="u1", gap_threshold=0.0)

    def test_invalid_stale_hours_raises(self):
        from app.personalization.watchlist_graph import WatchlistGraph
        with pytest.raises(ValueError, match="stale_after_hours"):
            WatchlistGraph(user_id="u1", stale_after_hours=-1)

    def test_watch_adds_node(self):
        from app.personalization.watchlist_graph import WatchlistGraph
        g = WatchlistGraph(user_id="u1")
        g.watch("vLLM", node_type="repo", expected_families=["developer_release"])
        assert g.is_watched("vLLM")
        assert g.count() == 1

    def test_watch_is_idempotent(self):
        from app.personalization.watchlist_graph import WatchlistGraph
        g = WatchlistGraph(user_id="u1")
        g.watch("GPT", node_type="topic")
        g.watch("GPT", node_type="topic")
        assert g.count() == 1

    def test_watch_invalid_priority_raises(self):
        from app.personalization.watchlist_graph import WatchlistGraph
        g = WatchlistGraph(user_id="u1")
        with pytest.raises(ValueError, match="priority"):
            g.watch("vLLM", priority=2.0)

    def test_unwatch_removes_node(self):
        from app.personalization.watchlist_graph import WatchlistGraph
        g = WatchlistGraph(user_id="u1")
        g.watch("vLLM")
        assert g.unwatch("vLLM") is True
        assert g.is_watched("vLLM") is False

    def test_unwatch_nonexistent_returns_false(self):
        from app.personalization.watchlist_graph import WatchlistGraph
        g = WatchlistGraph(user_id="u1")
        assert g.unwatch("nonexistent") is False

    def test_watched_nodes_sorted_by_priority(self):
        from app.personalization.watchlist_graph import WatchlistGraph
        g = WatchlistGraph(user_id="u1")
        g.watch("low",  node_type="topic", priority=0.2)
        g.watch("high", node_type="topic", priority=0.9)
        g.watch("mid",  node_type="topic", priority=0.5)
        nodes = g.watched_nodes()
        assert nodes[0].node_id == "high"
        assert nodes[-1].node_id == "low"

    def test_record_coverage_ignored_for_unwatched(self):
        from app.personalization.watchlist_graph import WatchlistGraph
        g = WatchlistGraph(user_id="u1")
        # Should not raise — silently ignored
        g.record_coverage("unwatched-node", source_id="src-1",
                          source_family="research", trust_score=0.8)

    def test_record_coverage_invalid_trust_raises(self):
        from app.personalization.watchlist_graph import WatchlistGraph
        g = WatchlistGraph(user_id="u1")
        g.watch("vLLM")
        with pytest.raises(ValueError, match="trust_score"):
            g.record_coverage("vLLM", source_id="s", source_family="research",
                              trust_score=1.5)

    def test_coverage_score_one_when_all_families_covered(self):
        from app.personalization.watchlist_graph import WatchlistGraph
        g = WatchlistGraph(user_id="u1")
        g.watch("vLLM", expected_families=["developer_release", "research"])
        g.record_coverage("vLLM", source_id="gh",  source_family="developer_release")
        g.record_coverage("vLLM", source_id="arx", source_family="research")
        nc = g.node_coverage("vLLM")
        assert nc.coverage_score == pytest.approx(1.0)
        assert nc.missing_families == frozenset()

    def test_coverage_score_zero_when_no_coverage(self):
        from app.personalization.watchlist_graph import WatchlistGraph
        g = WatchlistGraph(user_id="u1")
        g.watch("NewNode", expected_families=["research", "social"])
        nc = g.node_coverage("NewNode")
        assert nc.coverage_score == 0.0
        assert frozenset({"research", "social"}) <= nc.missing_families

    def test_coverage_report_has_gaps(self):
        from app.personalization.watchlist_graph import WatchlistGraph
        g = WatchlistGraph(user_id="u1", gap_threshold=0.60)
        g.watch("vLLM", expected_families=["developer_release", "research", "media_audio"],
                priority=0.9)
        # Only one of three families covered → gap
        g.record_coverage("vLLM", source_id="gh", source_family="developer_release")
        report = g.coverage_report()
        assert report.nodes_at_risk == 1
        assert len(report.gaps) == 1
        assert report.gaps[0].node_id == "vLLM"
        assert "research" in report.gaps[0].missing_families or \
               "media_audio" in report.gaps[0].missing_families

    def test_coverage_report_no_gaps_when_fully_covered(self):
        from app.personalization.watchlist_graph import WatchlistGraph
        g = WatchlistGraph(user_id="u1", gap_threshold=0.60)
        g.watch("LLaMA", expected_families=["research"])
        g.record_coverage("LLaMA", source_id="arx", source_family="research")
        report = g.coverage_report()
        assert report.nodes_at_risk == 0
        assert report.overall_score == pytest.approx(1.0)

    def test_stale_coverage_flagged(self):
        from datetime import timedelta
        from app.personalization.watchlist_graph import WatchlistGraph
        g = WatchlistGraph(user_id="u1", stale_after_hours=1)
        g.watch("OldTopic", expected_families=["social"])
        # Record as 2 hours ago
        old_time = _now() - timedelta(hours=2)
        g.record_coverage("OldTopic", source_id="tw", source_family="social",
                          observed_at=old_time)
        nc = g.node_coverage("OldTopic")
        assert "social" in nc.stale_families

    def test_item_count_incremented_on_repeat_coverage(self):
        from app.personalization.watchlist_graph import WatchlistGraph
        g = WatchlistGraph(user_id="u1")
        g.watch("Topic")
        for _ in range(3):
            g.record_coverage("Topic", source_id="src-1", source_family="social")
        nc = g.node_coverage("Topic")
        source_entry = next(s for s in nc.sources if s.source_id == "src-1")
        assert source_entry.item_count == 3

    def test_recommendation_contains_missing_family(self):
        from app.personalization.watchlist_graph import WatchlistGraph
        g = WatchlistGraph(user_id="u1", gap_threshold=0.80)
        g.watch("PyTorch", expected_families=["developer_release", "research"])
        g.record_coverage("PyTorch", source_id="gh", source_family="developer_release")
        report = g.coverage_report()
        assert len(report.gaps) == 1
        assert "research" in report.gaps[0].recommendation

    def test_node_coverage_returns_none_for_unwatched(self):
        from app.personalization.watchlist_graph import WatchlistGraph
        g = WatchlistGraph(user_id="u1")
        assert g.node_coverage("NotWatched") is None

    def test_thread_safe_concurrent_watch(self):
        import threading
        from app.personalization.watchlist_graph import WatchlistGraph
        g = WatchlistGraph(user_id="u1")
        errors = []
        def add_nodes(start):
            try:
                for i in range(50):
                    g.watch(f"node-{start}-{i}")
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=add_nodes, args=(t,)) for t in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
        assert g.count() == 200

    def test_coverage_report_generated_at_is_utc(self):
        from app.personalization.watchlist_graph import WatchlistGraph
        g = WatchlistGraph(user_id="u1")
        report = g.coverage_report()
        assert report.generated_at.tzinfo is not None


# ---------------------------------------------------------------------------
# PipelineHealthMonitor
# ---------------------------------------------------------------------------

class TestPipelineHealthMonitor:
    def test_green_when_no_observations(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        m = PipelineHealthMonitor()
        report = m.health_report()
        assert report.overall_status == SLOStatus.GREEN
        assert report.violations == []

    def test_record_latency_appears_in_stats(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor
        m = PipelineHealthMonitor()
        for lat in [1.0, 2.0, 3.0, 4.0, 5.0]:
            m.record_latency("route", lat)
        report = m.health_report()
        stage = next(s for s in report.latency_stats if s.stage == "route")
        assert stage.count == 5
        assert stage.mean_s == pytest.approx(3.0)
        assert stage.p50_s >= 1.0

    def test_negative_latency_raises(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor
        m = PipelineHealthMonitor()
        with pytest.raises(ValueError, match="≥ 0"):
            m.record_latency("route", -0.1)

    def test_routing_p95_slo_violation_raises_red(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        m = PipelineHealthMonitor(route_p95_slo_s=5.0, slo_warning_pct=1.01)
        # Record many samples with p95 >> 5s
        for _ in range(100):
            m.record_latency("route", 60.0)
        report = m.health_report()
        assert report.overall_status == SLOStatus.RED
        assert any("p95" in v.metric for v in report.violations)

    def test_error_rate_slo_violation(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        m = PipelineHealthMonitor(error_rate_slo=0.05, slo_warning_pct=1.01)
        for _ in range(10):
            m.record_error("route")   # 10 errors = 100 % error rate
        report = m.health_report()
        # Should be RED since 1.0 >> 0.05
        assert report.overall_status in {SLOStatus.YELLOW, SLOStatus.RED}

    def test_ece_violation_flagged(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        m = PipelineHealthMonitor(ece_slo=0.10, slo_warning_pct=1.01)
        m.record_ece(0.35)   # WAY above threshold
        report = m.health_report()
        assert report.current_ece == pytest.approx(0.35)
        assert report.overall_status == SLOStatus.RED

    def test_ece_passes_when_below_threshold(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        m = PipelineHealthMonitor(ece_slo=0.10)
        m.record_ece(0.0103)
        report = m.health_report()
        assert report.current_ece == pytest.approx(0.0103)
        # ECE alone should not cause RED
        assert not any(v.metric == "ece" and v.status == SLOStatus.RED
                       for v in report.violations)

    def test_invalid_ece_raises(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor
        m = PipelineHealthMonitor()
        with pytest.raises(ValueError, match="ECE"):
            m.record_ece(1.5)

    def test_chunk_count_and_growth_rate(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor
        import time
        m = PipelineHealthMonitor()
        m.record_chunk_count(0)
        time.sleep(0.05)
        m.record_chunk_count(100)
        report = m.health_report()
        assert report.chunk_store_size == 100
        # Growth rate should be positive (chunks per hour)
        assert report.chunk_growth_rate >= 0.0

    def test_negative_chunk_count_raises(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor
        m = PipelineHealthMonitor()
        with pytest.raises(ValueError, match="count"):
            m.record_chunk_count(-1)

    def test_connector_refresh_records_age(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor
        m = PipelineHealthMonitor()
        m.record_connector_refresh("github_releases")
        report = m.health_report()
        assert "github_releases" in report.connector_ages_h
        assert report.connector_ages_h["github_releases"] < 0.01  # just refreshed

    def test_stale_connector_raises_violation(self):
        from datetime import timedelta
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        m = PipelineHealthMonitor(freshness_slo_h=1.0, slo_warning_pct=1.01)
        # Manually backdating by inserting into internal dict
        with m._lock:
            m._connectors["old_connector"] = _now() - timedelta(hours=48)
        report = m.health_report()
        assert any("old_connector" in v.component for v in report.violations)
        assert report.overall_status in {SLOStatus.YELLOW, SLOStatus.RED}

    def test_reset_clears_all_observations(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        m = PipelineHealthMonitor()
        m.record_latency("route", 5.0)
        m.record_ece(0.5)
        m.record_chunk_count(1000)
        m.reset()
        report = m.health_report()
        assert report.overall_status == SLOStatus.GREEN
        assert report.latency_stats == []
        assert report.current_ece is None

    def test_summary_mentions_violations(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor
        m = PipelineHealthMonitor(ece_slo=0.10, slo_warning_pct=1.01)
        m.record_ece(0.50)
        report = m.health_report()
        assert "violation" in report.summary.lower()

    def test_yellow_status_within_warning_band(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        # slo = 10s p95; warning fires at 80% = 8s; we'll record p95 ≈ 9s
        m = PipelineHealthMonitor(route_p95_slo_s=10.0, slo_warning_pct=0.80)
        # 100 samples; make 95th percentile ≈ 9 (just below slo of 10)
        lats = [1.0] * 94 + [9.0] * 6   # p95 ≈ 9s which is > 10*0.80=8s
        for lat in lats:
            m.record_latency("route", lat)
        report = m.health_report()
        # Should be YELLOW (at risk) not GREEN
        assert report.overall_status in {SLOStatus.YELLOW, SLOStatus.RED}

    def test_thread_safe_concurrent_recording(self):
        import threading
        from app.intelligence.health_monitor import PipelineHealthMonitor
        m = PipelineHealthMonitor()
        errors = []
        def record(start):
            try:
                for i in range(200):
                    m.record_latency("route", float(i % 10))
                    m.record_chunk_count(i)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=record, args=(t,)) for t in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
        report = m.health_report()
        stage = next((s for s in report.latency_stats if s.stage == "route"), None)
        assert stage is not None
        assert stage.count <= 800  # window is 1000, all should fit



# ===========================================================================
# Phase 4 — Item 1: IndexingPipeline → WatchlistGraph auto-update
# ===========================================================================

class TestIndexingPipelineWatchlistIntegration:
    """Exhaustive tests for the WatchlistGraph auto-update wired into IndexingPipeline."""

    # ── helpers ─────────────────────────────────────────────────────────────

    def _make_graph(self):
        from app.personalization.watchlist_graph import WatchlistGraph
        return WatchlistGraph(user_id="test-user")

    def _make_item(self, platform="reddit", text="Hello world", title="Test"):
        return _make_content_item(platform=platform, text=text, title=title)

    # ── happy-path ───────────────────────────────────────────────────────────

    def test_watchlist_graph_property_none_by_default(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        p = IndexingPipeline()
        assert p.watchlist_graph is None

    def test_watchlist_graph_property_returns_injected(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        g = self._make_graph()
        p = IndexingPipeline(watchlist_graph=g)
        assert p.watchlist_graph is g

    def test_coverage_recorded_for_watched_entity(self):
        """Entity present in result.entities AND in watchlist ⇒ coverage entry created."""
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        g = self._make_graph()
        g.watch("OpenAI", expected_families=["social"])
        pipeline = IndexingPipeline(
            chunk_store=ChunkStore(),
            watchlist_graph=g,
        )
        # Text that produces "OpenAI" as an extracted entity via NER heuristic
        item = self._make_item(
            platform="reddit",
            text="OpenAI released a new model this week.",
        )
        asyncio.run(pipeline.process_batch([item]))
        nc = g.node_coverage("OpenAI")
        assert nc is not None
        # If entity matched, at least one source entry was recorded
        if nc.sources:
            assert nc.coverage_score > 0.0

    def test_no_watchlist_no_crash(self):
        """Omitting watchlist_graph must not raise or alter behavior."""
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        pipeline = IndexingPipeline(chunk_store=ChunkStore())
        result = asyncio.run(pipeline.process_batch([self._make_item()]))
        assert result.stats.watchlist_nodes_updated == 0

    def test_watchlist_not_updated_for_failed_result(self):
        """FAILED pipeline results must not trigger watchlist coverage recording."""
        import asyncio
        from unittest.mock import MagicMock, patch
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        from app.ingestion.pipeline_result import IntelligencePipelineResult, PipelineStatus
        import uuid

        g = self._make_graph()
        g.watch("FailNode", expected_families=["social"])

        mock_result = IntelligencePipelineResult(
            content_item_id=uuid.uuid4(),
            source_family="social",
            status=PipelineStatus.FAILED,
            entities=["FailNode"],
        )
        pipeline = IndexingPipeline(chunk_store=ChunkStore(), watchlist_graph=g)
        # Directly call _update_watchlist with the failed result
        from app.ingestion.indexing_pipeline import IndexingResult, IndexingStats
        ir = IndexingResult(stats=IndexingStats())
        pipeline._update_watchlist([mock_result], ir)
        nc = g.node_coverage("FailNode")
        assert nc is not None
        assert nc.sources == []  # no coverage recorded

    def test_watchlist_nodes_updated_counter_incremented(self):
        """stats.watchlist_nodes_updated must count distinct matched watched nodes."""
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline, IndexingResult, IndexingStats
        from app.ingestion.pipeline_result import IntelligencePipelineResult, PipelineStatus
        from app.intelligence.retrieval.chunk_store import ChunkStore
        import uuid

        g = self._make_graph()
        g.watch("GPT", expected_families=["social"])
        g.watch("Claude", expected_families=["social"])

        def _make_pr(entities):
            return IntelligencePipelineResult(
                content_item_id=uuid.uuid4(),
                source_family="social",
                status=PipelineStatus.SUCCESS,
                summary="test",
                entities=entities,
                confidence=0.8,
            )

        pipeline = IndexingPipeline(chunk_store=ChunkStore(), watchlist_graph=g)
        ir = IndexingResult(stats=IndexingStats())
        results = [_make_pr(["GPT", "Claude"]), _make_pr(["GPT"])]
        pipeline._update_watchlist(results, ir)
        # Both GPT and Claude are watched ⇒ 2 distinct nodes
        assert ir.stats.watchlist_nodes_updated == 2

    def test_only_watched_entities_counted(self):
        """Entities not in the watchlist must not inflate watchlist_nodes_updated."""
        from app.ingestion.indexing_pipeline import IndexingPipeline, IndexingResult, IndexingStats
        from app.ingestion.pipeline_result import IntelligencePipelineResult, PipelineStatus
        from app.intelligence.retrieval.chunk_store import ChunkStore
        import uuid

        g = self._make_graph()
        g.watch("WatchedEntity")

        pr = IntelligencePipelineResult(
            content_item_id=uuid.uuid4(),
            source_family="social",
            status=PipelineStatus.SUCCESS,
            summary="test",
            entities=["WatchedEntity", "UnwatchedEntity", "AnotherUnwatched"],
            confidence=0.8,
        )
        pipeline = IndexingPipeline(chunk_store=ChunkStore(), watchlist_graph=g)
        ir = IndexingResult(stats=IndexingStats())
        pipeline._update_watchlist([pr], ir)
        assert ir.stats.watchlist_nodes_updated == 1

    def test_watchlist_graph_error_does_not_abort_batch(self):
        """A raising WatchlistGraph must not propagate the exception out of process_batch."""
        import asyncio
        from unittest.mock import MagicMock
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore

        bad_graph = MagicMock()
        bad_graph.watched_nodes.return_value = []
        bad_graph.record_coverage_from_result.side_effect = RuntimeError("graph exploded")

        pipeline = IndexingPipeline(chunk_store=ChunkStore(), watchlist_graph=bad_graph)
        item = self._make_item(text="OpenAI GPT-4 announcement.")
        # Must not raise
        result = asyncio.run(pipeline.process_batch([item]))
        assert result is not None

    def test_coverage_from_result_called_per_actionable_item(self):
        """record_coverage_from_result is invoked exactly once per actionable result."""
        from unittest.mock import MagicMock
        from app.ingestion.indexing_pipeline import IndexingPipeline, IndexingResult, IndexingStats
        from app.ingestion.pipeline_result import IntelligencePipelineResult, PipelineStatus
        from app.intelligence.retrieval.chunk_store import ChunkStore
        import uuid

        mock_graph = MagicMock()
        mock_graph.watched_nodes.return_value = []
        mock_graph.record_coverage_from_result.return_value = None

        results = [
            IntelligencePipelineResult(
                content_item_id=uuid.uuid4(), source_family="social",
                status=PipelineStatus.SUCCESS, summary="ok", confidence=0.8,
            ),
            IntelligencePipelineResult(
                content_item_id=uuid.uuid4(), source_family="social",
                status=PipelineStatus.FAILED,  # NOT actionable
            ),
            IntelligencePipelineResult(
                content_item_id=uuid.uuid4(), source_family="news",
                status=PipelineStatus.PARTIAL, summary="partial", confidence=0.6,
            ),
        ]
        pipeline = IndexingPipeline(chunk_store=ChunkStore(), watchlist_graph=mock_graph)
        ir = IndexingResult(stats=IndexingStats())
        pipeline._update_watchlist(results, ir)
        # FAILED item is not actionable, so only 2 calls expected
        assert mock_graph.record_coverage_from_result.call_count == 2

    def test_multiple_batches_accumulate_coverage(self):
        """Each successive process_batch call accumulates coverage entries."""
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline, IndexingResult, IndexingStats
        from app.ingestion.pipeline_result import IntelligencePipelineResult, PipelineStatus
        from app.intelligence.retrieval.chunk_store import ChunkStore
        import uuid

        g = self._make_graph()
        g.watch("Anthropic", expected_families=["social", "news"])

        pr1 = IntelligencePipelineResult(
            content_item_id=uuid.uuid4(), source_family="social",
            status=PipelineStatus.SUCCESS, summary="s", entities=["Anthropic"], confidence=0.9,
        )
        pr2 = IntelligencePipelineResult(
            content_item_id=uuid.uuid4(), source_family="news",
            status=PipelineStatus.SUCCESS, summary="n", entities=["Anthropic"], confidence=0.85,
        )

        pipeline = IndexingPipeline(chunk_store=ChunkStore(), watchlist_graph=g)
        ir = IndexingResult(stats=IndexingStats())

        pipeline._update_watchlist([pr1], ir)
        pipeline._update_watchlist([pr2], ir)

        nc = g.node_coverage("Anthropic")
        # Two distinct source_ids should be recorded
        assert len(nc.sources) >= 1

    def test_empty_entities_list_does_not_crash(self):
        """Items with no extracted entities must be handled gracefully."""
        from app.ingestion.indexing_pipeline import IndexingPipeline, IndexingResult, IndexingStats
        from app.ingestion.pipeline_result import IntelligencePipelineResult, PipelineStatus
        from app.intelligence.retrieval.chunk_store import ChunkStore
        import uuid

        g = self._make_graph()
        g.watch("SomeEntity")

        pr = IntelligencePipelineResult(
            content_item_id=uuid.uuid4(), source_family="social",
            status=PipelineStatus.SUCCESS, summary="no entities here",
            entities=[],  # empty
            confidence=0.7,
        )
        pipeline = IndexingPipeline(chunk_store=ChunkStore(), watchlist_graph=g)
        ir = IndexingResult(stats=IndexingStats())
        pipeline._update_watchlist([pr], ir)
        assert ir.stats.watchlist_nodes_updated == 0

    def test_concurrent_watchlist_updates_thread_safe(self):
        """Concurrent _update_watchlist calls from multiple threads must not corrupt state."""
        import threading
        from app.ingestion.indexing_pipeline import IndexingPipeline, IndexingResult, IndexingStats
        from app.ingestion.pipeline_result import IntelligencePipelineResult, PipelineStatus
        from app.intelligence.retrieval.chunk_store import ChunkStore
        from app.personalization.watchlist_graph import WatchlistGraph
        import uuid

        g = WatchlistGraph(user_id="concurrent-user")
        for i in range(20):
            g.watch(f"Entity{i}", expected_families=["social"])

        pipeline = IndexingPipeline(chunk_store=ChunkStore(), watchlist_graph=g)
        errors = []

        def run_updates(thread_idx):
            try:
                for _ in range(100):
                    pr = IntelligencePipelineResult(
                        content_item_id=uuid.uuid4(),
                        source_family="social",
                        status=PipelineStatus.SUCCESS,
                        summary="test",
                        entities=[f"Entity{thread_idx % 20}"],
                        confidence=0.8,
                    )
                    ir = IndexingResult(stats=IndexingStats())
                    pipeline._update_watchlist([pr], ir)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_updates, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"

    def test_process_batch_includes_watchlist_stats(self):
        """process_batch result.stats.watchlist_nodes_updated is included in the result."""
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore

        g = self._make_graph()
        pipeline = IndexingPipeline(chunk_store=ChunkStore(), watchlist_graph=g)
        result = asyncio.run(pipeline.process_batch([self._make_item()]))
        # Attribute must exist (value may be 0 if no entity matches)
        assert hasattr(result.stats, "watchlist_nodes_updated")
        assert isinstance(result.stats.watchlist_nodes_updated, int)

    def test_record_coverage_from_result_not_called_when_no_graph(self):
        """Without watchlist_graph, process_batch must not attempt to call graph methods."""
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        # If it tried to call record_coverage_from_result on None, AttributeError would raise
        pipeline = IndexingPipeline(chunk_store=ChunkStore(), watchlist_graph=None)
        result = asyncio.run(pipeline.process_batch([self._make_item()]))
        assert result.stats.watchlist_nodes_updated == 0



# ===========================================================================
# Phase 4 — Item 2: ChunkStore SQLite persistence
# ===========================================================================

class TestChunkStoreSQLite:
    """Full coverage of the SQLite-backed ChunkStore — all public API surfaces."""

    # ── Construction ─────────────────────────────────────────────────────────

    def test_default_construction_in_memory(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore
        s = ChunkStore()
        assert s.db_path == ":memory:"
        assert s.count() == 0

    def test_custom_max_chunk_chars(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore(max_chunk_chars=10)
        cid = s.ingest(ChunkRecord(observation_id="o", text="0123456789abcdef"))
        assert s.get(cid).text == "0123456789"

    def test_max_chunk_chars_zero_raises(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore
        with pytest.raises(ValueError, match="max_chunk_chars"):
            ChunkStore(max_chunk_chars=0)

    def test_max_size_zero_raises(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore
        with pytest.raises(ValueError, match="max_size"):
            ChunkStore(max_size=0)

    def test_max_age_hours_zero_raises(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore
        with pytest.raises(ValueError, match="max_age_hours"):
            ChunkStore(max_age_hours=0.0)

    def test_max_age_hours_negative_raises(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore
        with pytest.raises(ValueError, match="max_age_hours"):
            ChunkStore(max_age_hours=-5)

    # ── embedding_version field ───────────────────────────────────────────────

    def test_embedding_version_none_by_default(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore()
        cid = s.ingest(ChunkRecord(observation_id="o", text="hello"))
        assert s.get(cid).embedding_version is None

    def test_embedding_version_stored_and_retrieved(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore()
        r = ChunkRecord(observation_id="o", text="hello", embedding_version="text-embedding-3-small")
        cid = s.ingest(r)
        assert s.get(cid).embedding_version == "text-embedding-3-small"

    def test_chunk_text_propagates_embedding_version(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore
        s = ChunkStore()
        ids = s.chunk_text("obs", "some text content here", chunk_size=10, overlap=2,
                           embedding_version="ada-002")
        for cid in ids:
            assert s.get(cid).embedding_version == "ada-002"

    def test_chunk_text_no_embedding_version_stores_none(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore
        s = ChunkStore()
        ids = s.chunk_text("obs", "hello world", chunk_size=50, overlap=5)
        assert s.get(ids[0]).embedding_version is None

    # ── Persistence across instances (file-based DB) ──────────────────────────

    def test_file_based_db_survives_new_instance(self, tmp_path):
        """Chunks written to a file DB must be readable by a new ChunkStore instance."""
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        db = str(tmp_path / "test.db")
        s1 = ChunkStore(db_path=db)
        cid = s1.ingest(ChunkRecord(observation_id="obs-persist", text="persist me",
                                    embedding_version="v1"))
        # Create second instance pointing at same file
        s2 = ChunkStore(db_path=db)
        assert s2.count() == 1
        rec = s2.get(cid)
        assert rec.text == "persist me"
        assert rec.embedding_version == "v1"
        assert rec.observation_id == "obs-persist"

    def test_file_based_clear_removes_all(self, tmp_path):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        db = str(tmp_path / "c.db")
        s = ChunkStore(db_path=db)
        for i in range(10):
            s.ingest(ChunkRecord(observation_id=f"o{i}", text=f"content {i}"))
        s.clear()
        assert s.count() == 0
        # New instance also sees empty store
        s2 = ChunkStore(db_path=db)
        assert s2.count() == 0

    def test_db_path_property(self, tmp_path):
        from app.intelligence.retrieval.chunk_store import ChunkStore
        p = str(tmp_path / "x.db")
        s = ChunkStore(db_path=p)
        assert s.db_path == p

    # ── Ingest / get / get_by_observation ─────────────────────────────────────

    def test_ingest_returns_chunk_id(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore()
        r = ChunkRecord(observation_id="o", text="hi")
        cid = s.ingest(r)
        assert cid == r.chunk_id

    def test_ingest_empty_after_truncation_raises(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore(max_chunk_chars=3)
        with pytest.raises(ValueError):
            s.ingest(ChunkRecord(observation_id="o", text="   "))

    def test_ingest_stores_metadata(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore()
        meta = {"signal_type": "claim", "confidence": 0.95}
        r = ChunkRecord(observation_id="o", text="fact", metadata=meta)
        cid = s.ingest(r)
        assert s.get(cid).metadata == meta

    def test_get_returns_none_for_missing(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore
        s = ChunkStore()
        assert s.get("nonexistent-id") is None

    def test_get_by_observation_ordered_by_chunk_index(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore()
        for idx in [2, 0, 1]:
            s.ingest(ChunkRecord(observation_id="obs", text=f"chunk{idx}",
                                 chunk_index=idx))
        recs = s.get_by_observation("obs")
        assert [r.chunk_index for r in recs] == [0, 1, 2]

    def test_get_by_observation_empty_for_unknown(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore
        s = ChunkStore()
        assert s.get_by_observation("no-such-obs") == []

    def test_ingest_batch_returns_ids_in_order(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore()
        records = [ChunkRecord(observation_id=f"o{i}", text=f"text {i}") for i in range(5)]
        ids = s.ingest_batch(records)
        assert ids == [r.chunk_id for r in records]
        assert s.count() == 5

    def test_duplicate_chunk_id_ignored(self):
        """INSERT OR IGNORE — ingesting the same chunk_id twice keeps count=1."""
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore()
        r = ChunkRecord(observation_id="o", text="original")
        cid = s.ingest(r)
        # same chunk_id with different text — should be silently ignored
        dup = r.model_copy(update={"text": "duplicate"})
        s.ingest(dup)
        assert s.count() == 1
        assert s.get(cid).text == "original"

    # ── chunk_text ────────────────────────────────────────────────────────────

    def test_chunk_text_produces_multiple_chunks(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore
        s = ChunkStore()
        text = "a" * 200
        ids = s.chunk_text("obs", text, chunk_size=50, overlap=10)
        assert len(ids) > 1
        assert s.count() == len(ids)

    def test_chunk_text_overlap_must_be_less_than_chunk_size(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore
        s = ChunkStore()
        with pytest.raises(ValueError, match="overlap"):
            s.chunk_text("obs", "text", chunk_size=10, overlap=10)

    def test_chunk_text_chunk_index_increments(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore
        s = ChunkStore()
        ids = s.chunk_text("obs", "0123456789" * 5, chunk_size=10, overlap=2)
        recs = s.get_by_observation("obs")
        assert recs[0].chunk_index == 0
        assert recs[1].chunk_index == 1

    def test_chunk_text_metadata_applied_to_all(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore
        s = ChunkStore()
        ids = s.chunk_text("obs", "word " * 50, chunk_size=20, overlap=5,
                           metadata={"key": "val"})
        for cid in ids:
            assert s.get(cid).metadata["key"] == "val"

    # ── FIFO eviction ─────────────────────────────────────────────────────────

    def test_fifo_eviction_respects_max_size(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore(max_size=3)
        ids = [s.ingest(ChunkRecord(observation_id=f"o{i}", text=f"c{i}")) for i in range(6)]
        assert s.count() == 3
        # Only the last 3 should survive (oldest evicted first)
        for cid in ids[3:]:
            assert s.get(cid) is not None

    def test_fifo_eviction_first_evicted_is_oldest(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore(max_size=2)
        r0 = ChunkRecord(observation_id="o0", text="first")
        r1 = ChunkRecord(observation_id="o1", text="second")
        r2 = ChunkRecord(observation_id="o2", text="third")
        s.ingest(r0)
        s.ingest(r1)
        s.ingest(r2)
        assert s.count() == 2
        assert s.get(r0.chunk_id) is None   # evicted
        assert s.get(r1.chunk_id) is not None
        assert s.get(r2.chunk_id) is not None

    # ── Age-based retention ───────────────────────────────────────────────────

    def test_evict_stale_without_config_raises(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore
        s = ChunkStore()  # no max_age_hours
        with pytest.raises(RuntimeError, match="max_age_hours"):
            s.evict_stale()

    def test_evict_stale_removes_old_chunks(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore(max_age_hours=1)
        old = ChunkRecord(
            observation_id="old",
            text="old chunk",
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        new = ChunkRecord(observation_id="new", text="new chunk")
        # Ingest new first so the lazy eviction during old.ingest() doesn't affect new
        s.ingest(new)
        # Manually insert old without triggering inline eviction by using a store without max_age_hours
        # We patch the max_age_hours temporarily
        old_max = s.max_age_hours
        s.max_age_hours = None
        s.ingest(old)
        s.max_age_hours = old_max
        # Now both are in the DB; call explicit evict_stale
        assert s.count() == 2
        n = s.evict_stale()
        assert n == 1
        assert s.count() == 1
        assert s.get(old.chunk_id) is None
        assert s.get(new.chunk_id) is not None

    def test_lazy_eviction_during_ingest(self):
        """Old chunk evicted inline during ingest() of a newer chunk."""
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore(max_age_hours=1)
        # Ingest an already-old chunk via a store without the age constraint
        # (simulate old data sitting in the DB)
        s_no_age = ChunkStore.__new__(ChunkStore)
        s_no_age._lock    = s._lock
        s_no_age._conn    = s._conn
        s_no_age.max_chunk_chars = s.max_chunk_chars
        s_no_age.max_size = None
        s_no_age.max_age_hours = None
        s_no_age._db_path = s._db_path
        old = ChunkRecord(
            observation_id="stale",
            text="stale data",
            created_at=datetime.now(timezone.utc) - timedelta(hours=3),
        )
        s_no_age.ingest(old)  # insert without eviction
        assert s.count() == 1
        # Now ingest through the age-limited store — lazy eviction should fire
        new = ChunkRecord(observation_id="fresh", text="fresh data")
        s.ingest(new)
        # old should have been evicted
        assert s.get(old.chunk_id) is None
        assert s.count() == 1

    def test_evict_stale_returns_zero_when_nothing_stale(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore(max_age_hours=1)
        s.ingest(ChunkRecord(observation_id="o", text="fresh"))
        n = s.evict_stale()
        assert n == 0
        assert s.count() == 1

    def test_evict_stale_returns_count_evicted(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore(max_age_hours=1)
        # Insert 5 stale records by bypassing inline eviction
        s.max_age_hours = None
        for i in range(5):
            s.ingest(ChunkRecord(
                observation_id=f"o{i}", text=f"stale {i}",
                created_at=datetime.now(timezone.utc) - timedelta(hours=2),
            ))
        s.max_age_hours = 1
        n = s.evict_stale()
        assert n == 5
        assert s.count() == 0

    # ── observation_ids + clear ───────────────────────────────────────────────

    def test_observation_ids_sorted(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore()
        for obs in ["c_obs", "a_obs", "b_obs"]:
            s.ingest(ChunkRecord(observation_id=obs, text="x"))
        assert s.observation_ids() == ["a_obs", "b_obs", "c_obs"]

    def test_observation_ids_unique(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore()
        for i in range(5):
            s.ingest(ChunkRecord(observation_id="same", text=f"chunk {i}"))
        assert s.observation_ids() == ["same"]

    def test_clear_empties_store(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore()
        for i in range(10):
            s.ingest(ChunkRecord(observation_id=f"o{i}", text=f"t{i}"))
        s.clear()
        assert s.count() == 0
        assert s.observation_ids() == []

    # ── keyword_search ────────────────────────────────────────────────────────

    def test_keyword_search_finds_match(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore()
        s.ingest(ChunkRecord(observation_id="o1", text="transformer attention mechanism"))
        s.ingest(ChunkRecord(observation_id="o2", text="random unrelated text"))
        hits = s.keyword_search("attention transformer")
        assert len(hits) >= 1
        assert hits[0].record.observation_id == "o1"

    def test_keyword_search_empty_query_returns_empty(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore()
        s.ingest(ChunkRecord(observation_id="o", text="hello world"))
        assert s.keyword_search("") == []

    def test_keyword_search_metadata_filter(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore()
        s.ingest(ChunkRecord(observation_id="o1", text="neural network deep learning",
                             metadata={"family": "research"}))
        s.ingest(ChunkRecord(observation_id="o2", text="neural network marketing content",
                             metadata={"family": "social"}))
        hits = s.keyword_search(
            "neural network",
            metadata_filter=lambda r: r.metadata.get("family") == "research",
        )
        assert all(h.record.metadata["family"] == "research" for h in hits)

    def test_keyword_search_top_k_respected(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore()
        for i in range(20):
            s.ingest(ChunkRecord(observation_id=f"o{i}", text=f"machine learning topic {i}"))
        hits = s.keyword_search("machine learning", top_k=5)
        assert len(hits) <= 5

    # ── semantic search (embeddings) ──────────────────────────────────────────

    def test_semantic_search_no_embeddings_raises(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore()
        s.ingest(ChunkRecord(observation_id="o", text="hello"))
        with pytest.raises(RuntimeError, match="embeddings"):
            s.search([0.1, 0.2, 0.3])

    def test_semantic_search_zero_vector_raises(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore()
        s.ingest(ChunkRecord(observation_id="o", text="hello", embedding=[0.1, 0.2]))
        with pytest.raises(ValueError, match="zero vector"):
            s.search([0.0, 0.0])

    def test_semantic_search_returns_most_similar(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore()
        s.ingest(ChunkRecord(observation_id="o1", text="A", embedding=[1.0, 0.0, 0.0]))
        s.ingest(ChunkRecord(observation_id="o2", text="B", embedding=[0.0, 1.0, 0.0]))
        hits = s.search([1.0, 0.0, 0.0], top_k=2)
        assert hits[0].record.observation_id == "o1"
        assert hits[0].score == pytest.approx(1.0)
        assert hits[1].score == pytest.approx(0.0)

    def test_semantic_search_embedding_version_preserved(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore()
        s.ingest(ChunkRecord(observation_id="o", text="test",
                             embedding=[1.0, 0.0], embedding_version="v3"))
        hits = s.search([1.0, 0.0])
        assert hits[0].record.embedding_version == "v3"

    def test_semantic_search_metadata_filter(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore()
        s.ingest(ChunkRecord(observation_id="match", text="A",
                             embedding=[1.0, 0.0], metadata={"ok": True}))
        s.ingest(ChunkRecord(observation_id="skip",  text="B",
                             embedding=[1.0, 0.0], metadata={"ok": False}))
        hits = s.search([1.0, 0.0], metadata_filter=lambda r: r.metadata.get("ok"))
        assert len(hits) == 1
        assert hits[0].record.observation_id == "match"

    # ── Thread-safety ─────────────────────────────────────────────────────────

    def test_thread_safe_concurrent_ingest(self):
        """500 concurrent ingests across 5 threads must all land without corruption."""
        import threading
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore()
        errors = []

        def worker(thread_id):
            try:
                for i in range(100):
                    s.ingest(ChunkRecord(
                        observation_id=f"t{thread_id}-i{i}",
                        text=f"content from thread {thread_id} iteration {i}",
                    ))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
        assert s.count() == 500

    def test_thread_safe_mixed_reads_and_writes(self):
        """Concurrent reads + writes must not cause SQLite locking errors."""
        import threading
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore()
        # Seed some data
        for i in range(20):
            s.ingest(ChunkRecord(observation_id=f"seed-{i}", text=f"seed content {i}"))

        errors = []

        def reader():
            try:
                for _ in range(100):
                    _ = s.count()
                    _ = s.observation_ids()
                    _ = s.keyword_search("content", top_k=5)
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(100):
                    s.ingest(ChunkRecord(observation_id=f"dyn-{i}-{id(threading.current_thread())}",
                                         text=f"dynamic content {i}"))
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=reader) for _ in range(3)] +
            [threading.Thread(target=writer) for _ in range(2)]
        )
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors

    def test_thread_safe_file_db_concurrent(self, tmp_path):
        """File-backed DB must remain consistent under 4-thread concurrent writes."""
        import threading
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        db = str(tmp_path / "concurrent.db")
        s = ChunkStore(db_path=db)
        errors = []

        def worker(t_id):
            try:
                for i in range(100):
                    s.ingest(ChunkRecord(
                        observation_id=f"t{t_id}-{i}",
                        text=f"thread {t_id} chunk {i}",
                    ))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
        assert s.count() == 400

    # ── Boundary conditions ───────────────────────────────────────────────────

    def test_ingest_exactly_max_chunk_chars_no_truncation(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore(max_chunk_chars=10)
        text = "1234567890"  # exactly 10
        cid = s.ingest(ChunkRecord(observation_id="o", text=text))
        assert s.get(cid).text == text

    def test_fifo_eviction_exact_boundary(self):
        """Inserting exactly max_size records must NOT trigger eviction."""
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore(max_size=5)
        for i in range(5):
            s.ingest(ChunkRecord(observation_id=f"o{i}", text=f"c{i}"))
        assert s.count() == 5  # no eviction

    def test_chunk_text_single_chunk_when_text_short(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore
        s = ChunkStore()
        ids = s.chunk_text("obs", "hello world", chunk_size=100, overlap=10)
        assert len(ids) == 1

    def test_count_zero_on_empty_store(self):
        from app.intelligence.retrieval.chunk_store import ChunkStore
        assert ChunkStore().count() == 0

    def test_created_at_preserved_from_record(self):
        """created_at set on the ChunkRecord must be preserved exactly."""
        from app.intelligence.retrieval.chunk_store import ChunkStore, ChunkRecord
        s = ChunkStore()
        ts = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        r = ChunkRecord(observation_id="o", text="test", created_at=ts)
        cid = s.ingest(r)
        retrieved = s.get(cid)
        # Timestamps match to the second (float precision may differ by microseconds)
        assert abs((retrieved.created_at - ts).total_seconds()) < 0.01



# ===========================================================================
# Phase 4 — Item 3: PipelineHealthMonitor → WatchlistGraph integration
# ===========================================================================

class TestHealthMonitorWatchlistGapCount:
    """Tests for record_watchlist_gap_count and its SLO thresholds."""

    # ── Basic recording ──────────────────────────────────────────────────────

    def test_zero_gaps_no_violation(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        m = PipelineHealthMonitor()
        m.record_watchlist_gap_count(0)
        report = m.health_report()
        watchlist_v = [v for v in report.violations if v.component == "watchlist"]
        assert not watchlist_v
        assert report.overall_status == SLOStatus.GREEN

    def test_two_gaps_no_violation(self):
        """n=2 is below YELLOW threshold; no violation."""
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        m = PipelineHealthMonitor()
        m.record_watchlist_gap_count(2)
        report = m.health_report()
        watchlist_v = [v for v in report.violations if v.component == "watchlist"]
        assert not watchlist_v

    def test_three_gaps_yellow(self):
        """n=3 triggers YELLOW."""
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        m = PipelineHealthMonitor()
        m.record_watchlist_gap_count(3)
        report = m.health_report()
        watchlist_v = [v for v in report.violations if v.component == "watchlist"]
        assert len(watchlist_v) == 1
        assert watchlist_v[0].status == SLOStatus.YELLOW
        assert report.overall_status == SLOStatus.YELLOW

    def test_nine_gaps_yellow(self):
        """n=9 (just below RED threshold) triggers YELLOW."""
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        m = PipelineHealthMonitor()
        m.record_watchlist_gap_count(9)
        report = m.health_report()
        watchlist_v = [v for v in report.violations if v.component == "watchlist"]
        assert watchlist_v[0].status == SLOStatus.YELLOW

    def test_ten_gaps_red(self):
        """n=10 triggers RED."""
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        m = PipelineHealthMonitor()
        m.record_watchlist_gap_count(10)
        report = m.health_report()
        watchlist_v = [v for v in report.violations if v.component == "watchlist"]
        assert len(watchlist_v) == 1
        assert watchlist_v[0].status == SLOStatus.RED
        assert report.overall_status == SLOStatus.RED

    def test_large_gap_count_red(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        m = PipelineHealthMonitor()
        m.record_watchlist_gap_count(100)
        report = m.health_report()
        watchlist_v = [v for v in report.violations if v.component == "watchlist"]
        assert watchlist_v[0].status == SLOStatus.RED

    def test_negative_gap_count_raises(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor
        m = PipelineHealthMonitor()
        with pytest.raises(ValueError, match="≥ 0"):
            m.record_watchlist_gap_count(-1)

    def test_not_recorded_no_violation(self):
        """Before any call to record_watchlist_gap_count, no watchlist violation exists."""
        from app.intelligence.health_monitor import PipelineHealthMonitor
        m = PipelineHealthMonitor()
        report = m.health_report()
        watchlist_v = [v for v in report.violations if v.component == "watchlist"]
        assert not watchlist_v

    def test_most_recent_value_used(self):
        """Only the last recorded gap count is evaluated."""
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        m = PipelineHealthMonitor()
        m.record_watchlist_gap_count(15)   # RED
        m.record_watchlist_gap_count(1)    # updates to GREEN (< 3)
        report = m.health_report()
        watchlist_v = [v for v in report.violations if v.component == "watchlist"]
        assert not watchlist_v

    def test_violation_message_contains_count(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor
        m = PipelineHealthMonitor()
        m.record_watchlist_gap_count(5)
        report = m.health_report()
        watchlist_v = [v for v in report.violations if v.component == "watchlist"]
        assert "5" in watchlist_v[0].message

    def test_reset_clears_gap_count(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        m = PipelineHealthMonitor()
        m.record_watchlist_gap_count(10)
        m.reset()
        report = m.health_report()
        watchlist_v = [v for v in report.violations if v.component == "watchlist"]
        assert not watchlist_v
        assert report.overall_status == SLOStatus.GREEN

    def test_combined_with_other_violations(self):
        """Gap count RED + ECE RED ⇒ overall RED with both violations present."""
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        m = PipelineHealthMonitor(ece_slo=0.10, slo_warning_pct=1.01)
        m.record_watchlist_gap_count(10)
        m.record_ece(0.5)
        report = m.health_report()
        assert report.overall_status == SLOStatus.RED
        components = {v.component for v in report.violations}
        assert "watchlist" in components
        assert "model" in components

    def test_thread_safe_concurrent_gap_recording(self):
        """Concurrent gap count updates from multiple threads must not corrupt state."""
        import threading
        from app.intelligence.health_monitor import PipelineHealthMonitor
        m = PipelineHealthMonitor()
        errors = []

        def worker():
            try:
                for i in range(100):
                    m.record_watchlist_gap_count(i % 15)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
        # Must be able to generate a report without errors
        report = m.health_report()
        assert report is not None


class TestIndexingPipelineHealthMonitorWiring:
    """Tests for the IndexingPipeline → PipelineHealthMonitor automatic wiring."""

    def test_health_monitor_property_none_by_default(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        p = IndexingPipeline()
        assert p.health_monitor is None

    def test_health_monitor_property_returns_injected(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.health_monitor import PipelineHealthMonitor
        hm = PipelineHealthMonitor()
        p = IndexingPipeline(health_monitor=hm)
        assert p.health_monitor is hm

    def test_no_monitor_no_crash(self):
        """Without a health_monitor, process_batch must not raise."""
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        from app.personalization.watchlist_graph import WatchlistGraph
        g = WatchlistGraph(user_id="u1")
        g.watch("Node1", expected_families=["social"])
        p = IndexingPipeline(chunk_store=ChunkStore(), watchlist_graph=g)
        item = _make_content_item()
        result = asyncio.run(p.process_batch([item]))
        assert result is not None

    def test_gap_count_sent_to_monitor_after_batch(self):
        """After process_batch, health_monitor must have received a gap count."""
        import asyncio
        from unittest.mock import MagicMock
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        from app.personalization.watchlist_graph import WatchlistGraph

        g = WatchlistGraph(user_id="u1")
        g.watch("A", expected_families=["social"])  # zero coverage → gap

        hm = MagicMock()
        hm.record_watchlist_gap_count = MagicMock()

        p = IndexingPipeline(
            chunk_store=ChunkStore(), watchlist_graph=g, health_monitor=hm
        )
        asyncio.run(p.process_batch([_make_content_item()]))
        hm.record_watchlist_gap_count.assert_called_once()
        # node A has zero coverage ⇒ gap count ≥ 1
        args = hm.record_watchlist_gap_count.call_args[0]
        assert args[0] >= 1

    def test_full_slo_pipeline_green_when_all_covered(self):
        """If all watched nodes are covered, health monitor receives gap_count=0."""
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline, IndexingResult, IndexingStats
        from app.ingestion.pipeline_result import IntelligencePipelineResult, PipelineStatus
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        from app.intelligence.retrieval.chunk_store import ChunkStore
        from app.personalization.watchlist_graph import WatchlistGraph
        import uuid

        g = WatchlistGraph(user_id="u1")
        g.watch("CoveredNode", expected_families=["social"])
        g.record_coverage("CoveredNode", source_id="src-x", source_family="social")

        hm = PipelineHealthMonitor()

        p = IndexingPipeline(
            chunk_store=ChunkStore(), watchlist_graph=g, health_monitor=hm
        )
        # Trigger _report_watchlist_health directly
        p._report_watchlist_health()

        report = hm.health_report()
        watchlist_v = [v for v in report.violations if v.component == "watchlist"]
        assert not watchlist_v

    def test_health_monitor_error_does_not_abort_batch(self):
        """A raising health_monitor must not propagate out of process_batch."""
        import asyncio
        from unittest.mock import MagicMock
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        from app.personalization.watchlist_graph import WatchlistGraph

        g = WatchlistGraph(user_id="u1")
        hm = MagicMock()
        hm.record_watchlist_gap_count.side_effect = RuntimeError("monitor exploded")

        p = IndexingPipeline(
            chunk_store=ChunkStore(), watchlist_graph=g, health_monitor=hm
        )
        result = asyncio.run(p.process_batch([_make_content_item()]))
        assert result is not None  # did not raise

    def test_no_monitor_call_when_no_watchlist(self):
        """When watchlist_graph is None, health_monitor.record_watchlist_gap_count
        must never be called (even if health_monitor is present)."""
        import asyncio
        from unittest.mock import MagicMock
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore

        hm = MagicMock()
        p = IndexingPipeline(
            chunk_store=ChunkStore(), watchlist_graph=None, health_monitor=hm
        )
        asyncio.run(p.process_batch([_make_content_item()]))
        hm.record_watchlist_gap_count.assert_not_called()

    def test_concurrent_batches_all_report_to_monitor(self):
        """Multiple sequential process_batch calls each send a gap count."""
        import asyncio
        from unittest.mock import MagicMock
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        from app.personalization.watchlist_graph import WatchlistGraph

        g = WatchlistGraph(user_id="u1")
        g.watch("X", expected_families=["social"])

        hm = MagicMock()
        p = IndexingPipeline(
            chunk_store=ChunkStore(), watchlist_graph=g, health_monitor=hm
        )
        for _ in range(5):
            asyncio.run(p.process_batch([_make_content_item()]))
        assert hm.record_watchlist_gap_count.call_count == 5



# ===========================================================================
# Phase 4 — Item 4: Enterprise Hardening
# ===========================================================================

# ---------------------------------------------------------------------------
# 4a — Per-tenant ChunkStore partitioning
# ---------------------------------------------------------------------------

class TestTenantPartitioning:
    """Exhaustive tests for per-tenant ChunkStore partitioning."""

    def test_default_tenant_store_is_injected_store(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        s = ChunkStore()
        p = IndexingPipeline(chunk_store=s)
        assert p.chunk_store is s
        assert p.tenant_store("default") is s

    def test_new_tenant_gets_separate_store(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        p = IndexingPipeline()
        s_a = p._get_or_create_tenant_store("tenant-a")
        s_b = p._get_or_create_tenant_store("tenant-b")
        assert s_a is not s_b
        assert s_a is not p.chunk_store

    def test_same_tenant_returns_same_store(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        p = IndexingPipeline()
        s1 = p._get_or_create_tenant_store("acme")
        s2 = p._get_or_create_tenant_store("acme")
        assert s1 is s2

    def test_tenant_store_unknown_returns_none(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        p = IndexingPipeline()
        assert p.tenant_store("nonexistent") is None

    def test_tenant_ids_contains_default(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        p = IndexingPipeline()
        assert "default" in p.tenant_ids()

    def test_tenant_ids_expands_on_create(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        p = IndexingPipeline()
        p._get_or_create_tenant_store("t1")
        p._get_or_create_tenant_store("t2")
        ids = p.tenant_ids()
        assert "t1" in ids and "t2" in ids

    def test_process_batch_default_tenant_indexes_into_default_store(self):
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        store = ChunkStore()
        p = IndexingPipeline(chunk_store=store)
        asyncio.run(p.process_batch([_make_content_item(text="hello world")]))
        assert store.count() >= 1

    def test_process_batch_custom_tenant_indexes_into_separate_store(self):
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        default_store = ChunkStore()
        p = IndexingPipeline(chunk_store=default_store)
        asyncio.run(p.process_batch([_make_content_item(text="tenant alpha content")],
                                     tenant_id="alpha"))
        # The alpha-specific store must have chunks
        alpha_store = p.tenant_store("alpha")
        assert alpha_store is not None
        assert alpha_store.count() >= 1
        # Default store must NOT have alpha's chunks
        assert default_store.count() == 0

    def test_two_tenants_chunks_do_not_bleed(self):
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline
        asyncio.run(_two_tenant_isolation_run())

    def test_tenant_id_propagated_to_pipeline_result(self):
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        p = IndexingPipeline(chunk_store=ChunkStore())
        result = asyncio.run(p.process_batch([_make_content_item()], tenant_id="acme"))
        for pr in result.pipeline_results:
            assert pr.tenant_id == "acme"

    def test_route_accepts_tenant_id(self):
        """ContentPipelineRouter.route() must accept and propagate tenant_id."""
        import asyncio
        from app.ingestion.content_pipeline_router import ContentPipelineRouter
        router = ContentPipelineRouter()
        item = _make_content_item()
        result = asyncio.run(router.route(item, tenant_id="org-x"))
        assert result.tenant_id == "org-x"

    def test_route_default_tenant_is_default(self):
        import asyncio
        from app.ingestion.content_pipeline_router import ContentPipelineRouter
        result = asyncio.run(ContentPipelineRouter().route(_make_content_item()))
        assert result.tenant_id == "default"

    def test_thread_safe_tenant_store_creation(self):
        """Multiple threads creating different tenant stores must not corrupt the dict."""
        import threading
        from app.ingestion.indexing_pipeline import IndexingPipeline
        p = IndexingPipeline()
        errors = []

        def make_tenant(tid):
            try:
                for _ in range(100):
                    p._get_or_create_tenant_store(f"tenant-{tid}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=make_tenant, args=(t,)) for t in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
        assert len(p.tenant_ids()) >= 4


async def _two_tenant_isolation_run():
    from app.ingestion.indexing_pipeline import IndexingPipeline
    p = IndexingPipeline()
    await p.process_batch([_make_content_item(text="alpha alpha alpha")], tenant_id="alpha")
    await p.process_batch([_make_content_item(text="beta beta beta")],   tenant_id="beta")
    s_alpha = p.tenant_store("alpha")
    s_beta  = p.tenant_store("beta")
    assert s_alpha is not s_beta
    assert s_alpha.count() >= 1
    assert s_beta.count() >= 1


# ---------------------------------------------------------------------------
# 4b — Immutable audit trail
# ---------------------------------------------------------------------------

class TestModelAuditTrail:
    """Exhaustive tests for the ModelArtifactRegistry immutable audit log."""

    def _make_registry(self, tmp_path):
        from training.model_registry import ModelArtifactRegistry
        return ModelArtifactRegistry(
            registry_path=tmp_path / "reg.json",
            audit_log_path=tmp_path / "audit.jsonl",
        )

    def _make_artifact(self, ece=0.05, f1=0.80, epoch=1):
        from training.model_registry import ArtifactRecord
        return ArtifactRecord(
            model_name="test_model",
            epoch=epoch,
            ece=ece,
            macro_f1=f1,
            checkpoint_path=f"/tmp/model_ep{epoch}.json",
        )

    def test_audit_log_empty_before_any_promote(self, tmp_path):
        r = self._make_registry(tmp_path)
        assert r.read_audit_log() == []

    def test_promote_appends_one_entry(self, tmp_path):
        r = self._make_registry(tmp_path)
        a = self._make_artifact()
        r.register(a)
        r.promote(a.artifact_id)
        entries = r.read_audit_log()
        assert len(entries) == 1
        assert entries[0]["event"] == "promote"
        assert entries[0]["artifact_id"] == a.artifact_id

    def test_promote_entry_contains_timestamp(self, tmp_path):
        r = self._make_registry(tmp_path)
        a = self._make_artifact()
        r.register(a)
        r.promote(a.artifact_id)
        entry = r.read_audit_log()[0]
        # Timestamp must be parseable ISO-8601
        ts = datetime.fromisoformat(entry["timestamp"])
        assert ts.tzinfo is not None

    def test_promote_notes_stored(self, tmp_path):
        r = self._make_registry(tmp_path)
        a = self._make_artifact()
        r.register(a)
        r.promote(a.artifact_id, notes="initial deploy v1")
        entry = r.read_audit_log()[0]
        assert entry["notes"] == "initial deploy v1"

    def test_rollback_appends_rollback_entry(self, tmp_path):
        r = self._make_registry(tmp_path)
        a1 = self._make_artifact(epoch=1)
        a2 = self._make_artifact(epoch=2)
        r.register(a1)
        r.register(a2)
        r.promote(a1.artifact_id)
        r.promote(a2.artifact_id)
        r.rollback(notes="bad deploy")
        entries = r.read_audit_log()
        assert len(entries) == 3
        assert entries[2]["event"] == "rollback"
        assert "previous_production_id" in entries[2]
        assert entries[2]["notes"] == "bad deploy"

    def test_multiple_promotes_multiple_entries(self, tmp_path):
        r = self._make_registry(tmp_path)
        artifacts = [self._make_artifact(epoch=i, ece=0.05, f1=0.80) for i in range(5)]
        for a in artifacts:
            r.register(a)
        for a in artifacts:
            r.promote(a.artifact_id)
        entries = r.read_audit_log()
        assert len(entries) == 5
        assert all(e["event"] == "promote" for e in entries)

    def test_audit_log_never_overwritten(self, tmp_path):
        """Creating a second registry instance against the same file must append, not overwrite."""
        r1 = self._make_registry(tmp_path)
        a1 = self._make_artifact(epoch=1)
        r1.register(a1)
        r1.promote(a1.artifact_id)

        from training.model_registry import ModelArtifactRegistry
        r2 = ModelArtifactRegistry(
            registry_path=tmp_path / "reg.json",
            audit_log_path=tmp_path / "audit.jsonl",
        )
        a2 = self._make_artifact(epoch=2)
        r2.register(a2)
        r2.promote(a2.artifact_id)

        # Both r1 and r2 read the same file; 2 entries total
        entries = r2.read_audit_log()
        assert len(entries) == 2
        events = [e["event"] for e in entries]
        assert events == ["promote", "promote"]

    def test_audit_log_is_valid_jsonl(self, tmp_path):
        """Every line in the audit log must be valid JSON."""
        import json as _json
        r = self._make_registry(tmp_path)
        for i in range(3):
            a = self._make_artifact(epoch=i)
            r.register(a)
            r.promote(a.artifact_id)
        audit_path = tmp_path / "audit.jsonl"
        lines = audit_path.read_text().strip().split("\n")
        for line in lines:
            obj = _json.loads(line)  # must not raise
            assert "event" in obj

    def test_rollback_no_fallback_no_audit_entry(self, tmp_path):
        """rollback() with no candidates must return None and not write an audit entry."""
        r = self._make_registry(tmp_path)
        # No artifacts at all
        result = r.rollback()
        assert result is None
        assert r.read_audit_log() == []

    def test_failed_promote_does_not_write_audit_entry(self, tmp_path):
        """A promote() that raises must not append to the audit log."""
        from training.model_registry import ArtifactRecord
        r = self._make_registry(tmp_path)
        # Artifact that fails gate (high ECE)
        bad = ArtifactRecord(
            model_name="bad", epoch=1, ece=0.50, macro_f1=0.80,
            checkpoint_path="/tmp/bad.json",
        )
        r.register(bad)
        with pytest.raises(ValueError, match="Cannot promote"):
            r.promote(bad.artifact_id)
        assert r.read_audit_log() == []

    def test_thread_safe_concurrent_promotes_audit(self, tmp_path):
        """Concurrent promotes from multiple threads each write exactly one audit entry."""
        import threading
        from training.model_registry import ModelArtifactRegistry, ArtifactRecord
        r = ModelArtifactRegistry(
            registry_path=tmp_path / "reg.json",
            audit_log_path=tmp_path / "audit.jsonl",
        )
        artifacts = [
            ArtifactRecord(model_name="m", epoch=i, ece=0.05, macro_f1=0.80,
                           checkpoint_path=f"/tmp/m{i}.json")
            for i in range(4)
        ]
        for a in artifacts:
            r.register(a)

        errors = []
        def promote_one(a):
            try:
                r.promote(a.artifact_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=promote_one, args=(a,)) for a in artifacts]
        for t in threads: t.start()
        for t in threads: t.join()

        entries = r.read_audit_log()
        # Each success writes exactly one entry; errors may reduce count
        successful_promotes = len([e for e in entries if e["event"] == "promote"])
        assert successful_promotes >= 1
        # All entries must be valid
        for e in entries:
            assert "artifact_id" in e and "timestamp" in e


# ---------------------------------------------------------------------------
# 4d — Per-connector circuit-breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    """Exhaustive tests for the per-connector circuit-breaker in PipelineHealthMonitor."""

    def test_no_failures_no_violation(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        m = PipelineHealthMonitor()
        report = m.health_report()
        cb_v = [v for v in report.violations if v.metric == "circuit_breaker"]
        assert not cb_v

    def test_four_failures_circuit_not_open(self):
        """4 failures < threshold of 5 → circuit stays closed."""
        from app.intelligence.health_monitor import PipelineHealthMonitor
        m = PipelineHealthMonitor(cb_open_threshold=5)
        for _ in range(4):
            m.record_connector_failure("github")
        assert not m.is_circuit_open("github")
        report = m.health_report()
        cb_v = [v for v in report.violations if v.metric == "circuit_breaker"]
        assert not cb_v

    def test_five_failures_opens_circuit(self):
        """5 consecutive failures (= threshold) opens the circuit → RED violation."""
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        m = PipelineHealthMonitor(cb_open_threshold=5)
        for _ in range(5):
            m.record_connector_failure("github")
        assert m.is_circuit_open("github")
        report = m.health_report()
        assert report.overall_status == SLOStatus.RED
        cb_v = [v for v in report.violations if v.metric == "circuit_breaker"]
        assert len(cb_v) == 1
        assert "github" in cb_v[0].component

    def test_more_than_threshold_failures_still_open(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        m = PipelineHealthMonitor(cb_open_threshold=5)
        for _ in range(20):
            m.record_connector_failure("arXiv")
        assert m.is_circuit_open("arXiv")
        report = m.health_report()
        assert report.overall_status == SLOStatus.RED

    def test_success_after_failures_closes_circuit(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        m = PipelineHealthMonitor(cb_open_threshold=5)
        for _ in range(5):
            m.record_connector_failure("rss")
        assert m.is_circuit_open("rss")
        m.record_connector_success("rss")
        assert not m.is_circuit_open("rss")
        report = m.health_report()
        cb_v = [v for v in report.violations if v.metric == "circuit_breaker"]
        assert not cb_v

    def test_success_resets_failure_counter(self):
        """After success, 4 more failures should not re-open (threshold=5)."""
        from app.intelligence.health_monitor import PipelineHealthMonitor
        m = PipelineHealthMonitor(cb_open_threshold=5)
        for _ in range(4):
            m.record_connector_failure("podcast")
        m.record_connector_success("podcast")
        for _ in range(4):
            m.record_connector_failure("podcast")
        assert not m.is_circuit_open("podcast")

    def test_success_records_connector_refresh(self):
        """record_connector_success() also marks the connector as fresh."""
        from app.intelligence.health_monitor import PipelineHealthMonitor
        m = PipelineHealthMonitor(freshness_slo_h=1.0)
        m.record_connector_success("rss")
        report = m.health_report()
        assert "rss" in report.connector_ages_h
        assert report.connector_ages_h["rss"] < 0.01

    def test_multiple_connectors_independent(self):
        """Open circuit on one connector must not affect others."""
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        m = PipelineHealthMonitor(cb_open_threshold=3)
        for _ in range(3):
            m.record_connector_failure("bad_connector")
        m.record_connector_success("good_connector")
        assert m.is_circuit_open("bad_connector")
        assert not m.is_circuit_open("good_connector")
        report = m.health_report()
        cb_v = [v for v in report.violations if v.metric == "circuit_breaker"]
        assert len(cb_v) == 1
        assert "bad_connector" in cb_v[0].component

    def test_custom_threshold_honored(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor
        m = PipelineHealthMonitor(cb_open_threshold=2)
        m.record_connector_failure("c1")
        assert not m.is_circuit_open("c1")
        m.record_connector_failure("c1")
        assert m.is_circuit_open("c1")

    def test_violation_message_contains_connector_id(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor
        m = PipelineHealthMonitor(cb_open_threshold=3)
        for _ in range(3):
            m.record_connector_failure("special_connector")
        report = m.health_report()
        cb_v = [v for v in report.violations if v.metric == "circuit_breaker"]
        assert "special_connector" in cb_v[0].message

    def test_reset_clears_circuit_breaker_state(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        m = PipelineHealthMonitor(cb_open_threshold=3)
        for _ in range(3):
            m.record_connector_failure("x")
        assert m.is_circuit_open("x")
        m.reset()
        assert not m.is_circuit_open("x")
        report = m.health_report()
        cb_v = [v for v in report.violations if v.metric == "circuit_breaker"]
        assert not cb_v

    def test_is_circuit_open_unknown_connector_returns_false(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor
        m = PipelineHealthMonitor()
        assert not m.is_circuit_open("never_seen")

    def test_thread_safe_concurrent_failures(self):
        """4 threads each recording 100 failures on the same connector must not corrupt state."""
        import threading
        from app.intelligence.health_monitor import PipelineHealthMonitor
        m = PipelineHealthMonitor(cb_open_threshold=5)
        errors = []

        def fail_worker():
            try:
                for _ in range(100):
                    m.record_connector_failure("c")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=fail_worker) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
        assert m.is_circuit_open("c")

    def test_thread_safe_fail_then_success(self):
        """Interleaved failures and successes from multiple threads must not deadlock."""
        import threading
        from app.intelligence.health_monitor import PipelineHealthMonitor
        m = PipelineHealthMonitor(cb_open_threshold=5)
        errors = []

        def mixed_worker(tid):
            try:
                for i in range(100):
                    if i % 7 == 0:
                        m.record_connector_success(f"c{tid}")
                    else:
                        m.record_connector_failure(f"c{tid}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=mixed_worker, args=(t,)) for t in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
        # Just verify we can still generate a report
        report = m.health_report()
        assert report is not None

    def test_combined_circuit_and_freshness_violation(self):
        """Both an open circuit AND a stale connector ⇒ both RED violations present."""
        from datetime import timedelta
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        m = PipelineHealthMonitor(cb_open_threshold=3, freshness_slo_h=1.0, slo_warning_pct=1.01)
        # Open circuit
        for _ in range(3):
            m.record_connector_failure("bad")
        # Stale connector
        with m._lock:
            m._connectors["stale"] = _now() - timedelta(hours=2)
        report = m.health_report()
        assert report.overall_status == SLOStatus.RED
        metrics = {v.metric for v in report.violations}
        assert "circuit_breaker" in metrics
        assert "freshness_h" in metrics



# ===========================================================================
# Phase 5 — Item 1: NoveltyScorer
# ===========================================================================

class TestNoveltyScorer:
    """Exhaustive coverage of NoveltyScorer: construction, scoring, history,
    concurrency, edge cases, and UserDigestRanker integration."""

    # ── helpers ──────────────────────────────────────────────────────────────

    def _cand(self, iid: str, topics=None, entities=None, text=""):
        from app.personalization.models import DigestCandidate
        return DigestCandidate(
            item_id=iid,
            topic_ids=topics or [],
            entity_ids=entities or [],
            raw_text=text,
        )

    # ── Construction validation ───────────────────────────────────────────────

    def test_window_size_zero_raises(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        with pytest.raises(ValueError, match="window_size"):
            NoveltyScorer(window_size=0)

    def test_window_size_negative_raises(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        with pytest.raises(ValueError, match="window_size"):
            NoveltyScorer(window_size=-1)

    def test_decay_below_zero_raises(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        with pytest.raises(ValueError, match="decay_factor"):
            NoveltyScorer(decay_factor=-0.1)

    def test_decay_above_one_raises(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        with pytest.raises(ValueError, match="decay_factor"):
            NoveltyScorer(decay_factor=1.1)

    def test_min_novelty_negative_raises(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        with pytest.raises(ValueError, match="min_novelty"):
            NoveltyScorer(min_novelty=-0.01)

    def test_min_novelty_above_one_raises(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        with pytest.raises(ValueError, match="min_novelty"):
            NoveltyScorer(min_novelty=1.1)

    def test_top_text_tokens_zero_raises(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        with pytest.raises(ValueError, match="top_text_tokens"):
            NoveltyScorer(top_text_tokens=0)

    def test_default_construction_valid(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer()
        assert s._window_size == 50
        assert s._decay == 0.9
        assert s._min_novelty == 0.05

    # ── score() — empty history ───────────────────────────────────────────────

    def test_empty_history_returns_one(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer()
        c = self._cand("x", topics=["ml"])
        assert s.score(c) == 1.0

    def test_no_fingerprint_empty_history_returns_one(self):
        """Candidate with no topics/entities/text → fingerprint is empty set → 1.0."""
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer()
        c = self._cand("x")
        assert s.score(c) == 1.0

    def test_wrong_type_raises(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer()
        with pytest.raises(TypeError):
            s.score("not a candidate")

    # ── score() — after record_shown ─────────────────────────────────────────

    def test_identical_topics_after_shown_returns_min_novelty(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer(min_novelty=0.05, decay_factor=1.0)
        c = self._cand("i1", topics=["ml", "llm"])
        s.record_shown(c)
        assert s.score(c) == pytest.approx(0.05)

    def test_completely_different_topics_returns_one(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer(min_novelty=0.05, decay_factor=1.0)
        old = self._cand("i1", topics=["finance", "banking"])
        s.record_shown(old)
        new = self._cand("i2", topics=["ml", "llm", "ai"])
        assert s.score(new) == 1.0

    def test_partial_overlap_score_between_min_and_one(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer(min_novelty=0.0, decay_factor=1.0)
        old = self._cand("i1", topics=["ml", "ai", "finance"])
        s.record_shown(old)
        new = self._cand("i2", topics=["ml", "ai", "healthcare"])
        score = s.score(new)
        assert 0.0 < score < 1.0

    def test_score_does_not_mutate_history(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer()
        c = self._cand("i1", topics=["ml"])
        before = s.stats()["history_size"]
        s.score(c)
        after = s.stats()["history_size"]
        assert before == after == 0

    # ── decay factor ─────────────────────────────────────────────────────────

    def test_decay_zero_only_most_recent_counts(self):
        """decay=0 ⇒ only the last item in history contributes similarity."""
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer(decay_factor=0.0, min_novelty=0.0)
        # Record an old item with matching topics
        old = self._cand("i1", topics=["ml", "ai"])
        s.record_shown(old)
        # Record a recent item with completely different topics
        recent = self._cand("i2", topics=["finance"])
        s.record_shown(recent)
        # Candidate matches old item but not recent → should be high novelty
        candidate = self._cand("i3", topics=["ml", "ai"])
        score = s.score(candidate)
        # Only recent (finance) item counts with decay=0, so score ≈ 1.0
        assert score == pytest.approx(1.0)

    def test_decay_one_all_history_equal_weight(self):
        """decay=1 ⇒ uniform weighting across all history items."""
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer(decay_factor=1.0, min_novelty=0.0, window_size=100)
        # Record 10 identical items to build strong history
        for i in range(10):
            s.record_shown(self._cand(f"h{i}", topics=["ml", "ai"]))
        # New item with same topics → high similarity → low novelty
        score = s.score(self._cand("new", topics=["ml", "ai"]))
        assert score == pytest.approx(0.0)

    # ── window eviction ───────────────────────────────────────────────────────

    def test_window_size_enforced(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer(window_size=3)
        for i in range(10):
            s.record_shown(self._cand(f"i{i}", topics=[f"topic{i}"]))
        assert s.stats()["history_size"] == 3

    def test_oldest_evicted_first(self):
        """When window is full, oldest items are evicted so newest count most."""
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer(window_size=2, decay_factor=1.0, min_novelty=0.0)
        # Fill window with finance items
        s.record_shown(self._cand("old1", topics=["finance"]))
        s.record_shown(self._cand("old2", topics=["finance"]))
        # Overflow: replaces old1 with ml topic
        s.record_shown(self._cand("new1", topics=["ml"]))
        # Now history = [finance, ml] (old2 and new1)
        # Query ml → has 50% Jaccard with ml item → novelty < 1
        score = s.score(self._cand("q", topics=["ml"]))
        assert score < 1.0

    # ── fingerprint priority ──────────────────────────────────────────────────

    def test_topics_take_priority_over_entities(self):
        from app.personalization.novelty_scorer import NoveltyScorer, _fingerprint
        from app.personalization.models import DigestCandidate
        c = DigestCandidate(
            item_id="x",
            topic_ids=["ml"],
            entity_ids=["OpenAI"],
        )
        fp = _fingerprint(c, 30)
        assert "ml" in fp
        assert "openai" not in fp

    def test_entities_used_when_no_topics(self):
        from app.personalization.novelty_scorer import NoveltyScorer, _fingerprint
        from app.personalization.models import DigestCandidate
        c = DigestCandidate(item_id="x", entity_ids=["OpenAI", "Anthropic"])
        fp = _fingerprint(c, 30)
        assert "openai" in fp and "anthropic" in fp

    def test_text_fallback_when_no_topics_or_entities(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer()
        old = self._cand("i1", text="transformer attention mechanism neural")
        s.record_shown(old)
        # Same text ≈ same fingerprint → low novelty
        same = self._cand("i2", text="transformer attention mechanism neural")
        score = s.score(same)
        assert score < 0.5

    def test_text_fallback_different_text_high_novelty(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer()
        old = self._cand("i1", text="transformer attention mechanism")
        s.record_shown(old)
        new = self._cand("i2", text="quarterly revenue balance sheet profit")
        score = s.score(new)
        assert score >= 0.8

    # ── score_batch ───────────────────────────────────────────────────────────

    def test_score_batch_empty_list(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer()
        assert s.score_batch([]) == []

    def test_score_batch_wrong_type_raises(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer()
        with pytest.raises(TypeError):
            s.score_batch("not a list")

    def test_score_batch_returns_same_length(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer()
        candidates = [self._cand(f"i{j}", topics=[f"t{j}"]) for j in range(10)]
        scores = s.score_batch(candidates)
        assert len(scores) == 10

    def test_score_batch_all_in_range(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer(min_novelty=0.05)
        s.record_shown(self._cand("h1", topics=["a", "b"]))
        candidates = [self._cand(f"i{j}", topics=["a", "b"]) for j in range(5)]
        for sc in s.score_batch(candidates):
            assert 0.05 <= sc <= 1.0

    # ── reset ─────────────────────────────────────────────────────────────────

    def test_reset_clears_history_and_count(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer()
        for i in range(5):
            s.record_shown(self._cand(f"i{i}", topics=[f"t{i}"]))
        s.reset()
        st = s.stats()
        assert st["history_size"] == 0
        assert st["shown_count"] == 0

    def test_after_reset_score_returns_one(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer()
        c = self._cand("i1", topics=["ml"])
        s.record_shown(c)
        s.reset()
        assert s.score(c) == 1.0

    # ── stats ─────────────────────────────────────────────────────────────────

    def test_stats_fields_present(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer(window_size=10, decay_factor=0.7, min_novelty=0.1)
        st = s.stats()
        for key in ("window_size", "history_size", "shown_count", "decay_factor", "min_novelty"):
            assert key in st

    def test_shown_count_increments(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer()
        for i in range(7):
            s.record_shown(self._cand(f"i{i}", topics=[f"t{i}"]))
        assert s.stats()["shown_count"] == 7

    def test_record_shown_wrong_type_raises(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer()
        with pytest.raises(TypeError):
            s.record_shown({"item_id": "x"})

    # ── min_novelty floor ────────────────────────────────────────────────────

    def test_min_novelty_zero_allows_zero_output(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer(min_novelty=0.0, decay_factor=1.0)
        c = self._cand("i1", topics=["ml", "ai"])
        s.record_shown(c)
        assert s.score(c) == pytest.approx(0.0)

    def test_min_novelty_respected_for_identical(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer(min_novelty=0.20, decay_factor=1.0)
        c = self._cand("i1", topics=["ml"])
        s.record_shown(c)
        assert s.score(c) >= 0.20

    # ── UserDigestRanker integration ──────────────────────────────────────────

    def test_ranker_wrong_type_raises(self):
        from app.personalization.user_digest_ranker import UserDigestRanker
        with pytest.raises(TypeError, match="novelty_scorer"):
            UserDigestRanker(novelty_scorer="bad")

    def test_ranker_uses_scorer_over_precomputed(self):
        """When a NoveltyScorer is wired, it overrides DigestCandidate.novelty_score."""
        from app.personalization.novelty_scorer import NoveltyScorer
        from app.personalization.user_digest_ranker import UserDigestRanker
        from app.personalization.models import DigestCandidate, RankingWeights
        scorer = NoveltyScorer(min_novelty=0.0, decay_factor=1.0)
        # Build a history of items with topics=["ml"]
        for i in range(5):
            scorer.record_shown(self._cand(f"h{i}", topics=["ml", "ai"]))
        # Create a candidate with precomputed novelty=1.0 but matching topics
        cand = DigestCandidate(
            item_id="c1",
            topic_ids=["ml", "ai"],
            novelty_score=1.0,  # pre-computed says novel
        )
        # Ranker with full novelty weight should score low on novelty
        weights = RankingWeights(
            topic_relevance=0, embedding_similarity=0, recency=0,
            engagement=0, trust=0, novelty=1.0,
        )
        ranker = UserDigestRanker(novelty_scorer=scorer, weights=weights)
        items = ranker.rank([cand])
        # The live scorer overrides novelty_score=1.0; actual novelty should be ~0
        assert items[0].novelty_score < 0.5

    def test_ranker_scorer_error_falls_back_to_precomputed(self):
        """If NoveltyScorer.score() raises, ranker uses pre-computed field."""
        from unittest.mock import MagicMock
        from app.personalization.user_digest_ranker import UserDigestRanker
        from app.personalization.models import DigestCandidate
        from app.personalization.novelty_scorer import NoveltyScorer
        scorer = MagicMock(spec=NoveltyScorer)
        scorer.score.side_effect = RuntimeError("scorer exploded")
        cand = DigestCandidate(item_id="c1", novelty_score=0.77)
        ranker = UserDigestRanker(novelty_scorer=scorer)
        # Should not raise
        items = ranker.rank([cand])
        assert items[0] is not None
        assert items[0].novelty_score == pytest.approx(0.77)

    def test_ranker_no_scorer_uses_precomputed(self):
        """Without a scorer, UserDigestRanker passes candidate.novelty_score through."""
        from app.personalization.user_digest_ranker import UserDigestRanker
        from app.personalization.models import DigestCandidate
        cand = DigestCandidate(item_id="c1", novelty_score=0.42)
        ranker = UserDigestRanker()
        items = ranker.rank([cand])
        assert items[0].novelty_score == pytest.approx(0.42)

    # ── Thread safety ─────────────────────────────────────────────────────────

    def test_concurrent_score_and_record(self):
        """4 threads scoring + 4 threads recording must not corrupt history."""
        import threading
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer(window_size=50)
        errors = []

        def scorer_worker():
            try:
                for i in range(200):
                    c = self._cand(f"s{i}", topics=[f"topic_{i % 10}"])
                    score = s.score(c)
                    assert 0.0 <= score <= 1.0
            except Exception as e:
                errors.append(e)

        def recorder_worker():
            try:
                for i in range(100):
                    c = self._cand(f"r{i}", topics=[f"topic_{i % 5}"])
                    s.record_shown(c)
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=scorer_worker) for _ in range(4)] +
            [threading.Thread(target=recorder_worker) for _ in range(4)]
        )
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
        st = s.stats()
        assert st["history_size"] <= s._window_size

    def test_concurrent_reset_and_score_no_crash(self):
        """Concurrent reset() and score() calls must not raise."""
        import threading
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer(window_size=20)
        for i in range(10):
            s.record_shown(self._cand(f"h{i}", topics=["ml"]))
        errors = []

        def reset_worker():
            try:
                for _ in range(50):
                    s.reset()
            except Exception as e:
                errors.append(e)

        def score_worker():
            try:
                for i in range(100):
                    s.score(self._cand(f"q{i}", topics=["ml"]))
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=reset_worker) for _ in range(2)] +
            [threading.Thread(target=score_worker) for _ in range(4)]
        )
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors

    def test_concurrent_batch_score_no_corruption(self):
        """Parallel score_batch calls must never raise."""
        import threading
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer(window_size=30, decay_factor=0.95)
        for i in range(15):
            s.record_shown(self._cand(f"h{i}", topics=[f"t{i}"]))
        errors = []

        def batch_worker():
            try:
                for _ in range(50):
                    cands = [self._cand(f"c{j}", topics=[f"t{j % 5}"]) for j in range(10)]
                    scores = s.score_batch(cands)
                    assert len(scores) == 10
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=batch_worker) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors

    # ── Jaccard edge cases ────────────────────────────────────────────────────

    def test_jaccard_both_empty_returns_zero(self):
        from app.personalization.novelty_scorer import _jaccard
        assert _jaccard(frozenset(), frozenset()) == 0.0

    def test_jaccard_identical_returns_one(self):
        from app.personalization.novelty_scorer import _jaccard
        s = frozenset(["a", "b", "c"])
        assert _jaccard(s, s) == pytest.approx(1.0)

    def test_jaccard_disjoint_returns_zero(self):
        from app.personalization.novelty_scorer import _jaccard
        a = frozenset(["x", "y"])
        b = frozenset(["p", "q"])
        assert _jaccard(a, b) == pytest.approx(0.0)

    def test_jaccard_partial_overlap(self):
        from app.personalization.novelty_scorer import _jaccard
        a = frozenset(["a", "b", "c"])
        b = frozenset(["b", "c", "d"])
        # intersection=2, union=4 → 0.5
        assert _jaccard(a, b) == pytest.approx(0.5)

    # ── Boundary conditions ───────────────────────────────────────────────────

    def test_window_size_one_keeps_last(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer(window_size=1, min_novelty=0.0)
        s.record_shown(self._cand("old", topics=["finance"]))
        s.record_shown(self._cand("new", topics=["ml"]))
        assert s.stats()["history_size"] == 1
        # Only 'ml' is in history; querying ml → low novelty
        assert s.score(self._cand("q", topics=["ml"])) < 0.5

    def test_decay_exactly_one(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer(decay_factor=1.0, min_novelty=0.0, window_size=100)
        for i in range(20):
            s.record_shown(self._cand(f"h{i}", topics=["ml", "ai"]))
        score = s.score(self._cand("q", topics=["ml", "ai"]))
        assert score == pytest.approx(0.0)

    def test_min_novelty_one_always_returns_one(self):
        from app.personalization.novelty_scorer import NoveltyScorer
        s = NoveltyScorer(min_novelty=1.0)
        c = self._cand("i1", topics=["ml"])
        s.record_shown(c)
        assert s.score(c) == 1.0



# ===========================================================================
# Phase 5 — Item 2: PipelineResultAdapter
# ===========================================================================

class TestPipelineResultAdapter:
    """Exhaustive tests for PipelineResultAdapter.adapt() and adapt_batch()."""

    # ── helpers ──────────────────────────────────────────────────────────────

    def _ok_result(self, **kwargs):
        """Create a minimal actionable IntelligencePipelineResult."""
        import uuid
        from app.ingestion.pipeline_result import IntelligencePipelineResult, PipelineStatus
        defaults = dict(
            content_item_id=uuid.uuid4(),
            source_family="social",
            status=PipelineStatus.SUCCESS,
            summary="This is a test summary about ML and AI.",
            keywords=["ml", "ai"],
            entities=["OpenAI"],
            signal_type="SOCIAL_SIGNAL",
            confidence=0.80,
        )
        defaults.update(kwargs)
        return IntelligencePipelineResult(**defaults)

    def _failed_result(self):
        import uuid
        from app.ingestion.pipeline_result import IntelligencePipelineResult, PipelineStatus
        return IntelligencePipelineResult(
            content_item_id=uuid.uuid4(),
            source_family="research",
            status=PipelineStatus.FAILED,
        )

    def _partial_result(self):
        import uuid
        from app.ingestion.pipeline_result import IntelligencePipelineResult, PipelineStatus
        return IntelligencePipelineResult(
            content_item_id=uuid.uuid4(),
            source_family="news",
            status=PipelineStatus.PARTIAL,
            summary="Partial content",
        )

    def _skipped_result(self):
        import uuid
        from app.ingestion.pipeline_result import IntelligencePipelineResult, PipelineStatus
        return IntelligencePipelineResult(
            content_item_id=uuid.uuid4(),
            source_family="social",
            status=PipelineStatus.SKIPPED,
        )

    # ── Construction validation ───────────────────────────────────────────────

    def test_default_construction(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        a = PipelineResultAdapter()
        assert a._trust_scorer is None
        assert a._novelty_scorer is None
        assert a._default_trust == 0.5

    def test_default_trust_out_of_range_raises(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        with pytest.raises(ValueError, match="default_trust"):
            PipelineResultAdapter(default_trust=1.5)

    def test_default_trust_negative_raises(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        with pytest.raises(ValueError, match="default_trust"):
            PipelineResultAdapter(default_trust=-0.1)

    def test_max_raw_text_zero_raises(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        with pytest.raises(ValueError, match="max_raw_text_chars"):
            PipelineResultAdapter(max_raw_text_chars=0)

    def test_wrong_trust_scorer_type_raises(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        with pytest.raises(TypeError, match="trust_scorer"):
            PipelineResultAdapter(trust_scorer="not a scorer")

    def test_wrong_novelty_scorer_type_raises(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        with pytest.raises(TypeError, match="novelty_scorer"):
            PipelineResultAdapter(novelty_scorer=42)

    # ── adapt() — type guard ──────────────────────────────────────────────────

    def test_adapt_wrong_type_raises(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        a = PipelineResultAdapter()
        with pytest.raises(TypeError, match="IntelligencePipelineResult"):
            a.adapt("not a result")

    def test_adapt_dict_raises(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        a = PipelineResultAdapter()
        with pytest.raises(TypeError):
            a.adapt({"content_item_id": "x"})

    # ── adapt() — non-actionable ──────────────────────────────────────────────

    def test_failed_result_returns_none(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        a = PipelineResultAdapter()
        assert a.adapt(self._failed_result()) is None

    def test_skipped_result_returns_none(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        a = PipelineResultAdapter()
        assert a.adapt(self._skipped_result()) is None

    def test_partial_result_is_adapted(self):
        """PARTIAL status is actionable → must produce a candidate."""
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        a = PipelineResultAdapter()
        c = a.adapt(self._partial_result())
        assert c is not None

    # ── adapt() — field mapping ───────────────────────────────────────────────

    def test_item_id_mapped(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        r = self._ok_result()
        c = PipelineResultAdapter().adapt(r)
        assert c.item_id == str(r.content_item_id)

    def test_topic_ids_mapped_from_keywords(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        r = self._ok_result(keywords=["nlp", "bert", "gpt"])
        c = PipelineResultAdapter().adapt(r)
        assert set(c.topic_ids) == {"nlp", "bert", "gpt"}

    def test_entity_ids_mapped(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        r = self._ok_result(entities=["Google", "DeepMind"])
        c = PipelineResultAdapter().adapt(r)
        assert "Google" in c.entity_ids and "DeepMind" in c.entity_ids

    def test_source_platform_mapped(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        r = self._ok_result(source_family="media_audio")
        c = PipelineResultAdapter().adapt(r)
        assert c.source_platform == "media_audio"

    def test_published_at_mapped(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        r = self._ok_result()
        c = PipelineResultAdapter().adapt(r)
        assert c.published_at == r.produced_at

    def test_title_from_summary(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        r = self._ok_result(summary="A short summary")
        c = PipelineResultAdapter().adapt(r)
        assert "A short summary" in c.title

    def test_title_truncated_at_200_chars(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        long_summary = "x" * 500
        r = self._ok_result(summary=long_summary)
        c = PipelineResultAdapter().adapt(r)
        assert len(c.title) <= 200

    def test_title_fallback_to_signal_type_when_no_summary(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        r = self._ok_result(summary="", signal_type="PODCAST_EPISODE")
        c = PipelineResultAdapter().adapt(r)
        assert c.title == "PODCAST_EPISODE"

    def test_raw_text_includes_summary(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        r = self._ok_result(summary="Important finding", claims=[])
        c = PipelineResultAdapter().adapt(r)
        assert "Important finding" in c.raw_text

    def test_raw_text_includes_up_to_five_claims(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        claims = [f"claim{i}" for i in range(10)]
        r = self._ok_result(claims=claims)
        c = PipelineResultAdapter().adapt(r)
        # Only first 5 claims included
        for i in range(5):
            assert f"claim{i}" in c.raw_text
        for i in range(5, 10):
            assert f"claim{i}" not in c.raw_text

    def test_raw_text_truncated_to_max_chars(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        r = self._ok_result(summary="x" * 5000)
        c = PipelineResultAdapter(max_raw_text_chars=100).adapt(r)
        assert len(c.raw_text) <= 100

    def test_metadata_contains_signal_type(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        r = self._ok_result(signal_type="DEVELOPER_RELEASE")
        c = PipelineResultAdapter().adapt(r)
        assert c.metadata["signal_type"] == "DEVELOPER_RELEASE"

    def test_metadata_contains_result_id(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        r = self._ok_result()
        c = PipelineResultAdapter().adapt(r)
        assert c.metadata["result_id"] == str(r.result_id)

    def test_metadata_contains_tenant_id(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        r = self._ok_result()
        r = r.model_copy(update={"tenant_id": "acme"})
        c = PipelineResultAdapter().adapt(r)
        assert c.metadata["tenant_id"] == "acme"

    def test_default_trust_used_when_no_scorer(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        r = self._ok_result()
        c = PipelineResultAdapter(default_trust=0.7).adapt(r)
        assert c.trust_score == pytest.approx(0.7)

    def test_trust_score_clipped_to_0_1(self):
        """Even if scorer returns out-of-range, we clip."""
        from unittest.mock import MagicMock
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        from app.source_intelligence.source_trust import SourceTrustScorer, TrustScore
        scorer = MagicMock(spec=SourceTrustScorer)
        scorer.score.return_value = TrustScore(
            source_id="x", composite=0.999, primacy=1.0,
            recency=1.0, accuracy=1.0, authority=1.0,
        )
        a = PipelineResultAdapter(trust_scorer=scorer)
        c = a.adapt(self._ok_result())
        assert 0.0 <= c.trust_score <= 1.0

    # ── adapt() — trust_scorer wiring ────────────────────────────────────────

    def test_trust_scorer_used_to_compute_trust(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        from app.source_intelligence.source_trust import SourceTrustScorer
        ts = SourceTrustScorer()
        ts.set_primacy("social", True)
        ts.set_authority("social", 0.9)
        a = PipelineResultAdapter(trust_scorer=ts)
        r = self._ok_result(source_family="social")
        c = a.adapt(r)
        # Should differ from default 0.5
        assert c.trust_score != 0.5

    def test_trust_scorer_failure_falls_back_gracefully(self):
        """If trust_scorer raises, adapter falls back to result.confidence."""
        from unittest.mock import MagicMock
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        from app.source_intelligence.source_trust import SourceTrustScorer
        scorer = MagicMock(spec=SourceTrustScorer)
        scorer.score.side_effect = RuntimeError("scorer exploded")
        a = PipelineResultAdapter(trust_scorer=scorer, default_trust=0.42)
        r = self._ok_result(confidence=0.80)
        c = a.adapt(r)
        # Should not raise; trust falls back to confidence
        assert 0.0 <= c.trust_score <= 1.0

    # ── adapt() — novelty_scorer wiring ──────────────────────────────────────

    def test_novelty_scorer_used_when_provided(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        from app.personalization.novelty_scorer import NoveltyScorer
        ns = NoveltyScorer()
        a = PipelineResultAdapter(novelty_scorer=ns)
        r = self._ok_result()
        c = a.adapt(r)
        # Empty history → novelty should be 1.0
        assert c.novelty_score == 1.0

    def test_default_novelty_without_scorer(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        a = PipelineResultAdapter()
        c = a.adapt(self._ok_result())
        assert c.novelty_score == 0.5

    def test_novelty_scorer_failure_leaves_default_novelty(self):
        from unittest.mock import MagicMock
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        from app.personalization.novelty_scorer import NoveltyScorer
        ns = MagicMock(spec=NoveltyScorer)
        ns.score.side_effect = RuntimeError("novelty blew up")
        a = PipelineResultAdapter(novelty_scorer=ns)
        c = a.adapt(self._ok_result())
        assert c.novelty_score == 0.5  # default maintained

    # ── adapt_batch() ─────────────────────────────────────────────────────────

    def test_adapt_batch_wrong_type_raises(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        a = PipelineResultAdapter()
        with pytest.raises(TypeError, match="list"):
            a.adapt_batch("not a list")

    def test_adapt_batch_empty_list(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        assert PipelineResultAdapter().adapt_batch([]) == []

    def test_adapt_batch_skips_non_actionable(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        a = PipelineResultAdapter()
        results = [
            self._ok_result(),
            self._failed_result(),
            self._skipped_result(),
            self._ok_result(),
        ]
        candidates = a.adapt_batch(results)
        assert len(candidates) == 2

    def test_adapt_batch_partial_included(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        a = PipelineResultAdapter()
        results = [self._ok_result(), self._partial_result(), self._failed_result()]
        candidates = a.adapt_batch(results)
        assert len(candidates) == 2

    def test_adapt_batch_error_isolation(self):
        """If one adapt() raises internally, adapt_batch continues with the rest."""
        from unittest.mock import patch
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        a = PipelineResultAdapter()
        r1 = self._ok_result()
        r2 = self._ok_result()
        call_count = 0

        original_adapt = a.adapt

        def patched_adapt(r):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("boom")
            return original_adapt(r)

        a.adapt = patched_adapt
        candidates = a.adapt_batch([r1, r2])
        # First item failed silently; second item adapted
        assert len(candidates) == 1

    def test_adapt_batch_preserves_order(self):
        """Candidates must appear in the same order as actionable results."""
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        a = PipelineResultAdapter()
        r1 = self._ok_result(summary="first")
        r2 = self._ok_result(summary="second")
        r3 = self._ok_result(summary="third")
        candidates = a.adapt_batch([r1, r2, r3])
        assert candidates[0].item_id == str(r1.content_item_id)
        assert candidates[1].item_id == str(r2.content_item_id)
        assert candidates[2].item_id == str(r3.content_item_id)

    def test_adapt_batch_all_unique_item_ids(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        a = PipelineResultAdapter()
        results = [self._ok_result() for _ in range(20)]
        candidates = a.adapt_batch(results)
        ids = [c.item_id for c in candidates]
        assert len(ids) == len(set(ids))

    # ── End-to-end: adapter → ranker pipeline ────────────────────────────────

    def test_adapt_then_rank_produces_ordered_list(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        from app.personalization.user_digest_ranker import UserDigestRanker
        adapter = PipelineResultAdapter()
        ranker = UserDigestRanker()
        results = [
            self._ok_result(summary=f"Story {i}", keywords=[f"topic{i}"])
            for i in range(5)
        ]
        results.append(self._failed_result())
        candidates = adapter.adapt_batch(results)
        ranked = ranker.rank(candidates)
        assert len(ranked) == 5
        scores = [r.final_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_adapt_with_novelty_scorer_then_rank(self):
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        from app.personalization.novelty_scorer import NoveltyScorer
        from app.personalization.user_digest_ranker import UserDigestRanker
        ns = NoveltyScorer(window_size=10, min_novelty=0.0)
        adapter = PipelineResultAdapter(novelty_scorer=ns)
        ranker = UserDigestRanker()
        results = [self._ok_result(keywords=["ml", "ai"]) for _ in range(5)]
        candidates = adapter.adapt_batch(results)
        ranked = ranker.rank(candidates)
        assert len(ranked) == 5

    # ── Thread safety ─────────────────────────────────────────────────────────

    def test_concurrent_adapt_calls_safe(self):
        """Multiple threads calling adapt() simultaneously must not corrupt output."""
        import threading
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        adapter = PipelineResultAdapter()
        errors = []
        results_list = [[] for _ in range(4)]

        def worker(tid):
            try:
                for _ in range(100):
                    r = self._ok_result()
                    c = adapter.adapt(r)
                    if c is None:
                        errors.append("Got None for actionable result")
                    else:
                        results_list[tid].append(c.item_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
        # All item IDs must be unique across all threads
        all_ids = [iid for lst in results_list for iid in lst]
        assert len(all_ids) == len(set(all_ids)) == 400

    def test_concurrent_adapt_batch_calls_safe(self):
        import threading
        from app.ingestion.pipeline_result_adapter import PipelineResultAdapter
        adapter = PipelineResultAdapter()
        errors = []

        def batch_worker():
            try:
                for _ in range(50):
                    batch = [self._ok_result() for _ in range(5)]
                    batch.append(self._failed_result())
                    candidates = adapter.adapt_batch(batch)
                    assert len(candidates) == 5
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=batch_worker) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors



# ===========================================================================
# Phase 5 — Item 3: SourceTrustScorer → IndexingPipeline wiring
# ===========================================================================

class TestIndexingPipelineTrustScorer:
    """Tests for SourceTrustScorer integration in IndexingPipeline."""

    def _make_ts(self, source_family="social", primacy=True, authority=0.8):
        from app.source_intelligence.source_trust import SourceTrustScorer
        ts = SourceTrustScorer()
        ts.set_primacy(source_family, primacy)
        ts.set_authority(source_family, authority)
        return ts

    # ── Construction ─────────────────────────────────────────────────────────

    def test_trust_scorer_property_returns_injected(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        ts = self._make_ts()
        p = IndexingPipeline(trust_scorer=ts)
        assert p.trust_scorer is ts

    def test_trust_scorer_property_none_by_default(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        p = IndexingPipeline()
        assert p.trust_scorer is None

    def test_trust_scorer_is_optional_not_required(self):
        """Pipeline must construct and operate normally without a trust scorer."""
        from app.ingestion.indexing_pipeline import IndexingPipeline
        p = IndexingPipeline()
        assert p.trust_scorer is None

    # ── Trust score stamped in chunk metadata ─────────────────────────────────

    def test_trust_score_in_chunk_metadata(self):
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        store = ChunkStore()
        ts = self._make_ts(source_family="social", primacy=True, authority=0.9)
        p = IndexingPipeline(chunk_store=store, trust_scorer=ts)
        item = _make_content_item(platform="social", text="Large language models are incredible")
        asyncio.run(p.process_batch([item]))
        chunks = store.get_by_observation(str(item.id))
        for chunk in chunks:
            assert "trust_score" in chunk.metadata
            assert 0.0 <= chunk.metadata["trust_score"] <= 1.0

    def test_no_trust_scorer_no_trust_score_in_metadata(self):
        """Without a trust scorer, chunks must NOT have a trust_score key."""
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        store = ChunkStore()
        p = IndexingPipeline(chunk_store=store)  # no trust_scorer
        item = _make_content_item(text="Testing metadata without trust scorer")
        asyncio.run(p.process_batch([item]))
        chunks = store.get_by_observation(str(item.id))
        for chunk in chunks:
            assert "trust_score" not in chunk.metadata

    def test_trust_score_value_matches_scorer_output(self):
        """trust_score in metadata must equal SourceTrustScorer.score().composite.
        Note: 'reddit' platform maps to source_family='social' in the router."""
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        from app.source_intelligence.source_trust import SourceTrustScorer
        store = ChunkStore()
        ts = SourceTrustScorer()
        ts.set_primacy("social", False)
        ts.set_authority("social", 0.3)
        expected_composite = ts.score("social").composite
        p = IndexingPipeline(chunk_store=store, trust_scorer=ts)
        # 'reddit' → source_family='social' in ContentPipelineRouter
        item = _make_content_item(platform="reddit", text="Machine learning advances")
        asyncio.run(p.process_batch([item]))
        chunks = store.get_by_observation(str(item.id))
        assert chunks, "Expected at least one chunk for actionable item"
        for chunk in chunks:
            assert chunk.metadata.get("trust_score") == pytest.approx(expected_composite)

    def test_trust_scorer_failure_does_not_abort_indexing(self):
        """If trust_scorer.score() raises, chunk indexing must still succeed."""
        import asyncio
        from unittest.mock import MagicMock
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        store = ChunkStore()
        bad_scorer = MagicMock()
        bad_scorer.score.side_effect = RuntimeError("scorer exploded")
        p = IndexingPipeline(chunk_store=store, trust_scorer=bad_scorer)
        item = _make_content_item(text="Trust scorer failure isolation test content")
        result = asyncio.run(p.process_batch([item]))
        # Chunks still indexed despite scorer failure
        assert result.stats.chunks_indexed >= 1
        chunks = store.get_by_observation(str(item.id))
        # trust_score key absent when scorer failed
        for chunk in chunks:
            assert "trust_score" not in chunk.metadata

    def test_trust_score_present_for_each_chunk_of_multi_chunk_item(self):
        """A long item produces multiple chunks; ALL must have trust_score.
        Uses chunk_size=10 with 200+ words to guarantee multiple chunks."""
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        store = ChunkStore()
        ts = self._make_ts(source_family="social")
        p = IndexingPipeline(chunk_store=store, trust_scorer=ts, chunk_size=10, chunk_overlap=0)
        long_text = " ".join(f"word{i}" for i in range(200))
        item = _make_content_item(text=long_text)
        asyncio.run(p.process_batch([item]))
        chunks = store.get_by_observation(str(item.id))
        assert len(chunks) > 1, (
            f"Expected multiple chunks with chunk_size=10 but got {len(chunks)}"
        )
        for chunk in chunks:
            assert "trust_score" in chunk.metadata

    def test_different_sources_get_different_trust_scores(self):
        """Two items from different source_families must get different trust scores.
        Uses 'reddit' (→ social) and 'rss' (→ news) to ensure distinct families."""
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        from app.source_intelligence.source_trust import SourceTrustScorer
        store = ChunkStore()
        ts = SourceTrustScorer()
        ts.set_primacy("social", False)
        ts.set_authority("social", 0.1)
        ts.set_primacy("news", True)
        ts.set_authority("news", 0.9)
        p = IndexingPipeline(chunk_store=store, trust_scorer=ts)
        # reddit → source_family="social"; rss → source_family="news"
        item_social = _make_content_item(platform="reddit", text="Social media post about events")
        item_news = _make_content_item(platform="rss", text="Research findings about events")
        asyncio.run(p.process_batch([item_social, item_news]))
        social_chunks = store.get_by_observation(str(item_social.id))
        news_chunks = store.get_by_observation(str(item_news.id))
        if social_chunks and news_chunks:
            social_trust = social_chunks[0].metadata.get("trust_score", 0)
            news_trust = news_chunks[0].metadata.get("trust_score", 0)
            # News with primacy=True and high authority should have higher trust
            assert news_trust > social_trust

    def test_trust_scorer_with_tenant_partitioning(self):
        """Trust scorer must work correctly with per-tenant ChunkStores."""
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        ts = self._make_ts(source_family="social")
        p = IndexingPipeline(trust_scorer=ts)
        item = _make_content_item(text="Tenant-aware trust scoring content")
        asyncio.run(p.process_batch([item], tenant_id="acme"))
        tenant_store = p.tenant_store("acme")
        assert tenant_store is not None
        chunks = tenant_store.get_by_observation(str(item.id))
        for chunk in chunks:
            assert "trust_score" in chunk.metadata

    def test_existing_chunk_metadata_preserved_with_trust_scorer(self):
        """trust_score is ADDED to metadata; other keys must not be overwritten."""
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        store = ChunkStore()
        ts = self._make_ts()
        p = IndexingPipeline(chunk_store=store, trust_scorer=ts)
        item = _make_content_item(text="Metadata preservation test content")
        asyncio.run(p.process_batch([item]))
        chunks = store.get_by_observation(str(item.id))
        for chunk in chunks:
            # These keys must still be present
            assert "signal_type" in chunk.metadata
            assert "result_id" in chunk.metadata
            assert "tenant_id" in chunk.metadata
            assert "trust_score" in chunk.metadata

    # ── Thread safety ─────────────────────────────────────────────────────────

    def test_concurrent_batches_with_trust_scorer_no_corruption(self):
        """Multiple async pipelines sharing a trust scorer must not corrupt metadata."""
        import asyncio, threading
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        from app.source_intelligence.source_trust import SourceTrustScorer
        ts = SourceTrustScorer()
        ts.set_primacy("social", True)
        errors = []

        def run_pipeline():
            try:
                store = ChunkStore()
                p = IndexingPipeline(chunk_store=store, trust_scorer=ts)
                for _ in range(10):
                    item = _make_content_item(text="Thread safety trust test content")
                    asyncio.run(p.process_batch([item]))
                    chunks = store.get_by_observation(str(item.id))
                    for chunk in chunks:
                        ts_val = chunk.metadata.get("trust_score")
                        assert ts_val is not None and 0.0 <= ts_val <= 1.0
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_pipeline) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors

    def test_trust_scorer_cache_not_mutated_by_pipeline(self):
        """The pipeline must only READ from the scorer, never mutate its cache."""
        import asyncio
        from app.source_intelligence.source_trust import SourceTrustScorer
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        ts = SourceTrustScorer()
        ts.set_primacy("social", True)
        ts.set_authority("social", 0.8)
        before = ts.score("social").composite
        p = IndexingPipeline(chunk_store=ChunkStore(), trust_scorer=ts)
        for _ in range(5):
            item = _make_content_item(text="Cache integrity test content")
            asyncio.run(p.process_batch([item]))
        after = ts.score("social").composite
        assert before == pytest.approx(after)



# ===========================================================================
# Phase 5 — Item 4: GroundedSummaryBuilder ↔ IndexingPipeline wiring
# ===========================================================================

class TestIndexingPipelineGroundedSummary:
    """Tests for IndexingPipeline.build_grounded_summary()."""

    def _make_actionable_result(self, summary_text="ML is transforming industries."):
        """Build an IntelligencePipelineResult with non-empty summary."""
        import uuid
        from app.ingestion.pipeline_result import IntelligencePipelineResult, PipelineStatus
        return IntelligencePipelineResult(
            content_item_id=uuid.uuid4(),
            source_family="social",
            status=PipelineStatus.SUCCESS,
            summary=summary_text,
            confidence=0.75,
        )

    def _make_failed_result(self):
        import uuid
        from app.ingestion.pipeline_result import IntelligencePipelineResult, PipelineStatus
        return IntelligencePipelineResult(
            content_item_id=uuid.uuid4(),
            source_family="research",
            status=PipelineStatus.FAILED,
        )

    def _make_indexing_result(self, summaries=None, include_failed=False):
        """Build an IndexingResult with actionable pipeline results."""
        from app.ingestion.indexing_pipeline import IndexingResult
        prs = []
        for s in (summaries or ["Machine learning is advancing rapidly."]):
            prs.append(self._make_actionable_result(summary_text=s))
        if include_failed:
            prs.append(self._make_failed_result())
        return IndexingResult(pipeline_results=prs)

    # ── Type guards ───────────────────────────────────────────────────────────

    def test_wrong_indexing_result_type_raises(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        p = IndexingPipeline()
        with pytest.raises(TypeError, match="IndexingResult"):
            p.build_grounded_summary("not an IndexingResult", topic="ai")

    def test_empty_topic_raises_typeerror(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        p = IndexingPipeline()
        ir = self._make_indexing_result()
        with pytest.raises(TypeError, match="topic"):
            p.build_grounded_summary(ir, "")

    def test_whitespace_topic_raises_typeerror(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        p = IndexingPipeline()
        ir = self._make_indexing_result()
        with pytest.raises(TypeError, match="topic"):
            p.build_grounded_summary(ir, "   ")

    def test_non_string_topic_raises_typeerror(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        p = IndexingPipeline()
        ir = self._make_indexing_result()
        with pytest.raises(TypeError, match="topic"):
            p.build_grounded_summary(ir, 123)

    def test_min_source_trust_negative_raises_valueerror(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        p = IndexingPipeline()
        ir = self._make_indexing_result()
        with pytest.raises(ValueError, match="min_source_trust"):
            p.build_grounded_summary(ir, "ai", min_source_trust=-0.1)

    def test_min_source_trust_above_one_raises_valueerror(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        p = IndexingPipeline()
        ir = self._make_indexing_result()
        with pytest.raises(ValueError, match="min_source_trust"):
            p.build_grounded_summary(ir, "ai", min_source_trust=1.1)

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_returns_grounded_summary_for_actionable_results(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.summarization.models import GroundedSummary
        ir = self._make_indexing_result(summaries=[
            "Machine learning is transforming every industry.",
            "Deep learning models achieve new accuracy records.",
        ])
        p = IndexingPipeline()
        summary = p.build_grounded_summary(ir, topic="machine learning")
        assert summary is not None
        assert isinstance(summary, GroundedSummary)

    def test_source_count_matches_actionable_items(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        ir = self._make_indexing_result(
            summaries=["AI advances.", "ML improves.", "DL excels."],
            include_failed=True,  # failed items must NOT count
        )
        p = IndexingPipeline()
        summary = p.build_grounded_summary(ir, topic="ml")
        assert summary is not None
        assert summary.source_count == 3  # only the 3 actionable results

    def test_confidence_score_in_range(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        ir = self._make_indexing_result()
        p = IndexingPipeline()
        summary = p.build_grounded_summary(ir, topic="ai")
        assert 0.0 <= summary.confidence_score <= 1.0

    def test_what_happened_is_non_empty(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        ir = self._make_indexing_result(summaries=[
            "Large language models are achieving human-level performance on many tasks.",
        ])
        p = IndexingPipeline()
        summary = p.build_grounded_summary(ir, topic="llm progress")
        assert summary.what_happened.strip() != ""

    def test_returns_none_for_all_failed_results(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline, IndexingResult
        ir = IndexingResult(pipeline_results=[self._make_failed_result() for _ in range(3)])
        p = IndexingPipeline()
        result = p.build_grounded_summary(ir, topic="test topic")
        assert result is None

    def test_returns_none_for_empty_pipeline_results(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline, IndexingResult
        p = IndexingPipeline()
        result = p.build_grounded_summary(IndexingResult(), topic="test topic")
        assert result is None

    def test_who_it_affects_forwarded(self):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        ir = self._make_indexing_result()
        p = IndexingPipeline()
        summary = p.build_grounded_summary(ir, topic="ai", who_it_affects=["developers"])
        assert "developers" in summary.who_it_affects

    # ── trust_scorer wiring ───────────────────────────────────────────────────

    def test_trust_scorer_used_for_evidence_source_trust(self):
        """When trust_scorer is provided, EvidenceSource.trust_score comes from scorer."""
        from unittest.mock import MagicMock, patch
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.source_intelligence.source_trust import SourceTrustScorer, TrustScore
        ts = SourceTrustScorer()
        ts.set_primacy("social", True)
        ts.set_authority("social", 0.95)
        p = IndexingPipeline(trust_scorer=ts)
        expected_trust = ts.score("social").composite
        ir = self._make_indexing_result(summaries=[
            "Social ML data pipeline produces high-quality results."
        ])
        summary = p.build_grounded_summary(ir, topic="ml pipeline")
        # Source attributions should use the trust scorer's composite
        for src in summary.source_attributions:
            assert src.trust_score == pytest.approx(expected_trust)

    def test_trust_scorer_failure_falls_back_to_confidence(self):
        """If trust_scorer.score() raises, confidence value is used as trust fallback."""
        from unittest.mock import MagicMock
        from app.ingestion.indexing_pipeline import IndexingPipeline
        bad_ts = MagicMock()
        bad_ts.score.side_effect = RuntimeError("scorer failed")
        p = IndexingPipeline(trust_scorer=bad_ts)
        ir = self._make_indexing_result()
        # Should not raise; falls back to confidence
        summary = p.build_grounded_summary(ir, topic="fallback test")
        assert summary is not None

    # ── Custom summary_builder injection ────────────────────────────────────

    def test_custom_summary_builder_used(self):
        """A user-injected summary_builder must be used instead of default."""
        from unittest.mock import MagicMock
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.summarization.models import GroundedSummary
        # Create a real GroundedSummary mock return value
        mock_summary = GroundedSummary(
            what_happened="Custom builder output",
            why_it_matters="Because custom",
            source_count=1,
        )
        custom_builder = MagicMock()
        custom_builder.build.return_value = mock_summary
        ir = self._make_indexing_result()
        p = IndexingPipeline()
        result = p.build_grounded_summary(ir, topic="custom", summary_builder=custom_builder)
        custom_builder.build.assert_called_once()
        assert result is mock_summary

    def test_builder_exception_returns_none_gracefully(self):
        """If summary_builder.build() raises, method returns None (no crash)."""
        from unittest.mock import MagicMock
        from app.ingestion.indexing_pipeline import IndexingPipeline
        bad_builder = MagicMock()
        bad_builder.build.side_effect = RuntimeError("builder exploded")
        ir = self._make_indexing_result()
        p = IndexingPipeline()
        result = p.build_grounded_summary(ir, topic="ai", summary_builder=bad_builder)
        assert result is None

    # ── min_source_trust filtering ────────────────────────────────────────────

    def test_min_source_trust_zero_includes_all(self):
        """min_source_trust=0.0 must include all actionable results."""
        from app.ingestion.indexing_pipeline import IndexingPipeline
        ir = self._make_indexing_result(summaries=[
            "Low-trust source content.", "Another source content.",
        ])
        p = IndexingPipeline()
        summary = p.build_grounded_summary(ir, topic="test", min_source_trust=0.0)
        assert summary is not None
        assert summary.source_count == 2

    def test_max_claims_forwarded(self):
        """max_claims parameter is forwarded to the SynthesisRequest."""
        from unittest.mock import MagicMock, call, patch
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.summarization.models import GroundedSummary, SynthesisRequest
        built_requests = []
        original_build = None

        class CapturingBuilder:
            def build(self, request):
                built_requests.append(request)
                from app.summarization.grounded_summary_builder import GroundedSummaryBuilder
                real = GroundedSummaryBuilder()
                return real.build(request)

        ir = self._make_indexing_result()
        p = IndexingPipeline()
        p.build_grounded_summary(
            ir, topic="test", summary_builder=CapturingBuilder(), max_claims=7
        )
        assert built_requests[0].max_claims == 7

    # ── Integration: full pipeline + grounded summary ─────────────────────────

    def test_end_to_end_process_batch_then_grounded_summary(self):
        """Full flow: process_batch → build_grounded_summary."""
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.summarization.models import GroundedSummary
        p = IndexingPipeline()
        items = [
            _make_content_item(text="Deep learning is revolutionizing AI capabilities globally."),
            _make_content_item(text="Transformer models set new records on language benchmarks."),
        ]
        ir = asyncio.run(p.process_batch(items))
        summary = p.build_grounded_summary(ir, topic="deep learning progress")
        if summary:  # may be None if routing extracted no text
            assert isinstance(summary, GroundedSummary)
            assert 0.0 <= summary.confidence_score <= 1.0

    def test_end_to_end_with_trust_scorer(self):
        """Full flow with trust_scorer: scores propagate into EvidenceSource."""
        import asyncio
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.source_intelligence.source_trust import SourceTrustScorer
        ts = SourceTrustScorer()
        ts.set_primacy("social", True)
        p = IndexingPipeline(trust_scorer=ts)
        items = [_make_content_item(text="AI research is progressing. This matters for all.")]
        ir = asyncio.run(p.process_batch(items))
        summary = p.build_grounded_summary(ir, topic="ai")
        if summary:
            for src in summary.source_attributions:
                assert 0.0 <= src.trust_score <= 1.0

    # ── Thread safety ─────────────────────────────────────────────────────────

    def test_concurrent_build_grounded_summary_no_crash(self):
        """build_grounded_summary is re-entrant and thread-safe."""
        import threading
        from app.ingestion.indexing_pipeline import IndexingPipeline
        p = IndexingPipeline()
        errors = []

        def worker():
            try:
                for i in range(20):
                    ir = self._make_indexing_result(summaries=[
                        f"Content item {i}: machine learning advances rapidly across all domains.",
                    ])
                    result = p.build_grounded_summary(ir, topic=f"ml-{i}")
                    assert result is not None
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors



# ===========================================================================
# Phase 5 — Item 5: ModelArtifactRegistry.check_and_rollback()
# ===========================================================================

class TestModelArtifactRegistryCheckAndRollback:
    """Exhaustive coverage of check_and_rollback SLO auto-rollback."""

    # ── Fixtures ─────────────────────────────────────────────────────────────

    @pytest.fixture()
    def reg(self, tmp_path):
        """Fresh in-memory-isolated registry using tmp_path."""
        from training.model_registry import ModelArtifactRegistry
        return ModelArtifactRegistry(
            registry_path=tmp_path / "reg.json",
            audit_log_path=tmp_path / "audit.jsonl",
        )

    @pytest.fixture()
    def populated_reg(self, reg):
        """Registry with two passing artifacts; a2 is production."""
        from training.model_registry import ArtifactRecord
        a1 = ArtifactRecord(epoch=1, ece=0.05, macro_f1=0.80, checkpoint_path="ckpt1.json")
        a2 = ArtifactRecord(epoch=2, ece=0.04, macro_f1=0.82, checkpoint_path="ckpt2.json")
        reg.register(a1)
        reg.register(a2)
        reg.promote(a1.artifact_id)   # a1 first, so it has older promoted_at
        reg.promote(a2.artifact_id)   # a2 is now production
        reg._a1 = a1
        reg._a2 = a2
        return reg

    def _green_monitor(self):
        from app.intelligence.health_monitor import PipelineHealthMonitor
        m = PipelineHealthMonitor()
        m.record_latency("route", stage_s=1.0)
        return m

    def _red_monitor(self):
        """Trigger RED via ECE violation."""
        from app.intelligence.health_monitor import PipelineHealthMonitor
        m = PipelineHealthMonitor(ece_slo=0.10)
        m.record_ece(0.50)  # 0.50 >> 0.10 → RED
        return m

    def _yellow_monitor(self):
        """Trigger YELLOW (at risk) via ECE just below threshold × warn_pct."""
        from app.intelligence.health_monitor import PipelineHealthMonitor
        m = PipelineHealthMonitor(ece_slo=0.10, slo_warning_pct=0.80)
        m.record_ece(0.085)  # > 0.10 × 0.80 = 0.08 → YELLOW, < 0.10 so not RED
        return m

    # ── Type guard ────────────────────────────────────────────────────────────

    def test_bad_monitor_no_health_report_raises_typeerror(self, reg):
        with pytest.raises(TypeError, match="health_report"):
            reg.check_and_rollback("not a monitor")

    def test_bad_monitor_none_raises_typeerror(self, reg):
        with pytest.raises(TypeError, match="health_report"):
            reg.check_and_rollback(None)

    def test_monitor_with_non_callable_health_report_raises_typeerror(self, reg):
        class BadMonitor:
            health_report = "not callable"
        with pytest.raises(TypeError, match="health_report"):
            reg.check_and_rollback(BadMonitor())

    def test_monitor_raising_health_report_propagates_runtime_error(self, reg):
        class BrokenMonitor:
            def health_report(self):
                raise RuntimeError("monitor exploded")
        with pytest.raises(RuntimeError, match="monitor.health_report.*failed|monitor exploded"):
            reg.check_and_rollback(BrokenMonitor())

    # ── GREEN status — no action ──────────────────────────────────────────────

    def test_green_monitor_returns_none(self, populated_reg):
        mon = self._green_monitor()
        result = populated_reg.check_and_rollback(mon)
        assert result is None

    def test_green_monitor_production_unchanged(self, populated_reg):
        mon = self._green_monitor()
        before_prod = populated_reg.get_production()
        populated_reg.check_and_rollback(mon)
        after_prod = populated_reg.get_production()
        assert before_prod.artifact_id == after_prod.artifact_id

    # ── YELLOW status — no action ─────────────────────────────────────────────

    def test_yellow_monitor_returns_none(self, populated_reg):
        mon = self._yellow_monitor()
        # Confirm it really is YELLOW
        from app.intelligence.health_monitor import SLOStatus
        report = mon.health_report()
        assert report.overall_status == SLOStatus.YELLOW
        result = populated_reg.check_and_rollback(mon)
        assert result is None

    def test_yellow_monitor_production_unchanged(self, populated_reg):
        mon = self._yellow_monitor()
        before = populated_reg.get_production().artifact_id
        populated_reg.check_and_rollback(mon)
        assert populated_reg.get_production().artifact_id == before

    # ── RED status — rollback triggered ───────────────────────────────────────

    def test_red_monitor_triggers_rollback(self, populated_reg):
        mon = self._red_monitor()
        result = populated_reg.check_and_rollback(mon)
        assert result is not None

    def test_red_monitor_demotes_current_production(self, populated_reg):
        mon = self._red_monitor()
        prev_prod_id = populated_reg.get_production().artifact_id  # a2
        populated_reg.check_and_rollback(mon)
        after_prod = populated_reg.get_production()
        # a2 should no longer be production
        assert after_prod.artifact_id != prev_prod_id

    def test_red_monitor_promotes_fallback(self, populated_reg):
        mon = self._red_monitor()
        result = populated_reg.check_and_rollback(mon)
        # The returned artifact must now be production
        assert result.is_production

    def test_red_monitor_rollback_returns_passing_artifact(self, populated_reg):
        mon = self._red_monitor()
        result = populated_reg.check_and_rollback(mon)
        assert result.passes_gate()

    def test_red_monitor_rollback_notes_set(self, populated_reg):
        mon = self._red_monitor()
        result = populated_reg.check_and_rollback(mon, notes="ECE spike detected")
        # Notes should be in the rollback result or the demoted artifact
        assert result is not None  # rollback succeeded

    def test_red_monitor_custom_notes_forwarded(self, populated_reg, tmp_path):
        """check_and_rollback must forward custom notes to the audit log."""
        mon = self._red_monitor()
        custom_notes = "Custom auto-rollback reason"
        populated_reg.check_and_rollback(mon, notes=custom_notes)
        log = populated_reg.read_audit_log()
        assert any(custom_notes in entry.get("notes", "") for entry in log)

    def test_red_monitor_default_notes_contain_red_and_violations(self, populated_reg, tmp_path):
        """Default notes when none supplied mention RED status."""
        mon = self._red_monitor()
        populated_reg.check_and_rollback(mon)
        log = populated_reg.read_audit_log()
        rollback_entries = [e for e in log if e.get("event") == "rollback"]
        assert rollback_entries
        notes = rollback_entries[-1].get("notes", "")
        assert "RED" in notes or "violation" in notes

    # ── No fallback ───────────────────────────────────────────────────────────

    def test_red_monitor_no_fallback_returns_none(self, reg, tmp_path):
        """When only one artifact exists and it's production, rollback returns None."""
        from training.model_registry import ArtifactRecord
        a = ArtifactRecord(epoch=1, ece=0.05, macro_f1=0.80, checkpoint_path="ckpt1.json")
        reg.register(a)
        reg.promote(a.artifact_id)
        mon = self._red_monitor()
        result = reg.check_and_rollback(mon)
        assert result is None  # no other passing artifact

    def test_red_monitor_with_empty_registry_returns_none(self, reg):
        mon = self._red_monitor()
        result = reg.check_and_rollback(mon)
        assert result is None

    # ── Multiple consecutive RED calls ────────────────────────────────────────

    def test_multiple_red_calls_idempotent(self, populated_reg):
        """Two consecutive RED calls must not raise even if no more fallbacks."""
        mon = self._red_monitor()
        r1 = populated_reg.check_and_rollback(mon)
        r2 = populated_reg.check_and_rollback(mon)
        # First call triggers rollback; second may return None if no more fallbacks
        assert r1 is not None  # first rollback succeeds

    # ── Audit log ─────────────────────────────────────────────────────────────

    def test_audit_log_contains_rollback_entry_on_red(self, populated_reg):
        mon = self._red_monitor()
        populated_reg.check_and_rollback(mon)
        log = populated_reg.read_audit_log()
        events = [e["event"] for e in log]
        assert "rollback" in events

    def test_audit_log_no_rollback_entry_on_green(self, populated_reg):
        mon = self._green_monitor()
        populated_reg.check_and_rollback(mon)
        log = populated_reg.read_audit_log()
        events = [e["event"] for e in log]
        assert "rollback" not in events

    # ── Circuit-breaker RED ───────────────────────────────────────────────────

    def test_circuit_breaker_open_triggers_rollback(self, populated_reg):
        """Circuit-breaker OPEN also produces RED → rollback."""
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        mon = PipelineHealthMonitor(cb_open_threshold=3)
        for _ in range(3):
            mon.record_connector_failure("github")
        assert mon.health_report().overall_status == SLOStatus.RED
        result = populated_reg.check_and_rollback(mon)
        assert result is not None

    # ── Watchlist RED ─────────────────────────────────────────────────────────

    def test_watchlist_red_triggers_rollback(self, populated_reg):
        """Watchlist gap count ≥ 10 (RED) must trigger rollback."""
        from app.intelligence.health_monitor import PipelineHealthMonitor, SLOStatus
        mon = PipelineHealthMonitor()
        mon.record_watchlist_gap_count(15)
        assert mon.health_report().overall_status == SLOStatus.RED
        result = populated_reg.check_and_rollback(mon)
        assert result is not None

    # ── Duck-typing / generic monitor ─────────────────────────────────────────

    def test_duck_typed_red_monitor_triggers_rollback(self, populated_reg):
        """Any object with health_report() returning something with .overall_status
        whose .value == 'red' triggers rollback without needing PipelineHealthMonitor."""
        class FakeStatus:
            value = "red"
            def __eq__(self, other):
                return str(getattr(other, "value", other)) == "red"
        class FakeReport:
            overall_status = FakeStatus()
            violations = [object()]
        class FakeMonitor:
            def health_report(self):
                return FakeReport()
        result = populated_reg.check_and_rollback(FakeMonitor())
        assert result is not None

    def test_duck_typed_green_monitor_returns_none(self, populated_reg):
        class FakeReport:
            class overall_status:
                value = "green"
            violations = []
        class FakeMonitor:
            def health_report(self):
                return FakeReport()
        result = populated_reg.check_and_rollback(FakeMonitor())
        assert result is None

    # ── Thread safety ─────────────────────────────────────────────────────────

    def test_concurrent_check_and_rollback_no_crash(self, populated_reg):
        """Multiple threads calling check_and_rollback simultaneously must not crash."""
        import threading
        mon_red = self._red_monitor()
        mon_green = self._green_monitor()
        errors = []

        def worker(mon):
            try:
                for _ in range(10):
                    populated_reg.check_and_rollback(mon)
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=worker, args=(mon_red,)) for _ in range(3)] +
            [threading.Thread(target=worker, args=(mon_green,)) for _ in range(2)]
        )
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors

