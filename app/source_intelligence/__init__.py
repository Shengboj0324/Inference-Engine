"""Source Intelligence Layer.

This package implements the source-aware intelligence capabilities described in
*Industrial_Advancement_Roadmap_For_Inference_Engine.md* Phase 1 and Phase 6.

Modules
-------
source_registry         — Capability-aware source registry (SourceSpec, SourceRegistryStore)
source_trust            — Source trust scoring (TrustScore, SourceTrustScorer)
source_discovery        — Auto-discovery of new relevant sources (SourceDiscoveryEngine)
coverage_planner        — Coverage gap analysis (CoveragePlanner)
coverage_graph          — Source coverage tracking across entity types (SourceCoverageGraph)
feed_expander           — Entity-to-feed auto-discovery (FeedExpander)
entity_to_source_mapper — Canonical entity → source mapping (EntityToSourceMapper)
change_monitor          — Source change detection (ChangeMonitor)
"""

from app.source_intelligence.source_registry import (
    SourceCapability,
    SourceFamily,
    SourceRegistryStore,
    SourceSpec,
)
from app.source_intelligence.source_trust import SourceTrustScorer, TrustScore
from app.source_intelligence.source_discovery import DiscoveredSource, SourceDiscoveryEngine
from app.source_intelligence.coverage_planner import CoverageGap, CoveragePlanner, GapSeverity
from app.source_intelligence.coverage_graph import (
    EntityCategory,
    EntityCoverageScore,
    SourceCoverageGraph,
    SourceFreshnessEntry,
)
from app.source_intelligence.feed_expander import FeedCandidate, FeedExpander
from app.source_intelligence.entity_to_source_mapper import EntitySourceMap, EntityToSourceMapper
from app.source_intelligence.change_monitor import ChangeEvent, ChangeMonitor

__all__ = [
    # source_registry
    "SourceCapability",
    "SourceFamily",
    "SourceRegistryStore",
    "SourceSpec",
    # source_trust
    "SourceTrustScorer",
    "TrustScore",
    # source_discovery
    "DiscoveredSource",
    "SourceDiscoveryEngine",
    # coverage_planner
    "CoverageGap",
    "CoveragePlanner",
    "GapSeverity",
    # coverage_graph (Phase 6)
    "EntityCategory",
    "EntityCoverageScore",
    "SourceCoverageGraph",
    "SourceFreshnessEntry",
    # feed_expander
    "FeedCandidate",
    "FeedExpander",
    # entity_to_source_mapper
    "EntitySourceMap",
    "EntityToSourceMapper",
    # change_monitor
    "ChangeEvent",
    "ChangeMonitor",
]

