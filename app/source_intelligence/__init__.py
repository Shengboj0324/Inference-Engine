"""Source Intelligence Layer.

This package implements the source-aware intelligence capabilities described in
*Industrial_Advancement_Roadmap_For_Inference_Engine.md* Phase 1.

Modules
-------
source_registry   — Capability-aware source registry (SourceSpec, SourceRegistryStore)
source_trust      — Source trust scoring (TrustScore, SourceTrustScorer)
source_discovery  — Auto-discovery of new relevant sources (SourceDiscoveryEngine)
coverage_planner  — Coverage gap analysis (CoveragePlanner)
feed_expander     — Entity-to-feed auto-discovery (FeedExpander)
entity_to_source_mapper — Canonical entity → source mapping (EntityToSourceMapper)
change_monitor    — Source change detection (ChangeMonitor)
"""

from app.source_intelligence.source_registry import (
    SourceCapability,
    SourceFamily,
    SourceRegistryStore,
    SourceSpec,
)
from app.source_intelligence.source_trust import SourceTrustScorer, TrustScore

__all__ = [
    "SourceCapability",
    "SourceFamily",
    "SourceRegistryStore",
    "SourceSpec",
    "SourceTrustScorer",
    "TrustScore",
]

