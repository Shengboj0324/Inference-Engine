"""Entity Resolution — Phases 2 + 6.

Resolves raw entity mentions to canonical IDs, clusters content items
about the same real-world event, maintains a temporal event graph, and
(Phase 6) orchestrates an event-first ingestion pipeline.

Public exports
--------------
Models: EntityType, CanonicalEntity, EventBundle, DedupeResult
Components: CanonicalEntityStore, AliasResolver, EventClusterer,
            CrossSourceDeduper, TemporalEventGraph
Phase 6: RawItem, ProcessedEvent, PipelineStats, EventFirstPipeline
"""

from app.entity_resolution.models import (
    CanonicalEntity,
    DedupeResult,
    EntityType,
    EventBundle,
)
from app.entity_resolution.canonical_entity_store import CanonicalEntityStore
from app.entity_resolution.alias_resolver import AliasResolver
from app.entity_resolution.event_clusterer import EventClusterer
from app.entity_resolution.cross_source_deduper import CrossSourceDeduper
from app.entity_resolution.temporal_event_graph import TemporalEventGraph
from app.entity_resolution.event_pipeline import (
    EventFirstPipeline,
    PipelineStats,
    ProcessedEvent,
    RawItem,
)

__all__ = [
    # Models
    "CanonicalEntity",
    "DedupeResult",
    "EntityType",
    "EventBundle",
    # Components
    "AliasResolver",
    "CanonicalEntityStore",
    "CrossSourceDeduper",
    "EventClusterer",
    "TemporalEventGraph",
    # Phase 6 — Event-First Pipeline
    "EventFirstPipeline",
    "PipelineStats",
    "ProcessedEvent",
    "RawItem",
]

