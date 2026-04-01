"""Entity Resolution — Phase 2 cross-source entity and event deduplication.

Resolves raw entity mentions to canonical IDs, clusters content items
about the same real-world event, and maintains a temporal event graph.

Public exports
--------------
Models: EntityType, CanonicalEntity, EventBundle, DedupeResult
Components: CanonicalEntityStore, AliasResolver, EventClusterer,
            CrossSourceDeduper, TemporalEventGraph
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
]

