"""Entity-to-source mapper.

Maintains a thread-safe bidirectional mapping between canonical entity names
(e.g. ``"OpenAI"``, ``"GPT-4"``) and the source IDs in ``SourceRegistryStore``
that are authoritative for that entity.

Use this mapper to:
- Resolve which sources should be queried when a new signal for an entity
  is detected downstream (in the Event Resolution layer, Phase 2).
- Drive the ``CoveragePlanner`` entity-gap checks.
- Feed the ``FeedExpander`` with entity-specific expansion targets.

The mapper is intentionally thin: it stores strings only and delegates
authority resolution to ``SourceRegistryStore``.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class EntitySourceMap:
    """An immutable snapshot of one entity → source mapping.

    Attributes:
        entity_name:   Canonical entity name (normalised to title-case).
        source_ids:    Ordered list of source IDs (highest trust first).
        primary_source_id: The single most authoritative source, or None.
    """

    def __init__(self, entity_name: str, source_ids: List[str], primary_source_id: Optional[str] = None) -> None:
        if not entity_name or not isinstance(entity_name, str):
            raise ValueError("'entity_name' must be a non-empty string")
        if not isinstance(source_ids, list):
            raise TypeError(f"'source_ids' must be a list, got {type(source_ids)!r}")
        self.entity_name = entity_name.strip()
        self.source_ids: List[str] = list(source_ids)
        self.primary_source_id: Optional[str] = primary_source_id

    def __repr__(self) -> str:
        return f"EntitySourceMap(entity={self.entity_name!r}, sources={self.source_ids!r})"


class EntityToSourceMapper:
    """Thread-safe registry mapping entity names to source IDs.

    All entity names are normalised (stripped, lowercased) for storage but
    returned in their original form via ``get_map()``.

    Methods
    -------
    add_mapping(entity, source_id, primary=False)
        Associate *source_id* with *entity*.  If *primary=True* marks this
        as the most authoritative source for that entity.
    remove_mapping(entity, source_id)
        Remove one source from an entity's set.
    get_sources(entity)
        Return source IDs for *entity* (empty list if unknown).
    get_entities_for_source(source_id)
        Reverse lookup: entities that reference *source_id*.
    get_map(entity)
        Return an ``EntitySourceMap`` snapshot.
    all_entities()
        Return all known entity names.
    """

    def __init__(self) -> None:
        # Normalised entity name → ordered list of source_ids
        self._fwd: Dict[str, List[str]] = {}
        # source_id → set of normalised entity names (reverse index)
        self._rev: Dict[str, Set[str]] = {}
        # Normalised entity → primary source_id
        self._primary: Dict[str, Optional[str]] = {}
        # Original casing store
        self._display: Dict[str, str] = {}
        self._lock = threading.Lock()

    def add_mapping(self, entity: str, source_id: str, *, primary: bool = False) -> None:
        """Associate *source_id* with *entity*.

        Args:
            entity:    Canonical entity name (any casing accepted).
            source_id: Source identifier string.
            primary:   If True, marks this as the most authoritative source.

        Raises:
            ValueError: If either argument is an empty string.
        """
        if not entity or not isinstance(entity, str):
            raise ValueError("'entity' must be a non-empty string")
        if not source_id or not isinstance(source_id, str):
            raise ValueError("'source_id' must be a non-empty string")

        t0 = time.perf_counter()
        norm = entity.strip().lower()
        with self._lock:
            self._display[norm] = entity.strip()
            sources = self._fwd.setdefault(norm, [])
            if source_id not in sources:
                sources.append(source_id)
            if primary:
                self._primary[norm] = source_id
            # Update reverse index
            self._rev.setdefault(source_id, set()).add(norm)
        logger.debug(
            "EntityToSourceMapper.add_mapping: entity=%r source=%r primary=%s latency_ms=%.2f",
            entity, source_id, primary, (time.perf_counter() - t0) * 1000,
        )

    def remove_mapping(self, entity: str, source_id: str) -> bool:
        """Remove *source_id* from *entity*'s source list.

        Returns:
            True if the mapping existed and was removed.
        """
        if not entity or not isinstance(entity, str):
            raise ValueError("'entity' must be a non-empty string")
        if not source_id or not isinstance(source_id, str):
            raise ValueError("'source_id' must be a non-empty string")

        norm = entity.strip().lower()
        with self._lock:
            sources = self._fwd.get(norm, [])
            if source_id not in sources:
                return False
            sources.remove(source_id)
            if self._primary.get(norm) == source_id:
                self._primary[norm] = sources[0] if sources else None
            rev_set = self._rev.get(source_id, set())
            rev_set.discard(norm)
        logger.debug("EntityToSourceMapper.remove_mapping: entity=%r source=%r", entity, source_id)
        return True

    def get_sources(self, entity: str) -> List[str]:
        """Return source IDs associated with *entity* (empty list if none)."""
        if not entity or not isinstance(entity, str):
            raise ValueError("'entity' must be a non-empty string")
        norm = entity.strip().lower()
        with self._lock:
            return list(self._fwd.get(norm, []))

    def get_entities_for_source(self, source_id: str) -> List[str]:
        """Reverse lookup: return entity names that reference *source_id*."""
        if not source_id or not isinstance(source_id, str):
            raise ValueError("'source_id' must be a non-empty string")
        with self._lock:
            norms = self._rev.get(source_id, set())
            return [self._display.get(n, n) for n in norms]

    def get_map(self, entity: str) -> Optional[EntitySourceMap]:
        """Return an ``EntitySourceMap`` snapshot for *entity*, or None."""
        if not entity or not isinstance(entity, str):
            raise ValueError("'entity' must be a non-empty string")
        norm = entity.strip().lower()
        with self._lock:
            sources = list(self._fwd.get(norm, []))
            if not sources:
                return None
            primary = self._primary.get(norm)
            display = self._display.get(norm, entity.strip())
        return EntitySourceMap(display, sources, primary)

    def all_entities(self) -> List[str]:
        """Return all registered entity display names, sorted."""
        with self._lock:
            return sorted(self._display.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._fwd)

