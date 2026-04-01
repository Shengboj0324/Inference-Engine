"""Canonical entity store.

Thread-safe in-memory store for ``CanonicalEntity`` objects with:
- CRUD operations keyed by ``entity_id``
- Alias → entity_id lookup index
- Fuzzy name matching via edit-distance (Levenshtein)
- Entity type filtering

All mutations acquire a ``threading.RLock`` for thread safety.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Iterable, Iterator, List, Optional

from app.entity_resolution.models import CanonicalEntity, EntityType

logger = logging.getLogger(__name__)


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between *a* and *b*."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Single-row DP
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        new_row = [i]
        for j, cb in enumerate(b, 1):
            new_row.append(min(row[j] + 1, new_row[j - 1] + 1, row[j - 1] + (ca != cb)))
        row = new_row
    return row[-1]


class CanonicalEntityStore:
    """Thread-safe in-memory store for canonical entity records.

    Args:
        max_entities: Maximum number of entities to store (0 = unlimited).
    """

    def __init__(self, max_entities: int = 0) -> None:
        if max_entities < 0:
            raise ValueError(f"'max_entities' must be >= 0, got {max_entities!r}")
        self._max_entities = max_entities
        self._entities: Dict[str, CanonicalEntity] = {}
        self._alias_index: Dict[str, str] = {}  # lowercase_alias → entity_id
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, entity: CanonicalEntity) -> None:
        """Add or replace an entity in the store.

        If the entity already exists, it is replaced and the alias index
        is updated to reflect any new aliases.

        Raises:
            TypeError: If *entity* is not a ``CanonicalEntity``.
            OverflowError: If ``max_entities`` limit is reached.
        """
        if not isinstance(entity, CanonicalEntity):
            raise TypeError(f"Expected CanonicalEntity, got {type(entity)!r}")
        with self._lock:
            if self._max_entities and entity.entity_id not in self._entities:
                if len(self._entities) >= self._max_entities:
                    raise OverflowError(f"CanonicalEntityStore is full ({self._max_entities} entities)")
            self._entities[entity.entity_id] = entity
            for name in entity.all_names():
                self._alias_index[name] = entity.entity_id
        logger.debug("CanonicalEntityStore: added entity %r", entity.entity_id)

    def get(self, entity_id: str) -> Optional[CanonicalEntity]:
        """Return entity by ID, or None."""
        with self._lock:
            return self._entities.get(entity_id)

    def remove(self, entity_id: str) -> bool:
        """Remove an entity and its alias entries.

        Returns:
            True if found and removed; False if not present.
        """
        with self._lock:
            entity = self._entities.pop(entity_id, None)
            if entity is None:
                return False
            for name in entity.all_names():
                self._alias_index.pop(name, None)
            return True

    def update_property(self, entity_id: str, key: str, value: object) -> bool:
        """Update a single property on an existing entity.

        Returns:
            True if the entity exists and was updated; False otherwise.
        """
        with self._lock:
            entity = self._entities.get(entity_id)
            if entity is None:
                return False
            new_props = {**entity.properties, key: value}
            updated = entity.model_copy(
                update={"properties": new_props, "updated_at": datetime.now(timezone.utc)}
            )
            self._entities[entity_id] = updated
            return True

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def resolve_alias(self, name: str) -> Optional[str]:
        """Return ``entity_id`` for an alias (exact match, case-insensitive).

        Args:
            name: Raw entity mention string.

        Returns:
            ``entity_id`` if found; None otherwise.
        """
        with self._lock:
            return self._alias_index.get(name.strip().lower())

    def fuzzy_match(self, name: str, max_distance: int = 2, entity_type: Optional[EntityType] = None) -> Optional[CanonicalEntity]:
        """Find the best fuzzy match for *name* within *max_distance*.

        Args:
            name:         Query name string.
            max_distance: Maximum allowed Levenshtein distance.
            entity_type:  Optional filter by entity type.

        Returns:
            Best-matching ``CanonicalEntity`` or None.
        """
        if not name:
            return None
        name_lower = name.strip().lower()
        best: Optional[CanonicalEntity] = None
        best_dist = max_distance + 1

        with self._lock:
            entities = list(self._entities.values())

        for entity in entities:
            if entity_type and entity.entity_type != entity_type:
                continue
            for alias in entity.all_names():
                dist = _levenshtein(name_lower, alias)
                if dist < best_dist:
                    best_dist = dist
                    best = entity

        return best if best_dist <= max_distance else None

    def list_by_type(self, entity_type: EntityType) -> List[CanonicalEntity]:
        """Return all entities of *entity_type*."""
        with self._lock:
            return [e for e in self._entities.values() if e.entity_type == entity_type]

    def all(self) -> List[CanonicalEntity]:
        """Return all stored entities (snapshot)."""
        with self._lock:
            return list(self._entities.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._entities)

    def __iter__(self) -> Iterator[CanonicalEntity]:
        with self._lock:
            entities = list(self._entities.values())
        return iter(entities)

    def __contains__(self, entity_id: str) -> bool:
        with self._lock:
            return entity_id in self._entities

