"""Temporal event graph.

Maintains a directed graph of ``EventBundle`` objects with typed edges:
  ``"causes"``    — event A caused event B
  ``"follows"``   — event A temporally precedes event B (no causal claim)
  ``"related"``   — related topic without strict ordering
  ``"contradicts"`` — events present conflicting claims

Supports:
- Timeline queries (events in time window, ordered)
- Entity-based filtering (events involving a given entity)
- Graph serialisation to adjacency dict
- BFS path queries

All mutations are thread-safe via ``threading.RLock``.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Dict, FrozenSet, Iterator, List, Optional, Set, Tuple

from app.entity_resolution.models import EventBundle

logger = logging.getLogger(__name__)

EDGE_TYPES = frozenset({"causes", "follows", "related", "contradicts"})


class TemporalEventGraph:
    """Directed graph of event bundles with temporal and semantic edges.

    Args:
        auto_follow_edges: Automatically add ``"follows"`` edges between
                           consecutive events involving shared entities.
    """

    def __init__(self, auto_follow_edges: bool = True) -> None:
        self._bundles: Dict[str, EventBundle] = {}
        self._edges: Dict[str, List[Tuple[str, str]]] = {}  # src → [(target, edge_type)]
        self._auto_follow = auto_follow_edges
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_bundle(self, bundle: EventBundle) -> None:
        """Add an event bundle to the graph.

        If ``auto_follow_edges`` is enabled, ``"follows"`` edges are
        automatically created to bundles sharing entity IDs that were
        added earlier (and have an earlier ``event_time``).

        Raises:
            TypeError: If *bundle* is not an ``EventBundle``.
        """
        if not isinstance(bundle, EventBundle):
            raise TypeError(f"Expected EventBundle, got {type(bundle)!r}")
        with self._lock:
            self._bundles[bundle.bundle_id] = bundle
            if bundle.bundle_id not in self._edges:
                self._edges[bundle.bundle_id] = []

            if self._auto_follow and bundle.event_time and bundle.entity_ids:
                self._create_follow_edges(bundle)

        logger.debug("TemporalEventGraph: added bundle %r", bundle.bundle_id)

    def add_edge(self, source_id: str, target_id: str, edge_type: str = "related") -> None:
        """Add a directed edge between two bundles.

        Args:
            source_id:  Source bundle ID.
            target_id:  Target bundle ID.
            edge_type:  One of ``"causes"``, ``"follows"``, ``"related"``,
                        ``"contradicts"``.

        Raises:
            KeyError:   If either bundle ID is not in the graph.
            ValueError: If *edge_type* is invalid.
        """
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"'edge_type' must be one of {EDGE_TYPES}, got {edge_type!r}")
        with self._lock:
            if source_id not in self._bundles:
                raise KeyError(f"Source bundle not in graph: {source_id!r}")
            if target_id not in self._bundles:
                raise KeyError(f"Target bundle not in graph: {target_id!r}")
            # Avoid duplicate edges
            existing = self._edges.setdefault(source_id, [])
            if not any(t == target_id and et == edge_type for t, et in existing):
                existing.append((target_id, edge_type))

    def remove_bundle(self, bundle_id: str) -> bool:
        """Remove a bundle and all its edges.

        Returns:
            True if found and removed; False otherwise.
        """
        with self._lock:
            if bundle_id not in self._bundles:
                return False
            del self._bundles[bundle_id]
            self._edges.pop(bundle_id, None)
            for src in self._edges:
                self._edges[src] = [(t, et) for t, et in self._edges[src] if t != bundle_id]
            return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_bundle(self, bundle_id: str) -> Optional[EventBundle]:
        with self._lock:
            return self._bundles.get(bundle_id)

    def timeline(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        entity_id: Optional[str] = None,
    ) -> List[EventBundle]:
        """Return bundles in chronological order, optionally filtered.

        Args:
            start:     Inclusive start datetime.
            end:       Inclusive end datetime.
            entity_id: Filter to bundles involving this entity.

        Returns:
            List of ``EventBundle`` sorted by event_time ascending.
        """
        with self._lock:
            bundles = list(self._bundles.values())

        filtered: List[EventBundle] = []
        for b in bundles:
            if entity_id and entity_id not in b.entity_ids:
                continue
            if b.event_time is None:
                filtered.append(b)
                continue
            bt = b.event_time
            if start and bt < start:
                continue
            if end and bt > end:
                continue
            filtered.append(b)

        filtered.sort(key=lambda b: b.event_time or datetime.min.replace(tzinfo=timezone.utc))
        return filtered

    def neighbours(self, bundle_id: str, edge_type: Optional[str] = None) -> List[EventBundle]:
        """Return adjacent bundles (outgoing edges from *bundle_id*).

        Args:
            bundle_id: Source bundle ID.
            edge_type: Filter by edge type (None = all).
        """
        with self._lock:
            edges = self._edges.get(bundle_id, [])
            bundles = self._bundles
        result: List[EventBundle] = []
        for target_id, et in edges:
            if edge_type is None or et == edge_type:
                b = bundles.get(target_id)
                if b:
                    result.append(b)
        return result

    def shortest_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """BFS shortest path from *source_id* to *target_id*.

        Returns:
            List of bundle IDs on the path, or None if unreachable.
        """
        with self._lock:
            if source_id not in self._bundles or target_id not in self._bundles:
                return None

        if source_id == target_id:
            return [source_id]

        visited: Set[str] = {source_id}
        queue: deque[List[str]] = deque([[source_id]])
        while queue:
            path = queue.popleft()
            current = path[-1]
            with self._lock:
                edges = list(self._edges.get(current, []))
            for target, _ in edges:
                if target == target_id:
                    return path + [target]
                if target not in visited:
                    visited.add(target)
                    queue.append(path + [target])
        return None

    def all_bundles(self) -> List[EventBundle]:
        with self._lock:
            return list(self._bundles.values())

    def edge_count(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._edges.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._bundles)

    def __iter__(self) -> Iterator[EventBundle]:
        with self._lock:
            bundles = list(self._bundles.values())
        return iter(bundles)

    def __contains__(self, bundle_id: str) -> bool:
        with self._lock:
            return bundle_id in self._bundles

    def to_adjacency_dict(self) -> Dict[str, List[Dict[str, str]]]:
        """Serialise to JSON-friendly format."""
        with self._lock:
            return {
                src: [{"target": t, "edge_type": et} for t, et in edges]
                for src, edges in self._edges.items()
            }

    # ------------------------------------------------------------------
    # Auto-follow edge creation
    # ------------------------------------------------------------------

    def _create_follow_edges(self, new_bundle: EventBundle) -> None:
        """Create ``"follows"`` edges from earlier bundles to *new_bundle*."""
        new_entities = frozenset(new_bundle.entity_ids)
        for bid, bundle in self._bundles.items():
            if bid == new_bundle.bundle_id:
                continue
            if bundle.event_time is None or new_bundle.event_time is None:
                continue
            if bundle.event_time >= new_bundle.event_time:
                continue
            shared = frozenset(bundle.entity_ids) & new_entities
            if not shared:
                continue
            existing = self._edges.setdefault(bid, [])
            if not any(t == new_bundle.bundle_id for t, _ in existing):
                existing.append((new_bundle.bundle_id, "follows"))

