"""User interest graph.

Maintains a directed weighted graph of topic nodes and typed edges.
Each node carries an *interest weight* ∈ [0, 1] representing how strongly
the user cares about that topic.  Edge weights quantify the strength of
the relationship between two topics.

Supported operations
--------------------
- Add / remove topics and typed edges
- Gradient-style weight update with clipping
- Temporal exponential decay across all weights
- Top-k interest query
- Neighborhood traversal (topics reachable from a given node)
- Serialization to / from a plain dict

Thread safety: all mutations and reads acquire a ``threading.RLock``.
"""

from __future__ import annotations

import logging
import math
import threading
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Set, Tuple

from app.personalization.models import InterestEdgeType, InterestWeight

logger = logging.getLogger(__name__)

_DEFAULT_INITIAL_WEIGHT = 0.5
_DEFAULT_DECAY_FACTOR = 0.995   # ≈ -0.5 % per day; halves in ~138 days


class InterestGraph:
    """Directed weighted graph of per-user topic interest.

    Nodes are topic_id strings.  Each node stores a mutable ``InterestWeight``.
    Edges are typed (``InterestEdgeType``) with a float weight [0, 1].

    Args:
        initial_weight:  Default interest weight for new topics.
        decay_factor:    Daily exponential decay factor applied by ``decay_all()``.
        min_weight:      Floor applied after any weight update.
        max_topics:      Maximum node count (0 = unlimited).
    """

    def __init__(
        self,
        initial_weight: float = _DEFAULT_INITIAL_WEIGHT,
        decay_factor: float = _DEFAULT_DECAY_FACTOR,
        min_weight: float = 0.0,
        max_topics: int = 0,
    ) -> None:
        if not (0.0 <= initial_weight <= 1.0):
            raise ValueError(f"'initial_weight' must be in [0, 1], got {initial_weight!r}")
        if not (0.0 < decay_factor <= 1.0):
            raise ValueError(f"'decay_factor' must be in (0, 1], got {decay_factor!r}")
        if not (0.0 <= min_weight <= 1.0):
            raise ValueError(f"'min_weight' must be in [0, 1], got {min_weight!r}")
        if max_topics < 0:
            raise ValueError(f"'max_topics' must be >= 0, got {max_topics!r}")

        self._initial = initial_weight
        self._decay = decay_factor
        self._min_weight = min_weight
        self._max_topics = max_topics

        # node → InterestWeight (mutable as plain dict, replaced on update)
        self._nodes: Dict[str, InterestWeight] = {}
        # (source, target) → (edge_type, edge_weight)
        self._edges: Dict[Tuple[str, str], Tuple[str, float]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_topic(self, topic_id: str, initial_weight: Optional[float] = None) -> None:
        """Add a topic node.  No-op if the topic already exists.

        Raises:
            ValueError: If *topic_id* is empty or *initial_weight* out of range.
            OverflowError: If ``max_topics`` limit is reached.
        """
        if not topic_id or not isinstance(topic_id, str):
            raise ValueError("'topic_id' must be a non-empty string")
        w = initial_weight if initial_weight is not None else self._initial
        if not (0.0 <= w <= 1.0):
            raise ValueError(f"'initial_weight' must be in [0, 1], got {w!r}")
        with self._lock:
            if topic_id in self._nodes:
                return
            if self._max_topics and len(self._nodes) >= self._max_topics:
                raise OverflowError(f"InterestGraph is full ({self._max_topics} topics)")
            self._nodes[topic_id] = InterestWeight(topic_id=topic_id, weight=w, update_count=0)
        logger.debug("InterestGraph: added topic %r (w=%.3f)", topic_id, w)

    def remove_topic(self, topic_id: str) -> bool:
        """Remove a topic and all its edges.

        Returns:
            True if the topic existed; False otherwise.
        """
        with self._lock:
            if topic_id not in self._nodes:
                return False
            del self._nodes[topic_id]
            # Remove connected edges
            to_delete = [k for k in self._edges if topic_id in k]
            for k in to_delete:
                del self._edges[k]
        return True

    def update_weight(self, topic_id: str, delta: float, learning_rate: float = 1.0) -> InterestWeight:
        """Apply a gradient-style weight update to *topic_id*.

        The topic is auto-created with ``initial_weight=0.5`` if absent.

        Args:
            topic_id:      Topic to update.
            delta:         Signed weight delta (positive → more interest).
            learning_rate: Scale factor for *delta* [0, 1].

        Returns:
            Updated ``InterestWeight``.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not topic_id:
            raise ValueError("'topic_id' must be non-empty")
        if not (0.0 <= learning_rate <= 1.0):
            raise ValueError(f"'learning_rate' must be in [0, 1], got {learning_rate!r}")
        with self._lock:
            if topic_id not in self._nodes:
                self.add_topic(topic_id)
            old = self._nodes[topic_id]
            new_weight = min(1.0, max(self._min_weight, old.weight + delta * learning_rate))
            count = old.update_count + 1
            confidence = min(1.0, count / 10.0)  # saturates at 10 updates
            updated = InterestWeight(
                topic_id=topic_id,
                weight=round(new_weight, 5),
                confidence=round(confidence, 5),
                decay_factor=old.decay_factor,
                last_updated=datetime.now(timezone.utc),
                update_count=count,
            )
            self._nodes[topic_id] = updated
        return updated

    def get_weight(self, topic_id: str) -> Optional[float]:
        """Return the current interest weight for *topic_id*, or None if absent."""
        with self._lock:
            iw = self._nodes.get(topic_id)
        return iw.weight if iw else None

    def get_interest_weight(self, topic_id: str) -> Optional[InterestWeight]:
        """Return the full ``InterestWeight`` object, or None if absent."""
        with self._lock:
            return self._nodes.get(topic_id)

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: InterestEdgeType = InterestEdgeType.RELATED_TO,
        weight: float = 0.5,
    ) -> None:
        """Add or update a directed edge *source* → *target*.

        Both nodes are auto-created if absent.

        Raises:
            ValueError: If *weight* not in [0, 1] or nodes are the same.
        """
        if source == target:
            raise ValueError("Self-loops are not allowed")
        if not (0.0 <= weight <= 1.0):
            raise ValueError(f"Edge 'weight' must be in [0, 1], got {weight!r}")
        if not isinstance(edge_type, InterestEdgeType):
            raise TypeError(f"'edge_type' must be InterestEdgeType, got {type(edge_type)!r}")
        with self._lock:
            if source not in self._nodes:
                self.add_topic(source)
            if target not in self._nodes:
                self.add_topic(target)
            self._edges[(source, target)] = (edge_type.value, weight)

    def remove_edge(self, source: str, target: str) -> bool:
        """Remove a directed edge.  Returns True if it existed."""
        with self._lock:
            key = (source, target)
            if key in self._edges:
                del self._edges[key]
                return True
        return False

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def top_interests(self, k: int = 10) -> List[InterestWeight]:
        """Return top-k topics sorted by weight descending.

        Raises:
            ValueError: If *k* ≤ 0.
        """
        if k <= 0:
            raise ValueError(f"'k' must be positive, got {k!r}")
        with self._lock:
            items = sorted(self._nodes.values(), key=lambda iw: iw.weight, reverse=True)
        return items[:k]

    def related_topics(
        self,
        topic_id: str,
        top_k: int = 5,
        edge_type: Optional[InterestEdgeType] = None,
        min_edge_weight: float = 0.0,
    ) -> List[InterestWeight]:
        """Return interest-weighted neighbors of *topic_id*.

        Args:
            topic_id:        Source node.
            top_k:           Maximum neighbors to return.
            edge_type:       Filter by edge type (None = all types).
            min_edge_weight: Minimum edge weight to include.

        Returns:
            List of ``InterestWeight`` for reachable neighbors, sorted by
            combined (edge_weight × node_weight) descending.
        """
        if top_k <= 0:
            raise ValueError(f"'top_k' must be positive, got {top_k!r}")
        with self._lock:
            candidates: List[Tuple[float, InterestWeight]] = []
            for (src, tgt), (et, ew) in self._edges.items():
                if src != topic_id:
                    continue
                if edge_type is not None and et != edge_type.value:
                    continue
                if ew < min_edge_weight:
                    continue
                node = self._nodes.get(tgt)
                if node:
                    candidates.append((ew * node.weight, node))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [iw for _, iw in candidates[:top_k]]

    def all_topics(self) -> List[str]:
        """Return list of all topic IDs."""
        with self._lock:
            return list(self._nodes.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._nodes)

    def __contains__(self, topic_id: str) -> bool:
        with self._lock:
            return topic_id in self._nodes

    def __iter__(self) -> Iterator[InterestWeight]:
        with self._lock:
            items = list(self._nodes.values())
        return iter(items)

    # ------------------------------------------------------------------
    # Decay
    # ------------------------------------------------------------------

    def decay_all(self, days_elapsed: float = 1.0) -> None:
        """Apply exponential decay to all topic weights.

        ``new_weight = old_weight × (decay_factor ^ days_elapsed)``

        Raises:
            ValueError: If *days_elapsed* is negative.
        """
        if days_elapsed < 0:
            raise ValueError(f"'days_elapsed' must be >= 0, got {days_elapsed!r}")
        factor = self._decay ** days_elapsed
        with self._lock:
            for tid, iw in list(self._nodes.items()):
                new_w = max(self._min_weight, iw.weight * factor)
                self._nodes[tid] = InterestWeight(
                    topic_id=tid,
                    weight=round(new_w, 6),
                    confidence=iw.confidence,
                    decay_factor=iw.decay_factor,
                    last_updated=iw.last_updated,
                    update_count=iw.update_count,
                )
        logger.debug("InterestGraph: decay_all(days=%.2f, factor=%.4f) applied", days_elapsed, factor)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize graph to a JSON-friendly dict."""
        with self._lock:
            return {
                "nodes": {
                    tid: {
                        "weight": iw.weight,
                        "confidence": iw.confidence,
                        "update_count": iw.update_count,
                    }
                    for tid, iw in self._nodes.items()
                },
                "edges": {
                    f"{src}→{tgt}": {"edge_type": et, "weight": ew}
                    for (src, tgt), (et, ew) in self._edges.items()
                },
            }

    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> "InterestGraph":
        """Reconstruct an ``InterestGraph`` from a ``to_dict()`` snapshot."""
        if not isinstance(data, dict):
            raise TypeError(f"'data' must be a dict, got {type(data)!r}")
        g = cls(**kwargs)
        for tid, info in data.get("nodes", {}).items():
            g.add_topic(tid, initial_weight=info.get("weight", 0.5))
            # Restore update_count via private injection
            old = g._nodes[tid]
            g._nodes[tid] = InterestWeight(
                topic_id=tid,
                weight=info.get("weight", 0.5),
                confidence=info.get("confidence", 0.5),
                update_count=info.get("update_count", 0),
                decay_factor=old.decay_factor,
                last_updated=old.last_updated,
            )
        for edge_key, einfo in data.get("edges", {}).items():
            src, _, tgt = edge_key.partition("→")
            try:
                et = InterestEdgeType(einfo.get("edge_type", "related_to"))
            except ValueError:
                et = InterestEdgeType.RELATED_TO
            g.add_edge(src, tgt, edge_type=et, weight=einfo.get("weight", 0.5))
        return g

