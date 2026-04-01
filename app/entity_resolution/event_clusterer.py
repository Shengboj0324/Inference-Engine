"""Event clusterer.

Groups ``ContentItem``-like dicts about the same real-world event into
``EventBundle`` objects using a multi-dimensional proximity score:

Score = w_entity × entity_overlap
      + w_title  × title_similarity
      + w_time   × time_proximity

All weights are configurable.  Items are clustered greedily: each new
item is assigned to the first bundle whose centroid score exceeds the
similarity threshold, or a new bundle is created.

Input items are plain dicts with keys:
    ``source_id``, ``title``, ``published_at`` (ISO str), ``entities`` (list[str]),
    ``platform`` (str), ``trust_score`` (float 0-1)
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.entity_resolution.models import EventBundle

logger = logging.getLogger(__name__)

_STOP_WORDS = frozenset(
    "a an the and or but in on at to for of with is are was were be been "
    "being have has had do does did will would could should may might must "
    "this that these those it its by from as into".split()
)
_PUNCT = re.compile(r"[^\w\s]")


def _tokenize(text: str) -> set[str]:
    tokens = _PUNCT.sub(" ", text.lower()).split()
    return {t for t in tokens if t not in _STOP_WORDS and len(t) > 2}


def _title_similarity(a: str, b: str) -> float:
    """Jaccard similarity on lowercased word tokens."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    intersection = len(ta & tb)
    union = len(ta | tb)
    return intersection / union if union else 0.0


def _entity_overlap(ea: List[str], eb: List[str]) -> float:
    """Jaccard similarity on entity sets."""
    sa = {e.lower() for e in ea}
    sb = {e.lower() for e in eb}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _time_proximity(dt_a: Optional[datetime], dt_b: Optional[datetime], half_life_hours: float = 48.0) -> float:
    """Exponential decay based on time difference in hours."""
    if dt_a is None or dt_b is None:
        return 0.5  # neutral when unknown
    diff_h = abs((dt_a - dt_b).total_seconds()) / 3600
    import math
    return math.exp(-diff_h / half_life_hours)


class EventClusterer:
    """Groups content items into event bundles.

    Args:
        similarity_threshold: Minimum score to merge into an existing bundle.
        weight_entity:        Weight for entity overlap component.
        weight_title:         Weight for title similarity component.
        weight_time:          Weight for time proximity component.
        time_half_life_hours: Half-life for time decay (hours).
        max_bundle_size:      Maximum items per bundle (0 = unlimited).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.30,
        weight_entity: float = 0.40,
        weight_title: float = 0.40,
        weight_time: float = 0.20,
        time_half_life_hours: float = 48.0,
        max_bundle_size: int = 0,
    ) -> None:
        if not (0.0 < similarity_threshold <= 1.0):
            raise ValueError(f"'similarity_threshold' must be in (0, 1], got {similarity_threshold!r}")
        weights = weight_entity + weight_title + weight_time
        if abs(weights - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {weights:.4f}")
        if time_half_life_hours <= 0:
            raise ValueError(f"'time_half_life_hours' must be positive, got {time_half_life_hours!r}")

        self._threshold = similarity_threshold
        self._w_entity = weight_entity
        self._w_title = weight_title
        self._w_time = weight_time
        self._half_life = time_half_life_hours
        self._max_size = max_bundle_size

    def cluster(self, items: List[Dict[str, Any]]) -> List[EventBundle]:
        """Cluster *items* into event bundles.

        Args:
            items: List of content item dicts.

        Returns:
            List of ``EventBundle`` sorted by event_time (newest first).

        Raises:
            TypeError: If *items* is not a list.
        """
        if not isinstance(items, list):
            raise TypeError(f"'items' must be a list, got {type(items)!r}")
        if not items:
            return []

        bundles: List[_MutableBundle] = []

        for item in items:
            best_bundle: Optional[_MutableBundle] = None
            best_score = self._threshold

            item_dt = self._parse_dt(item.get("published_at", ""))
            item_entities = list(item.get("entities", []))
            item_title = str(item.get("title", ""))

            for bundle in bundles:
                if self._max_size and bundle.size() >= self._max_size:
                    continue
                score = self._score(
                    item_title, item_entities, item_dt,
                    bundle.centroid_title, bundle.centroid_entities, bundle.centroid_dt,
                )
                if score > best_score:
                    best_score = score
                    best_bundle = bundle

            if best_bundle is not None:
                best_bundle.add(item, item_dt, item_entities, item_title)
            else:
                b = _MutableBundle(item_dt, item_entities, item_title)
                b.add(item, item_dt, item_entities, item_title)
                bundles.append(b)

        result = [b.to_bundle() for b in bundles]
        result.sort(key=lambda b: b.event_time or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        logger.debug("EventClusterer: %d items → %d bundles", len(items), len(result))
        return result

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(
        self,
        title_a: str, entities_a: List[str], dt_a: Optional[datetime],
        title_b: str, entities_b: List[str], dt_b: Optional[datetime],
    ) -> float:
        ts = _title_similarity(title_a, title_b)
        es = _entity_overlap(entities_a, entities_b)
        tps = _time_proximity(dt_a, dt_b, self._half_life)
        return self._w_title * ts + self._w_entity * es + self._w_time * tps

    @staticmethod
    def _parse_dt(s: str) -> Optional[datetime]:
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None


class _MutableBundle:
    """Internal mutable accumulator for clustering."""

    def __init__(self, dt: Optional[datetime], entities: List[str], title: str) -> None:
        self._items: List[Dict[str, Any]] = []
        self._dts: List[datetime] = []
        self._entity_counts: Dict[str, int] = {}
        self._titles: List[str] = []
        self.centroid_dt = dt
        self.centroid_entities: List[str] = list(entities)
        self.centroid_title = title

    def add(self, item: Dict[str, Any], dt: Optional[datetime], entities: List[str], title: str) -> None:
        self._items.append(item)
        if dt:
            self._dts.append(dt)
        self._titles.append(title)
        for e in entities:
            self._entity_counts[e.lower()] = self._entity_counts.get(e.lower(), 0) + 1
        # Update centroid
        self.centroid_dt = max(self._dts) if self._dts else None
        self.centroid_entities = [e for e, cnt in self._entity_counts.items() if cnt >= max(1, len(self._items) // 3)]
        self.centroid_title = self._titles[0]

    def size(self) -> int:
        return len(self._items)

    def to_bundle(self) -> EventBundle:
        bundle_id = hashlib.sha256(
            (self.centroid_title + str(self.centroid_dt)).encode()
        ).hexdigest()[:16]
        trust_scores = {
            item.get("source_id", f"item_{i}"): float(item.get("trust_score", 0.5))
            for i, item in enumerate(self._items)
        }
        primary = max(trust_scores, key=lambda k: trust_scores[k]) if trust_scores else ""
        return EventBundle(
            bundle_id=bundle_id,
            canonical_title=self.centroid_title,
            event_time=self.centroid_dt,
            entity_ids=self.centroid_entities,
            source_items=list(self._items),
            trust_scores=trust_scores,
            primary_item_id=primary,
        )

