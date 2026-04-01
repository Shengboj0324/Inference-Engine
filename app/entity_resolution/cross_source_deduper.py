"""Cross-source deduplicator.

Given an ``EventBundle`` that already clusters items about the same event,
removes duplicates by:

1. **Trust-first**: keep the item from the highest-trust source platform.
2. **Title similarity**: remove items whose title has Jaccard similarity
   ≥ *title_threshold* with the primary item's title.
3. **Content hash**: exact duplicates (same content hash) are always removed.

Produces a ``DedupeResult`` recording what was kept/removed and why.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Set

from app.entity_resolution.models import DedupeResult, EventBundle

logger = logging.getLogger(__name__)

_PUNCT = re.compile(r"[^\w\s]")
_STOP_WORDS = frozenset(
    "a an the and or but in on at to for of with is are was were".split()
)


def _title_tokens(title: str) -> frozenset[str]:
    tokens = _PUNCT.sub(" ", title.lower()).split()
    return frozenset(t for t in tokens if t not in _STOP_WORDS and len(t) > 2)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union)


def _content_hash(item: Dict[str, Any]) -> str:
    raw = f"{item.get('title', '')}|{item.get('raw_text', '')}|{item.get('source_url', '')}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


class CrossSourceDeduper:
    """Removes duplicate content items within an ``EventBundle``.

    Args:
        title_threshold:    Jaccard title similarity above which items are
                            considered duplicates (default 0.7).
        trust_weight:       Weight of trust score in primary selection (0-1).
        dedup_exact_hashes: Always remove exact content hash duplicates.
    """

    def __init__(
        self,
        title_threshold: float = 0.70,
        trust_weight: float = 0.8,
        dedup_exact_hashes: bool = True,
    ) -> None:
        if not (0.0 < title_threshold <= 1.0):
            raise ValueError(f"'title_threshold' must be in (0, 1], got {title_threshold!r}")
        if not (0.0 <= trust_weight <= 1.0):
            raise ValueError(f"'trust_weight' must be in [0, 1], got {trust_weight!r}")
        self._title_threshold = title_threshold
        self._trust_weight = trust_weight
        self._dedup_exact = dedup_exact_hashes

    def deduplicate(self, bundle: EventBundle) -> DedupeResult:
        """Remove duplicates from *bundle*.

        Args:
            bundle: ``EventBundle`` from ``EventClusterer``.

        Returns:
            ``DedupeResult`` describing what was kept/removed.

        Raises:
            TypeError: If *bundle* is not an ``EventBundle``.
        """
        if not isinstance(bundle, EventBundle):
            raise TypeError(f"Expected EventBundle, got {type(bundle)!r}")
        if bundle.size() <= 1:
            item_id = bundle.primary_item_id or (
                bundle.source_items[0].get("source_id", "item_0") if bundle.source_items else ""
            )
            return DedupeResult(
                bundle_id=bundle.bundle_id,
                kept_item_id=item_id,
                removed_ids=[],
                similarity_scores={},
                strategy="trust",
            )

        # Step 1: Select primary item (highest trust)
        primary_id = self._select_primary(bundle)
        primary_item = self._find_item(bundle.source_items, primary_id)
        primary_tokens = _title_tokens(primary_item.get("title", "")) if primary_item else frozenset()
        primary_hash = _content_hash(primary_item) if primary_item else ""

        removed: List[str] = []
        similarity_scores: Dict[str, float] = {}
        seen_hashes: Set[str] = {primary_hash}

        for item in bundle.source_items:
            item_id = item.get("source_id", "")
            if item_id == primary_id:
                continue
            item_hash = _content_hash(item)
            if self._dedup_exact and item_hash in seen_hashes:
                removed.append(item_id)
                similarity_scores[item_id] = 1.0
                continue
            seen_hashes.add(item_hash)

            # Title similarity check
            item_tokens = _title_tokens(item.get("title", ""))
            sim = _jaccard(primary_tokens, item_tokens)
            similarity_scores[item_id] = round(sim, 3)
            if sim >= self._title_threshold:
                removed.append(item_id)

        strategy = "combined" if self._dedup_exact else "similarity"
        logger.debug(
            "CrossSourceDeduper: bundle=%r kept=%r removed=%d items",
            bundle.bundle_id, primary_id, len(removed),
        )
        return DedupeResult(
            bundle_id=bundle.bundle_id,
            kept_item_id=primary_id,
            removed_ids=removed,
            similarity_scores=similarity_scores,
            strategy=strategy,
        )

    def deduplicate_batch(self, bundles: List[EventBundle]) -> List[DedupeResult]:
        """Deduplicate all bundles in a list.

        Raises:
            TypeError: If *bundles* is not a list.
        """
        if not isinstance(bundles, list):
            raise TypeError(f"Expected list, got {type(bundles)!r}")
        return [self.deduplicate(b) for b in bundles]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_primary(self, bundle: EventBundle) -> str:
        """Select the primary (best) item by trust score."""
        if bundle.primary_item_id and bundle.primary_item_id in bundle.trust_scores:
            return bundle.primary_item_id
        if bundle.trust_scores:
            return max(bundle.trust_scores, key=lambda k: bundle.trust_scores[k])
        return bundle.source_items[0].get("source_id", "item_0") if bundle.source_items else ""

    @staticmethod
    def _find_item(items: List[Dict[str, Any]], source_id: str) -> Optional[Dict[str, Any]]:
        for item in items:
            if item.get("source_id") == source_id:
                return item
        return None

