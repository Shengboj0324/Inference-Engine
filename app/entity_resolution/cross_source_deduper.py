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

    def deduplicate_cross_bundle(
        self,
        bundles: List[EventBundle],
    ) -> tuple[List[EventBundle], List[str]]:
        """Remove cross-bundle duplicates across source families.

        Two bundles are considered duplicates when their representative titles
        have Jaccard similarity ≥ ``title_threshold``.  The bundle whose primary
        item carries the higher trust score is retained; the lower-scoring bundle
        is discarded in full.

        Unlike ``deduplicate()`` (which operates *within* a single bundle),
        this method operates *across* all bundles, enabling de-duplication when
        the same event is captured by separate source families (e.g. a GitHub
        release that also appears on Reddit).

        Args:
            bundles: Ordered list of ``EventBundle`` objects to filter.

        Returns:
            Tuple of:
            - ``kept``: List of bundles not identified as cross-bundle
              duplicates.
            - ``removed_bundle_ids``: List of bundle_ids that were eliminated
              as cross-family duplicates.

        Raises:
            TypeError: If *bundles* is not a list.
        """
        if not isinstance(bundles, list):
            raise TypeError(f"Expected list, got {type(bundles)!r}")
        if len(bundles) <= 1:
            return list(bundles), []

        # Build representative tokens and trust score per bundle
        def _bundle_score(b: EventBundle) -> float:
            if b.trust_scores:
                return max(b.trust_scores.values())
            return 0.0

        def _bundle_title(b: EventBundle) -> str:
            if b.canonical_title:
                return b.canonical_title
            primary = self._find_item(b.source_items, b.primary_item_id or "")
            if primary:
                return primary.get("title", b.bundle_id)
            return b.source_items[0].get("title", b.bundle_id) if b.source_items else b.bundle_id

        bundle_tokens = [(_title_tokens(_bundle_title(b)), _bundle_score(b)) for b in bundles]
        removed_ids: List[str] = []
        kept_mask  = [True] * len(bundles)

        for i in range(len(bundles)):
            if not kept_mask[i]:
                continue
            for j in range(i + 1, len(bundles)):
                if not kept_mask[j]:
                    continue
                sim = _jaccard(bundle_tokens[i][0], bundle_tokens[j][0])
                if sim >= self._title_threshold:
                    # Drop the lower-trust bundle
                    if bundle_tokens[i][1] >= bundle_tokens[j][1]:
                        kept_mask[j] = False
                        removed_ids.append(bundles[j].bundle_id)
                        logger.debug(
                            "CrossSourceDeduper: cross-bundle dup removed=%r "
                            "(sim=%.2f vs retained=%r)",
                            bundles[j].bundle_id, sim, bundles[i].bundle_id,
                        )
                    else:
                        kept_mask[i] = False
                        removed_ids.append(bundles[i].bundle_id)
                        logger.debug(
                            "CrossSourceDeduper: cross-bundle dup removed=%r "
                            "(sim=%.2f vs retained=%r)",
                            bundles[i].bundle_id, sim, bundles[j].bundle_id,
                        )
                        break  # i is gone; stop checking j's against it

        kept = [b for b, ok in zip(bundles, kept_mask) if ok]
        return kept, removed_ids

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

