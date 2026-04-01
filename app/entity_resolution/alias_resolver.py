"""Alias resolver.

Resolves raw entity mention strings (``"GPT4"``, ``"gpt-4"``, ``"GPT 4"``)
to their canonical ``entity_id`` using:

1. Exact lookup (lowercased) in the ``CanonicalEntityStore``
2. Normalization pipeline: lowercase → strip punctuation → collapse spaces
3. Prefix/suffix token rules (``"OpenAI's GPT-4"`` → ``"openai/gpt-4"``)
4. Fuzzy Levenshtein fallback (configurable max distance)

The resolver is stateless after construction; it only reads from the store.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from app.entity_resolution.canonical_entity_store import CanonicalEntityStore
from app.entity_resolution.models import CanonicalEntity, EntityType

logger = logging.getLogger(__name__)

# Remove possessives, trailing punctuation, common filler words
_POSSESSIVE = re.compile(r"'s\b", re.IGNORECASE)
_PUNCT = re.compile(r"[.,;:!?\"'(){}\[\]<>]")
_FILLER_PREFIX = re.compile(
    r"^\s*(?:the\s+|a\s+|an\s+|new\s+|latest\s+|their\s+|its\s+|model\s+)?",
    re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")
_HYPHEN_VARIANTS = re.compile(r"[\s\-_]+")


class AliasResolver:
    """Resolves raw entity name strings to canonical entity IDs.

    Args:
        store:           ``CanonicalEntityStore`` to look up against.
        fuzzy_max_dist:  Maximum edit distance for fuzzy matching.
        use_fuzzy:       Enable fuzzy fallback (default True).
    """

    def __init__(
        self,
        store: CanonicalEntityStore,
        fuzzy_max_dist: int = 2,
        use_fuzzy: bool = True,
    ) -> None:
        if not isinstance(store, CanonicalEntityStore):
            raise TypeError(f"'store' must be CanonicalEntityStore, got {type(store)!r}")
        if fuzzy_max_dist < 0:
            raise ValueError(f"'fuzzy_max_dist' must be >= 0, got {fuzzy_max_dist!r}")
        self._store = store
        self._max_dist = fuzzy_max_dist
        self._use_fuzzy = use_fuzzy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        raw_name: str,
        entity_type: Optional[EntityType] = None,
    ) -> Optional[str]:
        """Resolve a raw mention to a canonical entity_id.

        Args:
            raw_name:    Raw entity mention text.
            entity_type: Optional type hint for fuzzy matching.

        Returns:
            Canonical ``entity_id`` string or None if unresolved.

        Raises:
            TypeError: If *raw_name* is not a string.
        """
        if not isinstance(raw_name, str):
            raise TypeError(f"'raw_name' must be str, got {type(raw_name)!r}")
        if not raw_name.strip():
            return None

        # Pass 1: exact lookup on raw (lowercased)
        entity_id = self._store.resolve_alias(raw_name)
        if entity_id:
            return entity_id

        # Pass 2: normalized forms
        for normalized in self._normalize_variants(raw_name):
            entity_id = self._store.resolve_alias(normalized)
            if entity_id:
                return entity_id

        # Pass 3: fuzzy fallback
        if self._use_fuzzy:
            best = self._store.fuzzy_match(raw_name, max_distance=self._max_dist, entity_type=entity_type)
            if best:
                logger.debug("AliasResolver: fuzzy match %r → %r", raw_name, best.entity_id)
                return best.entity_id

        logger.debug("AliasResolver: unresolved %r", raw_name)
        return None

    def resolve_entity(self, raw_name: str, entity_type: Optional[EntityType] = None) -> Optional[CanonicalEntity]:
        """Resolve to a full ``CanonicalEntity`` object (or None)."""
        entity_id = self.resolve(raw_name, entity_type)
        return self._store.get(entity_id) if entity_id else None

    def resolve_batch(self, names: List[str], entity_type: Optional[EntityType] = None) -> List[Optional[str]]:
        """Resolve a batch of raw names.

        Args:
            names:       List of raw mention strings.
            entity_type: Optional type hint applied to all.

        Returns:
            List of ``entity_id`` strings or ``None`` (parallel to *names*).
        """
        if not isinstance(names, list):
            raise TypeError(f"'names' must be a list, got {type(names)!r}")
        return [self.resolve(n, entity_type) for n in names]

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_variants(raw: str) -> List[str]:
        """Return a list of normalized forms to try (most specific first)."""
        variants: List[str] = []

        # Strip possessive and filler prefix
        cleaned = _POSSESSIVE.sub("", raw)
        cleaned = _FILLER_PREFIX.sub("", cleaned).strip()
        cleaned_lower = cleaned.lower()
        variants.append(cleaned_lower)

        # Strip punctuation
        no_punct = _PUNCT.sub("", cleaned_lower)
        no_punct = _WHITESPACE.sub(" ", no_punct).strip()
        if no_punct != cleaned_lower:
            variants.append(no_punct)

        # Hyphen/underscore/space equivalence
        normalized = _HYPHEN_VARIANTS.sub("-", no_punct)
        if normalized != no_punct:
            variants.append(normalized)

        space_form = _HYPHEN_VARIANTS.sub(" ", no_punct)
        if space_form not in variants:
            variants.append(space_form)

        no_space_form = _HYPHEN_VARIANTS.sub("", no_punct)
        if no_space_form not in variants:
            variants.append(no_space_form)

        return variants

