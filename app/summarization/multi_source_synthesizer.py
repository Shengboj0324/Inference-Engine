"""Multi-source synthesizer.

Provides two capabilities on top of ``GroundedSummaryBuilder``:

1. **Source deduplication** — removes near-duplicate ``EvidenceSource``
   objects before synthesis so the downstream pipeline is not skewed by
   repeated identical content from different platforms.

2. **Claim merging** — consolidates ``AttributedClaim`` objects that assert
   the same fact (high token-overlap) into a single, higher-confidence claim
   that carries the union of supporting source_ids.

3. **Full synthesis** — calls ``GroundedSummaryBuilder.build()`` on the
   de-duplicated, claim-merged request and returns a ``GroundedSummary``.

Deduplication algorithm
------------------------
Two sources are considered near-duplicates when their content_snippet
Jaccard-similarity exceeds ``dedup_threshold``.  The source with the higher
``trust_score`` is kept; in case of a tie, the earlier one (lower index) is
retained.  Deduplication is O(n²) and adequate for typical digest sizes
(≤ 100 sources).

Claim merging algorithm
------------------------
Claims are merged greedily: a new claim is started whenever its token
overlap with all existing merged claims is below ``merge_threshold``.
When two claims are merged:
- The text of the higher-confidence claim is kept.
- ``confidence`` = min(1.0, mean of both confidences × 1.05) (small boost
  because corroboration increases confidence).
- ``source_ids`` = union of both source_id lists.
- ``negation_detected`` = True if either input is negated (conservative).

Optional LLM path
-----------------
Passed through to ``GroundedSummaryBuilder``.  No direct LLM use in the
dedup / merge steps (pure heuristic).

Thread safety
-------------
``MultiSourceSynthesizer`` is stateless; all public methods are re-entrant.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from app.summarization.models import (
    AttributedClaim,
    EvidenceSource,
    GroundedSummary,
    SynthesisRequest,
)
from app.summarization.grounded_summary_builder import GroundedSummaryBuilder
from app.summarization.source_attribution import _tokenise

logger = logging.getLogger(__name__)

_DEFAULT_DEDUP_THRESHOLD: float = 0.70  # Jaccard similarity above which sources are dupes
_DEFAULT_MERGE_THRESHOLD: float = 0.60  # token-overlap above which claims are merged


def _source_jaccard(a: EvidenceSource, b: EvidenceSource) -> float:
    """Title + snippet Jaccard similarity between two sources."""
    tokens_a = _tokenise(a.content_snippet + " " + a.title)
    tokens_b = _tokenise(b.content_snippet + " " + b.title)
    if not tokens_a or not tokens_b:
        return 0.0
    inter = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(inter) / len(union)


def _claim_token_overlap(a: AttributedClaim, b: AttributedClaim) -> float:
    """Recall-based token overlap: |tokens_a ∩ tokens_b| / |tokens_a|."""
    ta = _tokenise(a.text)
    tb = _tokenise(b.text)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


class MultiSourceSynthesizer:
    """Deduplicates sources, merges claims, then builds a ``GroundedSummary``.

    Args:
        dedup_threshold: Jaccard similarity above which sources are near-dupes.
        merge_threshold: Token-overlap above which claims are consolidated.
        builder:         ``GroundedSummaryBuilder`` to delegate the final build.
        llm_router:      Optional LLM router passed to the builder.
    """

    def __init__(
        self,
        dedup_threshold: float = _DEFAULT_DEDUP_THRESHOLD,
        merge_threshold: float = _DEFAULT_MERGE_THRESHOLD,
        builder: Optional[GroundedSummaryBuilder] = None,
        llm_router: Optional[Any] = None,
    ) -> None:
        if not (0.0 < dedup_threshold <= 1.0):
            raise ValueError(f"'dedup_threshold' must be in (0, 1], got {dedup_threshold!r}")
        if not (0.0 < merge_threshold <= 1.0):
            raise ValueError(f"'merge_threshold' must be in (0, 1], got {merge_threshold!r}")
        if builder is not None and not isinstance(builder, GroundedSummaryBuilder):
            raise TypeError(f"'builder' must be GroundedSummaryBuilder or None, got {type(builder)!r}")

        self._dedup_t = dedup_threshold
        self._merge_t = merge_threshold
        self._builder = builder or GroundedSummaryBuilder(llm_router=llm_router)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, request: SynthesisRequest) -> GroundedSummary:
        """Deduplicate sources and synthesize a ``GroundedSummary``.

        Args:
            request: Validated ``SynthesisRequest`` (possibly with duplicate sources).

        Returns:
            ``GroundedSummary`` built from the deduplicated source set.

        Raises:
            TypeError: *request* is not a ``SynthesisRequest``.
        """
        if not isinstance(request, SynthesisRequest):
            raise TypeError(f"'request' must be SynthesisRequest, got {type(request)!r}")

        deduped = self.deduplicate_sources(request.sources)
        logger.info(
            "MultiSourceSynthesizer.synthesize: %d → %d sources after dedup",
            len(request.sources), len(deduped),
        )
        clean_request = SynthesisRequest(
            topic=request.topic,
            sources=deduped,
            context=request.context,
            max_claims=request.max_claims,
            min_source_trust=request.min_source_trust,
            who_it_affects=request.who_it_affects,
        )
        return self._builder.build(clean_request)

    def deduplicate_sources(self, sources: List[EvidenceSource]) -> List[EvidenceSource]:
        """Remove near-duplicate sources, keeping the highest-trust copy.

        Args:
            sources: List of ``EvidenceSource`` objects.

        Returns:
            Deduplicated list, preserving original ordering of kept items.

        Raises:
            TypeError: *sources* is not a list.
        """
        if not isinstance(sources, list):
            raise TypeError(f"'sources' must be a list, got {type(sources)!r}")
        if len(sources) <= 1:
            return list(sources)

        dropped: Set[int] = set()
        for i in range(len(sources)):
            if i in dropped:
                continue
            for j in range(i + 1, len(sources)):
                if j in dropped:
                    continue
                sim = _source_jaccard(sources[i], sources[j])
                if sim >= self._dedup_t:
                    # Drop the lower-trust one; ties → drop the later one (j)
                    if sources[j].trust_score > sources[i].trust_score:
                        dropped.add(i)
                        break
                    else:
                        dropped.add(j)

        result = [s for idx, s in enumerate(sources) if idx not in dropped]
        logger.debug(
            "deduplicate_sources: %d → %d (threshold=%.2f)",
            len(sources), len(result), self._dedup_t,
        )
        return result

    def merge_claims(self, claims: List[AttributedClaim]) -> List[AttributedClaim]:
        """Consolidate near-duplicate claims into corroborated single claims.

        Args:
            claims: ``AttributedClaim`` list (from one or more sources).

        Returns:
            Reduced list of merged ``AttributedClaim`` objects.

        Raises:
            TypeError: *claims* is not a list.
        """
        if not isinstance(claims, list):
            raise TypeError(f"'claims' must be a list, got {type(claims)!r}")
        if not claims:
            return []

        merged: List[AttributedClaim] = []
        for claim in claims:
            absorbed = False
            for idx, existing in enumerate(merged):
                overlap = _claim_token_overlap(claim, existing)
                if overlap >= self._merge_t:
                    # Keep the higher-confidence text, boost confidence slightly
                    if claim.confidence >= existing.confidence:
                        primary_text = claim.text
                        primary_type = claim.claim_type
                    else:
                        primary_text = existing.text
                        primary_type = existing.claim_type
                    merged_confidence = min(
                        1.0, (claim.confidence + existing.confidence) / 2.0 * 1.05
                    )
                    combined_source_ids = list(
                        dict.fromkeys(existing.source_ids + claim.source_ids)
                    )
                    merged[idx] = AttributedClaim(
                        claim_id=existing.claim_id,
                        text=primary_text,
                        claim_type=primary_type,
                        confidence=round(merged_confidence, 5),
                        source_ids=combined_source_ids,
                        negation_detected=existing.negation_detected or claim.negation_detected,
                        extracted_at=existing.extracted_at,
                    )
                    absorbed = True
                    break
            if not absorbed:
                merged.append(claim)

        logger.debug("merge_claims: %d → %d claims (threshold=%.2f)", len(claims), len(merged), self._merge_t)
        return merged

