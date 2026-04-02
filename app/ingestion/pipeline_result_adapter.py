"""Bridge between the ingestion layer and the personalization layer.

``PipelineResultAdapter`` converts :class:`IntelligencePipelineResult` objects
produced by :class:`~app.ingestion.indexing_pipeline.IndexingPipeline` into
:class:`~app.personalization.models.DigestCandidate` objects consumed by
:class:`~app.personalization.user_digest_ranker.UserDigestRanker`.

This is the sole canonical translation point between the two layers; nothing
else should duplicate this mapping logic.

Typical usage::

    adapter = PipelineResultAdapter(
        trust_scorer=my_trust_scorer,
        novelty_scorer=my_novelty_scorer,
    )
    candidates = adapter.adapt_batch(indexing_result.pipeline_results)
    ranked = ranker.rank(candidates)
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.ingestion.pipeline_result import IntelligencePipelineResult, PipelineStatus
from app.personalization.models import DigestCandidate

logger = logging.getLogger(__name__)

# Lazy imports kept out of the module body to avoid circular deps.
_TrustScorerType = None
_NoveltyScorerType = None


def _trust_scorer_class():
    global _TrustScorerType
    if _TrustScorerType is None:
        from app.source_intelligence.source_trust import SourceTrustScorer
        _TrustScorerType = SourceTrustScorer
    return _TrustScorerType


def _novelty_scorer_class():
    global _NoveltyScorerType
    if _NoveltyScorerType is None:
        from app.personalization.novelty_scorer import NoveltyScorer
        _NoveltyScorerType = NoveltyScorer
    return _NoveltyScorerType


class PipelineResultAdapter:
    """Converts ``IntelligencePipelineResult`` objects to ``DigestCandidate``.

    Args:
        trust_scorer:    Optional ``SourceTrustScorer`` used to compute live
                         trust scores from ``result.source_family``.  When
                         ``None``, ``result.confidence`` is used as the trust
                         score proxy.
        novelty_scorer:  Optional ``NoveltyScorer`` used to compute live
                         novelty scores.  When ``None``, ``novelty_score``
                         defaults to ``0.5`` (neutral).
        default_trust:   Fallback trust score when the trust_scorer raises or
                         is absent.  Must be in [0, 1].  Default: 0.5.
        max_raw_text_chars: Maximum characters taken from the raw text blob
                            for ``DigestCandidate.raw_text``.  Default: 2000.

    Raises:
        ValueError: If ``default_trust`` is outside [0, 1].
        TypeError:  If ``trust_scorer`` / ``novelty_scorer`` are the wrong types.
    """

    def __init__(
        self,
        trust_scorer=None,
        novelty_scorer=None,
        default_trust: float = 0.5,
        max_raw_text_chars: int = 2000,
    ) -> None:
        if trust_scorer is not None:
            cls = _trust_scorer_class()
            if not isinstance(trust_scorer, cls):
                raise TypeError(
                    f"'trust_scorer' must be SourceTrustScorer or None, "
                    f"got {type(trust_scorer)!r}"
                )
        if novelty_scorer is not None:
            cls = _novelty_scorer_class()
            if not isinstance(novelty_scorer, cls):
                raise TypeError(
                    f"'novelty_scorer' must be NoveltyScorer or None, "
                    f"got {type(novelty_scorer)!r}"
                )
        if not (0.0 <= default_trust <= 1.0):
            raise ValueError(f"'default_trust' must be in [0, 1], got {default_trust!r}")
        if max_raw_text_chars <= 0:
            raise ValueError(
                f"'max_raw_text_chars' must be positive, got {max_raw_text_chars!r}"
            )

        self._trust_scorer = trust_scorer
        self._novelty_scorer = novelty_scorer
        self._default_trust = default_trust
        self._max_raw = max_raw_text_chars


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def adapt(
        self, result: IntelligencePipelineResult
    ) -> Optional[DigestCandidate]:
        """Convert one ``IntelligencePipelineResult`` to a ``DigestCandidate``.

        Non-actionable results (``FAILED``, ``SKIPPED``) are skipped and
        ``None`` is returned.

        Args:
            result: The pipeline result to adapt.

        Returns:
            A ``DigestCandidate``, or ``None`` for non-actionable results.

        Raises:
            TypeError: If *result* is not an ``IntelligencePipelineResult``.
        """
        if not isinstance(result, IntelligencePipelineResult):
            raise TypeError(
                f"Expected IntelligencePipelineResult, got {type(result)!r}"
            )
        if not result.is_actionable():
            logger.debug(
                "PipelineResultAdapter: skipping non-actionable result %s (status=%s)",
                result.content_item_id, result.status,
            )
            return None

        # ── Trust score ────────────────────────────────────────────────
        trust = self._default_trust
        if self._trust_scorer is not None:
            try:
                ts = self._trust_scorer.score(result.source_family)
                trust = ts.composite
            except Exception as exc:
                logger.warning(
                    "PipelineResultAdapter: trust_scorer.score failed for "
                    "source_family=%r item=%s: %s",
                    result.source_family, result.content_item_id, exc,
                )
                trust = max(0.0, min(1.0, result.confidence)) if result.confidence else self._default_trust

        # ── Raw text for novelty fallback ──────────────────────────────
        raw_parts = []
        if result.summary:
            raw_parts.append(result.summary)
        raw_parts.extend(result.claims[:5])   # up to 5 claims
        raw_text = " ".join(raw_parts)[: self._max_raw]

        # ── Build candidate ────────────────────────────────────────────
        candidate = DigestCandidate(
            item_id=str(result.content_item_id),
            title=result.summary[:200] if result.summary else (result.signal_type or ""),
            topic_ids=list(result.keywords),
            entity_ids=list(result.entities),
            published_at=result.produced_at,
            trust_score=min(1.0, max(0.0, trust)),
            engagement_score=0.5,            # no engagement signal at this layer
            novelty_score=0.5,               # overwritten by scorer below
            source_platform=result.source_family,
            raw_text=raw_text,
            metadata={
                "signal_type":   result.signal_type or "",
                "result_id":     str(result.result_id),
                "tenant_id":     result.tenant_id,
                "status":        result.status.value,
                "pipeline_duration_s": result.pipeline_duration_s,
                "stages_run":    list(result.stages_run),
            },
        )

        # ── Novelty score ──────────────────────────────────────────────
        if self._novelty_scorer is not None:
            try:
                novelty = self._novelty_scorer.score(candidate)
                candidate = candidate.model_copy(update={"novelty_score": novelty})
            except Exception as exc:
                logger.warning(
                    "PipelineResultAdapter: novelty_scorer.score failed for "
                    "item=%s: %s",
                    result.content_item_id, exc,
                )

        logger.debug(
            "PipelineResultAdapter.adapt: item=%s trust=%.3f source=%s",
            candidate.item_id, candidate.trust_score, result.source_family,
        )
        return candidate

    def adapt_batch(
        self, results: List[IntelligencePipelineResult]
    ) -> List[DigestCandidate]:
        """Convert a list of pipeline results, skipping non-actionable ones.

        Errors for individual items are caught and logged; processing
        continues for remaining items.

        Args:
            results: List of ``IntelligencePipelineResult`` objects.

        Returns:
            List of ``DigestCandidate`` objects (non-actionable items omitted).

        Raises:
            TypeError: If *results* is not a list.
        """
        if not isinstance(results, list):
            raise TypeError(f"'results' must be a list, got {type(results)!r}")
        candidates: List[DigestCandidate] = []
        for r in results:
            try:
                c = self.adapt(r)
                if c is not None:
                    candidates.append(c)
            except Exception as exc:
                logger.warning(
                    "PipelineResultAdapter.adapt_batch: failed to adapt item %s: %s",
                    getattr(r, "content_item_id", "?"), exc,
                )
        return candidates

