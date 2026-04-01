"""Claim verifier.

Assigns a confidence score to text claims by measuring how well each claim
is supported by its evidence sources, and classifies claims into semantic
types (FACTUAL, ANNOUNCEMENT, BENCHMARK, SPECULATION, OPINION, COMPARATIVE).

Heuristic fast path
-------------------
Confidence = source_attribution_score × mean_source_trust
- source_attribution_score: fraction of claim tokens found in source snippets.
- mean_source_trust: average trust_score across the supporting sources.

``ClaimType`` classification uses compiled regex patterns ranked by
specificity (most specific first): BENCHMARK > ANNOUNCEMENT > COMPARATIVE >
SPECULATION > OPINION > FACTUAL (default).

Negation detection
------------------
A lightweight regex check for English negation markers is applied to each
claim.  ``AttributedClaim.negation_detected`` is set accordingly and the
confidence is slightly deflated (×0.90) to reflect the additional
uncertainty inherent in negated statements.

Optional LLM path
-----------------
When ``llm_router`` is provided and the heuristic confidence is below
``llm_threshold``, a structured verification prompt is sent.  The router
response is expected to be a JSON object
``{"confidence": float, "claim_type": str, "negation": bool}``.
Any parse error falls back to the heuristic result.

Thread safety
-------------
``ClaimVerifier`` is stateless; all public methods are re-entrant.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Set

from app.summarization.models import AttributedClaim, ClaimType, EvidenceSource
from app.summarization.source_attribution import SourceAttributor, _tokenise, _jaccard_recall

logger = logging.getLogger(__name__)

_DEFAULT_LLM_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# Claim-type classification patterns (ordered: most specific first)
# ---------------------------------------------------------------------------
_BENCHMARK_RE = re.compile(
    r"\b(\d+\.?\d*\s*%|score[sd]?|accuracy|precision|recall|f1|bleu|rouge|perplexity"
    r"|outperform|state.of.the.art|sota|benchmark)\b",
    re.IGNORECASE,
)
_ANNOUNCEMENT_RE = re.compile(
    r"\b(announc|launch(?:ing|ed)|introduc(?:ing|ed)|releas(?:ing|ed)|"
    r"unveil(?:ing|ed)|present(?:ing|ed)|debuting|debut)\b",
    re.IGNORECASE,
)
_COMPARATIVE_RE = re.compile(
    r"\b(better than|worse than|outperforms?|underperforms?|compared to|"
    r"versus|vs\.?|more than|less than|faster than|slower than)\b",
    re.IGNORECASE,
)
_SPECULATION_RE = re.compile(
    r"\b(might|may|could|possibly|potentially|reportedly|allegedly|"
    r"expected to|rumou?red|speculated|unconfirmed|sources say|"
    r"according to sources|plans? to|will|would)\b",
    re.IGNORECASE,
)
_OPINION_RE = re.compile(
    r"\b(believe[sd]?|think[s]?|opinion|feel[s]?|argue[sd]?|contend[s]?"
    r"|suggest[s]?|consider[s]?|regard[s]?|view[s]? as|i think|in my view)\b",
    re.IGNORECASE,
)
# Negation markers
_NEGATION_RE = re.compile(
    r"\b(not|no|never|neither|nor|cannot|can't|won't|isn't|aren't|"
    r"doesn't|didn't|hasn't|haven't|hadn't|wouldn't|shouldn't|couldn't)\b",
    re.IGNORECASE,
)


def _classify_claim_type(text: str) -> ClaimType:
    """Heuristic ClaimType classification — most specific pattern wins."""
    if _BENCHMARK_RE.search(text):
        return ClaimType.BENCHMARK
    if _ANNOUNCEMENT_RE.search(text):
        return ClaimType.ANNOUNCEMENT
    if _COMPARATIVE_RE.search(text):
        return ClaimType.COMPARATIVE
    if _SPECULATION_RE.search(text):
        return ClaimType.SPECULATION
    if _OPINION_RE.search(text):
        return ClaimType.OPINION
    return ClaimType.FACTUAL


class ClaimVerifier:
    """Scores and classifies text claims against evidence sources.

    Args:
        attributor:    ``SourceAttributor`` used to find supporting sources.
        llm_router:    Optional LLM router for borderline cases.
        llm_threshold: Heuristic confidence below which LLM is tried.
        negation_penalty: Confidence multiplier applied when negation is detected.
    """

    def __init__(
        self,
        attributor: Optional[SourceAttributor] = None,
        llm_router: Optional[Any] = None,
        llm_threshold: float = _DEFAULT_LLM_THRESHOLD,
        negation_penalty: float = 0.90,
    ) -> None:
        if attributor is not None and not isinstance(attributor, SourceAttributor):
            raise TypeError(f"'attributor' must be SourceAttributor or None, got {type(attributor)!r}")
        if not (0.0 <= llm_threshold <= 1.0):
            raise ValueError(f"'llm_threshold' must be in [0, 1], got {llm_threshold!r}")
        if not (0.0 < negation_penalty <= 1.0):
            raise ValueError(f"'negation_penalty' must be in (0, 1], got {negation_penalty!r}")

        self._attributor = attributor or SourceAttributor()
        self._router = llm_router
        self._llm_threshold = llm_threshold
        self._neg_penalty = negation_penalty

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_claim(self, claim_text: str, sources: List[EvidenceSource]) -> float:
        """Compute a confidence score for *claim_text* given *sources*.

        Args:
            claim_text: Text of the claim to verify.
            sources:    Evidence source pool.

        Returns:
            Confidence score in [0, 1].

        Raises:
            TypeError:  Wrong argument types.
            ValueError: *claim_text* is empty.
        """
        if not isinstance(claim_text, str):
            raise TypeError(f"'claim_text' must be str, got {type(claim_text)!r}")
        if not claim_text.strip():
            raise ValueError("'claim_text' must be non-empty")
        if not isinstance(sources, list):
            raise TypeError(f"'sources' must be a list, got {type(sources)!r}")

        confidence = self._heuristic_confidence(claim_text, sources)
        negated = bool(_NEGATION_RE.search(claim_text))
        if negated:
            confidence *= self._neg_penalty

        if self._router is not None and confidence < self._llm_threshold:
            confidence = self._llm_verify(claim_text, sources, confidence)

        return round(min(1.0, max(0.0, confidence)), 5)

    def classify_claim(self, text: str) -> ClaimType:
        """Return the heuristic ``ClaimType`` for *text*.

        Args:
            text: Claim text to classify.

        Raises:
            TypeError:  *text* is not a string.
            ValueError: *text* is empty.
        """
        if not isinstance(text, str):
            raise TypeError(f"'text' must be str, got {type(text)!r}")
        if not text.strip():
            raise ValueError("'text' must be non-empty")
        return _classify_claim_type(text)

    def verify_batch(
        self, claim_texts: List[str], sources: List[EvidenceSource]
    ) -> List[AttributedClaim]:
        """Verify a list of claim texts and return ``AttributedClaim`` objects.

        Args:
            claim_texts: List of claim strings.
            sources:     Shared evidence source pool.

        Returns:
            One ``AttributedClaim`` per input string.

        Raises:
            TypeError: Wrong argument types.
        """
        if not isinstance(claim_texts, list):
            raise TypeError(f"'claim_texts' must be a list, got {type(claim_texts)!r}")
        if not isinstance(sources, list):
            raise TypeError(f"'sources' must be a list, got {type(sources)!r}")

        results: List[AttributedClaim] = []
        for text in claim_texts:
            if not isinstance(text, str) or not text.strip():
                logger.warning("ClaimVerifier.verify_batch: skipping invalid claim %r", text)
                continue
            confidence = self.verify_claim(text, sources)
            claim_type = self.classify_claim(text)
            negated = bool(_NEGATION_RE.search(text))
            supporting = self._attributor.attribute_text(text, sources)
            results.append(
                AttributedClaim(
                    text=text,
                    claim_type=claim_type,
                    confidence=confidence,
                    source_ids=[s.source_id for s in supporting],
                    negation_detected=negated,
                )
            )
            logger.debug(
                "ClaimVerifier: %r → %s (conf=%.3f, neg=%s)",
                text[:60], claim_type.value, confidence, negated,
            )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _heuristic_confidence(self, text: str, sources: List[EvidenceSource]) -> float:
        """Attribution recall × mean source trust."""
        if not sources:
            return 0.0
        q_tokens = _tokenise(text)
        scores = []
        for src in sources:
            doc = _tokenise(src.content_snippet + " " + src.title)
            overlap = _jaccard_recall(q_tokens, doc)
            scores.append(overlap * src.trust_score)
        if not scores:
            return 0.0
        return min(1.0, max(scores))

    def _llm_verify(
        self, text: str, sources: List[EvidenceSource], fallback: float
    ) -> float:
        """LLM-based confidence refinement.  Returns *fallback* on any error."""
        import json
        snippets = "\n".join(
            f"[{s.source_id}] {s.content_snippet[:150]}" for s in sources[:5]
        )
        prompt = (
            f"Rate the confidence (0.0–1.0) that the following claim is supported "
            f"by the provided sources.  Reply ONLY with a JSON object:\n"
            f"{{\"confidence\": <float>}}\n\nClaim: {text}\n\nSources:\n{snippets}"
        )
        try:
            from app.llm.models import LLMMessage
            import asyncio, inspect
            resp = self._router.generate_for_signal(
                signal_type=None,
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.0,
                max_tokens=50,
            )
            if inspect.isawaitable(resp):
                resp = asyncio.get_event_loop().run_until_complete(resp)
            data = json.loads(resp)
            return float(data.get("confidence", fallback))
        except Exception as exc:
            logger.warning("ClaimVerifier._llm_verify failed (%s), using heuristic", exc)
            return fallback

