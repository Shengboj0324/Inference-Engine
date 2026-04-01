"""Grounded summary builder.

Orchestrates the full Phase 4 pipeline for a single ``SynthesisRequest``:

  SynthesisRequest
       │
       ├─ filter sources by min_source_trust
       ├─ extract key sentences (TF-IDF-style term-frequency ranking)
       ├─ ClaimVerifier  → AttributedClaim list
       ├─ SourceAttributor → per-claim source attribution
       ├─ ContradictionDetector → ContradictionPair list
       ├─ UncertaintyAnnotator  → UncertaintyAnnotation list
       └─ assemble GroundedSummary

Heuristic extractive fast path
-------------------------------
``_extract_key_sentences`` scores every sentence in source snippets by:

    score = Σ freq(token) × trust_boost

where ``freq(token)`` is the document-wide term frequency of non-stop-word
tokens, and ``trust_boost`` is the trust_score of the originating source.
The top-``n`` scoring sentences form the ``what_happened`` field.

``why_it_matters`` is synthesised by prepending a short template
("This matters because…") to the highest-scoring sentence that contains
a significance-signal token (``impact, effect, affect, mean, result,
implication, because, therefore, since, why``).  If none is found, the
second-highest scoring sentence is used as a fallback.

Overall confidence
------------------
``confidence_score = weighted_mean(source trust) × (1 − uncertainty_score)``

This rewards high-trust sources and deflates confidence proportionally to
detected uncertainty.

Thread safety
-------------
``GroundedSummaryBuilder`` holds no mutable state.  All public methods are
re-entrant.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from app.summarization.models import (
    AttributedClaim,
    EvidenceSource,
    GroundedSummary,
    SynthesisRequest,
)
from app.summarization.claim_verifier import ClaimVerifier
from app.summarization.contradiction_detector import ContradictionDetector
from app.summarization.source_attribution import SourceAttributor, _tokenise
from app.summarization.uncertainty_annotator import UncertaintyAnnotator

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SIGNIFICANCE_TOKENS = frozenset(
    "impact effect affect mean result implication because therefore "
    "since why matters matter significance important importance".split()
)
_STOP = frozenset(
    "the a an is are was were be been being have has had do does did "
    "will would could should may might shall can this that these those "
    "it its and or but if in on at to for of from with by about".split()
)
_DEFAULT_TOP_N_SENTENCES = 3


class GroundedSummaryBuilder:
    """Builds a ``GroundedSummary`` from a ``SynthesisRequest``.

    Args:
        attributor:  ``SourceAttributor`` instance (created with defaults if None).
        verifier:    ``ClaimVerifier`` instance (created with defaults if None).
        detector:    ``ContradictionDetector`` instance (created with defaults if None).
        annotator:   ``UncertaintyAnnotator`` instance (created with defaults if None).
        llm_router:  Optional LLM router passed through to sub-components.
        top_n_sentences: Number of key sentences to include in ``what_happened``.
    """

    def __init__(
        self,
        attributor: Optional[SourceAttributor] = None,
        verifier: Optional[ClaimVerifier] = None,
        detector: Optional[ContradictionDetector] = None,
        annotator: Optional[UncertaintyAnnotator] = None,
        llm_router: Optional[Any] = None,
        top_n_sentences: int = _DEFAULT_TOP_N_SENTENCES,
    ) -> None:
        if top_n_sentences < 1:
            raise ValueError(f"'top_n_sentences' must be ≥ 1, got {top_n_sentences!r}")

        self._attributor = attributor or SourceAttributor(llm_router=llm_router)
        self._verifier = verifier or ClaimVerifier(llm_router=llm_router)
        self._detector = detector or ContradictionDetector(llm_router=llm_router)
        self._annotator = annotator or UncertaintyAnnotator(llm_router=llm_router)
        self._top_n = top_n_sentences

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, request: SynthesisRequest) -> GroundedSummary:
        """Build a ``GroundedSummary`` from *request*.

        Args:
            request: Validated ``SynthesisRequest``.

        Returns:
            ``GroundedSummary`` with attribution, contradictions, and
            uncertainty annotations.

        Raises:
            TypeError:  *request* is not a ``SynthesisRequest``.
        """
        if not isinstance(request, SynthesisRequest):
            raise TypeError(f"'request' must be SynthesisRequest, got {type(request)!r}")

        # 1. Filter sources
        sources = [
            s for s in request.sources
            if s.trust_score >= request.min_source_trust
        ]
        if not sources:
            logger.warning(
                "GroundedSummaryBuilder: no sources pass min_trust=%.2f; using all",
                request.min_source_trust,
            )
            sources = list(request.sources)

        logger.info("GroundedSummaryBuilder.build: topic=%r sources=%d", request.topic, len(sources))

        # 2. Extract key sentences
        key_sentences = self._extract_key_sentences(sources, max_n=self._top_n * 2)

        # 3. Build what_happened / why_it_matters
        what_happened = self._build_what_happened(key_sentences, request.topic)
        why_it_matters = self._build_why_it_matters(key_sentences, request.topic)

        # 4. Extract and verify claims
        claim_texts = key_sentences[: request.max_claims]
        claims: List[AttributedClaim] = []
        if claim_texts:
            claims = self._verifier.verify_batch(claim_texts, sources)

        # 5. Detect contradictions
        contradictions = []
        if len(claims) >= 2:
            try:
                contradictions = self._detector.detect_contradictions(claims)
            except ValueError:
                pass  # < 2 valid claims after filtering

        # 6. Annotate uncertainty in the combined text
        combined_text = " ".join(s.content_snippet for s in sources if s.content_snippet)
        uncertainty_annotations = []
        if combined_text.strip():
            uncertainty_annotations = self._annotator.annotate(combined_text[:2000])

        # 7. Compute scores
        uncertainty_score = self._annotator.overall_uncertainty(uncertainty_annotations)
        confidence = self._compute_confidence(sources, uncertainty_score)

        return GroundedSummary(
            what_happened=what_happened,
            why_it_matters=why_it_matters,
            confidence_score=round(confidence, 5),
            source_attributions=sources,
            key_claims=claims[: request.max_claims],
            contradictions=contradictions,
            uncertainty_annotations=uncertainty_annotations,
            who_it_affects=request.who_it_affects,
            overall_uncertainty_score=round(uncertainty_score, 5),
            source_count=len(sources),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_key_sentences(
        self, sources: List[EvidenceSource], max_n: int = 6
    ) -> List[str]:
        """Return top-``max_n`` sentences by TF × trust score, deduplicated."""
        # Build corpus-level token frequency weighted by source trust
        tf: Counter = Counter()
        all_sentences: List[tuple] = []  # (score_base, sentence, trust)

        for src in sources:
            text = src.content_snippet.strip()
            if not text:
                continue
            tokens = [t for t in _tokenise(text) if t not in _STOP and len(t) >= 3]
            tf.update(tokens)

        for src in sources:
            text = src.content_snippet.strip()
            if not text:
                continue
            for sent in _SENTENCE_SPLIT_RE.split(text):
                sent = sent.strip()
                if len(sent) < 20:
                    continue
                tokens = [t for t in _tokenise(sent) if t not in _STOP and len(t) >= 3]
                score = sum(tf[t] for t in tokens) * (0.5 + src.trust_score)
                all_sentences.append((score, sent))

        all_sentences.sort(key=lambda x: x[0], reverse=True)
        seen: set = set()
        result: List[str] = []
        for _, sent in all_sentences:
            key = sent[:80].lower()
            if key not in seen:
                seen.add(key)
                result.append(sent)
                if len(result) >= max_n:
                    break
        return result

    def _build_what_happened(self, sentences: List[str], topic: str) -> str:
        if not sentences:
            return f"No content found for topic: {topic}."
        return " ".join(sentences[: self._top_n])

    def _build_why_it_matters(self, sentences: List[str], topic: str) -> str:
        significance_sent = next(
            (s for s in sentences if any(tok in _tokenise(s) for tok in _SIGNIFICANCE_TOKENS)),
            sentences[1] if len(sentences) > 1 else "",
        )
        if not significance_sent:
            return f"The significance of {topic} requires further analysis."
        return f"This matters because {significance_sent.lstrip('This matters because ').strip()}"

    def _compute_confidence(
        self, sources: List[EvidenceSource], uncertainty_score: float
    ) -> float:
        if not sources:
            return 0.0
        mean_trust = sum(s.trust_score for s in sources) / len(sources)
        return min(1.0, max(0.0, mean_trust * (1.0 - uncertainty_score * 0.5)))

