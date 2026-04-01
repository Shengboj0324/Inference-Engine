"""Source attribution.

Maps a text claim to the subset of ``EvidenceSource`` objects that support it,
using token-overlap Jaccard similarity as a dependency-free fast path.

Heuristic fast path
-------------------
1. Tokenise both the claim text and each source's ``content_snippet``
   (lower-case, strip punctuation, split on whitespace).
2. Compute Jaccard(claim_tokens ∩ source_tokens) / claim_tokens.
   (We use claim-recall so short snippets are not unfairly penalised.)
3. Return sources whose score ≥ ``min_overlap``, ranked by score DESC.
4. Compute a trust-weighted confidence from the returned sources.

Optional LLM path
-----------------
When an ``llm_router`` is injected and the heuristic confidence is below
``llm_threshold``, ``attribute_text`` calls the router with a structured
attribution prompt and merges the result with the heuristic attribution.
If the LLM call fails or the router is ``None``, the heuristic result is
returned unchanged (graceful degradation).

Thread safety
-------------
``SourceAttributor`` holds no mutable state; every public method is
re-entrant by construction.  If a subclass adds state, protect it with
``threading.RLock``.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, List, Optional, Set, Tuple

from app.summarization.models import EvidenceSource

logger = logging.getLogger(__name__)

# Minimum token-overlap ratio to consider a source attributable
_DEFAULT_MIN_OVERLAP: float = 0.10
# Below this heuristic confidence, try the LLM path (if router available)
_DEFAULT_LLM_THRESHOLD: float = 0.40
# Regex for punctuation stripping during tokenisation
_PUNCT = re.compile(r"[^\w\s]")


def _tokenise(text: str) -> Set[str]:
    """Lower-case, strip punctuation, return set of words (≥ 2 chars)."""
    cleaned = _PUNCT.sub(" ", text.lower())
    return {w for w in cleaned.split() if len(w) >= 2}


def _jaccard_recall(query_tokens: Set[str], doc_tokens: Set[str]) -> float:
    """Fraction of query tokens found in doc_tokens (claim-recall Jaccard)."""
    if not query_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens)


class SourceAttributor:
    """Maps text claims to supporting ``EvidenceSource`` objects.

    Args:
        min_overlap:    Minimum token-recall overlap to flag a source as
                        attributable (0 < min_overlap ≤ 1).
        llm_threshold:  Heuristic confidence below which the LLM path is
                        attempted when ``llm_router`` is provided.
        llm_router:     Optional LLM router; if ``None`` the heuristic path
                        is always used.
        max_sources:    Maximum sources returned per call.  0 = unlimited.
    """

    def __init__(
        self,
        min_overlap: float = _DEFAULT_MIN_OVERLAP,
        llm_threshold: float = _DEFAULT_LLM_THRESHOLD,
        llm_router: Optional[Any] = None,
        max_sources: int = 10,
    ) -> None:
        if not (0.0 < min_overlap <= 1.0):
            raise ValueError(f"'min_overlap' must be in (0, 1], got {min_overlap!r}")
        if not (0.0 <= llm_threshold <= 1.0):
            raise ValueError(f"'llm_threshold' must be in [0, 1], got {llm_threshold!r}")
        if max_sources < 0:
            raise ValueError(f"'max_sources' must be >= 0, got {max_sources!r}")

        self._min_overlap = min_overlap
        self._llm_threshold = llm_threshold
        self._router = llm_router
        self._max_sources = max_sources

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attribute_text(
        self, text: str, sources: List[EvidenceSource]
    ) -> List[EvidenceSource]:
        """Return sources that support *text*, ordered by relevance.

        Args:
            text:    Claim or sentence to attribute.
            sources: Candidate ``EvidenceSource`` pool.

        Returns:
            Filtered, ranked list of supporting ``EvidenceSource`` objects.

        Raises:
            TypeError:  If *text* is not a string or *sources* not a list.
            ValueError: If *text* is empty.
        """
        if not isinstance(text, str):
            raise TypeError(f"'text' must be a str, got {type(text)!r}")
        if not text.strip():
            raise ValueError("'text' must be a non-empty string")
        if not isinstance(sources, list):
            raise TypeError(f"'sources' must be a list, got {type(sources)!r}")

        ranked = self._heuristic_attribute(text, sources)
        heuristic_conf = self.compute_trust_weight(ranked) if ranked else 0.0

        if (
            self._router is not None
            and heuristic_conf < self._llm_threshold
            and sources
        ):
            try:
                llm_ids = self._llm_attribute(text, sources)
                if llm_ids:
                    id_set = set(llm_ids)
                    ranked = [s for s in sources if s.source_id in id_set]
                    logger.debug("SourceAttributor: LLM attribution returned %d sources", len(ranked))
            except Exception as exc:
                logger.warning("SourceAttributor: LLM attribution failed (%s), using heuristic", exc)

        limit = self._max_sources if self._max_sources > 0 else len(ranked)
        result = ranked[:limit]
        logger.debug("SourceAttributor: attribute_text → %d sources (text=%r)", len(result), text[:60])
        return result

    def rank_sources_by_relevance(
        self, query: str, sources: List[EvidenceSource]
    ) -> List[EvidenceSource]:
        """Return *sources* sorted by token-recall overlap with *query*, DESC.

        Args:
            query:   Free-text query.
            sources: Sources to rank.

        Raises:
            TypeError:  Wrong argument types.
            ValueError: Empty *query*.
        """
        if not isinstance(query, str):
            raise TypeError(f"'query' must be a str, got {type(query)!r}")
        if not query.strip():
            raise ValueError("'query' must be non-empty")
        if not isinstance(sources, list):
            raise TypeError(f"'sources' must be a list, got {type(sources)!r}")

        q_tokens = _tokenise(query)
        scored: List[Tuple[float, EvidenceSource]] = []
        for src in sources:
            score = _jaccard_recall(q_tokens, _tokenise(src.content_snippet + " " + src.title))
            scored.append((score, src))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored]

    def compute_trust_weight(self, sources: List[EvidenceSource]) -> float:
        """Return the mean trust score of *sources* (0.0 if empty).

        Args:
            sources: ``EvidenceSource`` objects.

        Raises:
            TypeError: If *sources* is not a list.
        """
        if not isinstance(sources, list):
            raise TypeError(f"'sources' must be a list, got {type(sources)!r}")
        if not sources:
            return 0.0
        return sum(s.trust_score for s in sources) / len(sources)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _heuristic_attribute(
        self, text: str, sources: List[EvidenceSource]
    ) -> List[EvidenceSource]:
        """Token-overlap attribution — heuristic fast path."""
        q_tokens = _tokenise(text)
        scored: List[Tuple[float, EvidenceSource]] = []
        for src in sources:
            snippet = (src.content_snippet + " " + src.title).strip()
            if not snippet:
                continue
            score = _jaccard_recall(q_tokens, _tokenise(snippet))
            if score >= self._min_overlap:
                scored.append((score, src))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored]

    def _llm_attribute(self, text: str, sources: List[EvidenceSource]) -> List[str]:
        """Ask the LLM router which source_ids support *text*.

        Returns a list of ``source_id`` strings from the router response.
        Falls back to empty list on any parse failure.
        """
        import json

        snippets = "\n".join(
            f"[{s.source_id}] {s.title}: {s.content_snippet[:200]}" for s in sources[:8]
        )
        prompt = (
            f"Which of the following sources support this claim?\n"
            f"Claim: {text}\n\nSources:\n{snippets}\n\n"
            f"Reply with a JSON array of source_id strings only. Example: [\"id1\",\"id2\"]"
        )
        try:
            from app.llm.models import LLMMessage
            response = self._router.generate_for_signal(
                signal_type=None,
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.0,
                max_tokens=200,
            )
            # Handle both coroutine (async router) and direct string
            import asyncio, inspect
            if inspect.isawaitable(response):
                response = asyncio.get_event_loop().run_until_complete(response)
            ids = json.loads(response)
            return [str(i) for i in ids] if isinstance(ids, list) else []
        except Exception as exc:
            logger.warning("SourceAttributor._llm_attribute parse error: %s", exc)
            return []

