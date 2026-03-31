"""Cross-encoder reranker for the RAG pipeline.

Scores each (query, candidate) pair independently and returns the top-k most
relevant ``NormalizedObservation`` instances for LLM generation context.

The default scoring heuristic is a deterministic TF-IDF overlap function that
requires no ML framework dependency and always completes in < 1 ms per pair.
In production, subclass ``Reranker`` and override ``_score_pair`` to call a
GPU-resident cross-encoder model (e.g. ``cross-encoder/ms-marco-MiniLM-L-6-v2``
via sentence-transformers).

Chunking parameters reference
------------------------------
Chunking occurs upstream in ``NormalizationEngine`` / the embedding path.
The constants below are the **mandated defaults** for any chunker wired into
this pipeline.  They are documented here (rather than in the chunker) because
the reranker's recall quality is sensitive to chunk granularity:

  CHUNK_SIZE    = 512 tokens
      Balances context-window utilisation vs. embedding dilution.  Larger
      chunks carry more context per retrieval unit, but dilute entity signals
      when a single chunk spans unrelated topics.  512 is the empirical sweet
      spot for 1 536-dim OpenAI embeddings on social-media content.

  CHUNK_OVERLAP = 64 tokens  (12.5% overlap)
      Ensures boundary terms (entities, product names) appear in at least one
      chunk even when they span a natural sentence boundary.  Without overlap,
      a query matching the last token of chunk N and the first of chunk N+1
      would score poorly on both; 64-token overlap eliminates this gap.
"""

from __future__ import annotations

import logging
import math
import re
from typing import List

from app.domain.normalized_models import NormalizedObservation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Upstream chunking contract
# ---------------------------------------------------------------------------

#: Mandated chunk size (tokens) for the embedding pipeline feeding this reranker.
CHUNK_SIZE: int = 512

#: Mandated overlap (tokens) between consecutive chunks.
CHUNK_OVERLAP: int = 64


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lower-case, punctuation-stripped word tokenizer (no external deps)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _tf_idf_score(query_tokens: List[str], doc_tokens: List[str]) -> float:
    """Compute a normalised TF-IDF overlap score between query and document.

    Uses a simplified IDF of ``1 + log(2)`` for every term that appears in the
    document (binary presence IDF).  This is intentionally lightweight so that
    the reranker meets its 150 ms SLA for 100 candidates without any caching.

    Returns:
        Score in [0.0, 1.0].
    """
    if not query_tokens or not doc_tokens:
        return 0.0

    from collections import Counter

    doc_counts = Counter(doc_tokens)
    doc_len = len(doc_tokens)
    score = 0.0
    for token in set(query_tokens):
        tf = doc_counts.get(token, 0) / doc_len if doc_len else 0.0
        idf = 1.0 + math.log(2.0) if tf > 0 else 0.0
        score += tf * idf

    # Normalise by number of unique query terms so that long queries do not
    # produce artificially high scores.
    unique_query = max(1, len(set(query_tokens)))
    return min(1.0, score / unique_query)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Reranker:
    """Relevance reranker for RAG candidate lists.

    Scores each ``(query, candidate)`` pair independently.

    Latency contract : ≤ 150 ms for up to 100 candidates.
    Error contract   : never raises; returns ``candidates[:top_k]`` on any
                       internal exception so the pipeline always makes progress.

    To plug in a GPU cross-encoder, subclass ``Reranker`` and override
    ``_score_pair`` — no other changes are needed.
    """

    def _score_pair(self, query: str, candidate_text: str) -> float:
        """Score a single (query, candidate) pair in [0.0, 1.0].

        Default: deterministic TF-IDF overlap.  Override for production
        cross-encoder inference.
        """
        return _tf_idf_score(_tokenize(query), _tokenize(candidate_text))

    def _candidate_text(self, candidate: NormalizedObservation) -> str:
        """Assemble the text representation of a candidate for scoring."""
        parts: List[str] = [candidate.title]
        if candidate.normalized_text:
            parts.append(candidate.normalized_text)
        parts.extend(candidate.topics or [])
        parts.extend(candidate.keywords or [])
        return " ".join(p for p in parts if p)

    def rerank(
        self,
        query: str,
        candidates: List[NormalizedObservation],
        top_k: int = 10,
    ) -> List[NormalizedObservation]:
        """Return the top-k candidates ordered by relevance to *query*.

        Args:
            query: The user query or observation merged text used for scoring.
            candidates: Pool of candidate observations retrieved by the retriever.
            top_k: Maximum number of candidates to return (clamped to
                   ``len(candidates)`` automatically).

        Returns:
            Up to *top_k* candidates, most relevant first.
        """
        if not candidates:
            return []
        effective_k = max(1, min(top_k, len(candidates)))
        try:
            scored = [
                (self._score_pair(query, self._candidate_text(c)), c)
                for c in candidates
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            result = [c for _, c in scored[:effective_k]]
            logger.debug(
                "Reranker: scored %d candidates, returning top %d",
                len(candidates),
                len(result),
            )
            return result
        except Exception as exc:
            logger.warning(
                "Reranker: scoring failed (%s); falling back to input order for top_%d",
                exc,
                effective_k,
            )
            return candidates[:effective_k]

