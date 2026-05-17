"""Local retrieval helper for the desktop sidecar's RAG layer.

Glues :class:`LocalEmbeddingProvider` to :class:`ContentStore.search_similar`
and applies :class:`DataResidencyGuard` to every snippet before it
leaves the store.  Designed to soft-fail: if the embedding provider is
unconfigured / the key permission is denied, :meth:`retrieve` returns
``[]`` instead of raising — callers (chat, RAG control routes) can then
proceed with their non-augmented path.

Snippets are truncated to ``max_chars`` so the chat-time context block
never blows past the LLM's context window for a single retrieved item.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable, List, Optional

from app.core.data_residency import DataResidencyGuard
from app.core.models import SourcePlatform
from app.local.content_store import ContentStore, get_content_store
from app.local.embedding_provider import (
    EmbeddingUnavailable,
    LocalEmbeddingProvider,
)
from app.local.retrieval_signals import (
    RetrievalSignalsStore,
    get_signals_store,
)

logger = logging.getLogger(__name__)

_DEFAULT_K = 6
_DEFAULT_SNIPPET_CHARS = 600

# Personalization weights.  Intentionally small so signals nudge the
# ranking but never override semantic similarity.  ``CITED_WEIGHT`` is
# applied to ``log(1 + cited)`` so the curve flattens quickly; a snippet
# the user has seen 10 times contributes only ~0.05 to the score, not
# enough to leapfrog a clearly better semantic match.
_CITED_WEIGHT = 0.02
_FEEDBACK_WEIGHT = 0.05
# Multiplier on the requested ``k`` so the re-rank has room to promote
# personalized hits that just missed the cosine top-k.  Capped to avoid
# unbounded SQL scans on large local corpora.
_OVERSAMPLE = 3
_MAX_OVERSAMPLE_HITS = 60


@dataclass(frozen=True)
class RetrievedSnippet:
    """A single PII-scrubbed result from the local vector search."""

    content_id: str
    title: str
    source_url: str
    source_platform: str
    published_at: float
    score: float
    text: str
    # Personalization is opt-in transparent: callers can show "boosted
    # because you upvoted this kind of thing" badges by inspecting these.
    base_score: float = 0.0
    personalization_bonus: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class LocalRAGRetriever:
    """BYOK-aware retrieval over the desktop :class:`ContentStore`."""

    def __init__(
        self,
        *,
        content: Optional[ContentStore] = None,
        embedder: Optional[LocalEmbeddingProvider] = None,
        guard: Optional[DataResidencyGuard] = None,
        signals: Optional[RetrievalSignalsStore] = None,
        max_snippet_chars: int = _DEFAULT_SNIPPET_CHARS,
        personalize: bool = True,
    ) -> None:
        self._content = content if content is not None else get_content_store()
        self._embedder = embedder if embedder is not None else LocalEmbeddingProvider()
        self._guard = guard if guard is not None else DataResidencyGuard()
        self._signals = signals if signals is not None else get_signals_store()
        self._max_chars = max(120, int(max_snippet_chars))
        self._personalize = bool(personalize)

    # ------------------------------------------------------------------
    # Capability probe
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """True when the embedding provider can be invoked right now."""
        try:
            return self._embedder.is_configured() and self._embedder.is_permitted()
        except Exception:  # noqa: BLE001 - capability probe must not raise
            return False

    def indexed_count(self) -> int:
        return self._content.count_with_embedding()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        *,
        k: int = _DEFAULT_K,
        since: Optional[datetime] = None,
        platforms: Optional[Iterable[SourcePlatform]] = None,
        min_score: float = 0.0,
        personalize: Optional[bool] = None,
    ) -> List[RetrievedSnippet]:
        """Embed ``query``, run ANN search, re-rank by personalization signals.

        Soft-fails to ``[]`` on missing key / denied permission / empty
        query / any embedding-provider exception.  Never raises.

        Personalization (when enabled and signals are available) pulls
        ``k * _OVERSAMPLE`` candidates from the vector store, computes
        a small additive bonus from ``cited_count`` / ``feedback_score``,
        and returns the top-``k`` by the *combined* score.  The semantic
        floor ``min_score`` is applied to the *base* score so a heavily
        upvoted but semantically irrelevant snippet still cannot leak
        into the result set.
        """
        text = (query or "").strip()
        if not text:
            return []
        try:
            vector = await self._embedder.embed_text(text)
        except EmbeddingUnavailable as exc:
            logger.debug("rag retrieve skipped: %s", exc)
            return []
        except Exception:  # noqa: BLE001 - third-party client surface
            logger.exception("rag query embedding failed")
            return []
        if not vector:
            return []
        do_personalize = self._personalize if personalize is None else bool(personalize)
        # Oversample so the re-rank has candidates to promote/demote.
        # When personalization is off we keep the original semantics by
        # asking for exactly ``k`` so the SQL ``LIMIT`` short-circuits
        # immediately at the ANN layer.
        search_k = min(_MAX_OVERSAMPLE_HITS, k * _OVERSAMPLE) if do_personalize else k
        hits = self._content.search_similar(
            vector, k=search_k, since=since, platforms=platforms, min_score=min_score,
        )
        if not hits:
            return []
        if not do_personalize:
            return [self._snippet_for(item, score, bonus=0.0) for item, score in hits]
        signals = {}
        try:
            signals = self._signals.get_signals([str(item.id) for item, _ in hits])
        except Exception:  # noqa: BLE001 - signals must never break retrieval
            logger.exception("retrieval signals load failed; ranking without them")
        rescored: List = []
        for item, score in hits:
            bonus = self._personalization_bonus(signals.get(str(item.id)))
            rescored.append((item, float(score), bonus))
        rescored.sort(key=lambda r: r[1] + r[2], reverse=True)
        return [
            self._snippet_for(item, base_score, bonus=bonus)
            for item, base_score, bonus in rescored[:k]
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _personalization_bonus(sig: Optional[dict]) -> float:
        """Map raw signal counters onto a bounded additive score bonus.

        The curve is intentionally flat: ``log(1+cited)`` saturates
        quickly, and the feedback weight is capped by the signals
        store's own clamp.  Worst-case bonus on a current cosine in
        ``[-1, 1]`` is ~``0.05 + 0.05 = 0.10`` \u2014 enough to break ties
        and promote near-misses, never enough to override a strong
        semantic mismatch.
        """
        if not sig:
            return 0.0
        cited = max(0, int(sig.get("cited", 0)))
        feedback = float(sig.get("feedback", 0.0))
        return _CITED_WEIGHT * math.log1p(cited) + _FEEDBACK_WEIGHT * math.tanh(feedback / 3.0)

    def _snippet_for(self, item, score: float, *, bonus: float = 0.0) -> RetrievedSnippet:
        # Prefer the body text; fall back to the title so we always have
        # *something* to inject into the prompt.
        body = (item.raw_text or item.title or "").strip()
        if len(body) > self._max_chars:
            body = body[: self._max_chars - 1].rstrip() + "\u2026"
        scrubbed_body, _ = self._guard.scrub_text(body)
        scrubbed_title, _ = self._guard.scrub_text(item.title or "")
        return RetrievedSnippet(
            content_id=str(item.id),
            title=scrubbed_title,
            source_url=item.source_url,
            source_platform=item.source_platform.value,
            published_at=item.published_at.timestamp(),
            score=float(score) + float(bonus),
            base_score=float(score),
            personalization_bonus=float(bonus),
            text=scrubbed_body,
        )

    # ------------------------------------------------------------------
    # Personalization signal recording
    # ------------------------------------------------------------------

    def record_citations(self, snippets: Iterable[RetrievedSnippet]) -> None:
        """Mark each snippet as cited so future queries can prefer them.

        Called by the chat path after a retrieval-augmented reply is
        produced.  Failures are swallowed because signal recording must
        never break the chat response.
        """
        for s in snippets:
            try:
                self._signals.record_citation(s.content_id)
            except Exception:  # noqa: BLE001 - signals are best-effort
                logger.exception("failed to record citation for %s", s.content_id)

    def record_feedback(self, content_id: str, score: float) -> float:
        """Apply explicit user feedback (+1 / -1) to a content id."""
        return self._signals.record_feedback(content_id, score)

    @property
    def signals_store(self) -> RetrievalSignalsStore:
        """Expose the signals store for transparency / debug routes."""
        return self._signals


_global_retriever: Optional[LocalRAGRetriever] = None


def get_rag_retriever() -> LocalRAGRetriever:
    """Process-wide :class:`LocalRAGRetriever` singleton."""
    global _global_retriever
    if _global_retriever is None:
        _global_retriever = LocalRAGRetriever()
    return _global_retriever


def reset_rag_retriever() -> None:
    """Drop the cached singleton (test-only)."""
    global _global_retriever
    _global_retriever = None
