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

logger = logging.getLogger(__name__)

_DEFAULT_K = 6
_DEFAULT_SNIPPET_CHARS = 600


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
        max_snippet_chars: int = _DEFAULT_SNIPPET_CHARS,
    ) -> None:
        self._content = content if content is not None else get_content_store()
        self._embedder = embedder if embedder is not None else LocalEmbeddingProvider()
        self._guard = guard if guard is not None else DataResidencyGuard()
        self._max_chars = max(120, int(max_snippet_chars))

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
    ) -> List[RetrievedSnippet]:
        """Embed ``query`` and return the top-``k`` PII-scrubbed snippets.

        Soft-fails to ``[]`` on missing key / denied permission / empty
        query / any embedding-provider exception.  Never raises.
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
        hits = self._content.search_similar(
            vector, k=k, since=since, platforms=platforms, min_score=min_score,
        )
        return [self._snippet_for(item, score) for item, score in hits]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _snippet_for(self, item, score: float) -> RetrievedSnippet:
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
            score=float(score),
            text=scrubbed_body,
        )


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
