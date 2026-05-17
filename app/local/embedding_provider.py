"""BYOK-aware embedding provider for the desktop ingestion pipeline.

Resolves the OpenAI (or compatible) embedding key from the Phase 2
:class:`KeyVault`, gates every call on the
:data:`Permission.USE_LLM_KEY` grant, and runs every input through
:meth:`DataResidencyGuard._scrub_text` *before* the text leaves the
process.  When no key is configured or permission has not been granted,
the provider raises :class:`EmbeddingUnavailable`; callers are expected
to treat that as a soft failure (item is still ingested, just without
an embedding) so the pipeline never blocks on missing credentials.

The provider is intentionally narrower than ``OpenAIEmbeddingClient``:
it only exposes ``embed_text`` / ``embed_batch`` returning raw float
vectors, because the ingestion layer does not need token-usage metering
(the LLM router does that for the chat path).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Protocol

from app.core.data_residency import DataResidencyGuard
from app.local.key_policy import normalise
from app.local.key_vault import KeyVault, get_key_vault
from app.local.permissions import Permission, PermissionError, PermissionManager

logger = logging.getLogger(__name__)


_DEFAULT_PROVIDER = "openai"
_DEFAULT_MODEL = "text-embedding-3-small"


class EmbeddingUnavailable(Exception):
    """Raised when no key is configured or permission has not been granted."""


class EmbeddingClient(Protocol):
    """Minimal async interface implemented by the real OpenAI client."""

    async def embed_text(self, text: str): ...
    async def embed_batch(self, texts: List[str]): ...


# Factory signature so tests can inject a fake embedding client without
# pulling in the openai package.  Receives ``(provider, api_key, model)``.
ClientFactory = "callable"


def _default_client_factory(provider: str, api_key: str, model: str):
    """Build the real async OpenAI embedding client on demand.

    Imported lazily so unit tests can swap in a stub factory and so the
    embedding code path stays optional at import time.
    """
    if provider != _DEFAULT_PROVIDER:
        raise EmbeddingUnavailable(
            f"embedding provider {provider!r} not supported (only "
            f"{_DEFAULT_PROVIDER!r} is wired into the local pipeline)"
        )
    from app.llm.openai_client import OpenAIEmbeddingClient  # noqa: PLC0415
    return OpenAIEmbeddingClient(api_key=api_key, model=model)


class LocalEmbeddingProvider:
    """BYOK embedding facade used by the desktop ingestion runtime."""

    def __init__(
        self,
        *,
        vault: Optional[KeyVault] = None,
        permissions: Optional[PermissionManager] = None,
        provider: str = _DEFAULT_PROVIDER,
        model: str = _DEFAULT_MODEL,
        client_factory=None,
    ) -> None:
        self._vault = vault if vault is not None else get_key_vault()
        self._permissions = permissions
        self._provider = normalise(provider)
        self._model = model
        self._factory = client_factory or _default_client_factory
        self._client_cache: Optional[EmbeddingClient] = None
        self._cached_key: Optional[str] = None

    # ------------------------------------------------------------------
    # Capability probes
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        """Identifier of the embedding model this provider issues calls against.

        Surfaced so the RAG layer can stamp ``embedding_version`` on each
        persisted vector and detect stale rows after a model upgrade
        without rescanning every payload.
        """
        return f"{self._provider}:{self._model}"

    def is_configured(self) -> bool:
        """True if a key is in the vault for the configured provider."""
        return bool(self._vault.get(self._provider))

    def is_permitted(self) -> bool:
        """True if no permission manager is set, or USE_LLM_KEY is granted."""
        if self._permissions is None:
            return True
        return self._permissions.is_allowed(Permission.USE_LLM_KEY)

    # ------------------------------------------------------------------
    # Public embedding API
    # ------------------------------------------------------------------

    async def embed_text(self, text: str) -> List[float]:
        client = self._require_client()
        safe = self._scrub(text)
        response = await client.embed_text(safe)
        return list(response.embedding)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        client = self._require_client()
        scrubbed = [self._scrub(t) for t in texts]
        responses = await client.embed_batch(scrubbed)
        return [list(r.embedding) for r in responses]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _scrub(text: str) -> str:
        """Strip PII before any byte leaves the machine."""
        if not text:
            return ""
        cleaned, _ = DataResidencyGuard._scrub_text(text)
        return cleaned

    def _require_client(self) -> EmbeddingClient:
        if self._permissions is not None:
            try:
                self._permissions.require(Permission.USE_LLM_KEY)
            except PermissionError as exc:
                raise EmbeddingUnavailable(str(exc)) from exc
        key = self._vault.get(self._provider)
        if not key:
            raise EmbeddingUnavailable(
                f"no API key stored for provider {self._provider!r}; "
                "configure one via PUT /api/v1/keys/{provider}"
            )
        if self._client_cache is None or self._cached_key != key:
            self._client_cache = self._factory(self._provider, key, self._model)
            self._cached_key = key
        return self._client_cache

    def invalidate_cache(self) -> None:
        """Drop the cached client (e.g. after a rotated key)."""
        self._client_cache = None
        self._cached_key = None
