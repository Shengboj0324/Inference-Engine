"""Tier 3.1 — pluggable embedding backend with version stamping.

A single ``embed_fn``-compatible callable that:

- **Prefers a real dense model** (``sentence-transformers``) when the package is
  installed, for production-quality semantic embeddings.
- **Falls back to the deterministic bag-of-words embedder**
  (:func:`app.core.text_embedding.bow_embed`) when the library is unavailable,
  so embedding always works with zero heavy dependencies (dev, tests, offline).
- **Stamps an ``embedding_version``** (e.g. ``"st:all-MiniLM-L6-v2"`` or
  ``"bow:512"``) so persisted vectors can be tagged and **stale vectors detected
  and re-embedded after a model upgrade** instead of silently mixing
  incompatible vector spaces.

Usage::

    backend = EmbeddingBackend()                 # real model if available
    store = ContextMemoryStore(embed_fn=backend) # backend is callable
    store.set_embedding_version(backend.embedding_version)
    ...
    if store.needs_reembed(backend.embedding_version):
        ...  # re-embed: the persisted vectors came from a different model
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from app.core.text_embedding import DEFAULT_DIM, bow_embed

logger = logging.getLogger(__name__)

_DEFAULT_ST_MODEL = "all-MiniLM-L6-v2"


class EmbeddingBackend:
    """Resolve the best available text embedder and stamp its version.

    Args:
        model_name: Sentence-transformer model to load when available.
        dim: Dimension of the bag-of-words fallback.
        prefer_real: When False, always use the deterministic fallback (useful
            for reproducible tests even on machines that have the ML stack).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_ST_MODEL,
        dim: int = DEFAULT_DIM,
        prefer_real: bool = True,
    ) -> None:
        self._dim = int(dim)
        self._model_name = model_name
        self._real = None
        if prefer_real:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415
                self._real = SentenceTransformer(model_name)
                self._version = f"st:{model_name}"
                logger.info("EmbeddingBackend: using sentence-transformer %s", model_name)
            except Exception:  # noqa: BLE001 - optional heavy dependency
                self._real = None
        if self._real is None:
            self._version = f"bow:{self._dim}"
            logger.info("EmbeddingBackend: using deterministic bag-of-words fallback (%s)",
                        self._version)

    @property
    def embedding_version(self) -> str:
        """Stable identifier of the active embedding space."""
        return self._version

    @property
    def uses_real_model(self) -> bool:
        return self._real is not None

    def embed(self, text: str) -> List[float]:
        if self._real is not None:
            try:
                vec = self._real.encode(text or "", normalize_embeddings=True)
                return [float(x) for x in vec]
            except Exception:  # noqa: BLE001 - degrade rather than fail
                logger.exception("EmbeddingBackend: real model failed; using fallback")
        return bow_embed(text, self._dim)

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        # Use the real model's native batched encode (far faster than per-item)
        # when available; otherwise fall back to the per-item deterministic path.
        if self._real is not None and texts:
            try:
                mat = self._real.encode(list(texts), normalize_embeddings=True)
                return [[float(x) for x in row] for row in mat]
            except Exception:  # noqa: BLE001 - degrade rather than fail
                logger.exception("EmbeddingBackend: batch encode failed; per-item fallback")
        return [self.embed(t) for t in texts]

    def __call__(self, text: str) -> List[float]:
        # So a backend instance can be passed directly as ``embed_fn=...``.
        return self.embed(text)


def detect_stale_versions(
    stored_versions: Dict[str, Optional[str]],
    current_version: str,
) -> List[str]:
    """Return the ids whose stored embedding version differs from ``current``.

    Args:
        stored_versions: Mapping of record id → the ``embedding_version`` the
            record's vector was computed under (``None`` if unstamped/legacy).
        current_version: The active backend's ``embedding_version``.

    Returns:
        Ids that must be re-embedded because their vectors live in a different
        (or unknown) embedding space.
    """
    return [rid for rid, v in stored_versions.items() if v != current_version]
