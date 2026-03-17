"""Per-user observation history store with cosine-similarity RAG retrieval.

``ContextMemoryStore`` persists per-user inference events as embedding vectors
and retrieves the nearest neighbours for a query text using cosine similarity.

The embedding function is injected at construction time so that:
- Production code passes the real OpenAI (or configured-provider) embed call.
- Unit tests pass a deterministic mock that avoids real API calls.

When the real embed function is not injected, the store derives a simple
bag-of-words TF-IDF-like embedding from the text so the store is always usable
without an API key (useful for local development).

Usage::

    store = ContextMemoryStore(embed_fn=my_embed_fn)
    await store.store(user_id, observation, inference)
    records = await store.retrieve(user_id, query_text, top_k=3)
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np

from app.domain.inference_models import SignalInference, SignalType
from app.domain.normalized_models import NormalizedObservation

logger = logging.getLogger(__name__)

# Maximum records retained globally (simple LRU-by-insertion-order eviction).
_MAX_RECORDS: int = 10_000


@dataclass
class MemoryRecord:
    """A single stored inference event for a user.

    Attributes:
        user_id: Owner of the record.
        observation_id: UUID of the source ``NormalizedObservation``.
        normalized_text: The (potentially truncated) observation text.
        signal_type: Top predicted signal type at storage time.
        confidence: Model confidence at storage time.
        created_at: UTC timestamp when the record was stored.
        score: Cosine-similarity score populated by ``retrieve()``
            (0.0 when the record is freshly stored).
    """

    user_id: UUID
    observation_id: UUID
    normalized_text: str
    signal_type: SignalType
    confidence: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    score: float = 0.0


class ContextMemoryStore:
    """Stores and retrieves per-user inference events via cosine similarity.

    Args:
        embed_fn: Callable that maps a text string to a ``List[float]``
            embedding vector.  Defaults to a lightweight bag-of-words fallback
            that does not require an external API.
        max_records: Maximum number of records retained across all users.
            Oldest records (by insertion order) are dropped when the limit is
            reached.
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        max_records: int = _MAX_RECORDS,
    ) -> None:
        """Initialise the store.

        Args:
            embed_fn: Embedding function.  If ``None`` the built-in
                bag-of-words fallback is used.
            max_records: Capacity limit before LRU eviction.
        """
        self._embed_fn: Callable[[str], List[float]] = embed_fn or _bow_embed
        self._max_records: int = max_records
        # Keyed by user_id for fast per-user lookup.
        self._records: Dict[str, List[MemoryRecord]] = {}
        self._embeddings: Dict[str, List[np.ndarray]] = {}
        self._total: int = 0

    async def store(
        self,
        user_id: UUID,
        observation: NormalizedObservation,
        inference: SignalInference,
    ) -> None:
        """Embed ``observation.normalized_text`` and upsert into the vector index.

        Only successful (non-abstained) inferences with a top prediction are
        stored.  The embed call is dispatched via ``asyncio.get_event_loop().
        run_in_executor`` so the async caller is not blocked if the embed
        function is synchronous.

        Args:
            user_id: Owner of the record.
            observation: Source observation whose text will be embedded.
            inference: Corresponding inference result.
        """
        if inference.top_prediction is None:
            return

        text: str = (observation.normalized_text or "")[:1200]
        if not text.strip():
            return

        uid: str = str(user_id)
        loop = asyncio.get_event_loop()
        vector: np.ndarray = np.array(
            await loop.run_in_executor(None, self._embed_fn, text),
            dtype=np.float32,
        )

        rec = MemoryRecord(
            user_id=user_id,
            observation_id=observation.id,
            normalized_text=text,
            signal_type=inference.top_prediction.signal_type,
            confidence=inference.top_prediction.probability,
        )

        if uid not in self._records:
            self._records[uid] = []
            self._embeddings[uid] = []

        self._records[uid].append(rec)
        self._embeddings[uid].append(vector)
        self._total += 1

        # Evict oldest record when capacity is exceeded.
        if self._total > self._max_records:
            for key in self._records:
                if self._records[key]:
                    self._records[key].pop(0)
                    self._embeddings[key].pop(0)
                    self._total -= 1
                    break

        logger.debug(
            "ContextMemoryStore.store: user=%s signal=%s total=%d",
            user_id,
            inference.top_prediction.signal_type.value,
            self._total,
        )

    async def retrieve(
        self,
        user_id: UUID,
        query_text: str,
        top_k: int = 5,
    ) -> List[MemoryRecord]:
        """Return the ``top_k`` most similar past records for ``user_id``.

        Similarity is cosine distance between the query embedding and each
        stored embedding.  Only records belonging to ``user_id`` are searched.

        Args:
            user_id: User whose history to search.
            query_text: Text to embed and compare against stored embeddings.
            top_k: Maximum number of records to return.

        Returns:
            List of ``MemoryRecord`` objects sorted by ``score`` descending,
            with ``score`` populated as the cosine similarity.  Returns an
            empty list when the user has no stored records.
        """
        uid: str = str(user_id)
        if uid not in self._records or not self._records[uid]:
            return []

        loop = asyncio.get_event_loop()
        query_vec: np.ndarray = np.array(
            await loop.run_in_executor(None, self._embed_fn, query_text),
            dtype=np.float32,
        )
        q_norm: float = float(np.linalg.norm(query_vec))
        if q_norm < 1e-9:
            return []

        scored: List[Tuple[float, MemoryRecord]] = []
        for vec, rec in zip(self._embeddings[uid], self._records[uid]):
            v_norm = float(np.linalg.norm(vec))
            if v_norm < 1e-9:
                continue
            similarity: float = float(np.dot(query_vec, vec) / (q_norm * v_norm))
            scored.append((similarity, rec))

        scored.sort(key=lambda t: t[0], reverse=True)
        results: List[MemoryRecord] = []
        for sim, rec in scored[:top_k]:
            import dataclasses
            result = dataclasses.replace(rec, score=sim)
            results.append(result)
        return results


# ---------------------------------------------------------------------------
# Built-in bag-of-words fallback embedding (no external API required)
# ---------------------------------------------------------------------------

_VOCAB_SIZE: int = 512  # Fixed dimension for reproducibility


def _bow_embed(text: str) -> List[float]:
    """Lightweight bag-of-words hashing embedding (512-dim, L2-normalised).

    This fallback is used when no ``embed_fn`` is injected.  It provides
    coarse semantic signal sufficient for development and tests.

    Args:
        text: Input string to embed.

    Returns:
        A 512-dimensional L2-normalised float list.
    """
    vec: List[float] = [0.0] * _VOCAB_SIZE
    tokens = text.lower().split()
    for token in tokens:
        idx = hash(token) % _VOCAB_SIZE
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]

