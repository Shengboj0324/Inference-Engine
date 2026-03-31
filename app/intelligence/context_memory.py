"""Per-user observation history, preference, and feedback stores.

``ContextMemoryStore`` persists per-user inference events as embedding vectors
and retrieves the nearest neighbours for a query text using cosine similarity.
It also maintains:

- **Preference store** — dismissed / acted-on signal-type counters that produce
  a per-user weight vector consumed by ``ActionRanker``.
- **Rolling inference history** — last 200 compressed ``SignalInference``
  summaries for each user (JSON-serialisable).
- **Rationale memory** — LLM rationale strings stored with their embeddings so
  the top-3 most semantically similar past rationales can be prepended as
  few-shot examples in the next adjudication prompt.
- **Adaptive noise filter thresholds** — per-user quality-score floor that
  tightens on high false-positive rates and loosens on recall drops.
- **Competitor alias registry** — user-defined name aliases merged into
  ``StrategicPriorities.competitors`` at inference time.
- **Preferred channel map** — ``SignalType`` → ``ResponseChannel`` learned from
  user action history.

``OutcomeFeedbackStore`` records per-``(user_id, signal_inference_id)`` outcome
labels (``OutcomeType``) and drives the adaptive threshold and federated
calibration updates.

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
import dataclasses
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np

from app.domain.inference_models import OutcomeType, SignalInference, SignalType
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

    #: Maximum compressed-inference records retained per user.
    _HISTORY_MAX: int = 200
    #: Maximum rationale records retained per user.
    _RATIONALE_MAX: int = 100
    #: Default per-user quality-score floor for the acquisition noise filter.
    _DEFAULT_NOISE_THRESHOLD: float = 0.3
    #: Hard lower bound for adaptive noise thresholds.
    _NOISE_MIN: float = 0.1
    #: Hard upper bound for adaptive noise thresholds.
    _NOISE_MAX: float = 0.7

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
        # Keyed by user_id (str) for fast per-user lookup.
        self._records: Dict[str, List[MemoryRecord]] = {}
        self._embeddings: Dict[str, List[np.ndarray]] = {}
        self._total: int = 0

        # ── Preference store ─────────────────────────────────────────────────
        # user_id → {signal_type.value → {"acted": int, "dismissed": int,
        #                                  "snoozed": int, "false_positive": int}}
        self._preferences: Dict[str, Dict[str, Dict[str, int]]] = {}

        # ── Rolling inference history (compressed) ───────────────────────────
        # user_id → list of dicts (JSON-serialisable), newest last
        self._inference_history: Dict[str, List[Dict[str, Any]]] = {}

        # ── Rationale memory (Step 4a) ────────────────────────────────────────
        # user_id → list of {"rationale": str, "signal_type": str,
        #                     "inferred_at": str, "embedding": List[float]}
        self._rationale_memory: Dict[str, List[Dict[str, Any]]] = {}

        # ── Adaptive noise thresholds (Step 4b) ──────────────────────────────
        # user_id → float (quality-score floor)
        self._noise_thresholds: Dict[str, float] = {}

        # ── Competitor aliases ────────────────────────────────────────────────
        # user_id → ordered unique list of alias strings
        self._competitor_aliases: Dict[str, List[str]] = {}

        # ── Preferred response channels ───────────────────────────────────────
        # user_id → {signal_type.value → channel_string}
        self._channel_prefs: Dict[str, Dict[str, str]] = {}

        # ── Source-embedding cache for drift detection (Step 4c) ─────────────
        # source_id → {"embedding": List[float], "seen_at": str}
        self._source_embeddings: Dict[str, Dict[str, Any]] = {}

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
            result = dataclasses.replace(rec, score=sim)
            results.append(result)
        return results

    # ── Preference store ──────────────────────────────────────────────────────

    def update_signal_preference(
        self,
        user_id: UUID,
        signal_type: SignalType,
        outcome: OutcomeType,
    ) -> None:
        """Record a user outcome against a specific ``signal_type``.

        Increments the outcome counter for ``signal_type`` so that
        ``get_signal_type_weights()`` can compute a dismissal-weighted priority
        vector.  All counters are JSON-serialisable integers.

        Args:
            user_id: Owner of the preference record.
            signal_type: Signal type the user acted on or dismissed.
            outcome: ``OutcomeType`` label provided by the user.
        """
        uid = str(user_id)
        st = signal_type.value
        if uid not in self._preferences:
            self._preferences[uid] = {}
        if st not in self._preferences[uid]:
            self._preferences[uid][st] = {
                "acted": 0, "dismissed": 0, "snoozed": 0, "false_positive": 0
            }
        key_map = {
            OutcomeType.ACTED_ON: "acted",
            OutcomeType.DISMISSED: "dismissed",
            OutcomeType.SNOOZED: "snoozed",
            OutcomeType.FALSE_POSITIVE: "false_positive",
        }
        self._preferences[uid][st][key_map[outcome]] += 1

    def get_signal_type_weights(self, user_id: UUID) -> Dict[str, float]:
        """Return a per-signal-type priority weight vector for ``user_id``.

        Weight formula::

            dismissal_frac = (dismissed + false_positive) /
                             max(1, acted + dismissed + snoozed + false_positive)
            weight = max(0.05, 1.0 - dismissal_frac * 0.95)

        A weight of ``1.0`` means "fully active"; ``0.05`` means "nearly
        suppressed".  Signal types with no feedback default to ``1.0``.

        Args:
            user_id: User whose preference vector to compute.

        Returns:
            Dict mapping ``SignalType.value`` strings to floats in ``[0.05, 1.0]``.
        """
        uid = str(user_id)
        prefs = self._preferences.get(uid, {})
        weights: Dict[str, float] = {}
        for st, counts in prefs.items():
            total = counts["acted"] + counts["dismissed"] + counts["snoozed"] + counts["false_positive"]
            neg = counts["dismissed"] + counts["false_positive"]
            dismissal_frac = neg / max(1, total)
            weights[st] = max(0.05, 1.0 - dismissal_frac * 0.95)
        return weights

    # ── Rolling inference history ─────────────────────────────────────────────

    def push_inference_result(
        self,
        user_id: UUID,
        signal_type: SignalType,
        probability: float,
        abstained: bool,
        inferred_at: datetime,
        source_platform: str,
    ) -> None:
        """Append a compressed ``SignalInference`` summary to the rolling history.

        Retains at most ``_HISTORY_MAX`` records per user (oldest evicted).
        All fields are JSON-serialisable primitives.

        Args:
            user_id: Owner of the history.
            signal_type: Top predicted signal type.
            probability: Calibrated probability of the top prediction.
            abstained: Whether the inference was abstained.
            inferred_at: UTC timestamp of the inference.
            source_platform: Platform value string (e.g. ``"reddit"``).
        """
        uid = str(user_id)
        if uid not in self._inference_history:
            self._inference_history[uid] = []
        record: Dict[str, Any] = {
            "signal_type": signal_type.value,
            "probability": round(float(probability), 4),
            "abstained": abstained,
            "inferred_at": inferred_at.isoformat(),
            "source_platform": source_platform,
        }
        history = self._inference_history[uid]
        history.append(record)
        if len(history) > self._HISTORY_MAX:
            history.pop(0)

    def get_inference_history(
        self, user_id: UUID, limit: int = 200
    ) -> List[Dict[str, Any]]:
        """Return the most recent compressed inference records for ``user_id``.

        Args:
            user_id: User whose history to retrieve.
            limit: Maximum number of records to return (newest last).

        Returns:
            List of dicts — newest last.  Empty list when no history exists.
        """
        uid = str(user_id)
        history = self._inference_history.get(uid, [])
        return history[-limit:]

    # ── Rationale memory (Step 4a) ────────────────────────────────────────────

    def store_rationale(
        self,
        user_id: UUID,
        rationale: str,
        signal_type: SignalType,
        embedding: List[float],
    ) -> None:
        """Store an LLM rationale string with its embedding for few-shot reuse.

        Retains at most ``_RATIONALE_MAX`` records per user; oldest evicted.

        Args:
            user_id: Owner of the record.
            rationale: Human-readable rationale string from ``SignalInference``.
            signal_type: Signal type that this rationale confirmed.
            embedding: Pre-computed embedding vector of ``rationale``.
        """
        uid = str(user_id)
        if uid not in self._rationale_memory:
            self._rationale_memory[uid] = []
        entry: Dict[str, Any] = {
            "rationale": rationale,
            "signal_type": signal_type.value,
            "inferred_at": datetime.now(timezone.utc).isoformat(),
            "embedding": [float(v) for v in embedding],
        }
        records = self._rationale_memory[uid]
        records.append(entry)
        if len(records) > self._RATIONALE_MAX:
            records.pop(0)

    def retrieve_similar_rationales(
        self,
        user_id: UUID,
        query_embedding: List[float],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return the top-k rationale records most similar to ``query_embedding``.

        Uses cosine similarity over the stored per-rationale embeddings.

        Args:
            user_id: User whose rationale memory to search.
            query_embedding: Embedding of the current observation text.
            top_k: Number of records to return.

        Returns:
            List of record dicts (with an added ``"score"`` key) sorted by
            cosine similarity descending.  Empty list if no records exist.
        """
        uid = str(user_id)
        records = self._rationale_memory.get(uid, [])
        if not records:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm < 1e-9:
            return []

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for rec in records:
            v = np.array(rec["embedding"], dtype=np.float32)
            v_norm = float(np.linalg.norm(v))
            if v_norm < 1e-9:
                continue
            sim = float(np.dot(q, v) / (q_norm * v_norm))
            scored.append((sim, rec))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [dict(rec, score=sim) for sim, rec in scored[:top_k]]

    # ── Adaptive noise-filter thresholds (Step 4b) ───────────────────────────

    def get_noise_threshold(self, user_id: UUID) -> float:
        """Return the per-user quality-score floor for the noise filter.

        Args:
            user_id: Target user.

        Returns:
            Float in ``[_NOISE_MIN, _NOISE_MAX]``; ``_DEFAULT_NOISE_THRESHOLD``
            for users with no recorded threshold.
        """
        return self._noise_thresholds.get(
            str(user_id), self._DEFAULT_NOISE_THRESHOLD
        )

    def update_noise_threshold(self, user_id: UUID, delta: float) -> None:
        """Adjust the per-user noise-filter threshold by ``delta``.

        Clamps the result to ``[_NOISE_MIN, _NOISE_MAX]``.  Positive ``delta``
        tightens the filter (fewer signals pass); negative loosens it.

        Args:
            user_id: Target user.
            delta: Additive change to the current threshold (e.g. ``+0.02``).
        """
        uid = str(user_id)
        current = self._noise_thresholds.get(uid, self._DEFAULT_NOISE_THRESHOLD)
        self._noise_thresholds[uid] = max(
            self._NOISE_MIN, min(self._NOISE_MAX, current + delta)
        )

    # ── Competitor aliases ────────────────────────────────────────────────────

    def add_competitor_alias(self, user_id: UUID, alias: str) -> None:
        """Add a competitor name alias for ``user_id``.

        Duplicate aliases (case-insensitive) are silently ignored.

        Args:
            user_id: Owner of the alias list.
            alias: Surface-form alias string (e.g. ``"MSFT"`` for Microsoft).
        """
        uid = str(user_id)
        if uid not in self._competitor_aliases:
            self._competitor_aliases[uid] = []
        lower = alias.lower()
        if not any(a.lower() == lower for a in self._competitor_aliases[uid]):
            self._competitor_aliases[uid].append(alias)

    def get_competitor_aliases(self, user_id: UUID) -> List[str]:
        """Return the user's registered competitor aliases.

        Args:
            user_id: Target user.

        Returns:
            List of alias strings (insertion order, deduplicated).
        """
        return list(self._competitor_aliases.get(str(user_id), []))

    # ── Preferred response channels ───────────────────────────────────────────

    def set_preferred_channel(
        self,
        user_id: UUID,
        signal_type: SignalType,
        channel: str,
    ) -> None:
        """Record that ``user_id`` prefers ``channel`` for ``signal_type``.

        Args:
            user_id: Owner.
            signal_type: Signal type whose preferred channel to set.
            channel: ``ResponseChannel`` string value (e.g. ``"direct_message"``).
        """
        uid = str(user_id)
        if uid not in self._channel_prefs:
            self._channel_prefs[uid] = {}
        self._channel_prefs[uid][signal_type.value] = channel

    def get_preferred_channels(self, user_id: UUID) -> Dict[str, str]:
        """Return the full preferred-channel map for ``user_id``.

        Args:
            user_id: Target user.

        Returns:
            Dict mapping ``SignalType.value`` strings to channel strings.
            Empty dict for users with no recorded preferences.
        """
        return dict(self._channel_prefs.get(str(user_id), {}))

    # ── Source-embedding cache for drift detection (Step 4c) ─────────────────

    def record_source_embedding(
        self,
        source_id: str,
        embedding: List[float],
    ) -> None:
        """Store the embedding for ``source_id`` (used for drift detection).

        Args:
            source_id: Platform source identifier (e.g. Reddit post ID).
            embedding: Normalised embedding vector of the observation text.
        """
        self._source_embeddings[source_id] = {
            "embedding": [float(v) for v in embedding],
            "seen_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_source_embedding(self, source_id: str) -> Optional[List[float]]:
        """Return the previously stored embedding for ``source_id``, or ``None``.

        Args:
            source_id: Platform source identifier.

        Returns:
            Embedding list or ``None`` if not previously seen.
        """
        entry = self._source_embeddings.get(source_id)
        if entry is None:
            return None
        return entry["embedding"]


# ---------------------------------------------------------------------------
# OutcomeFeedbackStore — records per-(user, inference) outcome labels
# ---------------------------------------------------------------------------


class OutcomeFeedbackStore:
    """Records user-provided outcome labels for delivered ``SignalInference``\\s.

    This store is the primary feedback signal for two adaptive subsystems:

    1. **Adaptive noise filter thresholds** — when the per-user false-positive
       rate in a recent window exceeds 20 %, ``record_outcome()`` calls
       ``context_memory.update_noise_threshold(user_id, +0.02)`` to tighten
       the quality-score floor.  If the FP rate is below 5 % (indicating good
       recall), the threshold is loosened by ``-0.02``.

    2. **Federated calibration** — callers should invoke
       ``ConfidenceCalibrator.update_user()`` with ``true_label=False`` whenever
       ``outcome == OutcomeType.FALSE_POSITIVE`` to drive the per-user
       temperature scalar upward (reducing that signal type's probability on
       subsequent inferences).

    All stored data is JSON-serialisable (strings, floats, booleans, ISO
    timestamps).

    Args:
        batch_size: Number of outcomes after which the adaptive threshold check
            is triggered.  Defaults to ``5`` (checked every 5 outcomes per user).
    """

    def __init__(self, batch_size: int = 5) -> None:
        """Initialise the feedback store.

        Args:
            batch_size: Adaptive-threshold trigger frequency per user.
        """
        # user_id_str → {inference_id_str → outcome_record}
        self._outcomes: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._batch_size: int = batch_size

    def record_outcome(
        self,
        user_id: UUID,
        signal_inference_id: UUID,
        outcome: OutcomeType,
        channel_used: Optional[str] = None,
        acted_at: Optional[datetime] = None,
        context_memory: Optional["ContextMemoryStore"] = None,
    ) -> None:
        """Record a user outcome for a delivered ``SignalInference``.

        Stores the outcome and — every ``batch_size`` outcomes — re-evaluates
        the per-user false-positive rate to potentially adjust the adaptive
        noise-filter threshold via ``context_memory``.

        Args:
            user_id: User who provided the feedback.
            signal_inference_id: UUID of the ``SignalInference`` being rated.
            outcome: User's outcome label (``OutcomeType``).
            channel_used: ``ResponseChannel`` string if the user acted on the
                signal (``None`` for dismissed / false-positive outcomes).
            acted_at: UTC timestamp of the outcome.  Defaults to now.
            context_memory: ``ContextMemoryStore`` whose adaptive threshold
                will be adjusted when the FP rate exceeds 20 % or drops
                below 5 %.  Pass ``None`` to skip threshold adjustment.
        """
        uid = str(user_id)
        iid = str(signal_inference_id)
        if uid not in self._outcomes:
            self._outcomes[uid] = {}

        record: Dict[str, Any] = {
            "outcome": outcome.value,
            "channel_used": channel_used,
            "acted_at": (acted_at or datetime.now(timezone.utc)).isoformat(),
            "inference_id": iid,
        }
        self._outcomes[uid][iid] = record

        # Adaptive threshold adjustment every batch_size outcomes
        if context_memory is not None:
            all_records = list(self._outcomes[uid].values())
            if len(all_records) % self._batch_size == 0:
                recent = all_records[-self._batch_size:]
                fp_count = sum(
                    1 for r in recent
                    if r["outcome"] == OutcomeType.FALSE_POSITIVE.value
                )
                fp_rate = fp_count / len(recent)
                if fp_rate > 0.20:
                    context_memory.update_noise_threshold(user_id, +0.02)
                    logger.debug(
                        "OutcomeFeedbackStore: FP rate=%.0f%% → tightened "
                        "noise threshold for user %s by +0.02",
                        fp_rate * 100, user_id,
                    )
                elif fp_rate < 0.05:
                    context_memory.update_noise_threshold(user_id, -0.02)
                    logger.debug(
                        "OutcomeFeedbackStore: FP rate=%.0f%% → loosened "
                        "noise threshold for user %s by -0.02",
                        fp_rate * 100, user_id,
                    )

    def get_outcome(
        self, user_id: UUID, signal_inference_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """Return the recorded outcome for ``(user_id, signal_inference_id)``.

        Args:
            user_id: User who provided the outcome.
            signal_inference_id: UUID of the ``SignalInference``.

        Returns:
            Dict with ``outcome``, ``channel_used``, ``acted_at`` keys, or
            ``None`` if no outcome has been recorded.
        """
        uid = str(user_id)
        iid = str(signal_inference_id)
        return self._outcomes.get(uid, {}).get(iid)

    def get_false_positive_rate(self, user_id: UUID, window: int = 100) -> float:
        """Return the fraction of false-positive outcomes in the last ``window``.

        Args:
            user_id: Target user.
            window: How many most-recent outcomes to consider.

        Returns:
            Float in ``[0.0, 1.0]``.  ``0.0`` when the user has no outcomes.
        """
        uid = str(user_id)
        all_records = list(self._outcomes.get(uid, {}).values())
        if not all_records:
            return 0.0
        recent = all_records[-window:]
        fp_count = sum(
            1 for r in recent
            if r["outcome"] == OutcomeType.FALSE_POSITIVE.value
        )
        return fp_count / len(recent)

    def get_recent_outcomes(
        self, user_id: UUID, window: int = 100
    ) -> List[Dict[str, Any]]:
        """Return the most recent ``window`` outcome records for ``user_id``.

        Args:
            user_id: Target user.
            window: Maximum number of records to return (newest last).

        Returns:
            List of outcome dicts; empty list if no outcomes recorded.
        """
        uid = str(user_id)
        all_records = list(self._outcomes.get(uid, {}).values())
        return all_records[-window:]


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

