"""Per-user observation history, preference, and feedback stores.

``ContextMemoryStore`` persists per-user inference events as embedding vectors
and retrieves the nearest neighbours for a query text using cosine similarity.
It also maintains:

- **Preference store** — dismissed / acted-on signal-type counters that produce
  a per-user weight vector consumed by ``ActionRanker``.
- **Rolling inference history** — last 200 compressed ``SignalInference``
  summaries for each user (JSON-serialisable); entries older than
  ``_HISTORY_TTL_DAYS`` days are automatically expired.
- **Rationale memory** — LLM rationale strings stored with their embeddings so
  the top-3 most semantically similar past rationales can be prepended as
  few-shot examples in the next adjudication prompt; entries older than
  ``_RATIONALE_TTL_DAYS`` days are automatically expired.
- **Adaptive noise filter thresholds** — per-user quality-score floor that
  tightens on high false-positive rates and loosens on recall drops.
- **Competitor alias registry** — user-defined name aliases merged into
  ``StrategicPriorities.competitors`` at inference time.
- **Preferred channel map** — ``SignalType`` → ``ResponseChannel`` learned from
  user action history.
- **Persistence** — the core per-user vector memory (observation records and
  their embeddings) plus all preference, history, profile, and metadata state
  are flushed to / restored from a JSON file via :meth:`persist` /
  :meth:`load_from_disk`, so semantic recall survives a process restart.
  Snapshot format ``"1.1"`` adds the ``records`` / ``embeddings`` buckets;
  older ``"1.0"`` snapshots still load (the new buckets default to empty).

``OutcomeFeedbackStore`` records per-``(user_id, signal_inference_id)`` outcome
labels (``OutcomeType``) and drives the adaptive threshold and federated
calibration updates.  Duplicate delivery (at-least-once semantics) is guarded
by an inference-id set — re-delivering the same ``signal_inference_id`` is a
no-op that does not double-count.

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
    store.persist(Path("data/context_memory.json"))
"""

import asyncio
import dataclasses
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import UUID

import numpy as np

from app.domain.inference_models import OutcomeType, SignalInference, SignalType
from app.domain.normalized_models import NormalizedObservation

logger = logging.getLogger(__name__)

# Maximum records retained globally (simple LRU-by-insertion-order eviction).
_MAX_RECORDS: int = 10_000

#: Number of days after which inference history entries are considered stale
#: and removed on the next write to that user's history bucket.
_HISTORY_TTL_DAYS: int = 90

#: Number of days after which rationale memory entries are considered stale
#: and removed on the next retrieval / write for that user's bucket.
_RATIONALE_TTL_DAYS: int = 180


class MemoryLeakError(Exception):
    """Raised when a memory read is attempted across user boundaries.

    The ``ContextMemoryStore`` is partitioned per ``user_id``.  Any read that
    supplies ``requesting_user_id`` different from the queried ``user_id`` is
    a security boundary violation and is rejected.  The audit log captures
    both ids and the attempted operation for downstream investigation.
    """


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

    All public write methods are thread-safe via a single ``threading.Lock``.
    State can be serialised to / restored from disk via :meth:`persist` /
    :meth:`load_from_disk`.

    Args:
        embed_fn: Callable that maps a text string to a ``List[float]``
            embedding vector.  Defaults to a lightweight bag-of-words fallback
            that does not require an external API.
        max_records: Maximum number of records retained across all users.
            Oldest records (by insertion order) are dropped when the limit is
            reached.
        history_ttl_days: Inference history entries older than this many days
            are pruned on the next write to that user's bucket.
        rationale_ttl_days: Rationale memory entries older than this many days
            are pruned on retrieval / the next write to that user's bucket.
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
        history_ttl_days: int = _HISTORY_TTL_DAYS,
        rationale_ttl_days: int = _RATIONALE_TTL_DAYS,
    ) -> None:
        """Initialise the store.

        Args:
            embed_fn: Embedding function.  If ``None`` the built-in
                bag-of-words fallback is used.
            max_records: Capacity limit before LRU eviction.
            history_ttl_days: TTL in days for inference history entries.
            rationale_ttl_days: TTL in days for rationale memory entries.
        """
        self._embed_fn: Callable[[str], List[float]] = embed_fn or _bow_embed
        self._max_records: int = max_records
        self._history_ttl_days: int = history_ttl_days
        self._rationale_ttl_days: int = rationale_ttl_days

        # Single non-reentrant lock protecting all mutable state.  All public
        # methods acquire it at most once and never call another locked method
        # while holding it (e.g. OutcomeFeedbackStore performs threshold
        # updates outside the lock), so a plain Lock is sufficient and avoids
        # masking accidental re-entrancy.
        self._lock: threading.Lock = threading.Lock()

        # Keyed by user_id (str) for fast per-user lookup.
        self._records: Dict[str, List[MemoryRecord]] = {}
        self._embeddings: Dict[str, List[np.ndarray]] = {}
        self._total: int = 0

        # ── Preference store ─────────────────────────────────────────────────
        # user_id → {signal_type.value → {"acted": int, "dismissed": int,
        #                                  "snoozed": int, "false_positive": int}}
        self._preferences: Dict[str, Dict[str, Dict[str, int]]] = {}

        # ── Rolling inference history (compressed) ───────────────────────────
        # user_id → list of dicts (JSON-serialisable), newest last.
        # Each dict has a mandatory "inferred_at" ISO-8601 key for TTL checks.
        self._inference_history: Dict[str, List[Dict[str, Any]]] = {}

        # ── Idempotency set for inference history ─────────────────────────────
        # user_id → set of inference_id strings already pushed.
        # Prevents double-counting on at-least-once message delivery.
        self._history_seen: Dict[str, Set[str]] = {}

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

        # ── Free-form user profile (Problem 4: detail retention) ─────────────
        # user_id → {"display_name": str, "interests": List[str],
        #            "preferences": Dict[str, Any], "updated_at": ISO-8601}
        # Every textual value is PII-scrubbed via DataResidencyGuard before
        # being written so the profile is safe to serialise and export.
        self._user_profiles: Dict[str, Dict[str, Any]] = {}

        # ── Dedicated per-user persona memory ────────────────────────────────
        # user_id → serialised UserPersonaProfile dict (see
        # app.personalization.user_persona).  Captures learned communication /
        # reasoning style, hobbies, stated characteristics, and acquisition
        # preferences.  Persisted alongside the rest of the per-user state so a
        # user's personalization survives restarts.
        self._user_personas: Dict[str, Dict[str, Any]] = {}

        # ── Optional similarity → probability calibrator (Tier 1.3) ──────────
        # When set, retrieve() maps each raw cosine similarity through the
        # calibrator so ``min_score`` is a calibrated relevance probability and
        # is comparable across embedding models.  ``None`` → raw cosine (default).
        self._sim_calibrator: Any = None

        # ── Embedding-version stamp (Tier 3.1) ───────────────────────────────
        # Records the embedding-space version the stored vectors were computed
        # under, so a model upgrade can be detected (see ``needs_reembed``).
        self._embedding_version: Optional[str] = None

        # ── Cross-user access audit trail ────────────────────────────────────
        # Append-only ring buffer of dicts describing every rejected read.
        # Capped at 1024 entries to bound memory growth.
        self._leak_audit: List[Dict[str, Any]] = []

    async def store(
        self,
        user_id: UUID,
        observation: NormalizedObservation,
        inference: SignalInference,
    ) -> None:
        """Embed ``observation.normalized_text`` and upsert into the vector index.

        Only successful (non-abstained) inferences with a top prediction are
        stored.  The embed call is dispatched via
        ``asyncio.get_running_loop().run_in_executor`` so the async caller is
        not blocked if the embed function is synchronous.

        Args:
            user_id: Owner of the record.
            observation: Source observation whose text will be embedded.
            inference: Corresponding inference result.
        """
        self._validate_user_id(user_id, op="store")
        if inference.top_prediction is None:
            return

        text: str = (observation.normalized_text or "")[:1200]
        if not text.strip():
            return

        # Canonical zero-egress scrub: never persist raw PII / secrets in the
        # user-scoped memory store, even when the upstream connector failed
        # to redact.  Logs the count when something was redacted.
        from app.core.data_residency import DataResidencyGuard
        scrubbed_text, _pii_n = DataResidencyGuard.scrub_text(text)
        if _pii_n > 0:
            logger.info(
                "ContextMemoryStore.store: scrubbed %d PII pattern(s) "
                "from observation text (user=%s)",
                _pii_n, user_id,
            )
        text = scrubbed_text

        uid: str = str(user_id)
        _t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
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

        with self._lock:
            if uid not in self._records:
                self._records[uid] = []
                self._embeddings[uid] = []

            self._records[uid].append(rec)
            self._embeddings[uid].append(vector)
            self._total += 1

            # Evict the GLOBALLY-oldest record(s) when capacity is exceeded.
            #
            # Records are appended in chronological order within each per-user
            # bucket, so the eviction candidate for a bucket is always its
            # front element (index 0).  The previous implementation popped the
            # front of the *first non-empty bucket in dict-iteration order*,
            # which under skewed multi-user load could evict a recent record
            # from one user while keeping a far older record from another.
            # We instead select the bucket whose front record has the smallest
            # ``created_at`` and evict that, guaranteeing true global-oldest
            # eviction and keeping ``_records`` / ``_embeddings`` in lockstep.
            while self._total > self._max_records:
                victim_uid: Optional[str] = None
                victim_ts: Optional[datetime] = None
                for key, recs in self._records.items():
                    if not recs:
                        continue
                    front_ts = recs[0].created_at
                    if victim_ts is None or front_ts < victim_ts:
                        victim_ts = front_ts
                        victim_uid = key
                if victim_uid is None:  # pragma: no cover - all buckets empty
                    break
                self._records[victim_uid].pop(0)
                self._embeddings[victim_uid].pop(0)
                self._total -= 1

        _elapsed_ms = (time.perf_counter() - _t0) * 1000
        logger.debug(
            "ContextMemoryStore.store: user=%s signal=%s total=%d latency_ms=%.1f",
            user_id,
            inference.top_prediction.signal_type.value,
            self._total,
            _elapsed_ms,
        )

    async def retrieve(
        self,
        user_id: UUID,
        query_text: str,
        top_k: int = 5,
        requesting_user_id: Optional[UUID] = None,
        recency_half_life_days: Optional[float] = None,
        signal_types: Optional[Set[SignalType]] = None,
        min_score: float = 0.0,
        mmr_lambda: Optional[float] = None,
    ) -> List[MemoryRecord]:
        """Return the ``top_k`` most similar past records for ``user_id``.

        Similarity is cosine distance between the query embedding and each
        stored embedding.  Only records belonging to ``user_id`` are searched.

        Args:
            user_id: User whose history to search.
            query_text: Text to embed and compare against stored embeddings.
            top_k: Maximum number of records to return.
            requesting_user_id: When supplied, must equal ``user_id``.  If it
                differs, the call is rejected with :class:`MemoryLeakError`
                and an audit entry is appended.  This is the canonical hook
                that callers (API layer, agents) should use to enforce
                cross-user isolation.
            recency_half_life_days: When set, each record's cosine similarity
                is multiplied by an exponential recency weight
                ``0.5 ** (age_days / half_life)`` so that, among
                similarly-relevant memories, more recent ones rank higher.
                ``None`` (default) preserves the original pure-cosine ranking.
            signal_types: Optional whitelist of ``SignalType``\\s; records whose
                ``signal_type`` is not in the set are excluded.  ``None`` keeps
                all types (default).
            min_score: Records whose final score is below this threshold are
                dropped.  Defaults to ``0.0`` (keep everything non-negative).

        Returns:
            List of ``MemoryRecord`` objects sorted by ``score`` descending.
            ``score`` is the cosine similarity, or the recency-weighted score
            when ``recency_half_life_days`` is supplied.  Returns an empty list
            when the user has no stored records.

        Raises:
            MemoryLeakError: If ``requesting_user_id`` is set and does not
                match ``user_id``.
        """
        self._validate_user_id(user_id, op="retrieve")
        if requesting_user_id is not None:
            self._validate_user_id(requesting_user_id, op="retrieve.requesting")
            if str(requesting_user_id) != str(user_id):
                self._record_leak_attempt(
                    op="retrieve",
                    queried=user_id,
                    requesting=requesting_user_id,
                )
                raise MemoryLeakError(
                    f"cross-user memory access denied: requesting_user_id="
                    f"{requesting_user_id} != user_id={user_id}"
                )
        uid: str = str(user_id)
        with self._lock:
            if uid not in self._records or not self._records[uid]:
                return []
            records_snapshot = list(self._records[uid])
            embeddings_snapshot = list(self._embeddings[uid])

        _t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        query_vec: np.ndarray = np.array(
            await loop.run_in_executor(None, self._embed_fn, query_text),
            dtype=np.float32,
        )
        q_norm: float = float(np.linalg.norm(query_vec))
        if q_norm < 1e-9:
            return []

        _now = datetime.now(timezone.utc)
        _calibrator = self._sim_calibrator
        scored: List[Tuple[float, MemoryRecord, np.ndarray]] = []
        for vec, rec in zip(embeddings_snapshot, records_snapshot):
            if signal_types is not None and rec.signal_type not in signal_types:
                continue
            v_norm = float(np.linalg.norm(vec))
            if v_norm < 1e-9:
                continue
            unit_vec = vec / v_norm
            similarity: float = float(np.dot(query_vec, vec) / (q_norm * v_norm))
            # Tier 1.3 — map raw cosine to a calibrated relevance probability so
            # ``min_score`` is portable across embedding models (opt-in).
            if _calibrator is not None:
                similarity = float(_calibrator.transform(similarity))
            score = similarity
            if recency_half_life_days is not None and recency_half_life_days > 0:
                created = rec.created_at
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_days = max(0.0, (_now - created).total_seconds() / 86400.0)
                recency_weight = 0.5 ** (age_days / recency_half_life_days)
                score = similarity * recency_weight
            if score < min_score:
                continue
            scored.append((score, rec, unit_vec))

        scored.sort(key=lambda t: t[0], reverse=True)
        results: List[MemoryRecord] = []
        if mmr_lambda is not None and 0.0 <= mmr_lambda <= 1.0 and scored:
            # Tier 3.3 — Maximal Marginal Relevance: greedily pick the candidate
            # maximising ``score - λ · max cosine-similarity to already-picked``,
            # so near-duplicate memories don't crowd out complementary context.
            # ``λ=0`` reduces to pure relevance; higher λ favours diversity.
            remaining = list(scored)
            selected_units: List[np.ndarray] = []
            first = remaining.pop(0)
            results.append(dataclasses.replace(first[1], score=first[0]))
            selected_units.append(first[2])
            while remaining and len(results) < top_k:
                best_idx, best_mmr = 0, None
                for i, (s, _rec, uv) in enumerate(remaining):
                    redundancy = max(float(np.dot(uv, su)) for su in selected_units)
                    mmr = s - mmr_lambda * redundancy
                    if best_mmr is None or mmr > best_mmr:
                        best_mmr, best_idx = mmr, i
                s, rec, uv = remaining.pop(best_idx)
                results.append(dataclasses.replace(rec, score=s))
                selected_units.append(uv)
        else:
            for sim, rec, _uv in scored[:top_k]:
                results.append(dataclasses.replace(rec, score=sim))

        _elapsed_ms = (time.perf_counter() - _t0) * 1000
        logger.debug(
            "ContextMemoryStore.retrieve: user=%s top_k=%d results=%d latency_ms=%.1f",
            user_id, top_k, len(results), _elapsed_ms,
        )
        return results

    # ── Preference store ──────────────────────────────────────────────────────

    def update_signal_preference(
        self,
        user_id: UUID,
        signal_type: SignalType,
        outcome: OutcomeType,
    ) -> None:
        """Record a user outcome against a specific ``signal_type``.

        Thread-safe.  Increments the outcome counter for ``signal_type`` so
        that ``get_signal_type_weights()`` can compute a dismissal-weighted
        priority vector.  All counters are JSON-serialisable integers.

        Args:
            user_id: Owner of the preference record.
            signal_type: Signal type the user acted on or dismissed.
            outcome: ``OutcomeType`` label provided by the user.
        """
        uid = str(user_id)
        st = signal_type.value
        key_map = {
            OutcomeType.ACTED_ON: "acted",
            OutcomeType.DISMISSED: "dismissed",
            OutcomeType.SNOOZED: "snoozed",
            OutcomeType.FALSE_POSITIVE: "false_positive",
        }
        _t0 = time.perf_counter()
        with self._lock:
            if uid not in self._preferences:
                self._preferences[uid] = {}
            if st not in self._preferences[uid]:
                self._preferences[uid][st] = {
                    "acted": 0, "dismissed": 0, "snoozed": 0, "false_positive": 0
                }
            self._preferences[uid][st][key_map[outcome]] += 1
        logger.debug(
            "ContextMemoryStore.update_signal_preference: user=%s signal=%s "
            "outcome=%s latency_ms=%.2f",
            user_id, st, outcome.value, (time.perf_counter() - _t0) * 1000,
        )

    def get_signal_type_weights(self, user_id: UUID) -> Dict[str, float]:
        """Return a per-signal-type priority weight vector for ``user_id``.

        Thread-safe.

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
        with self._lock:
            prefs_copy = {
                st: dict(counts)
                for st, counts in self._preferences.get(uid, {}).items()
            }
        weights: Dict[str, float] = {}
        for st, counts in prefs_copy.items():
            total = counts["acted"] + counts["dismissed"] + counts["snoozed"] + counts["false_positive"]
            neg = counts["dismissed"] + counts["false_positive"]
            dismissal_frac = neg / max(1, total)
            weights[st] = max(0.05, 1.0 - dismissal_frac * 0.95)
        logger.debug(
            "ContextMemoryStore.get_signal_type_weights: user=%s count=%d",
            user_id, len(weights),
        )
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
        inference_id: Optional[str] = None,
    ) -> None:
        """Append a compressed ``SignalInference`` summary to the rolling history.

        Thread-safe.  Idempotent: if ``inference_id`` is supplied and has
        already been recorded for this user, the call is a no-op (prevents
        double-counting on at-least-once message delivery).

        TTL: entries older than ``history_ttl_days`` are pruned before the new
        record is appended.

        Retains at most ``_HISTORY_MAX`` records per user (oldest evicted when
        both the TTL scan and the cap are insufficient).

        Args:
            user_id: Owner of the history.
            signal_type: Top predicted signal type.
            probability: Calibrated probability of the top prediction.
            abstained: Whether the inference was abstained.
            inferred_at: UTC timestamp of the inference.
            source_platform: Platform value string (e.g. ``"reddit"``).
            inference_id: Optional unique identifier for idempotency checks.
        """
        # Validate inputs at the boundary
        if not isinstance(signal_type, SignalType):
            raise TypeError(f"signal_type must be SignalType, got {type(signal_type)}")
        if not (0.0 <= probability <= 1.0):
            raise ValueError(f"probability must be in [0, 1], got {probability}")

        uid = str(user_id)
        _ttl_cutoff = datetime.now(timezone.utc) - timedelta(days=self._history_ttl_days)
        _t0 = time.perf_counter()

        with self._lock:
            # Idempotency guard
            if inference_id is not None:
                seen = self._history_seen.setdefault(uid, set())
                if inference_id in seen:
                    logger.debug(
                        "ContextMemoryStore.push_inference_result: duplicate "
                        "inference_id=%s for user=%s — skipped",
                        inference_id, user_id,
                    )
                    return
                seen.add(inference_id)

            if uid not in self._inference_history:
                self._inference_history[uid] = []
            history = self._inference_history[uid]

            # TTL-based expiry: remove entries older than _history_ttl_days
            _ttl_cutoff_iso = _ttl_cutoff.isoformat()
            history[:] = [
                r for r in history
                if r.get("inferred_at", "") >= _ttl_cutoff_iso
            ]

            record: Dict[str, Any] = {
                "signal_type": signal_type.value,
                "probability": round(float(probability), 4),
                "abstained": abstained,
                "inferred_at": inferred_at.isoformat(),
                "source_platform": source_platform,
            }
            if inference_id is not None:
                record["inference_id"] = inference_id
            history.append(record)
            if len(history) > self._HISTORY_MAX:
                history.pop(0)
            _size = len(history)

        logger.debug(
            "ContextMemoryStore.push_inference_result: user=%s signal=%s "
            "history_size=%d latency_ms=%.2f",
            user_id, signal_type.value, _size, (time.perf_counter() - _t0) * 1000,
        )

    def get_inference_history(
        self, user_id: UUID, limit: int = 200
    ) -> List[Dict[str, Any]]:
        """Return the most recent compressed inference records for ``user_id``.

        Thread-safe.

        Args:
            user_id: User whose history to retrieve.
            limit: Maximum number of records to return (newest last).

        Returns:
            List of dicts — newest last.  Empty list when no history exists.
        """
        uid = str(user_id)
        _ttl_cutoff_iso = (
            datetime.now(timezone.utc) - timedelta(days=self._history_ttl_days)
        ).isoformat()
        with self._lock:
            # Exclude TTL-expired entries on read as well as on write, so a
            # long-idle user's bucket (no recent writes to trigger pruning)
            # never returns stale history.
            history = [
                r for r in self._inference_history.get(uid, [])
                if r.get("inferred_at", "") >= _ttl_cutoff_iso
            ]
        logger.debug(
            "ContextMemoryStore.get_inference_history: user=%s limit=%d found=%d",
            user_id, limit, min(limit, len(history)),
        )
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

        Thread-safe.  TTL: entries older than ``rationale_ttl_days`` are pruned
        before the new record is appended.  Retains at most ``_RATIONALE_MAX``
        records per user; oldest evicted when the TTL scan is insufficient.

        Args:
            user_id: Owner of the record.
            rationale: Human-readable rationale string from ``SignalInference``.
            signal_type: Signal type that this rationale confirmed.
            embedding: Pre-computed embedding vector of ``rationale``.
        """
        uid = str(user_id)
        now_iso = datetime.now(timezone.utc).isoformat()
        _ttl_cutoff_iso = (
            datetime.now(timezone.utc) - timedelta(days=self._rationale_ttl_days)
        ).isoformat()
        _t0 = time.perf_counter()
        with self._lock:
            if uid not in self._rationale_memory:
                self._rationale_memory[uid] = []
            records = self._rationale_memory[uid]
            # TTL expiry
            records[:] = [
                r for r in records
                if r.get("inferred_at", "") >= _ttl_cutoff_iso
            ]
            entry: Dict[str, Any] = {
                "rationale": rationale,
                "signal_type": signal_type.value,
                "inferred_at": now_iso,
                "embedding": [float(v) for v in embedding],
            }
            records.append(entry)
            if len(records) > self._RATIONALE_MAX:
                records.pop(0)
            _size = len(records)
        logger.debug(
            "ContextMemoryStore.store_rationale: user=%s signal=%s "
            "rationale_count=%d latency_ms=%.2f",
            user_id, signal_type.value, _size, (time.perf_counter() - _t0) * 1000,
        )

    def retrieve_similar_rationales(
        self,
        user_id: UUID,
        query_embedding: List[float],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return the top-k rationale records most similar to ``query_embedding``.

        Thread-safe.  Uses cosine similarity over the stored per-rationale
        embeddings.  Stale entries (beyond TTL) are silently excluded from
        results by filtering at read time (they are also pruned on the next
        write).

        Args:
            user_id: User whose rationale memory to search.
            query_embedding: Embedding of the current observation text.
            top_k: Number of records to return.

        Returns:
            List of record dicts (with an added ``"score"`` key) sorted by
            cosine similarity descending.  Empty list if no records exist.
        """
        uid = str(user_id)
        _ttl_cutoff_iso = (
            datetime.now(timezone.utc) - timedelta(days=self._rationale_ttl_days)
        ).isoformat()
        _t0 = time.perf_counter()
        with self._lock:
            records_snapshot = [
                dict(r) for r in self._rationale_memory.get(uid, [])
                if r.get("inferred_at", "") >= _ttl_cutoff_iso
            ]
        if not records_snapshot:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm < 1e-9:
            return []

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for rec in records_snapshot:
            v = np.array(rec["embedding"], dtype=np.float32)
            v_norm = float(np.linalg.norm(v))
            if v_norm < 1e-9:
                continue
            sim = float(np.dot(q, v) / (q_norm * v_norm))
            scored.append((sim, rec))

        scored.sort(key=lambda t: t[0], reverse=True)
        results = [dict(rec, score=sim) for sim, rec in scored[:top_k]]
        logger.debug(
            "ContextMemoryStore.retrieve_similar_rationales: user=%s "
            "top_k=%d results=%d latency_ms=%.2f",
            user_id, top_k, len(results), (time.perf_counter() - _t0) * 1000,
        )
        return results

    # ── Adaptive noise-filter thresholds (Step 4b) ───────────────────────────

    def get_noise_threshold(self, user_id: UUID) -> float:
        """Return the per-user quality-score floor for the noise filter.

        Thread-safe.

        Args:
            user_id: Target user.

        Returns:
            Float in ``[_NOISE_MIN, _NOISE_MAX]``; ``_DEFAULT_NOISE_THRESHOLD``
            for users with no recorded threshold.
        """
        with self._lock:
            return self._noise_thresholds.get(
                str(user_id), self._DEFAULT_NOISE_THRESHOLD
            )

    def update_noise_threshold(self, user_id: UUID, delta: float) -> None:
        """Adjust the per-user noise-filter threshold by ``delta``.

        Thread-safe.  Clamps the result to ``[_NOISE_MIN, _NOISE_MAX]``.
        Positive ``delta`` tightens the filter (fewer signals pass); negative
        loosens it.

        Args:
            user_id: Target user.
            delta: Additive change to the current threshold (e.g. ``+0.02``).
        """
        uid = str(user_id)
        with self._lock:
            current = self._noise_thresholds.get(uid, self._DEFAULT_NOISE_THRESHOLD)
            self._noise_thresholds[uid] = max(
                self._NOISE_MIN, min(self._NOISE_MAX, current + delta)
            )
        logger.debug(
            "ContextMemoryStore.update_noise_threshold: user=%s delta=%.4f new=%.4f",
            user_id, delta, self._noise_thresholds[uid],
        )

    # ── Competitor aliases ────────────────────────────────────────────────────

    def add_competitor_alias(self, user_id: UUID, alias: str) -> None:
        """Add a competitor name alias for ``user_id``.

        Thread-safe.  Duplicate aliases (case-insensitive) are silently ignored.

        Args:
            user_id: Owner of the alias list.
            alias: Surface-form alias string (e.g. ``"MSFT"`` for Microsoft).
        """
        uid = str(user_id)
        with self._lock:
            if uid not in self._competitor_aliases:
                self._competitor_aliases[uid] = []
            lower = alias.lower()
            if not any(a.lower() == lower for a in self._competitor_aliases[uid]):
                self._competitor_aliases[uid].append(alias)

    def get_competitor_aliases(self, user_id: UUID) -> List[str]:
        """Return the user's registered competitor aliases.

        Thread-safe.

        Args:
            user_id: Target user.

        Returns:
            List of alias strings (insertion order, deduplicated).
        """
        with self._lock:
            return list(self._competitor_aliases.get(str(user_id), []))

    # ── Preferred response channels ───────────────────────────────────────────

    def set_preferred_channel(
        self,
        user_id: UUID,
        signal_type: SignalType,
        channel: str,
    ) -> None:
        """Record that ``user_id`` prefers ``channel`` for ``signal_type``.

        Thread-safe.

        Args:
            user_id: Owner.
            signal_type: Signal type whose preferred channel to set.
            channel: ``ResponseChannel`` string value (e.g. ``"direct_message"``).
        """
        uid = str(user_id)
        with self._lock:
            if uid not in self._channel_prefs:
                self._channel_prefs[uid] = {}
            self._channel_prefs[uid][signal_type.value] = channel

    def get_preferred_channels(self, user_id: UUID) -> Dict[str, str]:
        """Return the full preferred-channel map for ``user_id``.

        Thread-safe.

        Args:
            user_id: Target user.

        Returns:
            Dict mapping ``SignalType.value`` strings to channel strings.
            Empty dict for users with no recorded preferences.
        """
        with self._lock:
            return dict(self._channel_prefs.get(str(user_id), {}))

    # ── Source-embedding cache for drift detection (Step 4c) ─────────────────

    def record_source_embedding(
        self,
        source_id: str,
        embedding: List[float],
    ) -> None:
        """Store the embedding for ``source_id`` (used for drift detection).

        Thread-safe.

        Args:
            source_id: Platform source identifier (e.g. Reddit post ID).
            embedding: Normalised embedding vector of the observation text.
        """
        with self._lock:
            self._source_embeddings[source_id] = {
                "embedding": [float(v) for v in embedding],
                "seen_at": datetime.now(timezone.utc).isoformat(),
            }

    def get_source_embedding(self, source_id: str) -> Optional[List[float]]:
        """Return the previously stored embedding for ``source_id``, or ``None``.

        Thread-safe.

        Args:
            source_id: Platform source identifier.

        Returns:
            Embedding list or ``None`` if not previously seen.
        """
        with self._lock:
            entry = self._source_embeddings.get(source_id)
        if entry is None:
            return None
        return entry["embedding"]

    # ── Persistence ───────────────────────────────────────────────────────────

    def persist(self, path: Path) -> None:
        """Serialise all mutable state to ``path`` as a JSON file.

        The write is atomic: the payload is first written to a sibling
        ``<path>.tmp`` file and then renamed over ``path``.  This prevents
        partial writes from corrupting the persisted state on crash.

        Thread-safe.

        Args:
            path: Destination file path (created / overwritten).
        """
        path = Path(path)
        _t0 = time.perf_counter()
        with self._lock:
            # Serialise the core per-user vector memory (records + their
            # embeddings) so that semantic recall survives a process restart.
            # Prior to v1.1 these two buckets were silently omitted, so every
            # restart reset the observation memory to empty — the dominant
            # cause of the "insufficient memory" symptom.  Embeddings are
            # written as plain float lists; records keep the fields needed to
            # rebuild a fully-typed :class:`MemoryRecord` on load.
            records_payload: Dict[str, List[Dict[str, Any]]] = {}
            for u, recs in self._records.items():
                records_payload[u] = [
                    {
                        "observation_id": str(r.observation_id),
                        "normalized_text": r.normalized_text,
                        "signal_type": r.signal_type.value,
                        "confidence": float(r.confidence),
                        "created_at": r.created_at.isoformat(),
                    }
                    for r in recs
                ]
            embeddings_payload: Dict[str, List[List[float]]] = {
                u: [v.tolist() for v in vecs]
                for u, vecs in self._embeddings.items()
            }
            payload = {
                "version": "1.1",
                "records": records_payload,
                "embeddings": embeddings_payload,
                "preferences": self._preferences,
                "inference_history": self._inference_history,
                "history_seen": {
                    uid: list(seen)
                    for uid, seen in self._history_seen.items()
                },
                "rationale_memory": self._rationale_memory,
                "noise_thresholds": self._noise_thresholds,
                "competitor_aliases": self._competitor_aliases,
                "channel_prefs": self._channel_prefs,
                "source_embeddings": self._source_embeddings,
                "user_profiles": self._user_profiles,
                "user_personas": self._user_personas,
                "embedding_version": self._embedding_version,
            }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(path)
        logger.info(
            "ContextMemoryStore.persist: wrote %s in %.1f ms",
            path, (time.perf_counter() - _t0) * 1000,
        )

    def load_from_disk(self, path: Path) -> None:
        """Restore all mutable state from a file previously written by :meth:`persist`.

        Missing keys are silently ignored so that older snapshot files can be
        loaded into newer store versions without errors.

        Thread-safe.

        Args:
            path: Source file path.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the file is not valid JSON.
        """
        path = Path(path)
        _t0 = time.perf_counter()
        raw = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"ContextMemoryStore.load_from_disk: invalid JSON in {path}: {exc}") from exc
        with self._lock:
            self._preferences = payload.get("preferences", {})
            self._inference_history = payload.get("inference_history", {})
            seen_raw = payload.get("history_seen", {})
            self._history_seen = {uid: set(lst) for uid, lst in seen_raw.items()}
            self._rationale_memory = payload.get("rationale_memory", {})
            self._noise_thresholds = payload.get("noise_thresholds", {})
            self._competitor_aliases = payload.get("competitor_aliases", {})
            self._channel_prefs = payload.get("channel_prefs", {})
            self._source_embeddings = payload.get("source_embeddings", {})
            self._user_profiles = payload.get("user_profiles", {})
            self._user_personas = payload.get("user_personas", {})
            self._embedding_version = payload.get("embedding_version")

            # Restore the core vector memory (added in v1.1).  Snapshots written
            # by older versions simply lack these keys and load as empty, which
            # preserves backward compatibility.  We rebuild fully-typed
            # MemoryRecords and float32 embeddings, skip any record whose
            # signal_type is no longer a valid enum member (enum drift across
            # releases), and enforce the records/embeddings lockstep invariant.
            records_raw = payload.get("records", {}) or {}
            embeddings_raw = payload.get("embeddings", {}) or {}
            self._records = {}
            self._embeddings = {}
            _skipped = 0
            for u, recs in records_raw.items():
                vecs = embeddings_raw.get(u, [])
                n = min(len(recs), len(vecs))
                if n != len(recs) or n != len(vecs):
                    logger.warning(
                        "ContextMemoryStore.load_from_disk: records/embeddings "
                        "length mismatch for user=%s (%d records, %d embeddings)"
                        " — truncating to %d to preserve lockstep invariant",
                        u, len(recs), len(vecs), n,
                    )
                rebuilt: List[MemoryRecord] = []
                rebuilt_vecs: List[np.ndarray] = []
                for rec_d, vec in zip(recs[:n], vecs[:n]):
                    try:
                        signal_type = SignalType(rec_d["signal_type"])
                    except ValueError:
                        _skipped += 1
                        continue
                    try:
                        created_at = datetime.fromisoformat(rec_d["created_at"])
                    except (KeyError, ValueError):
                        created_at = datetime.now(timezone.utc)
                    rebuilt.append(
                        MemoryRecord(
                            user_id=UUID(u),
                            observation_id=UUID(rec_d["observation_id"]),
                            normalized_text=rec_d.get("normalized_text", ""),
                            signal_type=signal_type,
                            confidence=float(rec_d.get("confidence", 0.0)),
                            created_at=created_at,
                        )
                    )
                    rebuilt_vecs.append(np.array(vec, dtype=np.float32))
                if rebuilt:
                    self._records[u] = rebuilt
                    self._embeddings[u] = rebuilt_vecs
            self._total = sum(len(v) for v in self._records.values())
        logger.info(
            "ContextMemoryStore.load_from_disk: loaded %s in %.1f ms "
            "(records=%d users=%d skipped=%d)",
            path, (time.perf_counter() - _t0) * 1000,
            self._total, len(self._records), _skipped,
        )

    # ── Cross-user isolation helpers ─────────────────────────────────────────

    @staticmethod
    def _validate_user_id(user_id: Any, op: str) -> None:
        """Reject any non-``UUID`` user id supplied to read / write methods.

        Centralised type guard so that every public entry point fails closed
        with a consistent, audit-friendly ``TypeError`` whenever a caller
        passes ``None`` or a raw string id (which would otherwise silently
        coexist with legitimate ``UUID``-keyed buckets and create wildcard
        leakage risk).
        """
        if not isinstance(user_id, UUID):
            raise TypeError(
                f"ContextMemoryStore.{op}: user_id must be a uuid.UUID, "
                f"got {type(user_id).__name__!r}"
            )

    def _record_leak_attempt(
        self,
        op: str,
        queried: UUID,
        requesting: UUID,
    ) -> None:
        """Append a structured leak-attempt entry to the audit buffer."""
        entry = {
            "op": op,
            "queried_user_id": str(queried),
            "requesting_user_id": str(requesting),
            "at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._leak_audit.append(entry)
            if len(self._leak_audit) > 1024:
                self._leak_audit.pop(0)
        logger.warning(
            "ContextMemoryStore: cross-user access blocked (op=%s queried=%s requesting=%s)",
            op, queried, requesting,
        )

    def get_leak_audit(self) -> List[Dict[str, Any]]:
        """Return a snapshot copy of the cross-user access audit buffer."""
        with self._lock:
            return [dict(e) for e in self._leak_audit]

    def set_similarity_calibrator(self, calibrator: Any) -> None:
        """Attach (or clear) a similarity→probability calibrator for retrieve().

        ``calibrator`` must expose ``transform(sim: float) -> float`` (e.g.
        :class:`app.intelligence.similarity_calibration.SimilarityCalibrator`).
        Pass ``None`` to revert to raw cosine scoring.  Opt-in: does not change
        any other behaviour.
        """
        self._sim_calibrator = calibrator

    def set_embedding_version(self, version: Optional[str]) -> None:
        """Stamp the embedding-space version the current vectors belong to."""
        self._embedding_version = version

    def needs_reembed(self, current_version: str) -> bool:
        """True when stored vectors exist under a different embedding version.

        Use after loading a snapshot: if the active :class:`EmbeddingBackend`
        reports a different ``embedding_version`` than the one the persisted
        vectors were computed under (or the snapshot is unstamped/legacy), the
        vectors live in an incompatible space and should be re-embedded.
        """
        with self._lock:
            has_vectors = any(self._embeddings.values())
            stored = self._embedding_version
        return bool(has_vectors) and stored != current_version

    # ── GDPR / user-profile API (additive) ───────────────────────────────────

    def clear_user_data(self, user_id: UUID) -> Dict[str, int]:
        """Erase every byte of state associated with ``user_id``.

        Implements right-to-be-forgotten: removes records, embeddings,
        preferences, inference history, idempotency tokens, rationale
        memory, noise threshold, competitor aliases, channel preferences,
        and the user profile.  Idempotent and thread-safe.

        Returns:
            Counts of removed items per bucket, useful for compliance logs.
        """
        self._validate_user_id(user_id, op="clear_user_data")
        uid = str(user_id)
        with self._lock:
            removed = {
                "records": len(self._records.pop(uid, [])),
                "embeddings": len(self._embeddings.pop(uid, [])),
                "preferences": len(self._preferences.pop(uid, {})),
                "inference_history": len(self._inference_history.pop(uid, [])),
                "history_seen": len(self._history_seen.pop(uid, set())),
                "rationale_memory": len(self._rationale_memory.pop(uid, [])),
                "competitor_aliases": len(self._competitor_aliases.pop(uid, [])),
                "channel_prefs": len(self._channel_prefs.pop(uid, {})),
                "profile": 1 if self._user_profiles.pop(uid, None) else 0,
                "persona": 1 if self._user_personas.pop(uid, None) else 0,
            }
            self._noise_thresholds.pop(uid, None)
            self._total = sum(len(v) for v in self._records.values())
        logger.info(
            "ContextMemoryStore.clear_user_data: user=%s removed=%s",
            user_id, removed,
        )
        return removed

    def consolidate_user(
        self,
        user_id: UUID,
        target_k: int,
        summarize_fn: Optional[Callable[[List[str]], str]] = None,
        seed: int = 0,
    ) -> Dict[str, int]:
        """Cluster a user's records and keep one representative per cluster.

        Bounds per-user memory to ``target_k`` records while retaining the
        breadth of distinct topics (Tier 3.2) — far less lossy than dropping the
        oldest records.  When ``summarize_fn`` is given, each kept record's text
        is replaced by a summary of its cluster's member texts; otherwise the
        extractive medoid text is kept as-is.  No-op when the user already has
        ``<= target_k`` records.  Thread-safe.

        Returns counts: ``{"before": n, "after": m, "removed": n-m}``.
        """
        self._validate_user_id(user_id, op="consolidate_user")
        if target_k <= 0:
            raise ValueError("target_k must be positive")
        from app.intelligence.memory_consolidation import MemoryConsolidator
        uid = str(user_id)
        with self._lock:
            recs = self._records.get(uid, [])
            vecs = self._embeddings.get(uid, [])
            before = len(recs)
            if before <= target_k:
                return {"before": before, "after": before, "removed": 0}
            reps, labels = MemoryConsolidator(seed=seed).consolidate(
                [v.tolist() for v in vecs], target_k
            )
            new_recs: List[MemoryRecord] = []
            new_vecs: List[np.ndarray] = []
            for j, rep_idx in enumerate(reps):
                rec = recs[rep_idx]
                if summarize_fn is not None:
                    member_texts = [recs[i].normalized_text
                                    for i, lbl in enumerate(labels) if lbl == labels[rep_idx]]
                    try:
                        rec = dataclasses.replace(rec, normalized_text=summarize_fn(member_texts))
                    except Exception:  # noqa: BLE001 - summary is best-effort
                        logger.exception("consolidate_user: summarize_fn failed; keeping medoid text")
                new_recs.append(rec)
                new_vecs.append(vecs[rep_idx])
            self._records[uid] = new_recs
            self._embeddings[uid] = new_vecs
            self._total = sum(len(v) for v in self._records.values())
            after = len(new_recs)
        logger.info("ContextMemoryStore.consolidate_user: user=%s %d -> %d records",
                    user_id, before, after)
        return {"before": before, "after": after, "removed": before - after}

    def export_user_data(self, user_id: UUID) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of every byte stored for ``user_id``.

        Embeddings are returned as ``List[float]`` so the result is fully
        JSON-encodable.  Intended for GDPR data-portability requests.
        """
        self._validate_user_id(user_id, op="export_user_data")
        uid = str(user_id)
        with self._lock:
            records = [
                {
                    "observation_id": str(r.observation_id),
                    "normalized_text": r.normalized_text,
                    "signal_type": r.signal_type.value,
                    "confidence": r.confidence,
                    "created_at": r.created_at.isoformat(),
                }
                for r in self._records.get(uid, [])
            ]
            embeddings = [v.tolist() for v in self._embeddings.get(uid, [])]
            return {
                "user_id": uid,
                "records": records,
                "embeddings": embeddings,
                "preferences": dict(self._preferences.get(uid, {})),
                "inference_history": list(self._inference_history.get(uid, [])),
                "rationale_memory": list(self._rationale_memory.get(uid, [])),
                "noise_threshold": self._noise_thresholds.get(uid),
                "competitor_aliases": list(self._competitor_aliases.get(uid, [])),
                "channel_prefs": dict(self._channel_prefs.get(uid, {})),
                "profile": dict(self._user_profiles.get(uid, {})),
                "persona": dict(self._user_personas.get(uid, {})),
                "exported_at": datetime.now(timezone.utc).isoformat(),
            }

    def get_user_profile(self, user_id: UUID) -> Dict[str, Any]:
        """Return a defensive copy of the stored profile for ``user_id``."""
        self._validate_user_id(user_id, op="get_user_profile")
        uid = str(user_id)
        with self._lock:
            return dict(self._user_profiles.get(uid, {}))

    # ── Dedicated per-user persona memory ────────────────────────────────────

    def get_user_persona(self, user_id: UUID) -> "UserPersonaProfile":
        """Return the user's :class:`UserPersonaProfile` (reconstructed from disk-safe state).

        A fresh, empty profile is returned for users with no recorded persona,
        so callers can always rely on a non-``None`` object.  Mutating the
        returned object does **not** persist it — call :meth:`save_user_persona`
        or use :meth:`observe_user_persona` for the load-modify-store cycle.
        """
        self._validate_user_id(user_id, op="get_user_persona")
        from app.personalization.user_persona import UserPersonaProfile
        uid = str(user_id)
        with self._lock:
            stored = self._user_personas.get(uid)
            stored = dict(stored) if stored else None
        if stored:
            return UserPersonaProfile.from_dict(stored)
        return UserPersonaProfile(user_id=uid)

    def save_user_persona(self, user_id: UUID, persona: "UserPersonaProfile") -> None:
        """Persist (in-memory) the serialised state of ``persona`` for ``user_id``."""
        self._validate_user_id(user_id, op="save_user_persona")
        with self._lock:
            self._user_personas[str(user_id)] = persona.to_dict()

    def observe_user_persona(
        self,
        user_id: UUID,
        *,
        traits: Optional[Dict[str, float]] = None,
        hobbies: Optional[List[str]] = None,
        characteristics: Optional[Dict[str, str]] = None,
        acquisition: Optional[Dict[str, float]] = None,
        strength: float = 1.0,
    ) -> "UserPersonaProfile":
        """Load-modify-store convenience: fold observations into the persona.

        This is the canonical entry point used by the agent loop to teach the
        persona memory from a turn's signals (e.g. the user asked for a shorter
        answer → ``traits={"verbosity": 0.0}``; mentioned a hobby; stated a
        fact).  All updates use the confidence-weighted, recency-decayed
        estimator in :class:`UserPersonaProfile`.  Returns the updated persona.
        """
        self._validate_user_id(user_id, op="observe_user_persona")
        persona = self.get_user_persona(user_id)
        for name, value in (traits or {}).items():
            persona.observe_trait(name, value, strength=strength)
        for tag in (hobbies or []):
            persona.observe_hobby(tag, strength=strength)
        for key, value in (characteristics or {}).items():
            persona.observe_characteristic(key, value, strength=strength)
        for source, value in (acquisition or {}).items():
            persona.observe_acquisition(source, value, strength=strength)
        self.save_user_persona(user_id, persona)
        return persona

    def update_user_profile(
        self,
        user_id: UUID,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge ``updates`` into the user profile, scrubbing PII as we go.

        String values and string elements of list values are passed through
        :meth:`DataResidencyGuard.scrub_text` before being written so that
        no raw PII survives in the profile bucket.  ``None`` values delete
        their key.  Returns the updated profile (defensive copy).
        """
        self._validate_user_id(user_id, op="update_user_profile")
        if not isinstance(updates, dict):
            raise TypeError(
                f"update_user_profile: updates must be a dict, "
                f"got {type(updates).__name__!r}"
            )
        from app.core.data_residency import DataResidencyGuard
        uid = str(user_id)
        with self._lock:
            profile = self._user_profiles.setdefault(uid, {})
            for k, v in updates.items():
                if v is None:
                    profile.pop(k, None)
                elif isinstance(v, str):
                    profile[k], _ = DataResidencyGuard.scrub_text(v)
                elif isinstance(v, list):
                    profile[k] = [
                        DataResidencyGuard.scrub_text(item)[0]
                        if isinstance(item, str) else item
                        for item in v
                    ]
                else:
                    profile[k] = v
            profile["updated_at"] = datetime.now(timezone.utc).isoformat()
            return dict(profile)


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
    timestamps).  ``record_outcome()`` is **idempotent**: re-delivering the same
    ``signal_inference_id`` for a user is a no-op.

    All public methods are thread-safe via a single ``threading.Lock``.

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
        self._lock: threading.Lock = threading.Lock()

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

        Idempotent: re-delivering the same ``(user_id, signal_inference_id)``
        pair is a no-op that does not double-count outcomes or corrupt adaptive
        threshold state.

        Stores the outcome and — every ``batch_size`` outcomes — re-evaluates
        the per-user false-positive rate to potentially adjust the adaptive
        noise-filter threshold via ``context_memory``.

        Thread-safe.

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
        _t0 = time.perf_counter()

        _should_adjust_threshold = False
        _fp_rate: float = 0.0
        _all_count: int = 0

        with self._lock:
            if uid not in self._outcomes:
                self._outcomes[uid] = {}

            # Idempotency: skip if this inference has already been recorded.
            if iid in self._outcomes[uid]:
                logger.debug(
                    "OutcomeFeedbackStore.record_outcome: duplicate "
                    "inference_id=%s for user=%s — skipped",
                    iid, user_id,
                )
                return

            record: Dict[str, Any] = {
                "outcome": outcome.value,
                "channel_used": channel_used,
                "acted_at": (acted_at or datetime.now(timezone.utc)).isoformat(),
                "inference_id": iid,
            }
            self._outcomes[uid][iid] = record
            _all_count = len(self._outcomes[uid])

            if context_memory is not None and _all_count % self._batch_size == 0:
                all_records = list(self._outcomes[uid].values())
                recent = all_records[-self._batch_size:]
                fp_count = sum(
                    1 for r in recent
                    if r["outcome"] == OutcomeType.FALSE_POSITIVE.value
                )
                _fp_rate = fp_count / len(recent)
                _should_adjust_threshold = True

        # Perform threshold update outside the lock (update_noise_threshold is
        # itself thread-safe and avoids nested locking).
        if _should_adjust_threshold and context_memory is not None:
            if _fp_rate > 0.20:
                context_memory.update_noise_threshold(user_id, +0.02)
                logger.debug(
                    "OutcomeFeedbackStore: FP rate=%.0f%% → tightened "
                    "noise threshold for user %s by +0.02",
                    _fp_rate * 100, user_id,
                )
            elif _fp_rate < 0.05:
                context_memory.update_noise_threshold(user_id, -0.02)
                logger.debug(
                    "OutcomeFeedbackStore: FP rate=%.0f%% → loosened "
                    "noise threshold for user %s by -0.02",
                    _fp_rate * 100, user_id,
                )

        logger.debug(
            "OutcomeFeedbackStore.record_outcome: user=%s outcome=%s "
            "total=%d latency_ms=%.2f",
            user_id, outcome.value, _all_count, (time.perf_counter() - _t0) * 1000,
        )

    def get_outcome(
        self, user_id: UUID, signal_inference_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """Return the recorded outcome for ``(user_id, signal_inference_id)``.

        Thread-safe.

        Args:
            user_id: User who provided the outcome.
            signal_inference_id: UUID of the ``SignalInference``.

        Returns:
            Dict with ``outcome``, ``channel_used``, ``acted_at`` keys, or
            ``None`` if no outcome has been recorded.
        """
        uid = str(user_id)
        iid = str(signal_inference_id)
        with self._lock:
            return dict(self._outcomes.get(uid, {}).get(iid) or {}) or None  # type: ignore[return-value]

    def get_false_positive_rate(self, user_id: UUID, window: int = 100) -> float:
        """Return the fraction of false-positive outcomes in the last ``window``.

        Thread-safe.

        Args:
            user_id: Target user.
            window: How many most-recent outcomes to consider.

        Returns:
            Float in ``[0.0, 1.0]``.  ``0.0`` when the user has no outcomes.
        """
        uid = str(user_id)
        with self._lock:
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

        Thread-safe.

        Args:
            user_id: Target user.
            window: Maximum number of records to return (newest last).

        Returns:
            List of outcome dicts; empty list if no outcomes recorded.
        """
        uid = str(user_id)
        with self._lock:
            all_records = list(self._outcomes.get(uid, {}).values())
        return all_records[-window:]


# ---------------------------------------------------------------------------
# Built-in bag-of-words fallback embedding (no external API required)
# ---------------------------------------------------------------------------

_VOCAB_SIZE: int = 512  # Fixed dimension for reproducibility

# The deterministic bag-of-words fallback embedder lives in
# ``app.core.text_embedding`` so there is a single implementation shared by the
# context store, the embedding backend, and candidate retrieval.  Re-exported
# here under the historical private names for backward compatibility.
from app.core.text_embedding import bow_embed as _bow_embed  # noqa: E402
from app.core.text_embedding import stable_token_bucket as _stable_token_bucket  # noqa: E402

