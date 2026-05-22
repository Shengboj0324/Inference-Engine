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
import threading
import time
from dataclasses import asdict, dataclass, field
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

# Exponential half-life for the personalization bonus.  After ~30 days
# without a fresh citation or thumbs the contribution of a signal row
# has decayed to half; after ~90 days it is ~1/8.  Picked to match the
# typical "feed I'm working out of this quarter" cadence without
# making engagement from yesterday meaningless.
_BONUS_HALFLIFE_SECONDS = 30 * 24 * 3600.0
# Decay floor so very old signals still contribute *something* (1%) of
# their original weight rather than being silently dropped; keeps the
# transparency drawer's numbers consistent with the ranking they drive.
_BONUS_DECAY_FLOOR = 0.01


def _apply_source_balance(
    snippets: List["RetrievedSnippet"],
    k: int,
    max_per_platform: Optional[int],
) -> List["RetrievedSnippet"]:
    """Limit how many snippets a single platform may contribute.

    Preserves the input order (which is already score-sorted) and keeps
    at most ``max_per_platform`` items per ``source_platform``.  When
    ``max_per_platform`` is ``None`` the cap is disabled and the
    existing top-``k`` slice is returned unchanged so callers that do
    not opt in see identical behaviour to before.
    """
    if max_per_platform is None or max_per_platform <= 0:
        return list(snippets[:k])
    counts: dict = {}
    out: List["RetrievedSnippet"] = []
    for s in snippets:
        plat = s.source_platform
        if counts.get(plat, 0) >= max_per_platform:
            continue
        out.append(s)
        counts[plat] = counts.get(plat, 0) + 1
        if len(out) >= k:
            break
    return out


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


@dataclass
class PersonalizationTelemetry:
    """Counters exposed via ``GET /api/v1/rag/telemetry``.

    All counters are monotonic per-process; the desktop sidecar is
    expected to be re-launched on upgrade and the UI tolerates a
    counter reset.  Persisting them on disk would force a second
    write per chat turn and offers no operational value.
    """

    queries_total: int = 0
    queries_personalized: int = 0  # personalize=True AND signals were applied
    queries_reordered: int = 0     # personalize=True AND the top-k order changed
    queries_promoted_to_top: int = 0  # the #1 result changed
    signal_lookups_failed: int = 0  # signals DB threw; ranking continued

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
        bonus_halflife_seconds: float = _BONUS_HALFLIFE_SECONDS,
        calibrator: Optional[object] = None,
    ) -> None:
        self._content = content if content is not None else get_content_store()
        self._embedder = embedder if embedder is not None else LocalEmbeddingProvider()
        self._guard = guard if guard is not None else DataResidencyGuard()
        self._signals = signals if signals is not None else get_signals_store()
        self._max_chars = max(120, int(max_snippet_chars))
        self._personalize = bool(personalize)
        self._halflife = max(1.0, float(bonus_halflife_seconds))
        self._telemetry = PersonalizationTelemetry()
        self._telemetry_lock = threading.Lock()
        # Tier 1.3 — optional similarity→probability calibrator.  When set, raw
        # cosine scores are mapped to calibrated relevance probabilities so the
        # snippet ``score`` is comparable across embedding models.  Calibration
        # is monotonic, so result *ordering* is unchanged; only the score scale
        # (and the meaning of any downstream threshold) changes.  ``None`` →
        # raw cosine, identical to prior behaviour.
        self._calibrator = calibrator

    # ------------------------------------------------------------------
    # Capability probe
    # ------------------------------------------------------------------

    def set_similarity_calibrator(self, calibrator: Optional[object]) -> None:
        """Attach (or clear) a similarity→probability calibrator.

        ``calibrator`` must expose ``transform(score: float) -> float``.  Pass
        ``None`` to revert to raw cosine scoring.  Opt-in; never raises.
        """
        self._calibrator = calibrator

    def _calibrate(self, score: float) -> float:
        """Map a raw cosine score to a calibrated probability (soft-fail)."""
        cal = self._calibrator
        if cal is None:
            return float(score)
        try:
            return float(cal.transform(float(score)))
        except Exception:  # noqa: BLE001 - calibration must never break retrieval
            logger.exception("similarity calibration failed; using raw score")
            return float(score)

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
        max_per_platform: Optional[int] = None,
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
            self._bump_telemetry(queries_total=1)
            return []
        if not do_personalize:
            self._bump_telemetry(queries_total=1)
            snippets = [self._snippet_for(item, self._calibrate(score), bonus=0.0)
                        for item, score in hits]
            return _apply_source_balance(snippets, k, max_per_platform)
        signals = {}
        signal_lookup_failed = False
        try:
            signals = self._signals.get_signals([str(item.id) for item, _ in hits])
        except Exception:  # noqa: BLE001 - signals must never break retrieval
            logger.exception("retrieval signals load failed; ranking without them")
            signal_lookup_failed = True
        now = time.time()
        rescored: List = []
        for item, score in hits:
            bonus = self._personalization_bonus(
                signals.get(str(item.id)), halflife=self._halflife, now=now,
            )
            rescored.append((item, self._calibrate(float(score)), bonus))
        rescored.sort(key=lambda r: r[1] + r[2], reverse=True)
        # Telemetry: count this query, and whether personalization actually
        # changed the order or moved a new item into the top slot.  Computed
        # *before* truncating to ``k`` so the baseline tail is comparable.
        baseline_ids = [str(item.id) for item, _ in hits[:k]]
        reranked_ids = [str(item.id) for item, _, _ in rescored[:k]]
        applied = bool(signals) and any(b > 0.0 or b < 0.0 for _, _, b in rescored[:k])
        reordered = reranked_ids != baseline_ids
        promoted = bool(baseline_ids and reranked_ids
                        and baseline_ids[0] != reranked_ids[0])
        self._bump_telemetry(
            queries_total=1,
            queries_personalized=int(applied),
            queries_reordered=int(reordered),
            queries_promoted_to_top=int(promoted),
            signal_lookups_failed=int(signal_lookup_failed),
        )
        rescored_snippets = [
            self._snippet_for(item, base_score, bonus=bonus)
            for item, base_score, bonus in rescored
        ]
        return _apply_source_balance(rescored_snippets, k, max_per_platform)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _personalization_bonus(
        sig: Optional[dict],
        *,
        halflife: float = _BONUS_HALFLIFE_SECONDS,
        now: Optional[float] = None,
    ) -> float:
        """Map raw signal counters onto a bounded, time-decayed score bonus.

        The curve is intentionally flat: ``log(1+cited)`` saturates
        quickly, and the feedback weight is capped by the signals
        store's own clamp.  Worst-case un-decayed bonus on a current
        cosine in ``[-1, 1]`` is ~``0.05 + 0.05 = 0.10`` \u2014 enough to
        break ties and promote near-misses, never enough to override a
        strong semantic mismatch.

        A multiplicative exponential decay on ``(now - last_seen)`` is
        applied so engagement from months ago contributes a fraction of
        its original weight.  The decay is floored at
        :data:`_BONUS_DECAY_FLOOR` so very old signals do not vanish
        entirely (the transparency drawer would otherwise show a row
        the ranker treats as zero).
        """
        if not sig:
            return 0.0
        cited = max(0, int(sig.get("cited", 0)))
        feedback = float(sig.get("feedback", 0.0))
        raw = (
            _CITED_WEIGHT * math.log1p(cited)
            + _FEEDBACK_WEIGHT * math.tanh(feedback / 3.0)
        )
        if raw == 0.0:
            return 0.0
        last_seen = float(sig.get("last_seen", 0.0) or 0.0)
        if last_seen <= 0.0:
            return raw
        age = max(0.0, (now if now is not None else time.time()) - last_seen)
        decay = math.exp(-age * math.log(2.0) / max(1.0, float(halflife)))
        return raw * max(_BONUS_DECAY_FLOOR, decay)

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _bump_telemetry(self, **deltas: int) -> None:
        """Increment telemetry counters under the instance lock.

        Kept private; the public surface is :meth:`telemetry_snapshot`
        which returns an immutable dict copy.
        """
        with self._telemetry_lock:
            for name, delta in deltas.items():
                if not delta:
                    continue
                setattr(
                    self._telemetry, name,
                    getattr(self._telemetry, name) + int(delta),
                )

    def telemetry_snapshot(self) -> dict:
        """Return a copy of the personalization telemetry counters."""
        with self._telemetry_lock:
            return self._telemetry.to_dict()

    def reset_telemetry(self) -> None:
        """Zero every counter.  Used by the dedicated telemetry route."""
        with self._telemetry_lock:
            self._telemetry = PersonalizationTelemetry()

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


def _load_persisted_calibrator() -> Optional[object]:
    """Load a trained similarity calibrator from the user-data dir, if present.

    Operators place a ``similarity_calibrator.json`` (produced by the training
    notebook, Tier 1.3) under the user-data dir to enable calibrated retrieval
    scores.  Absent / malformed file → ``None`` (raw cosine).  Never raises.
    """
    try:
        from app.local.user_data_dir import get_user_data_dir
        from app.intelligence.similarity_calibration import SimilarityCalibrator
        path = get_user_data_dir() / "similarity_calibrator.json"
        if path.exists():
            cal = SimilarityCalibrator.load(path)
            logger.info("rag retriever: loaded similarity calibrator from %s", path)
            return cal
    except Exception:  # noqa: BLE001 - calibrator is optional; never block startup
        logger.exception("rag retriever: failed to load similarity calibrator")
    return None


def get_rag_retriever() -> LocalRAGRetriever:
    """Process-wide :class:`LocalRAGRetriever` singleton."""
    global _global_retriever
    if _global_retriever is None:
        _global_retriever = LocalRAGRetriever(calibrator=_load_persisted_calibrator())
    return _global_retriever


def reset_rag_retriever() -> None:
    """Drop the cached singleton (test-only)."""
    global _global_retriever
    _global_retriever = None
