"""Source trust scoring.

Computes a composite trust score for each source along four orthogonal
dimensions:

1. **Primacy**     — Is this source the *origin* of the content (primary
                     source) or a re-publisher / aggregator (derivative)?
2. **Recency**     — How fresh is the source's typical publication latency?
                     Faster is more trusted for time-sensitive signals.
3. **Accuracy**    — Historical accuracy as measured by confirmed vs.
                     false-positive inferences derived from this source.
4. **Authority**   — Domain authority proxy: citation count for research,
                     GitHub stars for developer sources, subscriber count
                     for podcasts (all normalised to [0, 1]).

The composite score is a weighted sum in [0, 1]:
    score = w_primacy * primacy + w_recency * recency
          + w_accuracy * accuracy + w_authority * authority

Weights default to (0.35, 0.20, 0.30, 0.15) and can be overridden at
construction time.
"""

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS: Tuple[float, float, float, float] = (0.35, 0.20, 0.30, 0.15)


@dataclass(frozen=True)
class TrustScore:
    """Immutable trust assessment for a single source.

    Attributes:
        source_id:  Source identifier matching ``SourceSpec.source_id``.
        composite:  Weighted composite score in [0.0, 1.0].
        primacy:    Primary-source indicator score (0 or 1 in practice).
        recency:    Normalised recency score [0, 1].
        accuracy:   Historical accuracy [0, 1].
        authority:  Authority proxy [0, 1].
    """

    source_id: str
    composite: float
    primacy: float
    recency: float
    accuracy: float
    authority: float

    def __post_init__(self) -> None:
        for attr in ("composite", "primacy", "recency", "accuracy", "authority"):
            v = getattr(self, attr)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"TrustScore.{attr}={v!r} is not in [0, 1]")


class SourceTrustScorer:
    """Computes and caches trust scores for registered sources.

    Thread-safe: all mutable state (accuracy history, star/citation counts,
    cached scores) is protected by ``_lock``.

    The scorer maintains:
    - ``_primacy``   : Dict[source_id, bool]  — set by caller
    - ``_accuracy``  : Dict[source_id, Tuple[int, int]]  — (confirmed, total)
    - ``_authority`` : Dict[source_id, float]  — normalised [0, 1]
    - ``_latency_s`` : Dict[source_id, float]  — median pub latency seconds
    - ``_cache``     : Dict[source_id, TrustScore]  — invalidated on update
    """

    def __init__(
        self,
        weights: Optional[Tuple[float, float, float, float]] = None,
        max_latency_hours: float = 48.0,
    ) -> None:
        """Initialise the scorer.

        Args:
            weights: (w_primacy, w_recency, w_accuracy, w_authority).
                     Must sum to 1.0.  Default: (0.35, 0.20, 0.30, 0.15).
            max_latency_hours: Latency (hours) mapped to recency=0.0.
                               Sources faster than 1 minute → recency ≈ 1.0.
        """
        _w = weights or _DEFAULT_WEIGHTS
        if len(_w) != 4:
            raise ValueError("'weights' must have exactly 4 elements")
        total = sum(_w)
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"'weights' must sum to 1.0, got {total:.6f}")
        if any(w < 0 for w in _w):
            raise ValueError("All weights must be non-negative")
        if max_latency_hours <= 0:
            raise ValueError("'max_latency_hours' must be positive")

        self._w_primacy, self._w_recency, self._w_accuracy, self._w_authority = _w
        self._max_latency_s = max_latency_hours * 3600.0
        self._lock = threading.Lock()
        self._primacy: Dict[str, bool] = {}
        self._accuracy: Dict[str, Tuple[int, int]] = {}  # (confirmed, total)
        self._authority: Dict[str, float] = {}
        self._latency_s: Dict[str, float] = {}
        self._cache: Dict[str, TrustScore] = {}

    # ------------------------------------------------------------------
    # Setters (all invalidate the per-source cache entry)
    # ------------------------------------------------------------------

    def set_primacy(self, source_id: str, is_primary: bool) -> None:
        """Mark *source_id* as a primary (True) or derivative (False) source."""
        self._validate_source_id(source_id)
        with self._lock:
            self._primacy[source_id] = bool(is_primary)
            self._cache.pop(source_id, None)
        logger.debug("SourceTrustScorer.set_primacy: %r → %s", source_id, is_primary)

    def record_outcome(self, source_id: str, *, confirmed: bool) -> None:
        """Record one signal outcome (confirmed correct vs. false positive).

        Args:
            source_id: Source that produced the signal.
            confirmed: True = signal was acted-on / correct; False = FP.
        """
        self._validate_source_id(source_id)
        t0 = time.perf_counter()
        with self._lock:
            old_confirmed, old_total = self._accuracy.get(source_id, (0, 0))
            self._accuracy[source_id] = (
                old_confirmed + (1 if confirmed else 0),
                old_total + 1,
            )
            self._cache.pop(source_id, None)
        logger.debug(
            "SourceTrustScorer.record_outcome: source=%r confirmed=%s latency_ms=%.2f",
            source_id, confirmed, (time.perf_counter() - t0) * 1000,
        )

    def set_authority(self, source_id: str, raw_authority: float) -> None:
        """Set the normalised authority score for *source_id*.

        Args:
            raw_authority: Value in [0.0, 1.0].

        Raises:
            ValueError: If *raw_authority* is not in [0, 1].
        """
        self._validate_source_id(source_id)
        if not (0.0 <= raw_authority <= 1.0):
            raise ValueError(f"'raw_authority' must be in [0, 1], got {raw_authority!r}")
        with self._lock:
            self._authority[source_id] = raw_authority
            self._cache.pop(source_id, None)

    def set_latency(self, source_id: str, median_latency_seconds: float) -> None:
        """Record the typical publication latency for *source_id*.

        Args:
            median_latency_seconds: Median seconds between event and content appearing.

        Raises:
            ValueError: If value is negative.
        """
        self._validate_source_id(source_id)
        if median_latency_seconds < 0:
            raise ValueError(f"'median_latency_seconds' must be ≥ 0, got {median_latency_seconds!r}")
        with self._lock:
            self._latency_s[source_id] = median_latency_seconds
            self._cache.pop(source_id, None)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, source_id: str) -> TrustScore:
        """Compute (or retrieve from cache) the trust score for *source_id*.

        Returns:
            ``TrustScore`` with all four dimensions and composite score.
        """
        self._validate_source_id(source_id)
        t0 = time.perf_counter()
        with self._lock:
            cached = self._cache.get(source_id)
            if cached is not None:
                return cached
            primacy_raw = self._primacy.get(source_id, False)
            acc_confirmed, acc_total = self._accuracy.get(source_id, (0, 0))
            authority_raw = self._authority.get(source_id, 0.5)  # default neutral
            latency = self._latency_s.get(source_id, self._max_latency_s)

        primacy_score = 1.0 if primacy_raw else 0.3
        accuracy_score = (acc_confirmed / acc_total) if acc_total >= 10 else 0.5
        recency_score = max(0.0, 1.0 - (latency / self._max_latency_s))
        authority_score = authority_raw

        composite = (
            self._w_primacy * primacy_score
            + self._w_recency * recency_score
            + self._w_accuracy * accuracy_score
            + self._w_authority * authority_score
        )
        composite = max(0.0, min(1.0, composite))

        ts = TrustScore(
            source_id=source_id,
            composite=composite,
            primacy=primacy_score,
            recency=recency_score,
            accuracy=accuracy_score,
            authority=authority_score,
        )
        with self._lock:
            self._cache[source_id] = ts
        logger.debug(
            "SourceTrustScorer.score: source=%r composite=%.3f latency_ms=%.2f",
            source_id, composite, (time.perf_counter() - t0) * 1000,
        )
        return ts

    def score_many(self, source_ids: list) -> Dict[str, TrustScore]:
        """Compute trust scores for multiple sources.

        Args:
            source_ids: List of source identifier strings.

        Returns:
            Dict mapping each source_id to its ``TrustScore``.
        """
        if not isinstance(source_ids, list):
            raise TypeError(f"'source_ids' must be a list, got {type(source_ids)!r}")
        return {sid: self.score(sid) for sid in source_ids}

    @staticmethod
    def normalise_authority_from_stars(stars: int, *, scale: int = 10_000) -> float:
        """Map a GitHub star count to a normalised [0, 1] authority score.

        Uses log-sigmoid scaling so that the curve is steep at low counts
        and asymptotes near 1.0 for highly popular repos.

        Args:
            stars: Raw GitHub star count (≥ 0).
            scale: Star count that maps to ≈ 0.73 (sigmoid midpoint).

        Returns:
            Float in [0.0, 1.0].
        """
        if stars < 0:
            raise ValueError(f"'stars' must be ≥ 0, got {stars!r}")
        if scale <= 0:
            raise ValueError(f"'scale' must be > 0, got {scale!r}")
        x = math.log1p(stars) / math.log1p(scale)
        return max(0.0, min(1.0, x))

    @staticmethod
    def normalise_authority_from_citations(citations: int, *, scale: int = 1_000) -> float:
        """Map a citation count to a normalised [0, 1] authority score."""
        if citations < 0:
            raise ValueError(f"'citations' must be ≥ 0, got {citations!r}")
        if scale <= 0:
            raise ValueError(f"'scale' must be > 0, got {scale!r}")
        return max(0.0, min(1.0, math.log1p(citations) / math.log1p(scale)))

    @staticmethod
    def _validate_source_id(source_id: str) -> None:
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"'source_id' must be a non-empty string, got {source_id!r}")

