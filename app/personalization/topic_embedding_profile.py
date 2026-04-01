"""Topic embedding profile.

Maintains a per-user *centroid* embedding — a running representation of
what the user is interested in — updated online via Exponential Moving
Average (EMA) as feedback arrives.

The centroid is used by the digest ranker to compute semantic similarity
between the user profile and each candidate article.

All arithmetic is pure Python (no numpy required) so the module has zero
extra dependencies.

Key concepts
------------
- **EMA update**: ``centroid ← α·new_embedding + (1−α)·centroid``
  where ``α`` (``ema_alpha``) controls how quickly the profile adapts.
  Smaller α = slower adaptation (more stable); larger α = faster drift.
- **Cosine similarity**: dot(a, b) / (|a| · |b|), clamped to [−1, 1].
- **Drift score**: ``1 − cosine_similarity(embedding, centroid)``
  — high drift means the new item is far from the user's current profile.
- **Stability**: the profile is considered "stable" after
  ``min_updates_before_stable`` feedback events.

Thread safety: all mutations acquire a ``threading.Lock``.
"""

from __future__ import annotations

import logging
import math
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-Python linear algebra helpers
# ---------------------------------------------------------------------------

def _dot(a: List[float], b: List[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def _magnitude(v: List[float]) -> float:
    return math.sqrt(sum(vi * vi for vi in v))


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Return cosine similarity in [−1, 1]; 0.0 if either vector is zero."""
    if len(a) != len(b) or not a:
        return 0.0
    mag_a = _magnitude(a)
    mag_b = _magnitude(b)
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    raw = _dot(a, b) / (mag_a * mag_b)
    return max(-1.0, min(1.0, raw))


def _ema_update(old: List[float], new: List[float], alpha: float) -> List[float]:
    """EMA: α·new + (1−α)·old, element-wise."""
    return [alpha * n + (1.0 - alpha) * o for o, n in zip(old, new)]


# ---------------------------------------------------------------------------
# TopicEmbeddingProfile
# ---------------------------------------------------------------------------


class TopicEmbeddingProfile:
    """Online EMA-updated representation of a user's topical interests.

    Args:
        dim:                       Embedding dimensionality.
        ema_alpha:                 EMA smoothing factor [0, 1].
                                   0 → never adapt; 1 → replace each time.
        min_updates_before_stable: Updates required before ``is_stable``.
    """

    def __init__(
        self,
        dim: int,
        ema_alpha: float = 0.2,
        min_updates_before_stable: int = 5,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"'dim' must be positive, got {dim!r}")
        if not (0.0 <= ema_alpha <= 1.0):
            raise ValueError(f"'ema_alpha' must be in [0, 1], got {ema_alpha!r}")
        if min_updates_before_stable <= 0:
            raise ValueError(f"'min_updates_before_stable' must be positive, got {min_updates_before_stable!r}")

        self._dim = dim
        self._alpha = ema_alpha
        self._min_stable = min_updates_before_stable
        self._centroid: List[float] = [0.0] * dim
        self._update_count: int = 0
        self._last_updated: Optional[datetime] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, embedding: List[float], weight: float = 1.0) -> None:
        """Update the centroid with a new embedding observation.

        For the first update the centroid is set directly to *embedding*.
        For subsequent updates EMA is applied.  The *weight* parameter
        scales how much this observation contributes (heavier feedback
        events, e.g. SAVE, should supply weight > 1.0).

        Args:
            embedding: Dense float vector of length ``dim``.
            weight:    Relative importance [0, ∞).  Values > 1.0 increase
                       the effective alpha for this update.

        Raises:
            TypeError:  If *embedding* is not a list of floats.
            ValueError: If *embedding* length differs from ``dim``.
        """
        if not isinstance(embedding, list):
            raise TypeError(f"'embedding' must be a list, got {type(embedding)!r}")
        if len(embedding) != self._dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self._dim}, got {len(embedding)}"
            )
        if weight < 0.0:
            raise ValueError(f"'weight' must be >= 0, got {weight!r}")

        effective_alpha = min(1.0, self._alpha * max(1.0, weight))

        with self._lock:
            if self._update_count == 0:
                self._centroid = list(embedding)
            else:
                self._centroid = _ema_update(self._centroid, embedding, effective_alpha)
            self._update_count += 1
            self._last_updated = datetime.now(timezone.utc)

        logger.debug(
            "TopicEmbeddingProfile: update #%d (alpha=%.3f, weight=%.2f)",
            self._update_count, effective_alpha, weight,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def similarity(self, embedding: List[float]) -> float:
        """Cosine similarity between *embedding* and current centroid.

        Returns:
            Float in [−1, 1]; 0.0 if the profile has not been updated yet.

        Raises:
            ValueError: If *embedding* length differs from ``dim``.
        """
        if not isinstance(embedding, list):
            raise TypeError(f"'embedding' must be a list, got {type(embedding)!r}")
        if len(embedding) != self._dim:
            raise ValueError(f"Embedding dim mismatch: expected {self._dim}, got {len(embedding)}")
        with self._lock:
            if self._update_count == 0:
                return 0.0
            return _cosine_similarity(self._centroid, embedding)

    def drift_score(self, embedding: List[float]) -> float:
        """Measure how different *embedding* is from the centroid.

        Returns:
            Float in [0, 1]; 0 = identical, 1 = orthogonal/opposite.
        """
        return max(0.0, 1.0 - self.similarity(embedding))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def centroid(self) -> List[float]:
        """A copy of the current centroid vector."""
        with self._lock:
            return list(self._centroid)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def update_count(self) -> int:
        with self._lock:
            return self._update_count

    @property
    def is_stable(self) -> bool:
        """True once enough updates have been applied."""
        with self._lock:
            return self._update_count >= self._min_stable

    @property
    def last_updated(self) -> Optional[datetime]:
        with self._lock:
            return self._last_updated

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset profile to the zero centroid state."""
        with self._lock:
            self._centroid = [0.0] * self._dim
            self._update_count = 0
            self._last_updated = None
        logger.info("TopicEmbeddingProfile: reset")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Return a JSON-serializable snapshot of the profile."""
        with self._lock:
            return {
                "dim": self._dim,
                "ema_alpha": self._alpha,
                "min_updates_before_stable": self._min_stable,
                "centroid": list(self._centroid),
                "update_count": self._update_count,
                "last_updated": self._last_updated.isoformat() if self._last_updated else None,
            }

    @classmethod
    def from_dict(cls, data: dict) -> "TopicEmbeddingProfile":
        """Reconstruct a ``TopicEmbeddingProfile`` from a ``to_dict()`` snapshot."""
        if not isinstance(data, dict):
            raise TypeError(f"'data' must be a dict, got {type(data)!r}")
        profile = cls(
            dim=data["dim"],
            ema_alpha=data.get("ema_alpha", 0.2),
            min_updates_before_stable=data.get("min_updates_before_stable", 5),
        )
        centroid = data.get("centroid", [])
        if centroid and len(centroid) == profile._dim:
            with profile._lock:
                profile._centroid = list(centroid)
                profile._update_count = data.get("update_count", 0)
                raw_dt = data.get("last_updated")
                profile._last_updated = datetime.fromisoformat(raw_dt) if raw_dt else None
        return profile

