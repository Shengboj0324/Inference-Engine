"""SourceVolatilityProfile — adaptive recrawl interval optimisation.

Tracks how frequently a source produces *novel* content (i.e. items that do
**not** match the BloomFilter / deduplication layer) and computes an optimal
recrawl interval that balances:

* **Freshness** — crawling often enough not to miss breaking events.
* **Efficiency** — avoiding redundant fetches on low-churn sources.
* **Trust weighting** — high-authority sources are crawled more eagerly.
* **User-interest** — sources matching watchlist entities are boosted.

Algorithm
---------
The base interval is derived from the observed mean inter-arrival time of
novel items (``novel_items_per_hour``).  A Kalman-style exponential moving
average filters out burst noise::

    novelty_ema ← α · novelty_ema + (1 − α) · latest_novelty_fraction

The raw interval is then scaled by trust and user-interest factors::

    interval = clamp(base / (trust_weight * interest_factor), MIN, MAX)

``VolatilityRegistry`` manages one profile per source and exposes
``next_crawl_at(source_id)`` for use by ``AcquisitionScheduler``.

Usage::

    registry = VolatilityRegistry()
    registry.record_crawl("arxiv-cs", novel_items=12, total_items=25,
                          trust_score=0.85, user_interest=0.9)
    dt = registry.next_crawl_at("arxiv-cs")
    # dt → datetime (UTC) — recommended next crawl time
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_MIN_INTERVAL_MINUTES = 5
_MAX_INTERVAL_MINUTES = 1_440       # 24 hours
_DEFAULT_INTERVAL_MINUTES = 60
_EMA_ALPHA = 0.3                    # novelty EMA smoothing factor


@dataclass
class CrawlObservation:
    """A single crawl result contributed to the volatility model."""
    timestamp: datetime             # UTC
    novel_items: int
    total_items: int
    trust_score: float              # ∈ [0, 1]
    user_interest: float            # ∈ [0, 1]; 0 = source not on any watchlist

    @property
    def novelty_fraction(self) -> float:
        if self.total_items == 0:
            return 0.0
        return max(0.0, min(1.0, self.novel_items / self.total_items))


class SourceVolatilityProfile:
    """Adaptive recrawl model for a single source.

    Attributes:
        source_id: Identifier matching ``SourceSpec.source_id``.
    """

    def __init__(self, source_id: str) -> None:
        self.source_id = source_id
        self._lock = threading.Lock()
        self._novelty_ema: float = 0.5          # start at neutral
        self._last_crawl: Optional[datetime] = None
        self._observation_count: int = 0
        self._last_trust: float = 0.5
        self._last_interest: float = 0.5

    def record_crawl(
        self,
        novel_items: int,
        total_items: int,
        trust_score: float = 0.5,
        user_interest: float = 0.5,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Update the volatility model with the result of one crawl.

        Args:
            novel_items:   Number of items that were not already in the store.
            total_items:   Total items returned by the connector.
            trust_score:   Authority score of this source (∈ [0, 1]).
            user_interest: Mean relevance to active watchlists (∈ [0, 1]).
            timestamp:     Crawl completion time (UTC); defaults to now().
        """
        ts = timestamp or datetime.now(tz=timezone.utc)
        obs = CrawlObservation(ts, novel_items, total_items, trust_score, user_interest)
        with self._lock:
            self._novelty_ema = (
                _EMA_ALPHA * obs.novelty_fraction
                + (1 - _EMA_ALPHA) * self._novelty_ema
            )
            self._last_crawl = ts
            self._observation_count += 1
            self._last_trust = max(0.01, trust_score)
            self._last_interest = max(0.1, user_interest)
        logger.debug(
            "VolatilityProfile[%s]: novelty_ema=%.3f interval_min=%.0f",
            self.source_id,
            self._novelty_ema,
            self.recommended_interval_minutes,
        )

    @property
    def novelty_ema(self) -> float:
        with self._lock:
            return self._novelty_ema

    @property
    def recommended_interval_minutes(self) -> float:
        """Compute the recommended recrawl interval in minutes.

        High novelty + high trust + high user interest → shorter interval.
        Low novelty (source is stale) → longer interval.
        """
        with self._lock:
            if self._observation_count == 0:
                return float(_DEFAULT_INTERVAL_MINUTES)

            # Base interval: inversely proportional to novelty EMA.
            # novelty_ema=1.0 → base=MIN; novelty_ema~0 → base=MAX
            novelty = max(0.01, self._novelty_ema)
            base = _DEFAULT_INTERVAL_MINUTES / novelty

            # Trust scaling: high-trust sources crawled more eagerly.
            trust_factor = 0.5 + 0.5 * self._last_trust   # ∈ [0.5, 1.0]

            # User-interest scaling: watchlist-relevant sources boosted.
            interest_factor = 0.5 + 0.5 * self._last_interest  # ∈ [0.5, 1.0]

            interval = base / (trust_factor * interest_factor)
            return float(max(_MIN_INTERVAL_MINUTES, min(_MAX_INTERVAL_MINUTES, interval)))

    def next_crawl_at(self, from_time: Optional[datetime] = None) -> datetime:
        """Return the UTC datetime at which the next crawl should be scheduled."""
        anchor = from_time or self._last_crawl or datetime.now(tz=timezone.utc)
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=timezone.utc)
        return anchor + timedelta(minutes=self.recommended_interval_minutes)

    def is_overdue(self, now: Optional[datetime] = None) -> bool:
        """True if the source is past its recommended next-crawl time."""
        now = now or datetime.now(tz=timezone.utc)
        return now >= self.next_crawl_at()

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "novelty_ema": round(self._novelty_ema, 4),
            "recommended_interval_minutes": round(self.recommended_interval_minutes, 1),
            "observation_count": self._observation_count,
            "last_crawl": self._last_crawl.isoformat() if self._last_crawl else None,
        }


class VolatilityRegistry:
    """Thread-safe registry of ``SourceVolatilityProfile`` objects."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._profiles: Dict[str, SourceVolatilityProfile] = {}

    def _get_or_create(self, source_id: str) -> SourceVolatilityProfile:
        if source_id not in self._profiles:
            self._profiles[source_id] = SourceVolatilityProfile(source_id)
        return self._profiles[source_id]

    def record_crawl(
        self,
        source_id: str,
        novel_items: int,
        total_items: int,
        trust_score: float = 0.5,
        user_interest: float = 0.5,
        timestamp: Optional[datetime] = None,
    ) -> None:
        with self._lock:
            self._get_or_create(source_id).record_crawl(
                novel_items, total_items, trust_score, user_interest, timestamp
            )

    def next_crawl_at(self, source_id: str) -> Optional[datetime]:
        """Return the recommended next crawl datetime for ``source_id``, or None."""
        with self._lock:
            profile = self._profiles.get(source_id)
        return profile.next_crawl_at() if profile else None

    def overdue_sources(self, now: Optional[datetime] = None) -> list[str]:
        """Return sorted list of source IDs whose recommended crawl time has passed."""
        now = now or datetime.now(tz=timezone.utc)
        with self._lock:
            profiles = list(self._profiles.values())
        return sorted(p.source_id for p in profiles if p.is_overdue(now))

    def get(self, source_id: str) -> Optional[SourceVolatilityProfile]:
        with self._lock:
            return self._profiles.get(source_id)

    def summary(self) -> list[dict]:
        with self._lock:
            return [p.to_dict() for p in self._profiles.values()]

