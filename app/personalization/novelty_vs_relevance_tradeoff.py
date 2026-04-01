"""Novelty-vs-relevance tradeoff controller.

Maintains a dynamic α ∈ [min_alpha, max_alpha] that blends *novelty*
(exploration) and *relevance* (exploitation) scores in the digest:

    blended = α · novelty_score + (1 − α) · relevance_score

α is updated after each feedback event:
- **Exploration signals** (CLICK on unfamiliar topic, SCROLL_PAST on
  familiar topic) → α increases by ``novelty_push_rate``
- **Exploitation signals** (SAVE, READ_COMPLETE, SHARE) → α decreases
  by ``relevance_pull_rate``
- **Neutral** (LIKE, DISMISS) → no change to α

When ``auto_adapt=False``, α is frozen at its initial value and
``update_from_feedback()`` is a no-op.

The class is also used to detect "topic fatigue": if a user keeps
engaging with the same top-3 topics over the last ``exploration_window``
events, α is nudged upward to encourage diversity.

Thread safety: ``_alpha`` and ``_recent_topics`` are protected by a
``threading.Lock``.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter, deque
from typing import Deque, Dict, List, Optional

from app.personalization.models import FeedbackEvent, FeedbackType, NoveltyRelevanceConfig

logger = logging.getLogger(__name__)

# Feedback types that signal the user wants *more* exploration (less familiar content)
_PUSH_TYPES = frozenset({FeedbackType.CLICK, FeedbackType.SCROLL_PAST})
# Feedback types that signal the user is happy with familiar content
_PULL_TYPES = frozenset({FeedbackType.SAVE, FeedbackType.READ_COMPLETE, FeedbackType.SHARE})
# Concentration ratio above which topic-fatigue nudge fires
_FATIGUE_CONCENTRATION = 0.60   # top-3 topics > 60 % of events → push novelty


class NoveltyRelevanceTradeoff:
    """Dynamic α controller for novelty-vs-relevance blending.

    Args:
        config: ``NoveltyRelevanceConfig`` with initial α and adaptation rates.
    """

    def __init__(self, config: Optional[NoveltyRelevanceConfig] = None) -> None:
        self._cfg = config or NoveltyRelevanceConfig()
        if not isinstance(self._cfg, NoveltyRelevanceConfig):
            raise TypeError(f"'config' must be NoveltyRelevanceConfig, got {type(self._cfg)!r}")
        self._alpha: float = self._cfg.alpha
        self._recent_topics: Deque[str] = deque(maxlen=self._cfg.exploration_window)
        self._event_count: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @property
    def current_alpha(self) -> float:
        """Current α value."""
        with self._lock:
            return self._alpha

    def blend_scores(self, relevance_score: float, novelty_score: float, alpha: Optional[float] = None) -> float:
        """Blend relevance and novelty scores.

        ``result = α · novelty + (1 − α) · relevance``

        Args:
            relevance_score: Topic/embedding relevance in [0, 1].
            novelty_score:   Content novelty in [0, 1].
            alpha:           Override α (uses ``current_alpha`` if None).

        Returns:
            Blended score in [0, 1].

        Raises:
            ValueError: If scores are outside [0, 1].
        """
        for name, val in (("relevance_score", relevance_score), ("novelty_score", novelty_score)):
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"'{name}' must be in [0, 1], got {val!r}")
        a = alpha if alpha is not None else self.current_alpha
        if not (0.0 <= a <= 1.0):
            raise ValueError(f"'alpha' must be in [0, 1], got {a!r}")
        return round(a * novelty_score + (1.0 - a) * relevance_score, 6)

    def update_from_feedback(self, event: FeedbackEvent) -> float:
        """Update α based on a feedback event and return the new value.

        Args:
            event: Validated ``FeedbackEvent``.

        Returns:
            New α after update.

        Raises:
            TypeError: If *event* is not a ``FeedbackEvent``.
        """
        if not isinstance(event, FeedbackEvent):
            raise TypeError(f"Expected FeedbackEvent, got {type(event)!r}")

        with self._lock:
            # Record topics for fatigue detection
            for tid in event.topic_ids:
                self._recent_topics.append(tid)
            self._event_count += 1

            if not self._cfg.auto_adapt:
                return self._alpha

            ft = event.feedback_type
            if ft in _PUSH_TYPES:
                self._alpha = min(self._cfg.max_alpha, self._alpha + self._cfg.novelty_push_rate)
            elif ft in _PULL_TYPES:
                self._alpha = max(self._cfg.min_alpha, self._alpha - self._cfg.relevance_pull_rate)

            # Topic-fatigue nudge
            if len(self._recent_topics) >= self._cfg.exploration_window:
                concentration = self._compute_concentration()
                if concentration > _FATIGUE_CONCENTRATION:
                    nudge = self._cfg.novelty_push_rate * 0.5
                    self._alpha = min(self._cfg.max_alpha, self._alpha + nudge)
                    logger.debug(
                        "NoveltyRelevanceTradeoff: fatigue nudge fired (conc=%.2f, α→%.3f)",
                        concentration, self._alpha,
                    )

            logger.debug(
                "NoveltyRelevanceTradeoff: %s → α=%.3f",
                event.feedback_type.value, self._alpha,
            )
            return self._alpha

    def compute_alpha(self, recent_events: List[FeedbackEvent]) -> float:
        """Stateless α computation from a list of recent events.

        Does *not* mutate internal state.  Useful for batch re-computation.

        Args:
            recent_events: Recent ``FeedbackEvent`` objects (chronological).

        Returns:
            Suggested α in [min_alpha, max_alpha].

        Raises:
            TypeError: If *recent_events* is not a list.
        """
        if not isinstance(recent_events, list):
            raise TypeError(f"'recent_events' must be a list, got {type(recent_events)!r}")

        alpha = self._cfg.alpha
        topic_buf: Deque[str] = deque(maxlen=self._cfg.exploration_window)

        for event in recent_events:
            ft = event.feedback_type
            topic_buf.extend(event.topic_ids)
            if ft in _PUSH_TYPES:
                alpha = min(self._cfg.max_alpha, alpha + self._cfg.novelty_push_rate)
            elif ft in _PULL_TYPES:
                alpha = max(self._cfg.min_alpha, alpha - self._cfg.relevance_pull_rate)

        return round(alpha, 5)

    def reset(self) -> None:
        """Reset α to the configured default and clear topic history."""
        with self._lock:
            self._alpha = self._cfg.alpha
            self._recent_topics.clear()
            self._event_count = 0
        logger.info("NoveltyRelevanceTradeoff: reset to α=%.3f", self._cfg.alpha)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_concentration(self) -> float:
        """Fraction of recent topic slots occupied by the top-3 topics."""
        if not self._recent_topics:
            return 0.0
        counts = Counter(self._recent_topics)
        top3_count = sum(v for _, v in counts.most_common(3))
        return top3_count / len(self._recent_topics)

    # ------------------------------------------------------------------
    # Stats / introspection
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, object]:
        """Return current controller statistics."""
        with self._lock:
            conc = self._compute_concentration() if self._recent_topics else 0.0
            return {
                "current_alpha": self._alpha,
                "event_count": self._event_count,
                "min_alpha": self._cfg.min_alpha,
                "max_alpha": self._cfg.max_alpha,
                "auto_adapt": self._cfg.auto_adapt,
                "topic_concentration": round(conc, 3),
                "recent_topic_count": len(self._recent_topics),
            }

