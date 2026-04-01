"""Feedback learner.

Processes explicit and implicit user feedback events and translates them
into weight updates for the ``InterestGraph`` and the
``TopicEmbeddingProfile``.

Signal hierarchy (effective weight delta)
-----------------------------------------
SAVE / SHARE   → strong positive  (+0.20 / +0.18)
LIKE           → moderate positive (+0.15)
READ_COMPLETE  → moderate positive (+0.10) — scaled by dwell time
CLICK          → weak positive    (+0.05)
SCROLL_PAST    → weak negative    (−0.03)
DISMISS        → moderate negative (−0.12)

Implicit dwell-time scaling
---------------------------
For ``READ_COMPLETE`` events, the effective delta is scaled by:
    ``min(1.0, dwell_seconds / expected_read_seconds)``
where ``expected_read_seconds`` defaults to 240 s (≈ 1 000-word article).

Thread safety: ``_history`` and metrics are protected by ``threading.Lock``.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

from app.personalization.interest_graph import InterestGraph
from app.personalization.models import FEEDBACK_DELTA, FeedbackEvent, FeedbackType, InterestWeight
from app.personalization.topic_embedding_profile import TopicEmbeddingProfile

logger = logging.getLogger(__name__)

_EXPECTED_READ_SECONDS: float = 240.0   # typical 1 000-word read
_HISTORY_MAXLEN: int = 10_000


class FeedbackLearner:
    """Translates ``FeedbackEvent`` objects into interest-graph weight updates.

    Args:
        interest_graph:         ``InterestGraph`` to update.
        embedding_profile:      ``TopicEmbeddingProfile`` to update.
        learning_rate:          Global scale factor for weight deltas [0, 1].
        expected_read_seconds:  Reference read time for dwell-time scaling.
        history_maxlen:         Maximum events kept in the internal history.
    """

    def __init__(
        self,
        interest_graph: InterestGraph,
        embedding_profile: Optional[TopicEmbeddingProfile] = None,
        learning_rate: float = 1.0,
        expected_read_seconds: float = _EXPECTED_READ_SECONDS,
        history_maxlen: int = _HISTORY_MAXLEN,
    ) -> None:
        if not isinstance(interest_graph, InterestGraph):
            raise TypeError(f"'interest_graph' must be InterestGraph, got {type(interest_graph)!r}")
        if embedding_profile is not None and not isinstance(embedding_profile, TopicEmbeddingProfile):
            raise TypeError(f"'embedding_profile' must be TopicEmbeddingProfile or None")
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError(f"'learning_rate' must be in (0, 1], got {learning_rate!r}")
        if expected_read_seconds <= 0:
            raise ValueError(f"'expected_read_seconds' must be positive, got {expected_read_seconds!r}")
        if history_maxlen <= 0:
            raise ValueError(f"'history_maxlen' must be positive, got {history_maxlen!r}")

        self._graph = interest_graph
        self._profile = embedding_profile
        self._lr = learning_rate
        self._ref_read_s = expected_read_seconds
        self._history: Deque[FeedbackEvent] = deque(maxlen=history_maxlen)
        self._counts: Dict[str, int] = {}   # feedback_type.value → count
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_feedback(self, event: FeedbackEvent) -> List[InterestWeight]:
        """Process a single ``FeedbackEvent`` and update the interest graph.

        Args:
            event: Validated ``FeedbackEvent``.

        Returns:
            List of ``InterestWeight`` objects (one per affected topic).

        Raises:
            TypeError: If *event* is not a ``FeedbackEvent``.
        """
        if not isinstance(event, FeedbackEvent):
            raise TypeError(f"Expected FeedbackEvent, got {type(event)!r}")

        delta = self._effective_delta(event)
        updated: List[InterestWeight] = []

        for topic_id in event.topic_ids:
            iw = self._graph.update_weight(topic_id, delta, learning_rate=self._lr)
            updated.append(iw)

        # Update embedding profile if embedding is provided
        if self._profile and event.embedding:
            emb_weight = max(0.1, abs(delta) / 0.20)  # heavier for strong signals
            try:
                self._profile.update(event.embedding, weight=emb_weight)
            except (ValueError, TypeError) as exc:
                logger.warning("FeedbackLearner: embedding update failed (%s)", exc)

        with self._lock:
            self._history.append(event)
            key = event.feedback_type.value
            self._counts[key] = self._counts.get(key, 0) + 1

        logger.debug(
            "FeedbackLearner: %s on %s (Δ=%.4f, topics=%d)",
            event.feedback_type.value, event.item_id, delta, len(event.topic_ids),
        )
        return updated

    def process_batch(self, events: List[FeedbackEvent]) -> Dict[str, int]:
        """Process a batch of events.

        Args:
            events: List of ``FeedbackEvent`` objects.

        Returns:
            Dict with ``{"processed": N, "topics_updated": M}`` stats.

        Raises:
            TypeError: If *events* is not a list.
        """
        if not isinstance(events, list):
            raise TypeError(f"'events' must be a list, got {type(events)!r}")
        topics_set: set = set()
        for event in events:
            updated = self.process_feedback(event)
            topics_set.update(iw.topic_id for iw in updated)
        return {"processed": len(events), "topics_updated": len(topics_set)}

    def get_history(self, limit: Optional[int] = None) -> List[FeedbackEvent]:
        """Return recent feedback events (newest first).

        Args:
            limit: Maximum number of events to return (None = all).

        Returns:
            List of ``FeedbackEvent`` objects.
        """
        with self._lock:
            history = list(self._history)
        history.reverse()
        return history[:limit] if limit is not None else history

    def summary_stats(self) -> Dict[str, object]:
        """Return aggregate statistics across all processed events."""
        with self._lock:
            counts = dict(self._counts)
            total = sum(counts.values())
        positive = sum(v for k, v in counts.items() if FEEDBACK_DELTA.get(k, 0) > 0)
        negative = sum(v for k, v in counts.items() if FEEDBACK_DELTA.get(k, 0) < 0)
        return {
            "total_events": total,
            "positive_events": positive,
            "negative_events": negative,
            "by_type": counts,
            "top_interests": [
                {"topic_id": iw.topic_id, "weight": iw.weight}
                for iw in self._graph.top_interests(k=5)
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _effective_delta(self, event: FeedbackEvent) -> float:
        """Compute effective weight delta, optionally scaled by dwell time."""
        base = FEEDBACK_DELTA.get(event.feedback_type.value, 0.0)
        if event.feedback_type == FeedbackType.READ_COMPLETE and event.implicit_dwell_seconds is not None:
            dwell_scale = min(1.0, event.implicit_dwell_seconds / self._ref_read_s)
            base = base * dwell_scale
        return base

