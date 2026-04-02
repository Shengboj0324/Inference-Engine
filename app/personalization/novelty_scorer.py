"""Content novelty scorer.

Computes a novelty score in [0, 1] for each ``DigestCandidate`` by comparing
its topic/entity fingerprint against a sliding window of recently shown items.

Algorithm
---------
For each candidate we compute a *weighted Jaccard similarity* against every
item in the history window, weighting more recent items more heavily via
exponential decay:

    novelty = 1 − weighted_mean(Jaccard(candidate, history_item), w_i=decay^i)

where ``i = 0`` is the most recent item.

When ``topic_ids`` are available they form the fingerprint set.  If the
candidate has no topics but has ``entity_ids``, those are used instead.  As a
final fallback the top-N token stems from ``raw_text`` are used.

The scorer is intentionally stateless per-score call; state is only mutated
by ``record_shown()``.  All mutable state is protected by ``threading.Lock``.

Public API
----------
    NoveltyScorer(window_size, decay_factor, min_novelty, top_text_tokens)
    .score(candidate)              → float in [min_novelty, 1.0]
    .score_batch(candidates)       → List[float]
    .record_shown(candidate)       → None  (add to window)
    .reset()                       → None
    .stats()                       → Dict[str, object]
"""

from __future__ import annotations

import logging
import re
import threading
from collections import deque
from typing import Deque, Dict, FrozenSet, List, Optional

from app.personalization.models import DigestCandidate

logger = logging.getLogger(__name__)

_STOP: FrozenSet[str] = frozenset(
    "the a an is are was were be been being have has had do does did "
    "will would could should may might shall can this that these those "
    "it its and or but if in on at to for of from with by about".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _text_fingerprint(text: str, top_n: int) -> FrozenSet[str]:
    """Return the top-N most-frequent non-stop tokens from *text*."""
    tokens = _TOKEN_RE.findall(text.lower())
    counts: Dict[str, int] = {}
    for t in tokens:
        if t not in _STOP and len(t) >= 3:
            counts[t] = counts.get(t, 0) + 1
    sorted_tokens = sorted(counts, key=lambda k: -counts[k])
    return frozenset(sorted_tokens[:top_n])


def _fingerprint(candidate: DigestCandidate, top_text_tokens: int) -> FrozenSet[str]:
    """Return a hashable fingerprint set for *candidate*."""
    if candidate.topic_ids:
        return frozenset(t.lower() for t in candidate.topic_ids)
    if candidate.entity_ids:
        return frozenset(e.lower() for e in candidate.entity_ids)
    if candidate.raw_text:
        return _text_fingerprint(candidate.raw_text, top_text_tokens)
    return frozenset()


def _jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    """Jaccard similarity of two sets; returns 0.0 when both are empty."""
    if not a and not b:
        return 0.0
    union = a | b
    inter = a & b
    return len(inter) / len(union)


class NoveltyScorer:
    """Sliding-window content novelty scorer.

    Args:
        window_size:     Maximum number of recently-shown items to retain.
        decay_factor:    Exponential decay applied to older history items.
                         1.0 = uniform; 0.0 = only the most recent item counts.
        min_novelty:     Floor for the returned novelty score (avoids zero).
        top_text_tokens: When falling back to raw text, use the top-N tokens.

    Raises:
        ValueError: On out-of-range construction parameters.
    """

    def __init__(
        self,
        window_size:     int   = 50,
        decay_factor:    float = 0.9,
        min_novelty:     float = 0.05,
        top_text_tokens: int   = 30,
    ) -> None:
        if window_size <= 0:
            raise ValueError(f"'window_size' must be > 0, got {window_size!r}")
        if not (0.0 <= decay_factor <= 1.0):
            raise ValueError(f"'decay_factor' must be in [0, 1], got {decay_factor!r}")
        if not (0.0 <= min_novelty <= 1.0):
            raise ValueError(f"'min_novelty' must be in [0, 1], got {min_novelty!r}")
        if top_text_tokens <= 0:
            raise ValueError(f"'top_text_tokens' must be > 0, got {top_text_tokens!r}")

        self._window_size     = window_size
        self._decay           = decay_factor
        self._min_novelty     = min_novelty
        self._top_text_tokens = top_text_tokens
        self._history: Deque[FrozenSet[str]] = deque(maxlen=window_size)
        self._shown_count: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, candidate: DigestCandidate) -> float:
        """Compute the novelty score for *candidate*.

        Does **not** mutate history — call :meth:`record_shown` separately
        when the item is actually included in a delivered digest.

        Args:
            candidate: ``DigestCandidate`` to score.

        Returns:
            Float in [``min_novelty``, 1.0].  Returns 1.0 when history is empty.

        Raises:
            TypeError: If *candidate* is not a ``DigestCandidate``.
        """
        if not isinstance(candidate, DigestCandidate):
            raise TypeError(f"Expected DigestCandidate, got {type(candidate)!r}")

        fp = _fingerprint(candidate, self._top_text_tokens)

        with self._lock:
            history_snapshot = list(self._history)

        if not history_snapshot:
            return 1.0

        total_weight = 0.0
        weighted_sim = 0.0
        for i, past_fp in enumerate(reversed(history_snapshot)):  # newest first
            w = self._decay ** i
            sim = _jaccard(fp, past_fp)
            weighted_sim += w * sim
            total_weight += w

        avg_sim = weighted_sim / total_weight if total_weight > 0 else 0.0
        novelty = max(self._min_novelty, min(1.0, 1.0 - avg_sim))
        logger.debug("NoveltyScorer: item=%s novelty=%.3f", candidate.item_id, novelty)
        return novelty

    def score_batch(self, candidates: List[DigestCandidate]) -> List[float]:
        """Score a list of candidates in order.

        Args:
            candidates: List of ``DigestCandidate`` objects.

        Returns:
            List of novelty floats in the same order as *candidates*.

        Raises:
            TypeError: If *candidates* is not a list.
        """
        if not isinstance(candidates, list):
            raise TypeError(f"'candidates' must be a list, got {type(candidates)!r}")
        return [self.score(c) for c in candidates]

    def record_shown(self, candidate: DigestCandidate) -> None:
        """Mark *candidate* as shown; adds its fingerprint to the history window.

        Args:
            candidate: ``DigestCandidate`` that was delivered to the user.

        Raises:
            TypeError: If *candidate* is not a ``DigestCandidate``.
        """
        if not isinstance(candidate, DigestCandidate):
            raise TypeError(f"Expected DigestCandidate, got {type(candidate)!r}")
        fp = _fingerprint(candidate, self._top_text_tokens)
        with self._lock:
            self._history.append(fp)
            self._shown_count += 1
        logger.debug(
            "NoveltyScorer.record_shown: item=%s history_size=%d",
            candidate.item_id, len(self._history),
        )

    def reset(self) -> None:
        """Clear history and reset the shown counter."""
        with self._lock:
            self._history.clear()
            self._shown_count = 0
        logger.info("NoveltyScorer: reset")

    def stats(self) -> Dict[str, object]:
        """Return current scorer statistics.

        Returns:
            Dict with ``window_size``, ``history_size``, ``shown_count``,
            ``decay_factor``, ``min_novelty``.
        """
        with self._lock:
            return {
                "window_size":   self._window_size,
                "history_size":  len(self._history),
                "shown_count":   self._shown_count,
                "decay_factor":  self._decay,
                "min_novelty":   self._min_novelty,
            }

