"""User digest ranker.

Produces a ranked ``List[RankedDigestItem]`` from a pool of
``DigestCandidate`` objects by compositing six scoring signals:

Signal                 Default weight  Source
─────────────────────  ─────────────  ──────────────────────────────
topic_relevance        0.30           Jaccard(item_topics, top-user-topics)
embedding_similarity   0.25           Cosine(item_embedding, user_centroid)
recency                0.20           Exponential decay on age in hours
engagement             0.10           Candidate.engagement_score (pass-through)
trust                  0.10           Candidate.trust_score (pass-through)
novelty                0.05           Candidate.novelty_score (pass-through)

The six scores are combined via a weighted average.  An optional
``personalization_boost`` (derived from ``FeedbackLearner`` history)
is blended on top with a configurable ``boost_weight``.

An instance of ``NoveltyRelevanceTradeoff`` can be supplied to apply
the dynamic α-blending of topic_relevance and novelty scores.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, List, Optional

from app.personalization.interest_graph import InterestGraph
from app.personalization.models import (
    DigestCandidate,
    FeedbackEvent,
    RankedDigestItem,
    RankingWeights,
)
from app.personalization.topic_embedding_profile import TopicEmbeddingProfile, _cosine_similarity
from app.personalization.novelty_scorer import NoveltyScorer

logger = logging.getLogger(__name__)

_RECENCY_HALF_LIFE_HOURS: float = 24.0   # score halves every 24 h


def _recency_score(published_at: Optional[datetime], half_life_hours: float = _RECENCY_HALF_LIFE_HOURS) -> float:
    """Exponential decay score based on item age."""
    if published_at is None:
        return 0.5  # neutral when unknown
    now = datetime.now(timezone.utc)
    age_hours = max(0.0, (now - published_at).total_seconds() / 3600.0)
    return math.exp(-age_hours / half_life_hours)


def _topic_relevance(item_topics: List[str], top_topics: List[str]) -> float:
    """Fraction of item topics that appear in user's top interests."""
    if not item_topics or not top_topics:
        return 0.0
    top_set = set(t.lower() for t in top_topics)
    item_set = set(t.lower() for t in item_topics)
    intersection = len(item_set & top_set)
    return intersection / len(top_set)


class UserDigestRanker:
    """Multi-signal digest ranker personalized to a user's interest graph.

    Args:
        interest_graph:       Per-user ``InterestGraph``; may be None (scores zero).
        embedding_profile:    Per-user ``TopicEmbeddingProfile``; may be None.
        weights:              ``RankingWeights`` (normalized automatically).
        novelty_tradeoff:     Optional ``NoveltyRelevanceTradeoff`` for dynamic α.
        novelty_scorer:       Optional live ``NoveltyScorer`` (overrides pre-computed
                              ``DigestCandidate.novelty_score``).
        watchlist_graph:      Optional ``WatchlistGraph``.  Items whose ``topic_ids``
                              or ``entity_ids`` match a watched node receive an
                              additional boost proportional to that node's priority.
        feedback_learner:     Optional ``FeedbackLearner``.  When provided,
                              :meth:`apply_feedback` delegates to it so that the
                              embedded ``InterestGraph`` is updated after each user
                              interaction.  The same ``InterestGraph`` instance
                              should be shared between this ranker and the learner.
        top_interest_k:       How many top interests to use for topic relevance.
        recency_half_life_h:  Half-life in hours for recency score.
        boost_weight:         Weight of personalization_boost in final score [0, 1].
        watchlist_boost_weight: Additional weight for watchlist-matched items [0, 1].
    """

    def __init__(
        self,
        interest_graph: Optional[InterestGraph] = None,
        embedding_profile: Optional[TopicEmbeddingProfile] = None,
        weights: Optional[RankingWeights] = None,
        novelty_tradeoff: Optional[Any] = None,
        novelty_scorer: Optional[NoveltyScorer] = None,
        watchlist_graph: Optional[Any] = None,
        feedback_learner: Optional[Any] = None,
        top_interest_k: int = 20,
        recency_half_life_h: float = _RECENCY_HALF_LIFE_HOURS,
        boost_weight: float = 0.1,
        watchlist_boost_weight: float = 0.10,
    ) -> None:
        if interest_graph is not None and not isinstance(interest_graph, InterestGraph):
            raise TypeError(f"'interest_graph' must be InterestGraph or None")
        if embedding_profile is not None and not isinstance(embedding_profile, TopicEmbeddingProfile):
            raise TypeError(f"'embedding_profile' must be TopicEmbeddingProfile or None")
        if novelty_scorer is not None and not isinstance(novelty_scorer, NoveltyScorer):
            raise TypeError(f"'novelty_scorer' must be NoveltyScorer or None")
        if top_interest_k <= 0:
            raise ValueError(f"'top_interest_k' must be positive, got {top_interest_k!r}")
        if recency_half_life_h <= 0:
            raise ValueError(f"'recency_half_life_h' must be positive, got {recency_half_life_h!r}")
        if not (0.0 <= boost_weight <= 1.0):
            raise ValueError(f"'boost_weight' must be in [0, 1], got {boost_weight!r}")
        if not (0.0 <= watchlist_boost_weight <= 1.0):
            raise ValueError(
                f"'watchlist_boost_weight' must be in [0, 1], got {watchlist_boost_weight!r}"
            )

        self._graph = interest_graph
        self._profile = embedding_profile
        self._weights = (weights or RankingWeights()).normalized()
        self._tradeoff = novelty_tradeoff
        self._novelty_scorer = novelty_scorer
        # WatchlistGraph — duck-typed so tests can inject plain mocks
        self._watchlist_graph = watchlist_graph
        # FeedbackLearner — duck-typed; must have .process_feedback(event)
        self._feedback_learner = feedback_learner
        self._top_k = top_interest_k
        self._half_life = recency_half_life_h
        self._boost_weight = boost_weight
        self._watchlist_boost_weight = watchlist_boost_weight

    def rank(
        self,
        candidates: List[DigestCandidate],
        top_k: Optional[int] = None,
    ) -> List[RankedDigestItem]:
        """Rank *candidates* by personalized multi-signal score.

        Args:
            candidates: List of ``DigestCandidate`` objects.
            top_k:      Maximum items to return (None = return all).

        Returns:
            List of ``RankedDigestItem`` sorted by ``final_score`` descending.

        Raises:
            TypeError:  If *candidates* is not a list.
            ValueError: If *top_k* ≤ 0.
        """
        if not isinstance(candidates, list):
            raise TypeError(f"'candidates' must be a list, got {type(candidates)!r}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"'top_k' must be positive, got {top_k!r}")

        if not candidates:
            return []

        top_topics = (
            [iw.topic_id for iw in self._graph.top_interests(k=self._top_k)]
            if self._graph else []
        )

        scored: List[tuple[float, RankedDigestItem]] = []
        for candidate in candidates:
            item = self._score_candidate(candidate, top_topics)
            scored.append((item.final_score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        limit = top_k if top_k is not None else len(scored)

        result: List[RankedDigestItem] = []
        for rank_idx, (_, item) in enumerate(scored[:limit], start=1):
            result.append(item.model_copy(update={"rank": rank_idx}))

        logger.debug(
            "UserDigestRanker: ranked %d/%d candidates",
            len(result), len(candidates),
        )
        return result

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _score_candidate(
        self, candidate: DigestCandidate, top_topics: List[str]
    ) -> RankedDigestItem:
        w = self._weights

        # 1. Topic relevance
        t_rel = _topic_relevance(candidate.topic_ids, top_topics)

        # 2. Embedding similarity
        e_sim = 0.0
        if self._profile and candidate.embedding:
            try:
                raw = self._profile.similarity(candidate.embedding)
                e_sim = (raw + 1.0) / 2.0  # map [-1,1] → [0,1]
            except (ValueError, TypeError):
                e_sim = 0.0

        # 3. Recency
        rec = _recency_score(candidate.published_at, self._half_life)

        # 4–6. Pass-through signals
        eng = candidate.engagement_score
        tru = candidate.trust_score
        # Use live NoveltyScorer if provided; fall back to pre-computed field
        if self._novelty_scorer is not None:
            try:
                nov = self._novelty_scorer.score(candidate)
            except Exception as exc:
                logger.warning(
                    "UserDigestRanker: NoveltyScorer.score failed for item=%s: %s",
                    candidate.item_id, exc,
                )
                nov = candidate.novelty_score
        else:
            nov = candidate.novelty_score

        # Apply novelty-relevance tradeoff if configured
        if self._tradeoff is not None:
            alpha = getattr(self._tradeoff, "current_alpha", 0.3)
            blended_relevance = self._tradeoff.blend_scores(t_rel, nov, alpha)
            t_rel = blended_relevance
            nov = 0.0  # folded into t_rel; zero out so total weight unchanged

        base_score = (
            w.topic_relevance * t_rel
            + w.embedding_similarity * e_sim
            + w.recency * rec
            + w.engagement * eng
            + w.trust * tru
            + w.novelty * nov
        )

        # Personalization boost: proportional to weight sum of matched topics
        boost = 0.0
        if self._graph and candidate.topic_ids:
            match_weights = [
                self._graph.get_weight(tid) or 0.0
                for tid in candidate.topic_ids
            ]
            if match_weights:
                boost = min(1.0, sum(match_weights) / len(match_weights))

        # Watchlist boost: items whose topic/entity IDs match a watched node
        # receive a boost proportional to that node's priority.
        watchlist_boost = 0.0
        if self._watchlist_graph is not None:
            try:
                candidate_ids = set(candidate.topic_ids) | set(candidate.entity_ids)
                if candidate_ids:
                    watched = self._watchlist_graph.watched_nodes()
                    matched = [n for n in watched if n.node_id in candidate_ids]
                    if matched:
                        watchlist_boost = min(
                            1.0,
                            sum(n.priority for n in matched) / len(matched),
                        )
            except Exception as exc:
                logger.warning(
                    "UserDigestRanker: watchlist_graph.watched_nodes() failed "
                    "for item=%s: %s", candidate.item_id, exc,
                )

        # Combine: base_score → personalization boost → watchlist boost
        # Order of blending: first apply personalization boost, then watchlist
        after_boost = min(1.0, base_score * (1.0 - self._boost_weight) + boost * self._boost_weight)
        final = min(
            1.0,
            after_boost * (1.0 - self._watchlist_boost_weight)
            + watchlist_boost * self._watchlist_boost_weight,
        )

        return RankedDigestItem(
            item_id=candidate.item_id,
            rank=1,  # placeholder — overwritten by rank()
            final_score=round(final, 5),
            relevance_score=round(t_rel, 5),
            novelty_score=round(nov, 5),
            recency_score=round(rec, 5),
            engagement_score=round(eng, 5),
            trust_score=round(tru, 5),
            personalization_boost=round(boost, 5),
            score_breakdown={
                "topic_relevance": round(t_rel, 4),
                "embedding_similarity": round(e_sim, 4),
                "recency": round(rec, 4),
                "engagement": round(eng, 4),
                "trust": round(tru, 4),
                "novelty": round(nov, 4),
                "boost": round(boost, 4),
                "watchlist_boost": round(watchlist_boost, 4),
            },
        )

    # ------------------------------------------------------------------
    # Feedback integration
    # ------------------------------------------------------------------

    @property
    def watchlist_graph(self) -> Optional[Any]:
        """The ``WatchlistGraph`` injected at construction, or ``None``."""
        return self._watchlist_graph

    @property
    def feedback_learner(self) -> Optional[Any]:
        """The ``FeedbackLearner`` injected at construction, or ``None``."""
        return self._feedback_learner

    def apply_feedback(self, event: "FeedbackEvent") -> None:
        """Process a user feedback event and update interest graph + tradeoff.

        Delegates to the embedded ``FeedbackLearner`` (if provided) to update
        the ``InterestGraph`` weights.  If a ``NoveltyRelevanceTradeoff`` is
        also present, it is updated so that future ``rank()`` calls reflect the
        shifted exploration-vs-exploitation balance.

        This is the canonical integration point between user interactions and
        the personalization layer.

        Args:
            event: A ``FeedbackEvent`` representing one user interaction.

        Raises:
            TypeError: If *event* is not a ``FeedbackEvent``.

        Notes:
            Errors from the ``FeedbackLearner`` or ``NoveltyRelevanceTradeoff``
            are caught, logged, and do NOT propagate — the caller must not assume
            these subsystems are infallible.
        """
        if not isinstance(event, FeedbackEvent):
            raise TypeError(f"Expected FeedbackEvent, got {type(event)!r}")

        # Update interest graph via FeedbackLearner
        if self._feedback_learner is not None:
            try:
                self._feedback_learner.process_feedback(event)
            except Exception as exc:
                logger.warning(
                    "UserDigestRanker.apply_feedback: FeedbackLearner failed "
                    "for user=%s item=%s: %s",
                    event.user_id, event.item_id, exc,
                )

        # Update novelty-vs-relevance tradeoff alpha
        if self._tradeoff is not None:
            try:
                self._tradeoff.update_from_feedback(event)
            except Exception as exc:
                logger.warning(
                    "UserDigestRanker.apply_feedback: NoveltyRelevanceTradeoff "
                    "update failed for user=%s: %s",
                    event.user_id, exc,
                )

