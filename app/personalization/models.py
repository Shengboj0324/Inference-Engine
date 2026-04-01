"""Shared Pydantic models for the personalization / user-adaptation package.

All models follow the Phase 2 conventions:
- ``frozen=True`` on value-object models (InterestWeight, FeedbackEvent, RankedDigestItem)
- Mutable configuration models (DigestCandidate, NoveltyRelevanceConfig, RankingWeights)
- Field-level validation with descriptive messages
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class FeedbackType(str, Enum):
    """User feedback signal type — explicit or implicit."""

    LIKE = "like"                    # Explicit: thumbs up / star
    DISMISS = "dismiss"              # Explicit: hide / not interested
    SAVE = "save"                    # Explicit: bookmark / save for later
    SHARE = "share"                  # Explicit: forwarded to others
    CLICK = "click"                  # Implicit: opened item
    READ_COMPLETE = "read_complete"  # Implicit: read to the end
    SCROLL_PAST = "scroll_past"      # Implicit: skipped without clicking


class InterestEdgeType(str, Enum):
    """Typed relationship between nodes in the interest graph."""

    SUBTOPIC_OF = "subtopic_of"          # Fine-grained → broad category
    RELATED_TO = "related_to"            # Peer topic association
    CO_OCCURS_WITH = "co_occurs_with"    # Frequently seen together
    CAUSED_BY = "caused_by"              # Causal relationship


#: How much each feedback type nudges a topic weight (may be negative).
FEEDBACK_DELTA: Dict[str, float] = {
    FeedbackType.LIKE.value:          +0.15,
    FeedbackType.DISMISS.value:       -0.12,
    FeedbackType.SAVE.value:          +0.20,
    FeedbackType.SHARE.value:         +0.18,
    FeedbackType.CLICK.value:         +0.05,
    FeedbackType.READ_COMPLETE.value: +0.10,
    FeedbackType.SCROLL_PAST.value:   -0.03,
}


class InterestWeight(BaseModel, frozen=True):
    """Per-topic interest weight for a single user.

    Attributes:
        topic_id:      Topic identifier string.
        weight:        Interest weight in [0, 1].
        confidence:    Reliability of this weight estimate [0, 1].
        decay_factor:  Daily multiplicative decay [0, 1].
        last_updated:  UTC datetime of last update.
        update_count:  Total feedback events that shaped this weight.
    """

    topic_id: str = Field(..., min_length=1)
    weight: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    decay_factor: float = Field(default=0.995, ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    update_count: int = Field(default=0, ge=0)


class FeedbackEvent(BaseModel, frozen=True):
    """A single user feedback event on a content item.

    Attributes:
        user_id:                 User identifier.
        item_id:                 Content item identifier.
        feedback_type:           Explicit or implicit feedback type.
        topic_ids:               Topics associated with this item.
        entity_ids:              Canonical entity IDs in this item.
        timestamp:               UTC time of event.
        implicit_dwell_seconds:  Seconds user spent reading (implicit).
        source_platform:         Platform where item was consumed.
        embedding:               Optional item embedding for profile update.
    """

    user_id: str = Field(..., min_length=1)
    item_id: str = Field(..., min_length=1)
    feedback_type: FeedbackType
    topic_ids: List[str] = Field(default_factory=list)
    entity_ids: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    implicit_dwell_seconds: Optional[float] = Field(None, ge=0.0)
    source_platform: str = ""
    embedding: List[float] = Field(default_factory=list)


class DigestCandidate(BaseModel):
    """A content item candidate for inclusion in a personalized digest.

    Attributes:
        item_id:          Unique content item identifier.
        title:            Display title.
        topic_ids:        Topic IDs extracted from content.
        entity_ids:       Canonical entity IDs in content.
        embedding:        Dense vector (empty if unavailable).
        published_at:     UTC publication time.
        trust_score:      Source trust [0, 1].
        engagement_score: Social engagement [0, 1].
        novelty_score:    Pre-computed novelty from Phase 2 [0, 1].
        source_platform:  Platform name.
        raw_text:         Truncated raw text for fallback embedding.
        metadata:         Arbitrary extra metadata.
    """

    item_id: str = Field(..., min_length=1)
    title: str = ""
    topic_ids: List[str] = Field(default_factory=list)
    entity_ids: List[str] = Field(default_factory=list)
    embedding: List[float] = Field(default_factory=list)
    published_at: Optional[datetime] = None
    trust_score: float = Field(default=0.5, ge=0.0, le=1.0)
    engagement_score: float = Field(default=0.5, ge=0.0, le=1.0)
    novelty_score: float = Field(default=0.5, ge=0.0, le=1.0)
    source_platform: str = ""
    raw_text: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RankedDigestItem(BaseModel, frozen=True):
    """A single ranked item in a personalized digest output.

    Attributes:
        item_id:               Content item identifier.
        rank:                  1-indexed position in digest.
        final_score:           Composite personalization score [0, 1].
        relevance_score:       Topic + embedding relevance [0, 1].
        novelty_score:         Content novelty [0, 1].
        recency_score:         Time-based freshness [0, 1].
        engagement_score:      Social engagement signal [0, 1].
        trust_score:           Source trust [0, 1].
        personalization_boost: Extra boost from feedback history [0, 1].
        score_breakdown:       Per-signal score dict.
    """

    item_id: str
    rank: int = Field(..., ge=1)
    final_score: float = Field(..., ge=0.0, le=1.0)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty_score: float = Field(default=0.0, ge=0.0, le=1.0)
    recency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    engagement_score: float = Field(default=0.0, ge=0.0, le=1.0)
    trust_score: float = Field(default=0.0, ge=0.0, le=1.0)
    personalization_boost: float = Field(default=0.0, ge=0.0, le=1.0)
    score_breakdown: Dict[str, float] = Field(default_factory=dict)




class NoveltyRelevanceConfig(BaseModel):
    """Configuration for the dynamic novelty-vs-relevance tradeoff.

    Attributes:
        alpha:              Blend factor [0, 1]; 0 = pure relevance, 1 = pure novelty.
        auto_adapt:         Auto-adjust alpha from engagement feedback.
        novelty_push_rate:  Alpha increase when user explores new topics.
        relevance_pull_rate: Alpha decrease when user saves / reads-to-end.
        min_alpha:          Floor for auto-adapted alpha.
        max_alpha:          Ceiling for auto-adapted alpha.
        exploration_window: Recent events window for trend detection.
    """

    alpha: float = Field(default=0.30, ge=0.0, le=1.0)
    auto_adapt: bool = True
    novelty_push_rate: float = Field(default=0.05, ge=0.0, le=0.5)
    relevance_pull_rate: float = Field(default=0.03, ge=0.0, le=0.5)
    min_alpha: float = Field(default=0.05, ge=0.0, le=1.0)
    max_alpha: float = Field(default=0.80, ge=0.0, le=1.0)
    exploration_window: int = Field(default=20, ge=1)

    @field_validator("max_alpha")
    @classmethod
    def _max_ge_min(cls, v: float, info) -> float:
        min_alpha = info.data.get("min_alpha", 0.05)
        if v < min_alpha:
            raise ValueError(f"'max_alpha' ({v}) must be >= 'min_alpha' ({min_alpha})")
        return v


class RankingWeights(BaseModel):
    """Configurable signal weights for the digest ranker.

    All weights must be non-negative.  Use ``normalized()`` to get
    a copy that sums exactly to 1.0.

    Attributes:
        topic_relevance:      Weight for topic overlap with interest graph.
        embedding_similarity: Weight for embedding cosine similarity.
        recency:              Weight for time-based freshness.
        engagement:           Weight for social engagement signals.
        trust:                Weight for source trust score.
        novelty:              Weight for content novelty (from Phase 2).
    """

    topic_relevance: float = Field(default=0.30, ge=0.0)
    embedding_similarity: float = Field(default=0.25, ge=0.0)
    recency: float = Field(default=0.20, ge=0.0)
    engagement: float = Field(default=0.10, ge=0.0)
    trust: float = Field(default=0.10, ge=0.0)
    novelty: float = Field(default=0.05, ge=0.0)

    def total(self) -> float:
        """Return the raw sum of all weights."""
        return self.topic_relevance + self.embedding_similarity + self.recency + self.engagement + self.trust + self.novelty

    def normalized(self) -> "RankingWeights":
        """Return a copy with weights normalized to sum to 1.0."""
        t = self.total()
        if t == 0.0:
            raise ValueError("All ranking weights are zero; cannot normalize")
        return RankingWeights(
            topic_relevance=self.topic_relevance / t,
            embedding_similarity=self.embedding_similarity / t,
            recency=self.recency / t,
            engagement=self.engagement / t,
            trust=self.trust / t,
            novelty=self.novelty / t,
        )
