"""Personalization & User Adaptation — Phase 3.

Builds a per-user interest graph, maintains a topic-embedding profile,
learns online from explicit and implicit feedback signals, and reranks
digest candidates with a configurable multi-signal ranker.

Public exports
--------------
Models:
    FeedbackType, InterestEdgeType, FEEDBACK_DELTA,
    InterestWeight, FeedbackEvent, DigestCandidate,
    RankedDigestItem, NoveltyRelevanceConfig, RankingWeights

Components:
    InterestGraph, TopicEmbeddingProfile, FeedbackLearner,
    UserDigestRanker, NoveltyRelevanceTradeoff
"""

from app.personalization.models import (
    FEEDBACK_DELTA,
    DigestCandidate,
    FeedbackEvent,
    FeedbackType,
    InterestEdgeType,
    InterestWeight,
    NoveltyRelevanceConfig,
    RankedDigestItem,
    RankingWeights,
)
from app.personalization.interest_graph import InterestGraph
from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
from app.personalization.feedback_learner import FeedbackLearner
from app.personalization.user_digest_ranker import UserDigestRanker
from app.personalization.novelty_vs_relevance_tradeoff import NoveltyRelevanceTradeoff
from app.personalization.user_persona import (
    REASONING_TRAITS,
    STYLE_TRAITS,
    Trait,
    UserPersonaProfile,
)
from app.personalization.persona_bayes import BetaBinomialTrait, PopulationPrior
from app.personalization.persona_bandit import ThompsonDirectiveSelector
from app.personalization.persona_predictor import NextStylePredictor

__all__ = [
    # Models
    "FEEDBACK_DELTA",
    "DigestCandidate",
    "FeedbackEvent",
    "FeedbackType",
    "InterestEdgeType",
    "InterestWeight",
    "NoveltyRelevanceConfig",
    "RankedDigestItem",
    "RankingWeights",
    # Components
    "FeedbackLearner",
    "InterestGraph",
    "NoveltyRelevanceTradeoff",
    "TopicEmbeddingProfile",
    "UserDigestRanker",
    # Persona memory
    "STYLE_TRAITS",
    "REASONING_TRAITS",
    "Trait",
    "UserPersonaProfile",
    # Bayesian / bandit / predictive extensions
    "BetaBinomialTrait",
    "PopulationPrior",
    "ThompsonDirectiveSelector",
    "NextStylePredictor",
]

