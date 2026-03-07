"""Action scoring service - calculates multi-dimensional signal scores.

This module implements the scoring logic that prioritizes signals in the action queue.
It uses a composite scoring formula that balances multiple factors:
- Intent strength: How clear is the business intent?
- Reach potential: How many people could this impact?
- Urgency: How time-sensitive is this?
- Brand fit: How well does this align with our brand?
- Conversion likelihood: How likely is this to convert?
- Risk: What are the potential downsides?

Design principles:
- Transparent scoring with explainable components
- Configurable weights for different use cases
- Fast computation for real-time classification
- Learning-ready (weights can be optimized from outcomes)
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, Optional

from app.core.models import ContentItem, SourcePlatform
from app.core.signal_models import SignalType

logger = logging.getLogger(__name__)


class ActionScorer:
    """Calculate composite action scores for signal prioritization.

    The action score is a weighted combination of multiple factors:
    action_score = w1*intent + w2*reach + w3*urgency + w4*brand_fit + w5*conversion - w6*risk

    All scores are normalized to 0-1 scale for consistency.

    Attributes:
        weights: Scoring weights (can be tuned via learning loop)
    """

    # Default weights (can be overridden)
    DEFAULT_WEIGHTS = {
        'intent': 0.25,  # How clear is the business intent?
        'reach': 0.20,  # Potential audience size
        'urgency': 0.20,  # Time sensitivity
        'brand_fit': 0.15,  # Alignment with brand
        'conversion': 0.15,  # Likelihood to convert
        'risk': 0.05,  # Potential downside (negative weight)
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Initialize action scorer.

        Args:
            weights: Optional custom weights (defaults to DEFAULT_WEIGHTS)
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

        # Validate weights sum to ~1.0
        weight_sum = sum(abs(v) for v in self.weights.values())
        if not (0.95 <= weight_sum <= 1.05):
            logger.warning(
                f"Weights sum to {weight_sum:.2f}, expected ~1.0. "
                "Normalizing weights."
            )
            # Normalize
            for key in self.weights:
                self.weights[key] /= weight_sum

    async def calculate_action_score(
        self,
        item: ContentItem,
        signal_type: SignalType,
    ) -> Dict[str, float]:
        """Calculate all scores for a content item.

        Args:
            item: Content item to score
            signal_type: Classified signal type

        Returns:
            Dictionary with all scores:
            - action_score: Composite score (0-1)
            - urgency_score: Time sensitivity (0-1)
            - impact_score: Business value potential (0-1)
            - confidence_score: Classification confidence (0-1)
            - component_scores: Individual component scores
        """
        # Calculate individual components
        intent = self._calculate_intent_strength(item, signal_type)
        reach = self._calculate_reach(item)
        urgency = self._calculate_urgency(item, signal_type)
        brand_fit = self._calculate_brand_fit(item)
        conversion = self._calculate_conversion_likelihood(item, signal_type)
        risk = self._calculate_risk(item)

        # Calculate composite action score
        action_score = (
            self.weights['intent'] * intent +
            self.weights['reach'] * reach +
            self.weights['urgency'] * urgency +
            self.weights['brand_fit'] * brand_fit +
            self.weights['conversion'] * conversion -
            self.weights['risk'] * risk
        )

        # Clamp to [0, 1]
        action_score = max(0.0, min(1.0, action_score))

        # Impact score is combination of reach and conversion
        impact_score = (reach + conversion) / 2.0

        # Confidence score is combination of intent and brand_fit
        confidence_score = (intent + brand_fit) / 2.0

        logger.debug(
            f"Scored item {item.id}: action={action_score:.2f}, "
            f"urgency={urgency:.2f}, impact={impact_score:.2f}, "
            f"confidence={confidence_score:.2f}"
        )

        return {
            'action_score': action_score,
            'urgency_score': urgency,
            'impact_score': impact_score,
            'confidence_score': confidence_score,
            'component_scores': {
                'intent': intent,
                'reach': reach,
                'urgency': urgency,
                'brand_fit': brand_fit,
                'conversion': conversion,
                'risk': risk,
            },
        }

    def _calculate_intent_strength(
        self,
        item: ContentItem,
        signal_type: SignalType,
    ) -> float:
        """Calculate intent strength score (0-1).

        How clear and strong is the business intent?

        Args:
            item: Content item
            signal_type: Signal type

        Returns:
            Intent strength (0-1)
        """
        # Base score by signal type
        # Some signal types have inherently stronger intent
        base_scores = {
            SignalType.LEAD_OPPORTUNITY: 0.9,  # Very clear intent
            SignalType.CHURN_RISK: 0.85,  # Clear dissatisfaction
            SignalType.COMPETITOR_WEAKNESS: 0.75,  # Moderate intent
            SignalType.PRODUCT_CONFUSION: 0.7,  # Question, not commitment
            SignalType.FEATURE_REQUEST_PATTERN: 0.65,  # Wish, not need
            SignalType.TREND_TO_CONTENT: 0.5,  # Informational
        }

        score = base_scores.get(signal_type, 0.5)

        # Boost for explicit keywords
        text = (item.title + " " + (item.raw_text or "")).lower()

        high_intent_keywords = [
            'need', 'looking for', 'switching', 'alternative',
            'recommend', 'suggest', 'help', 'urgent'
        ]

        for keyword in high_intent_keywords:
            if keyword in text:
                score = min(1.0, score + 0.05)

        return score

    def _calculate_reach(self, item: ContentItem) -> float:
        """Calculate reach potential score (0-1).

        How many people could this impact?
        Based on author followers, engagement, and platform.

        Args:
            item: Content item

        Returns:
            Reach score (0-1)
        """
        score = 0.5  # Base score

        # Platform reach multipliers
        platform_multipliers = {
            SourcePlatform.REDDIT: 1.1,  # Good reach in communities
            SourcePlatform.YOUTUBE: 1.3,  # High viral potential
            SourcePlatform.FACEBOOK: 1.2,  # Large audience
            SourcePlatform.INSTAGRAM: 1.2,  # Visual content reach
        }

        multiplier = platform_multipliers.get(item.source_platform, 1.0)
        score *= multiplier

        # Engagement boost (from metadata)
        if item.metadata and 'engagement_count' in item.metadata:
            engagement_count = item.metadata['engagement_count']
            # Logarithmic scale: 10 engagements = 0.1, 100 = 0.2, 1000 = 0.3
            engagement_boost = min(0.3, math.log10(engagement_count + 1) / 10)
            score += engagement_boost

        # Author influence (if available in metadata)
        if item.metadata and 'author_followers' in item.metadata:
            followers = item.metadata['author_followers']
            # Logarithmic scale
            follower_boost = min(0.2, math.log10(followers + 1) / 20)
            score += follower_boost

        return min(1.0, score)

    def _calculate_urgency(
        self,
        item: ContentItem,
        signal_type: SignalType,
    ) -> float:
        """Calculate urgency score (0-1).

        How time-sensitive is this signal?
        Based on signal type, recency, and content indicators.

        Args:
            item: Content item
            signal_type: Signal type

        Returns:
            Urgency score (0-1)
        """
        # Base urgency by signal type
        base_urgency = {
            SignalType.CHURN_RISK: 0.95,  # Critical
            SignalType.SUPPORT_ESCALATION: 0.9,  # Very urgent
            SignalType.LEAD_OPPORTUNITY: 0.8,  # Time-sensitive
            SignalType.MISINFORMATION_RISK: 0.75,  # Needs quick response
            SignalType.COMPETITOR_WEAKNESS: 0.5,  # Moderate
            SignalType.PRODUCT_CONFUSION: 0.6,  # Should respond soon
            SignalType.TREND_TO_CONTENT: 0.4,  # Can wait
            SignalType.FEATURE_REQUEST_PATTERN: 0.3,  # Low urgency
        }

        score = base_urgency.get(signal_type, 0.5)

        # Recency boost
        if item.published_at:
            age_hours = (datetime.utcnow() - item.published_at).total_seconds() / 3600

            # Decay urgency over time
            # Fresh content (< 1 hour) gets full score
            # After 24 hours, urgency drops by 50%
            if age_hours < 1:
                recency_factor = 1.0
            elif age_hours < 24:
                recency_factor = 1.0 - (age_hours / 48)  # Gradual decay
            else:
                recency_factor = 0.5  # Older content less urgent

            score *= recency_factor

        # Urgency keywords
        text = (item.title + " " + (item.raw_text or "")).lower()
        urgency_keywords = ['urgent', 'asap', 'immediately', 'now', 'today']

        for keyword in urgency_keywords:
            if keyword in text:
                score = min(1.0, score + 0.1)
                break

        return min(1.0, score)

    def _calculate_brand_fit(self, item: ContentItem) -> float:
        """Calculate brand fit score (0-1).

        How well does this align with our brand and values?

        Args:
            item: Content item

        Returns:
            Brand fit score (0-1)
        """
        # Default moderate fit
        score = 0.7

        # Platform fit
        # Some platforms align better with certain brands
        professional_platforms = {
            SourcePlatform.REDDIT,  # Tech-savvy communities
            SourcePlatform.YOUTUBE,  # Professional content
        }

        if item.source_platform in professional_platforms:
            score += 0.1

        # Content quality indicators
        text = item.raw_text or item.title

        # Positive indicators
        if text and len(text) > 100:  # Substantive content
            score += 0.05

        # Negative indicators (reduce brand fit)
        negative_keywords = [
            'spam', 'scam', 'fake', 'clickbait',
            'offensive', 'inappropriate'
        ]

        text_lower = text.lower() if text else ""
        for keyword in negative_keywords:
            if keyword in text_lower:
                score -= 0.2
                break

        return max(0.0, min(1.0, score))

    def _calculate_conversion_likelihood(
        self,
        item: ContentItem,
        signal_type: SignalType,
    ) -> float:
        """Calculate conversion likelihood score (0-1).

        How likely is this to lead to a business outcome?
        (signup, demo, sale, etc.)

        Args:
            item: Content item
            signal_type: Signal type

        Returns:
            Conversion likelihood (0-1)
        """
        # Base conversion likelihood by signal type
        base_conversion = {
            SignalType.LEAD_OPPORTUNITY: 0.7,  # High conversion potential
            SignalType.PRODUCT_CONFUSION: 0.5,  # Moderate (support -> conversion)
            SignalType.COMPETITOR_WEAKNESS: 0.6,  # Good opportunity
            SignalType.CHURN_RISK: 0.4,  # Retention, not new conversion
            SignalType.FEATURE_REQUEST_PATTERN: 0.3,  # Low direct conversion
            SignalType.TREND_TO_CONTENT: 0.2,  # Indirect conversion
        }

        score = base_conversion.get(signal_type, 0.4)

        # Buying intent keywords
        text = (item.title + " " + (item.raw_text or "")).lower()
        buying_keywords = [
            'buy', 'purchase', 'pricing', 'cost', 'demo',
            'trial', 'sign up', 'get started', 'budget'
        ]

        for keyword in buying_keywords:
            if keyword in text:
                score = min(1.0, score + 0.15)
                break

        return score

    def _calculate_risk(self, item: ContentItem) -> float:
        """Calculate risk score (0-1).

        What are the potential downsides of engaging?
        Higher risk = lower action score (negative weight)

        Args:
            item: Content item

        Returns:
            Risk score (0-1, higher = more risky)
        """
        risk = 0.1  # Base low risk

        text = (item.title + " " + (item.raw_text or "")).lower()

        # High-risk indicators
        risk_indicators = [
            ('legal', 0.3),
            ('lawsuit', 0.4),
            ('regulation', 0.2),
            ('controversy', 0.25),
            ('scandal', 0.3),
            ('offensive', 0.35),
            ('political', 0.2),
        ]

        for keyword, risk_increase in risk_indicators:
            if keyword in text:
                risk += risk_increase

        # Very negative sentiment = higher risk (from metadata)
        if item.metadata and 'sentiment_score' in item.metadata:
            sentiment = item.metadata['sentiment_score']
            if sentiment < -0.7:
                risk += 0.15

        return min(1.0, risk)

