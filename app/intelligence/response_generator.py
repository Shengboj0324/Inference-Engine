"""Response generation engine with multi-variant generation and quality scoring.

This module orchestrates LLM-based response generation with A/B testing capabilities.
It generates multiple response variants with different tones and scores them for quality.
"""

import logging
import re
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from app.core.signal_models import (
    ActionableSignal,
    ResponseTone,
)
from app.intelligence.response_playbook import (
    ResponseChannel,
    ResponsePlaybook,
)
from app.llm.router import LLMRouter

logger = logging.getLogger(__name__)


class ResponseVariant(BaseModel):
    """A single response variant with quality scores."""

    content: str
    tone: ResponseTone
    channel: ResponseChannel
    clarity_score: float = Field(ge=0.0, le=1.0)
    tone_match_score: float = Field(ge=0.0, le=1.0)
    length_score: float = Field(ge=0.0, le=1.0)
    engagement_score: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


class ResponseGenerator:
    """Generate and score response variants for actionable signals.

    This class orchestrates the response generation pipeline:
    1. Select appropriate tones for variants
    2. Generate responses using LLM with playbook templates
    3. Score each variant for quality
    4. Return ranked variants
    """

    def __init__(
        self,
        router: Optional[LLMRouter] = None,
        playbook: Optional[ResponsePlaybook] = None,
        enable_quality_scoring: bool = True,
    ):
        """Initialize response generator.

        Args:
            router: LLM router for generation (optional, will create if not provided)
            playbook: Response playbook (optional, will create if not provided)
            enable_quality_scoring: Whether to enable quality scoring
        """
        self.router = router or LLMRouter()
        self.playbook = playbook or ResponsePlaybook()
        self.enable_quality_scoring = enable_quality_scoring

        # Quality scoring weights
        self.clarity_weight = 0.25
        self.tone_weight = 0.25
        self.length_weight = 0.20
        self.engagement_weight = 0.30

    async def generate_variants(
        self,
        signal: ActionableSignal,
        num_variants: int = 3,
        tones: Optional[List[ResponseTone]] = None,
        channel: ResponseChannel = ResponseChannel.REDDIT,
    ) -> List[ResponseVariant]:
        """Generate multiple response variants for a signal.

        Args:
            signal: Signal to respond to
            num_variants: Number of variants to generate (1-5)
            tones: Specific tones to use (defaults to signal's suggested tone + variations)
            channel: Target channel for responses

        Returns:
            List of response variants with quality scores
        """
        # Determine tones to use
        if not tones:
            tones = self._select_tones_for_variants(signal, num_variants)

        variants = []

        for i, tone in enumerate(tones[:num_variants], start=1):
            try:
                variant = await self._generate_single_variant(
                    signal=signal,
                    tone=tone,
                    channel=channel,
                    variant_number=i,
                )
                variants.append(variant)

                logger.info(
                    "Generated variant %d/%d for signal %s (tone=%s, score=%.2f)",
                    i, num_variants, signal.id, tone.value, variant.overall_score
                )

            except Exception as e:  # pylint: disable=broad-except
                logger.error("Failed to generate variant %d: %s", i, e)
                continue

        # Sort by overall score (highest first)
        variants.sort(key=lambda v: v.overall_score, reverse=True)

        best_score = variants[0].overall_score if variants else 0
        logger.info(
            "Generated %d variants for signal %s, best score: %.2f",
            len(variants), signal.id, best_score
        )

        return variants



    def _score_clarity(self, content: str) -> float:
        """Score response clarity based on structure and readability.

        Args:
            content: Response content

        Returns:
            Clarity score (0.0-1.0)
        """
        score = 0.5  # Base score

        # Check for proper sentence structure
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) >= 2:
            score += 0.15  # Multiple sentences

        # Check for capitalization
        if content and content[0].isupper():
            score += 0.10

        # Check for punctuation
        if any(p in content for p in ['.', '!', '?']):
            score += 0.10

        # Penalize very short responses
        if len(content) < 50:
            score -= 0.20

        # Penalize very long run-on sentences
        avg_sentence_length = len(content) / max(len(sentences), 1)
        if avg_sentence_length > 200:
            score -= 0.15

        return max(0.0, min(1.0, score))

    def _score_tone_match(self, content: str, tone: ResponseTone) -> float:
        """Score how well content matches the desired tone.

        Args:
            content: Response content
            tone: Target tone

        Returns:
            Tone match score (0.0-1.0)
        """
        score = 0.5  # Base score
        content_lower = content.lower()

        if tone == ResponseTone.PROFESSIONAL:
            # Professional indicators
            professional_words = ["however", "therefore", "appreciate", "regarding", "please"]
            score += sum(0.05 for word in professional_words if word in content_lower)

            # Penalize informal language
            informal_words = ["hey", "yeah", "gonna", "wanna", "lol"]
            score -= sum(0.1 for word in informal_words if word in content_lower)

        elif tone == ResponseTone.HELPFUL:
            # Helpful indicators
            friendly_words = ["hey", "thanks", "happy", "glad", "love", "awesome"]
            score += sum(0.06 for word in friendly_words if word in content_lower)

        elif tone == ResponseTone.TECHNICAL:
            # Technical indicators (longer words, specific terminology)
            words = content.split()
            avg_word_length = sum(len(w) for w in words) / max(len(words), 1)
            if avg_word_length > 5.5:
                score += 0.15

        elif tone == ResponseTone.SUPPORTIVE:
            # Supportive indicators
            supportive_words = ["understand", "appreciate", "sorry", "help", "support"]
            score += sum(0.06 for word in supportive_words if word in content_lower)

        elif tone == ResponseTone.FOUNDER_VOICE:
            # Founder voice indicators
            founder_words = ["recommend", "should", "best", "proven", "expert", "we", "our"]
            score += sum(0.05 for word in founder_words if word in content_lower)

        return max(0.0, min(1.0, score))

    def _score_length(self, content: str, channel: ResponseChannel) -> float:
        """Score response length appropriateness for channel.

        Args:
            content: Response content
            channel: Target channel

        Returns:
            Length score (0.0-1.0)
        """
        length = len(content)

        # Channel-specific ideal ranges
        ideal_ranges = {
            ResponseChannel.TWITTER: (150, 280),
            ResponseChannel.REDDIT: (200, 1500),
            ResponseChannel.LINKEDIN: (300, 2000),
            ResponseChannel.EMAIL: (200, 800),
            ResponseChannel.DM: (100, 400),
        }

        min_ideal, max_ideal = ideal_ranges.get(channel, (100, 1000))

        if min_ideal <= length <= max_ideal:
            return 1.0
        if length < min_ideal:
            # Too short
            ratio = length / min_ideal
            return max(0.3, ratio)
        # Too long
        if length > max_ideal * 1.5:
            return 0.3
        if length > max_ideal:
            excess_ratio = (length - max_ideal) / max_ideal
            return max(0.5, 1.0 - excess_ratio)

        return 0.7

    def _score_engagement_potential(
        self,
        content: str,
        signal: ActionableSignal,  # pylint: disable=unused-argument
    ) -> float:
        """Score potential for engagement.

        Args:
            content: Response content
            signal: Original signal (for context)

        Returns:
            Engagement score (0.0-1.0)
        """
        score = 0.5  # Base score
        content_lower = content.lower()

        # Positive indicators
        engagement_words = ["question", "?", "thoughts", "feedback", "share", "learn"]
        score += sum(0.05 for word in engagement_words if word in content_lower)

        # Check for questions
        if "?" in content:
            score += 0.10

        # Penalize overly salesy language
        salesy_words = ["buy", "purchase", "limited time", "act now", "don't miss"]
        score -= sum(0.08 for word in salesy_words if word in content_lower)

        # Reward specific examples or details
        if any(indicator in content_lower for indicator in ["for example", "such as", "like"]):
            score += 0.08

        return max(0.0, min(1.0, score))


    def _select_tones_for_variants(
        self,
        signal: ActionableSignal,
        num_variants: int,
    ) -> List[ResponseTone]:
        """Select tones for generating variants.

        Args:
            signal: Signal to respond to
            num_variants: Number of variants needed

        Returns:
            List of tones to use
        """
        tones = []

        # Start with suggested tone
        if signal.suggested_tone:
            tones.append(signal.suggested_tone)

        # Add complementary tones based on signal type
        complementary_tones = {
            ResponseTone.PROFESSIONAL: [ResponseTone.HELPFUL, ResponseTone.FOUNDER_VOICE],
            ResponseTone.HELPFUL: [ResponseTone.PROFESSIONAL, ResponseTone.SUPPORTIVE],
            ResponseTone.TECHNICAL: [ResponseTone.PROFESSIONAL, ResponseTone.FOUNDER_VOICE],
            ResponseTone.SUPPORTIVE: [ResponseTone.HELPFUL, ResponseTone.PROFESSIONAL],
            ResponseTone.FOUNDER_VOICE: [ResponseTone.PROFESSIONAL, ResponseTone.TECHNICAL],
        }

        # Add complementary tones
        if signal.suggested_tone and signal.suggested_tone in complementary_tones:
            for tone in complementary_tones[signal.suggested_tone]:
                if tone not in tones:
                    tones.append(tone)

        # Fill remaining slots with other tones
        all_tones = list(ResponseTone)
        for tone in all_tones:
            if len(tones) >= num_variants:
                break
            if tone not in tones:
                tones.append(tone)

        return tones[:num_variants]

    async def _generate_single_variant(
        self,
        signal: ActionableSignal,
        tone: ResponseTone,
        channel: ResponseChannel,
        variant_number: int,  # pylint: disable=unused-argument
    ) -> ResponseVariant:
        """Generate a single response variant.

        Args:
            signal: Signal to respond to
            tone: Tone for this variant
            channel: Target channel
            variant_number: Variant number (for tracking)

        Returns:
            Response variant with quality scores

        Note:
            This is a stub implementation. Full LLM integration will be added in Phase 2.5.
        """
        # Build prompts using playbook (for future LLM integration)
        _ = self.playbook.build_prompt(
            signal=signal,
            tone=tone,
            channel=channel,
        )

        # Stub: In production, this would call the LLM router
        # For now, return a placeholder variant for testing
        content = f"[Generated response for {signal.signal_type.value} in {tone.value} tone]"

        # Score the generated content
        clarity_score = self._score_clarity(content)
        tone_score = self._score_tone_match(content, tone)
        length_score = self._score_length(content, channel)
        engagement_score = self._score_engagement_potential(content, signal)

        overall_score = (
            clarity_score * self.clarity_weight
            + tone_score * self.tone_weight
            + length_score * self.length_weight
            + engagement_score * self.engagement_weight
        )

        return ResponseVariant(
            content=content,
            tone=tone,
            channel=channel,
            clarity_score=clarity_score,
            tone_match_score=tone_score,
            length_score=length_score,
            engagement_score=engagement_score,
            overall_score=overall_score,
        )
