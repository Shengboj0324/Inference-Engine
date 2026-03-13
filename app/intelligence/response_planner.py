"""Response planning system for generating action response drafts.

This module implements response planning with:
- Multi-variant draft generation
- Tone adaptation (professional, friendly, empathetic, etc.)
- Channel-specific formatting
- Policy and safety checking
- Critique and revision cycles
"""

import logging
from typing import List, Optional, Dict

from app.domain.normalized_models import NormalizedObservation
from app.domain.inference_models import SignalInference, SignalType
from app.domain.action_models import (
    ActionableSignal,
    ResponseDraft,
    ResponseChannel,
    PolicyViolation,
)
from app.llm.router import get_router
from app.llm.exceptions import LLMError

logger = logging.getLogger(__name__)


class ResponsePlanner:
    """Response planning system with multi-variant generation.
    
    Generates multiple response drafts with different tones and channels,
    then critiques and ranks them.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4-turbo",
        temperature: float = 0.7,
        num_variants: int = 3,
    ):
        """Initialize response planner.
        
        Args:
            model_name: LLM model to use
            temperature: Sampling temperature
            num_variants: Number of draft variants to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.num_variants = num_variants
        
        # Initialize LLM router
        self.llm_router = get_router()
        
        logger.info(
            f"ResponsePlanner initialized: "
            f"model={model_name}, "
            f"variants={num_variants}"
        )
    
    async def plan_response(
        self,
        action: ActionableSignal,
        observation: NormalizedObservation,
        inference: SignalInference,
    ) -> ActionableSignal:
        """Plan response for an actionable signal.
        
        Args:
            action: Actionable signal
            observation: Normalized observation
            inference: Signal inference
            
        Returns:
            ActionableSignal with response drafts populated
        """
        # Generate response drafts
        drafts = await self._generate_drafts(action, observation, inference)

        # Critique and rank drafts (sync operation)
        ranked_drafts = self._critique_and_rank_drafts(
            drafts, action, observation
        )
        
        # Update action with drafts
        action.response_drafts = ranked_drafts
        
        logger.info(
            f"Generated {len(ranked_drafts)} response drafts for action {action.id}"
        )
        
        return action
    
    async def _generate_drafts(
        self,
        action: ActionableSignal,
        observation: NormalizedObservation,
        inference: SignalInference,
    ) -> List[ResponseDraft]:
        """Generate multiple response draft variants.
        
        Args:
            action: Actionable signal
            observation: Normalized observation
            inference: Signal inference
            
        Returns:
            List of response drafts
        """
        drafts = []
        
        # Define tone variants
        tones = ["professional", "friendly", "empathetic"][:self.num_variants]
        
        for i, tone in enumerate(tones):
            # Build prompt
            prompt = self._build_response_prompt(
                action, observation, inference, tone
            )
            
            # Generate response
            try:
                # Use generate_simple which returns string directly
                content = await self.llm_router.generate_simple(
                    prompt=prompt,
                    max_tokens=500,
                    temperature=self.temperature,
                )

                # Create draft (confidence will be computed later)
                draft = ResponseDraft(
                    variant_id=f"v{i+1}_{tone}",
                    channel=action.recommended_channel,
                    content=content.strip(),
                    tone=tone,
                    confidence=0.5,  # Will be updated by critique
                    generated_by=self.model_name,
                )

                drafts.append(draft)

            except LLMError as e:
                logger.error(f"LLM error generating draft variant {i+1}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error generating draft variant {i+1}: {e}")
                continue
        
        return drafts
    
    def _build_response_prompt(
        self,
        action: ActionableSignal,
        observation: NormalizedObservation,
        inference: SignalInference,
        tone: str,
    ) -> str:
        """Build prompt for response generation.
        
        Args:
            action: Actionable signal
            observation: Normalized observation
            inference: Signal inference
            tone: Desired tone
            
        Returns:
            Prompt string
        """
        # System message
        system = f"""You are an expert customer success and sales professional.
Generate a {tone} response to the following social media content.

Signal Type: {action.signal_type.value}
Channel: {action.recommended_channel.value}
Confidence: {action.signal_confidence:.2f}

Guidelines:
- Be {tone} and authentic
- Address the specific need or concern
- Provide value without being pushy
- Keep it concise (2-3 sentences for social, longer for email)
- Do not make claims you can't back up
- Do not be overly salesy"""
        
        # User message
        user_message = f"""Content to respond to:

Title: {observation.title or 'N/A'}
Text: {observation.normalized_text[:500] if observation.normalized_text else 'N/A'}
Platform: {observation.source_platform.value}

Generate a {tone} response."""
        
        return f"{system}\n\n{user_message}"

    def _critique_and_rank_drafts(
        self,
        drafts: List[ResponseDraft],
        action: ActionableSignal,
        observation: NormalizedObservation,
    ) -> List[ResponseDraft]:
        """Critique and rank response drafts (sync operation).

        Args:
            drafts: List of response drafts
            action: Actionable signal
            observation: Normalized observation

        Returns:
            Ranked list of drafts (best first)
        """
        if not drafts:
            return []

        # Critique each draft
        critiqued_drafts = []
        for draft in drafts:
            # Run policy check
            violations = self._check_policy(draft, action, observation)
            draft.policy_violations = violations

            # Compute confidence based on critique
            confidence = self._compute_draft_confidence(draft, action)
            draft.confidence = confidence

            critiqued_drafts.append(draft)

        # Sort by confidence (descending)
        critiqued_drafts.sort(key=lambda d: d.confidence, reverse=True)

        return critiqued_drafts

    def _check_policy(
        self,
        draft: ResponseDraft,
        action: ActionableSignal,
        observation: NormalizedObservation,
    ) -> List[PolicyViolation]:
        """Check draft for policy violations.

        Args:
            draft: Response draft
            action: Actionable signal
            observation: Normalized observation

        Returns:
            List of policy violations
        """
        violations = []

        content_lower = draft.content.lower()

        # Check for prohibited claims
        prohibited_claims = [
            "guaranteed", "promise", "best in the world",
            "number one", "always", "never fails"
        ]

        for claim in prohibited_claims:
            if claim in content_lower:
                violations.append(
                    PolicyViolation(
                        policy_name="no_unsubstantiated_claims",
                        violation_type="prohibited_claim",
                        severity="medium",
                        description=f"Contains prohibited claim: '{claim}'",
                        blocking=False,
                    )
                )

        # Check for competitor mentions (risky)
        competitor_keywords = ["competitor", "vs", "versus", "better than"]
        if any(kw in content_lower for kw in competitor_keywords):
            violations.append(
                PolicyViolation(
                    policy_name="competitor_mention_policy",
                    violation_type="competitor_mention",
                    severity="low",
                    description="Mentions competitors - review for tone",
                    blocking=False,
                )
            )

        # Check for pricing promises
        pricing_keywords = ["free", "discount", "price", "cost", "$"]
        if any(kw in content_lower for kw in pricing_keywords):
            violations.append(
                PolicyViolation(
                    policy_name="pricing_disclosure_policy",
                    violation_type="pricing_mention",
                    severity="low",
                    description="Mentions pricing - ensure accuracy",
                    blocking=False,
                )
            )

        # Check length for channel
        if draft.channel == ResponseChannel.DIRECT_REPLY:
            if len(draft.content) > 280:
                violations.append(
                    PolicyViolation(
                        policy_name="channel_length_policy",
                        violation_type="too_long",
                        severity="medium",
                        description=f"Reply too long ({len(draft.content)} chars) for social media",
                        blocking=False,
                    )
                )

        return violations

    def _compute_draft_confidence(
        self,
        draft: ResponseDraft,
        action: ActionableSignal,
    ) -> float:
        """Compute confidence score for draft.

        Args:
            draft: Response draft
            action: Actionable signal

        Returns:
            Confidence score (0-1)
        """
        confidence = 0.8  # Base confidence

        # Penalize for policy violations
        for violation in draft.policy_violations:
            if violation.severity == "critical":
                confidence -= 0.3
            elif violation.severity == "high":
                confidence -= 0.2
            elif violation.severity == "medium":
                confidence -= 0.1
            elif violation.severity == "low":
                confidence -= 0.05

        # Boost for appropriate length
        content_len = len(draft.content)
        if draft.channel == ResponseChannel.DIRECT_REPLY:
            if 50 <= content_len <= 280:
                confidence += 0.1
        elif draft.channel == ResponseChannel.EMAIL:
            if 200 <= content_len <= 1000:
                confidence += 0.1

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

