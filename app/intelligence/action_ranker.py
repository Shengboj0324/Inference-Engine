"""Action ranking system for prioritizing actionable signals.

This module implements multi-dimensional action ranking:
- Priority scoring (overall priority)
- Opportunity scoring (business value potential)
- Urgency scoring (time sensitivity)
- Risk scoring (risk if not addressed)

Combines multiple signals to produce a calibrated priority score.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timezone

from app.domain.normalized_models import NormalizedObservation
from app.domain.inference_models import SignalInference, SignalType
from app.domain.action_models import ActionableSignal, ActionPriority, ActionStatus, ResponseChannel
from app.core.models import SourcePlatform

logger = logging.getLogger(__name__)


class ActionRanker:
    """Multi-dimensional action ranking system.
    
    Scores signals across 4 dimensions:
    1. Opportunity: Business value potential
    2. Urgency: Time sensitivity
    3. Risk: Risk if not addressed
    4. Priority: Overall priority (weighted combination)
    """
    
    def __init__(
        self,
        opportunity_weight: float = 0.35,
        urgency_weight: float = 0.30,
        risk_weight: float = 0.35,
        min_confidence_threshold: float = 0.5,
    ):
        """Initialize action ranker.
        
        Args:
            opportunity_weight: Weight for opportunity score
            urgency_weight: Weight for urgency score
            risk_weight: Weight for risk score
            min_confidence_threshold: Minimum confidence to create action
        """
        self.opportunity_weight = opportunity_weight
        self.urgency_weight = urgency_weight
        self.risk_weight = risk_weight
        self.min_confidence_threshold = min_confidence_threshold
        
        # Validate weights sum to 1.0
        total_weight = opportunity_weight + urgency_weight + risk_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Weights sum to {total_weight}, not 1.0. "
                f"Normalizing weights."
            )
            self.opportunity_weight /= total_weight
            self.urgency_weight /= total_weight
            self.risk_weight /= total_weight
        
        logger.info(
            f"ActionRanker initialized: "
            f"opp={self.opportunity_weight:.2f}, "
            f"urg={self.urgency_weight:.2f}, "
            f"risk={self.risk_weight:.2f}"
        )
    
    def rank_action(
        self,
        inference: SignalInference,
        observation: NormalizedObservation,
    ) -> Optional[ActionableSignal]:
        """Rank a signal inference and create actionable signal if worthy.
        
        Args:
            inference: Signal inference from Phase 2
            observation: Normalized observation
            
        Returns:
            ActionableSignal if worthy of action, None otherwise
        """
        # Check if inference is abstained
        if inference.abstained:
            logger.debug(f"Skipping abstained inference {inference.id}")
            return None
        
        # Check if confidence meets threshold
        if not inference.top_prediction:
            logger.debug(f"No top prediction for inference {inference.id}")
            return None
        
        if inference.top_prediction.probability < self.min_confidence_threshold:
            logger.debug(
                f"Confidence {inference.top_prediction.probability:.2f} "
                f"below threshold {self.min_confidence_threshold}"
            )
            return None
        
        # Compute scores
        opportunity_score = self._compute_opportunity_score(inference, observation)
        urgency_score = self._compute_urgency_score(inference, observation)
        risk_score = self._compute_risk_score(inference, observation)
        
        # Compute overall priority score
        priority_score = (
            self.opportunity_weight * opportunity_score +
            self.urgency_weight * urgency_score +
            self.risk_weight * risk_score
        )
        
        # Determine priority level
        priority = self._determine_priority_level(priority_score)
        
        # Determine recommended channel
        recommended_channel = self._determine_channel(inference, observation)
        
        # Create actionable signal
        action = ActionableSignal(
            signal_inference_id=inference.id,
            normalized_observation_id=observation.id,
            user_id=observation.user_id,
            signal_type=inference.top_prediction.signal_type,
            signal_confidence=inference.top_prediction.probability,
            priority=priority,
            priority_score=priority_score,
            opportunity_score=opportunity_score,
            urgency_score=urgency_score,
            risk_score=risk_score,
            recommended_channel=recommended_channel,
            status=ActionStatus.NEW,
        )
        
        logger.info(
            f"Created action {action.id}: "
            f"type={action.signal_type.value}, "
            f"priority={priority.value}, "
            f"score={priority_score:.2f}"
        )
        
        return action
    
    def rank_batch(
        self,
        inferences: List[SignalInference],
        observations: Dict[str, NormalizedObservation],
    ) -> List[ActionableSignal]:
        """Rank a batch of inferences.
        
        Args:
            inferences: List of signal inferences
            observations: Dict mapping observation ID to observation
            
        Returns:
            List of actionable signals, sorted by priority score
        """
        actions = []
        
        for inference in inferences:
            observation = observations.get(str(inference.normalized_observation_id))
            if not observation:
                logger.warning(
                    f"No observation found for inference {inference.id}"
                )
                continue
            
            action = self.rank_action(inference, observation)
            if action:
                actions.append(action)
        
        # Sort by priority score (descending)
        actions.sort(key=lambda a: a.priority_score, reverse=True)
        
        logger.info(f"Ranked {len(actions)} actions from {len(inferences)} inferences")
        return actions

    def _compute_opportunity_score(
        self,
        inference: SignalInference,
        observation: NormalizedObservation,
    ) -> float:
        """Compute business opportunity score.

        Args:
            inference: Signal inference
            observation: Normalized observation

        Returns:
            Opportunity score (0-1)
        """
        score = 0.5  # Base score

        if not inference.top_prediction:
            return score

        signal_type = inference.top_prediction.signal_type

        # High opportunity signals
        if signal_type in [
            SignalType.ALTERNATIVE_SEEKING,
            SignalType.COMPETITOR_MENTION,
            SignalType.EXPANSION_OPPORTUNITY,
            SignalType.UPSELL_OPPORTUNITY,
            SignalType.PARTNERSHIP_OPPORTUNITY,
        ]:
            score = 0.9

        # Medium opportunity signals
        elif signal_type in [
            SignalType.FEATURE_REQUEST,
            SignalType.INTEGRATION_REQUEST,
            SignalType.PRICE_SENSITIVITY,
        ]:
            score = 0.7

        # Low opportunity signals
        elif signal_type in [
            SignalType.SUPPORT_REQUEST,
            SignalType.BUG_REPORT,
            SignalType.COMPLAINT,
        ]:
            score = 0.4

        # Boost for high engagement
        if observation.engagement_velocity and observation.engagement_velocity > 10:
            score = min(1.0, score + 0.1)

        # Boost for high virality
        if observation.virality_score and observation.virality_score > 0.5:
            score = min(1.0, score + 0.1)

        return score

    def _compute_urgency_score(
        self,
        inference: SignalInference,
        observation: NormalizedObservation,
    ) -> float:
        """Compute time sensitivity score.

        Args:
            inference: Signal inference
            observation: Normalized observation

        Returns:
            Urgency score (0-1)
        """
        score = 0.5  # Base score

        if not inference.top_prediction:
            return score

        signal_type = inference.top_prediction.signal_type

        # High urgency signals
        if signal_type in [
            SignalType.CHURN_RISK,
            SignalType.SECURITY_CONCERN,
            SignalType.LEGAL_RISK,
            SignalType.REPUTATION_RISK,
        ]:
            score = 0.9

        # Medium urgency signals
        elif signal_type in [
            SignalType.COMPLAINT,
            SignalType.BUG_REPORT,
            SignalType.ALTERNATIVE_SEEKING,
        ]:
            score = 0.7

        # Low urgency signals
        elif signal_type in [
            SignalType.PRAISE,
            SignalType.FEATURE_REQUEST,
        ]:
            score = 0.3

        # Boost for recent content
        if observation.published_at:
            hours_old = (
                datetime.now(timezone.utc) - observation.published_at
            ).total_seconds() / 3600

            if hours_old < 1:
                score = min(1.0, score + 0.2)
            elif hours_old < 6:
                score = min(1.0, score + 0.1)
            elif hours_old > 48:
                score = max(0.0, score - 0.2)

        # Boost for high engagement velocity
        if observation.engagement_velocity and observation.engagement_velocity > 20:
            score = min(1.0, score + 0.1)

        return score

    def _compute_risk_score(
        self,
        inference: SignalInference,
        observation: NormalizedObservation,
    ) -> float:
        """Compute risk score if not addressed.

        Args:
            inference: Signal inference
            observation: Normalized observation

        Returns:
            Risk score (0-1)
        """
        score = 0.3  # Base score

        if not inference.top_prediction:
            return score

        signal_type = inference.top_prediction.signal_type

        # High risk signals
        if signal_type in [
            SignalType.CHURN_RISK,
            SignalType.SECURITY_CONCERN,
            SignalType.LEGAL_RISK,
            SignalType.REPUTATION_RISK,
        ]:
            score = 0.95

        # Medium risk signals
        elif signal_type in [
            SignalType.COMPLAINT,
            SignalType.COMPETITOR_MENTION,
            SignalType.ALTERNATIVE_SEEKING,
        ]:
            score = 0.6

        # Low risk signals
        elif signal_type in [
            SignalType.PRAISE,
            SignalType.FEATURE_REQUEST,
            SignalType.SUPPORT_REQUEST,
        ]:
            score = 0.2

        # Boost for public visibility (high-reach platforms)
        public_platforms = [
            SourcePlatform.REDDIT,
            SourcePlatform.YOUTUBE,
            SourcePlatform.TIKTOK,
            SourcePlatform.FACEBOOK,
            SourcePlatform.INSTAGRAM,
        ]
        if observation.source_platform in public_platforms:
            score = min(1.0, score + 0.1)

        # Boost for high virality (public spread)
        if observation.virality_score and observation.virality_score > 0.7:
            score = min(1.0, score + 0.15)

        return score

    def _determine_priority_level(self, priority_score: float) -> ActionPriority:
        """Determine priority level from score.

        Args:
            priority_score: Overall priority score

        Returns:
            ActionPriority enum
        """
        if priority_score >= 0.8:
            return ActionPriority.CRITICAL
        elif priority_score >= 0.6:
            return ActionPriority.HIGH
        elif priority_score >= 0.4:
            return ActionPriority.MEDIUM
        elif priority_score >= 0.2:
            return ActionPriority.LOW
        else:
            return ActionPriority.MONITOR

    def _determine_channel(
        self,
        inference: SignalInference,
        observation: NormalizedObservation,
    ) -> ResponseChannel:
        """Determine recommended response channel.

        Args:
            inference: Signal inference
            observation: Normalized observation

        Returns:
            ResponseChannel enum
        """
        if not inference.top_prediction:
            return ResponseChannel.NO_RESPONSE

        signal_type = inference.top_prediction.signal_type

        # Direct reply for public signals
        if signal_type in [
            SignalType.ALTERNATIVE_SEEKING,
            SignalType.COMPETITOR_MENTION,
            SignalType.PRAISE,
        ]:
            return ResponseChannel.DIRECT_REPLY

        # Direct message for sensitive signals
        if signal_type in [
            SignalType.CHURN_RISK,
            SignalType.COMPLAINT,
            SignalType.SECURITY_CONCERN,
        ]:
            return ResponseChannel.DIRECT_MESSAGE

        # Email for business opportunities
        if signal_type in [
            SignalType.PARTNERSHIP_OPPORTUNITY,
            SignalType.EXPANSION_OPPORTUNITY,
            SignalType.UPSELL_OPPORTUNITY,
        ]:
            return ResponseChannel.EMAIL

        # Internal ticket for support/bugs
        if signal_type in [
            SignalType.SUPPORT_REQUEST,
            SignalType.BUG_REPORT,
            SignalType.FEATURE_REQUEST,
        ]:
            return ResponseChannel.INTERNAL_TICKET

        # Default to no response
        return ResponseChannel.NO_RESPONSE

