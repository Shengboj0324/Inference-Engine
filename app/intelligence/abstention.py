"""Abstention decision logic for uncertain predictions.

This module implements abstention logic based on:
- Confidence thresholds
- Uncertainty quantification
- Prediction disagreement
- Context completeness
- Policy constraints

Abstention is a feature, not a bug. Wrong action is worse than no action.
"""

import logging
from typing import Optional, List
from dataclasses import dataclass

from app.domain.normalized_models import NormalizedObservation
from app.domain.inference_models import (
    SignalInference,
    SignalPrediction,
    AbstentionReason,
)
from app.intelligence.candidate_retrieval import SignalCandidate

logger = logging.getLogger(__name__)


@dataclass
class AbstentionThresholds:
    """Thresholds for abstention decisions."""
    
    min_confidence: float = 0.7  # Minimum confidence to not abstain
    max_disagreement: float = 0.3  # Maximum prediction disagreement
    min_completeness: float = 0.5  # Minimum context completeness
    min_quality: float = 0.4  # Minimum content quality


class AbstentionDecider:
    """Decides whether to abstain from making a prediction.
    
    Implements multiple abstention criteria to ensure high-quality predictions.
    """
    
    def __init__(
        self,
        thresholds: Optional[AbstentionThresholds] = None,
    ):
        """Initialize abstention decider.
        
        Args:
            thresholds: Abstention thresholds
        """
        self.thresholds = thresholds or AbstentionThresholds()
        
        logger.info(
            f"AbstentionDecider initialized: "
            f"min_confidence={self.thresholds.min_confidence}"
        )
    
    def should_abstain(
        self,
        inference: SignalInference,
        observation: NormalizedObservation,
        candidates: Optional[List[SignalCandidate]] = None,
    ) -> tuple[bool, Optional[AbstentionReason], str]:
        """Decide whether to abstain from this prediction.
        
        Args:
            inference: Signal inference
            observation: Normalized observation
            candidates: Candidate signals from retrieval
            
        Returns:
            Tuple of (should_abstain, reason, explanation)
        """
        # Check if already abstained
        if inference.abstained:
            return True, inference.abstention_reason, inference.rationale or "Already abstained"
        
        # Check confidence threshold using the calibrated confidence interval upper bound.
        # We use the CI *upper* bound (most optimistic estimate) — if even the upper bound
        # is below the threshold, we abstain.  This is stricter than using the raw probability
        # and correctly uses the Calibrator's output rather than raw model logits.
        if inference.top_prediction:
            # Prefer calibrated CI upper bound when available; fall back to raw probability.
            ci_upper = (
                inference.calibration_metrics.confidence_interval_upper
                if inference.calibration_metrics is not None
                and inference.calibration_metrics.confidence_interval_upper is not None
                else inference.top_prediction.probability
            )
            if ci_upper < self.thresholds.min_confidence:
                return (
                    True,
                    AbstentionReason.LOW_CONFIDENCE,
                    f"Calibrated CI upper bound {ci_upper:.2f} below threshold "
                    f"{self.thresholds.min_confidence} (raw probability "
                    f"{inference.top_prediction.probability:.2f})",
                )
        else:
            return (
                True,
                AbstentionReason.LOW_CONFIDENCE,
                "No top prediction available",
            )
        
        # Check prediction disagreement
        if len(inference.predictions) > 1:
            disagreement = self._compute_disagreement(inference.predictions)
            if disagreement > self.thresholds.max_disagreement:
                return (
                    True,
                    AbstentionReason.AMBIGUOUS_MULTI_LABEL,
                    f"Prediction disagreement {disagreement:.2f} exceeds threshold {self.thresholds.max_disagreement}"
                )
        
        # Check context completeness
        if observation.completeness_score is not None:
            if observation.completeness_score < self.thresholds.min_completeness:
                return (
                    True,
                    AbstentionReason.INSUFFICIENT_CONTEXT,
                    f"Completeness {observation.completeness_score:.2f} below threshold {self.thresholds.min_completeness}"
                )
        
        # Check content quality
        if observation.quality_score is not None:
            if observation.quality_score < self.thresholds.min_quality:
                return (
                    True,
                    AbstentionReason.SPAM_OR_NOISE,
                    f"Quality {observation.quality_score:.2f} below threshold {self.thresholds.min_quality}"
                )

        # Check for unsafe content
        if self._is_unsafe_content(observation):
            return (
                True,
                AbstentionReason.UNSAFE_TO_CLASSIFY,
                "Content flagged as potentially unsafe"
            )

        # Check for policy violations
        if self._has_policy_violation(observation):
            return (
                True,
                AbstentionReason.UNSAFE_TO_CLASSIFY,
                "Content violates policy constraints"
            )

        # Check if out of scope
        if self._is_out_of_scope(observation, inference):
            return (
                True,
                AbstentionReason.OUT_OF_DISTRIBUTION,
                "Content is out of scope for actionable signals"
            )
        
        # No abstention needed
        return False, None, ""
    
    def _compute_disagreement(self, predictions: List[SignalPrediction]) -> float:
        """Compute disagreement between predictions.
        
        Args:
            predictions: List of predictions
            
        Returns:
            Disagreement score (0-1, higher means more disagreement)
        """
        if len(predictions) < 2:
            return 0.0
        
        # Sort by probability
        sorted_preds = sorted(predictions, key=lambda p: p.probability, reverse=True)
        
        # Compute difference between top 2 predictions
        top_prob = sorted_preds[0].probability
        second_prob = sorted_preds[1].probability
        
        # Disagreement is 1 - difference
        # If top is 0.9 and second is 0.1, disagreement is low (0.2)
        # If top is 0.6 and second is 0.5, disagreement is high (0.9)
        disagreement = 1.0 - (top_prob - second_prob)
        
        return disagreement
    
    def _is_unsafe_content(self, observation: NormalizedObservation) -> bool:
        """Check if content is unsafe.
        
        Args:
            observation: Normalized observation
            
        Returns:
            True if unsafe
        """
        # Simple keyword-based check
        # In production, use a proper content moderation model
        unsafe_keywords = [
            "violence", "hate", "illegal", "explicit", "nsfw"
        ]

        text = (observation.normalized_text or "").lower()
        return any(keyword in text for keyword in unsafe_keywords)
    
    def _has_policy_violation(self, observation: NormalizedObservation) -> bool:
        """Check if content violates policies.
        
        Args:
            observation: Normalized observation
            
        Returns:
            True if policy violation detected
        """
        # Placeholder - would check against policy rules
        return False
    
    def _is_out_of_scope(
        self, observation: NormalizedObservation, inference: SignalInference
    ) -> bool:
        """Check if content is out of scope.
        
        Args:
            observation: Normalized observation
            inference: Signal inference
            
        Returns:
            True if out of scope
        """
        # Check if prediction is NOT_ACTIONABLE or UNCLEAR
        if inference.top_prediction:
            from app.domain.inference_models import SignalType
            if inference.top_prediction.signal_type in [
                SignalType.NOT_ACTIONABLE,
                SignalType.UNCLEAR,
            ]:
                return True
        
        return False

