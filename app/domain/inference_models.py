"""Inference models - Layer 3 of domain architecture.

Inference models represent ML/LLM interpretation results with:
- Calibrated confidence scores
- Evidence spans and rationale
- Abstention support (when confidence is too low)
- Multi-label predictions
- Uncertainty quantification
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class SignalType(str, Enum):
    """Types of actionable signals."""
    
    # Customer signals
    SUPPORT_REQUEST = "support_request"
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    COMPLAINT = "complaint"
    PRAISE = "praise"
    
    # Market signals
    COMPETITOR_MENTION = "competitor_mention"
    ALTERNATIVE_SEEKING = "alternative_seeking"
    PRICE_SENSITIVITY = "price_sensitivity"
    INTEGRATION_REQUEST = "integration_request"
    
    # Risk signals
    CHURN_RISK = "churn_risk"
    SECURITY_CONCERN = "security_concern"
    LEGAL_RISK = "legal_risk"
    REPUTATION_RISK = "reputation_risk"
    
    # Opportunity signals
    EXPANSION_OPPORTUNITY = "expansion_opportunity"
    UPSELL_OPPORTUNITY = "upsell_opportunity"
    PARTNERSHIP_OPPORTUNITY = "partnership_opportunity"
    
    # Meta
    UNCLEAR = "unclear"
    NOT_ACTIONABLE = "not_actionable"


class AbstentionReason(str, Enum):
    """Reasons for abstaining from making a prediction."""
    
    LOW_CONFIDENCE = "low_confidence"  # Model confidence below threshold
    AMBIGUOUS_MULTI_LABEL = "ambiguous_multi_label"  # Multiple labels equally likely
    INSUFFICIENT_CONTEXT = "insufficient_context"  # Need thread/conversation context
    OUT_OF_DISTRIBUTION = "out_of_distribution"  # Content unlike training data
    UNSAFE_TO_CLASSIFY = "unsafe_to_classify"  # High-risk content (legal, political)
    LANGUAGE_BARRIER = "language_barrier"  # Translation quality too low
    SPAM_OR_NOISE = "spam_or_noise"  # Content quality too low


class EvidenceSpan(BaseModel):
    """Evidence span in text supporting a prediction."""
    
    text: str
    start_char: int
    end_char: int
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class SignalPrediction(BaseModel):
    """Single signal type prediction with calibrated confidence."""
    
    signal_type: SignalType
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated probability (not raw model output)"
    )
    evidence_spans: List[EvidenceSpan] = Field(default_factory=list)
    
    @field_validator('probability')
    @classmethod
    def validate_probability(cls, v: float) -> float:
        """Ensure probability is valid."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Probability must be between 0.0 and 1.0, got {v}")
        return v


class CalibrationMetrics(BaseModel):
    """Calibration quality metrics for this inference."""
    
    expected_calibration_error: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="ECE score for this model on validation set"
    )
    brier_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Brier score (lower is better)"
    )
    confidence_interval_lower: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence_interval_upper: Optional[float] = Field(None, ge=0.0, le=1.0)


class SignalInference(BaseModel):
    """Signal inference result - Layer 3 output.
    
    This represents the ML/LLM interpretation of a normalized observation.
    Includes calibrated confidence, evidence, and abstention support.
    """

    # Identity
    id: UUID = Field(default_factory=uuid4)
    normalized_observation_id: UUID
    user_id: UUID
    
    # Inference results
    predictions: List[SignalPrediction] = Field(
        default_factory=list,
        description="All signal predictions above minimum threshold"
    )
    top_prediction: Optional[SignalPrediction] = Field(
        None,
        description="Highest probability prediction (if any)"
    )
    
    # Abstention
    abstained: bool = False
    abstention_reason: Optional[AbstentionReason] = None
    abstention_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence in the abstention decision itself"
    )
    
    # Rationale and evidence
    rationale: Optional[str] = Field(
        None,
        description="Human-readable explanation of the inference"
    )
    evidence_summary: Optional[str] = Field(
        None,
        description="Summary of key evidence supporting the prediction"
    )
    
    # Calibration
    calibration_metrics: Optional[CalibrationMetrics] = None
    
    # Model provenance
    model_name: str = Field(..., description="Model used for inference")
    model_version: str
    inference_method: str = Field(
        ...,
        description="embedding_retrieval, llm_zero_shot, llm_few_shot, ensemble, etc."
    )
    
    # Timestamps
    inferred_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Metadata
    inference_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model-specific metadata: token count, latency, temperature, etc."
    )
    
    class Config:
        """Pydantic config."""
        
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "signal_type": "feature_request",
                        "probability": 0.87,
                        "evidence_spans": [
                            {
                                "text": "would love to see dark mode",
                                "start_char": 45,
                                "end_char": 73,
                                "relevance_score": 0.92
                            }
                        ]
                    }
                ],
                "abstained": False,
                "model_name": "gpt-4-turbo",
                "model_version": "2024-01-15",
                "inference_method": "llm_few_shot"
            }
        }

