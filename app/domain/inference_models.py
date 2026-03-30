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

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    MALFORMED_OUTPUT = "malformed_output"  # LLM output could not be parsed/validated


class StrategicPriorities(BaseModel):
    """User-defined GTM priorities that bias inference scoring.

    All fields are optional; absent fields receive system defaults.
    Populated from ``User.strategic_priorities`` JSON at request time.

    Attributes:
        competitors: Surface forms of competitors to up-weight in scoring.
            When the observation mentions any of these names the ``urgency_score``
            multiplier is applied during adjudication.
        focus_areas: Product or domain keywords (e.g. "permissions", "pricing")
            that indicate high relevance for this user's business context.
        tone: Preferred response tone for ``DraftResponse`` generation.
            One of ``"assertive"``, ``"empathetic"``, or ``"neutral"`` (default).
        urgency_weight: Multiplier applied to the raw ``urgency_score`` when
            persisting an ``ActionableSignalDB`` for this user (default 1.0).
        impact_weight: Multiplier applied to the raw ``impact_score`` (default 1.0).
    """

    competitors: List[str] = Field(default_factory=list)
    focus_areas: List[str] = Field(default_factory=list)
    tone: str = Field(
        default="neutral",
        description="One of 'assertive', 'empathetic', 'neutral'",
    )
    urgency_weight: float = Field(default=1.0, ge=0.5, le=3.0)
    impact_weight: float = Field(default=1.0, ge=0.5, le=3.0)

    @classmethod
    def from_db_json(cls, raw: Optional[Dict[str, Any]]) -> "StrategicPriorities":
        """Construct from the raw ``User.strategic_priorities`` JSON blob.

        Returns a default ``StrategicPriorities`` when *raw* is ``None`` or
        otherwise unparseable so callers never need to ``None``-guard.
        """
        if not raw:
            return cls()
        try:
            return cls(**{k: v for k, v in raw.items() if k in cls.model_fields})
        except Exception:
            return cls()


class UserContext(BaseModel):
    """Typed wrapper around per-user context injected into the inference prompt.

    Constructed once per API request from the authenticated ``User`` record and
    threaded through ``InferencePipeline.run()`` → ``LLMAdjudicator.adjudicate()``.

    Attributes:
        user_id: UUID of the authenticated user (for logging).
        strategic_priorities: Parsed GTM priorities for prompt injection.
    """

    user_id: UUID
    strategic_priorities: StrategicPriorities = Field(
        default_factory=StrategicPriorities
    )

    @classmethod
    def from_user(cls, user: Any) -> "UserContext":
        """Build a ``UserContext`` from a ``User`` DB model instance.

        Args:
            user: ``app.core.db_models.User`` instance.

        Returns:
            Fully populated ``UserContext``; never raises.
        """
        return cls(
            user_id=user.id,
            strategic_priorities=StrategicPriorities.from_db_json(
                getattr(user, "strategic_priorities", None)
            ),
        )


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
    """Calibration quality metrics attached to a signal inference.

    ECE and Brier score are batch metrics — they require ground-truth labels
    from a validation set and are computed by app/evals/calibration_eval.py.
    They are None for single-inference objects and only populated when the
    model has been evaluated against a labelled holdout.

    confidence_interval_lower / _upper are per-inference approximations
    derived from the Bernoulli standard deviation of the top prediction's
    probability and are always populated when top_prediction is present.
    """

    expected_calibration_error: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "ECE over a labelled validation batch (None for single-inference objects). "
            "Lower is better; < 0.05 is well-calibrated."
        ),
    )
    brier_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Mean squared probability error over a labelled batch (None for single-inference). "
            "Lower is better; 0.0 is perfect."
        ),
    )
    confidence_interval_lower: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Lower bound of the per-inference Bernoulli confidence interval.",
    )
    confidence_interval_upper: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Upper bound of the per-inference Bernoulli confidence interval.",
    )


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
    
    model_config = ConfigDict(
        json_schema_extra={
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
                                "relevance_score": 0.92,
                            }
                        ],
                    }
                ],
                "abstained": False,
                "model_name": "gpt-4-turbo",
                "model_version": "2024-01-15",
                "inference_method": "llm_few_shot",
            }
        }
    )

