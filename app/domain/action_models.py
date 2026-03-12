"""Action models - Layer 4 of domain architecture.

Action models represent operational units for human/agent action with:
- Priority, opportunity, urgency, risk scores
- Response plans and draft variants
- Policy constraints and safety checks
- Status tracking and outcome logging
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from app.domain.inference_models import SignalType


class ActionPriority(str, Enum):
    """Action priority levels."""
    
    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"  # Action needed within hours
    MEDIUM = "medium"  # Action needed within days
    LOW = "low"  # Action can wait
    MONITOR = "monitor"  # No action, just monitor


class ActionStatus(str, Enum):
    """Action status tracking."""
    
    NEW = "new"  # Just created
    QUEUED = "queued"  # In action queue
    IN_PROGRESS = "in_progress"  # Being worked on
    PENDING_REVIEW = "pending_review"  # Awaiting human review
    APPROVED = "approved"  # Approved for execution
    EXECUTED = "executed"  # Action taken
    COMPLETED = "completed"  # Outcome verified
    REJECTED = "rejected"  # Rejected by human
    CANCELLED = "cancelled"  # Cancelled
    FAILED = "failed"  # Execution failed


class ResponseChannel(str, Enum):
    """Response channel types."""
    
    DIRECT_REPLY = "direct_reply"  # Reply to the content
    DIRECT_MESSAGE = "direct_message"  # Private message
    EMAIL = "email"  # Email outreach
    INTERNAL_TICKET = "internal_ticket"  # Create internal ticket
    SLACK_NOTIFICATION = "slack_notification"  # Notify team
    NO_RESPONSE = "no_response"  # Monitor only


class PolicyViolation(BaseModel):
    """Policy violation detected."""
    
    policy_name: str
    violation_type: str
    severity: str  # low, medium, high, critical
    description: str
    blocking: bool = False  # Whether this blocks action execution


class ResponseDraft(BaseModel):
    """Draft response variant."""
    
    variant_id: str
    channel: ResponseChannel
    content: str
    tone: str = Field(..., description="professional, friendly, empathetic, etc.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    policy_violations: List[PolicyViolation] = Field(default_factory=list)
    
    # Metadata
    generated_by: str = Field(..., description="Model or human identifier")
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class ActionOutcome(BaseModel):
    """Outcome of executed action."""
    
    executed_at: datetime
    executed_by: str  # User ID or agent identifier
    response_sent: bool = False
    response_channel: Optional[ResponseChannel] = None
    response_content: Optional[str] = None
    
    # Outcome metrics
    customer_replied: bool = False
    issue_resolved: bool = False
    escalated: bool = False
    conversion_achieved: bool = False
    
    # Feedback
    human_rating: Optional[int] = Field(None, ge=1, le=5)
    human_feedback: Optional[str] = None
    
    # Metadata
    outcome_metadata: Dict[str, Any] = Field(default_factory=dict)


class ActionableSignal(BaseModel):
    """Actionable signal - Layer 4 operational unit.
    
    This represents a business-actionable opportunity or risk that requires
    human or agent response. Includes priority scoring, response plans, and
    outcome tracking.
    """

    # Identity
    id: UUID = Field(default_factory=uuid4)
    signal_inference_id: UUID
    normalized_observation_id: UUID
    user_id: UUID
    
    # Signal classification
    signal_type: SignalType
    signal_confidence: float = Field(..., ge=0.0, le=1.0)
    
    # Priority scoring
    priority: ActionPriority
    priority_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall priority score (0=lowest, 1=highest)"
    )
    opportunity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Business opportunity potential"
    )
    urgency_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Time sensitivity"
    )
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Risk if not addressed"
    )
    
    # Response planning
    recommended_channel: ResponseChannel
    response_drafts: List[ResponseDraft] = Field(default_factory=list)
    requires_human_review: bool = True
    
    # Policy and safety
    policy_violations: List[PolicyViolation] = Field(default_factory=list)
    safe_to_auto_respond: bool = False
    
    # Status tracking
    status: ActionStatus = ActionStatus.NEW
    assigned_to: Optional[UUID] = Field(None, description="User ID of assignee")
    
    # Outcome
    outcome: Optional[ActionOutcome] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    due_by: Optional[datetime] = None
    
    # Metadata
    action_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific metadata: SLA, escalation rules, etc."
    )
    
    @field_validator('priority_score', 'opportunity_score', 'urgency_score', 'risk_score', 'signal_confidence')
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Ensure scores are in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {v}")
        return v
    
    class Config:
        """Pydantic config."""
        
        validate_assignment = True  # Allow status updates

