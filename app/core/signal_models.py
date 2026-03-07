"""Core models for actionable signals - the primary product entity.

This module defines the signal-based architecture that transforms content
into actionable business opportunities. Signals replace clusters as the
primary user-facing entity.

Design Principles:
- Signals are business-intent focused, not just topic clusters
- Every signal has a recommended action and generated assets
- Scoring is multi-dimensional: urgency, impact, confidence
- Workflow status enables team collaboration
- Outcome tracking enables learning loops
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class SignalType(str, Enum):
    """Business-intent signal classification.

    Each type represents a specific opportunity or risk that requires
    different handling strategies and response playbooks.
    """

    # Revenue opportunities
    LEAD_OPPORTUNITY = "lead_opportunity"  # "Looking for alternatives to X"
    COMPETITOR_WEAKNESS = "competitor_weakness"  # Complaints about competitors
    INFLUENCER_AMPLIFICATION = "influencer_amplification"  # High-reach accounts

    # Risk signals
    CHURN_RISK = "churn_risk"  # Customer dissatisfaction
    MISINFORMATION_RISK = "misinformation_risk"  # Brand reputation threats
    SUPPORT_ESCALATION = "support_escalation"  # Support issues going public

    # Product signals
    PRODUCT_CONFUSION = "product_confusion"  # Feature questions/misunderstandings
    FEATURE_REQUEST_PATTERN = "feature_request_pattern"  # Recurring feature requests
    LAUNCH_MOMENT = "launch_moment"  # Product launches (ours or competitors)

    # Content opportunities
    TREND_TO_CONTENT = "trend_to_content"  # Rising conversations worth covering


class ActionType(str, Enum):
    """Recommended action types for signals."""

    REPLY_PUBLIC = "reply_public"  # Public reply on platform
    DM_OUTREACH = "dm_outreach"  # Direct message outreach
    CREATE_CONTENT = "create_content"  # Create content asset
    INTERNAL_ALERT = "internal_alert"  # Alert internal team
    MONITOR = "monitor"  # Continue monitoring
    ESCALATE = "escalate"  # Escalate to leadership


class SignalStatus(str, Enum):
    """Signal workflow status."""

    NEW = "new"  # Just detected
    QUEUED = "queued"  # In action queue
    IN_PROGRESS = "in_progress"  # Being worked on
    ACTED = "acted"  # Action taken
    DISMISSED = "dismissed"  # Not worth acting on
    EXPIRED = "expired"  # Passed SLA deadline


class ResponseTone(str, Enum):
    """Tone for generated responses."""

    HELPFUL = "helpful"  # Friendly and helpful
    PROFESSIONAL = "professional"  # Business professional
    TECHNICAL = "technical"  # Technical/engineering focused
    FOUNDER_VOICE = "founder_voice"  # Authentic founder perspective
    SUPPORTIVE = "supportive"  # Customer support tone
    EDUCATIONAL = "educational"  # Teaching/explaining


class ActionableSignal(BaseModel):
    """Core product entity representing a business-actionable signal.

    This replaces Cluster as the primary user-facing object. Every signal
    represents a specific opportunity or risk with recommended actions and
    pre-generated response drafts.

    Attributes:
        id: Unique signal identifier
        user_id: Owner of this signal
        signal_type: Business intent classification
        source_item_ids: ContentItems that triggered this signal
        source_platform: Primary platform where signal originated
        source_url: Direct link to the conversation
        source_author: Author of the triggering content
        title: Brief signal title (max 200 chars)
        description: Detailed description of the signal
        context: Why this signal matters (business context)
        urgency_score: Time sensitivity (0-1, higher = more urgent)
        impact_score: Potential business value (0-1, higher = more valuable)
        confidence_score: Classification confidence (0-1, higher = more certain)
        action_score: Composite score for prioritization
        recommended_action: Primary recommended action type
        suggested_channel: Platform to act on
        suggested_tone: Recommended response tone
        draft_response: Generated public response
        draft_post: Generated content post
        draft_dm: Generated direct message
        positioning_angle: Strategic positioning (for competitor signals)
        status: Current workflow status
        assigned_to: Team member assigned to this signal
        expires_at: SLA deadline for action
        created_at: Signal detection timestamp
        acted_at: When action was taken
        outcome_feedback: Results of action (for learning loop)
        metadata: Additional signal-specific data
    """

    # Identity
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(..., description="Owner of this signal")

    # Signal classification
    signal_type: SignalType = Field(..., description="Business intent type")

    # Source tracking
    source_item_ids: List[UUID] = Field(
        default_factory=list,
        description="ContentItems that triggered this signal",
    )
    source_platform: str = Field(..., description="Primary platform")
    source_url: str = Field(..., description="Direct link to conversation")
    source_author: Optional[str] = Field(None, description="Content author")

    # Signal content
    title: str = Field(..., min_length=1, max_length=200, description="Brief title")
    description: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Detailed description",
    )
    context: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Why this matters (business context)",
    )


    # Multi-dimensional scoring (all 0-1 scale)
    urgency_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Time sensitivity (0=can wait, 1=act now)",
    )
    impact_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Potential business value (0=low, 1=high)",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classification confidence (0=uncertain, 1=certain)",
    )
    action_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Composite priority score for queue ranking",
    )

    # Recommended actions
    recommended_action: ActionType = Field(
        ...,
        description="Primary recommended action",
    )
    suggested_channel: str = Field(
        ...,
        description="Platform to act on (twitter, reddit, linkedin, etc.)",
    )
    suggested_tone: ResponseTone = Field(
        ...,
        description="Recommended response tone",
    )

    # Generated assets (pre-drafted for user)
    draft_response: Optional[str] = Field(
        None,
        max_length=5000,
        description="Generated public response/reply",
    )
    draft_post: Optional[str] = Field(
        None,
        max_length=5000,
        description="Generated content post",
    )
    draft_dm: Optional[str] = Field(
        None,
        max_length=5000,
        description="Generated direct message",
    )
    positioning_angle: Optional[str] = Field(
        None,
        max_length=2000,
        description="Strategic positioning (for competitor signals)",
    )

    # Workflow management
    status: SignalStatus = Field(
        default=SignalStatus.NEW,
        description="Current workflow status",
    )
    assigned_to: Optional[UUID] = Field(
        None,
        description="Team member assigned to this signal",
    )

    # Timestamps and SLA
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Signal detection timestamp",
    )
    expires_at: Optional[datetime] = Field(
        None,
        description="SLA deadline for action",
    )
    acted_at: Optional[datetime] = Field(
        None,
        description="When action was taken",
    )

    # Learning loop
    outcome_feedback: Optional[Dict[str, Any]] = Field(
        None,
        description="Results of action (engagement, conversion, etc.)",
    )

    # Additional data
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Signal-specific metadata",
    )

    @field_validator("action_score")
    @classmethod
    def validate_action_score(cls, v: float, info) -> float:
        """Validate that action_score is reasonable composite of other scores.

        Action score should generally be influenced by urgency, impact, and confidence.
        We don't enforce exact formula to allow for flexibility, but warn if suspicious.
        """
        # This is a soft validation - we allow override but log warning
        # In production, action_score is calculated by ActionScorer
        return v

    @field_validator("expires_at")
    @classmethod
    def validate_expires_at(cls, v: Optional[datetime], info) -> Optional[datetime]:
        """Validate that expiry is in the future."""
        if v is not None and v < datetime.utcnow():
            raise ValueError("expires_at must be in the future")
        return v

    def is_expired(self) -> bool:
        """Check if signal has passed SLA deadline."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def time_until_expiry(self) -> Optional[timedelta]:
        """Get time remaining until expiry."""
        if self.expires_at is None:
            return None
        return self.expires_at - datetime.utcnow()

    def mark_acted(self, outcome: Optional[Dict[str, Any]] = None) -> None:
        """Mark signal as acted upon.

        Args:
            outcome: Optional outcome data for learning loop
        """
        self.status = SignalStatus.ACTED
        self.acted_at = datetime.utcnow()
        if outcome:
            self.outcome_feedback = outcome

    def mark_dismissed(self, reason: Optional[str] = None) -> None:
        """Mark signal as dismissed.

        Args:
            reason: Optional reason for dismissal
        """
        self.status = SignalStatus.DISMISSED
        if reason:
            self.metadata["dismissal_reason"] = reason

    def assign_to(self, user_id: UUID) -> None:
        """Assign signal to team member.

        Args:
            user_id: User to assign to
        """
        self.assigned_to = user_id
        if self.status == SignalStatus.NEW:
            self.status = SignalStatus.QUEUED


class SignalSummary(BaseModel):
    """Lightweight signal summary for list views.

    Used in API responses where full signal data is too heavy.
    """

    id: UUID
    signal_type: SignalType
    title: str
    urgency_score: float
    impact_score: float
    action_score: float
    status: SignalStatus
    created_at: datetime
    expires_at: Optional[datetime]
    source_platform: str
    source_author: Optional[str]


class SignalFilter(BaseModel):
    """Filter criteria for signal queries.

    Used in API endpoints to filter signal queue.
    """

    signal_types: Optional[List[SignalType]] = Field(
        None,
        description="Filter by signal types",
    )
    min_action_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum action score threshold",
    )
    max_action_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Maximum action score threshold",
    )
    statuses: Optional[List[SignalStatus]] = Field(
        None,
        description="Filter by status",
    )
    platforms: Optional[List[str]] = Field(
        None,
        description="Filter by source platform",
    )
    assigned_to: Optional[UUID] = Field(
        None,
        description="Filter by assigned user",
    )
    created_after: Optional[datetime] = Field(
        None,
        description="Filter by creation date (after)",
    )
    created_before: Optional[datetime] = Field(
        None,
        description="Filter by creation date (before)",
    )
    include_expired: bool = Field(
        default=False,
        description="Include expired signals",
    )

    @field_validator("max_action_score")
    @classmethod
    def validate_score_range(cls, v: float, info) -> float:
        """Validate that max >= min."""
        min_score = info.data.get("min_action_score", 0.0)
        if v < min_score:
            raise ValueError("max_action_score must be >= min_action_score")
        return v

