"""SQLAlchemy database models."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base
from app.core.models import MediaType, SourcePlatform
from app.core.signal_models import SignalType, ActionType, SignalStatus, ResponseTone


class User(Base):
    """User account model."""

    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # ── Personalization ───────────────────────────────────────────────────────
    # Free-form JSON blob storing the user's GTM strategic priorities.
    # Schema (all keys optional):
    #   {
    #     "competitors":   ["Notion", "Coda"],        # surface forms to up-weight
    #     "focus_areas":   ["permissions", "pricing"], # product/domain focus
    #     "tone":          "assertive",                # preferred response tone
    #     "urgency_weight": 1.4,                       # multiplier for urgency_score
    #     "impact_weight":  1.2                        # multiplier for impact_score
    #   }
    # Null means "use system defaults".  Validated at read-time by UserContext.
    strategic_priorities: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True, default=None,
        comment="User GTM priorities: competitors, focus_areas, tone, score weights",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    interest_profile: Mapped["InterestProfile"] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    platform_configs: Mapped[List["PlatformConfigDB"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    content_items: Mapped[List["ContentItemDB"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    clusters: Mapped[List["ClusterDB"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    actionable_signals: Mapped[List["ActionableSignalDB"]] = relationship(
        back_populates="user",
        foreign_keys="ActionableSignalDB.user_id",
        cascade="all, delete-orphan",
    )


class InterestProfile(Base):
    """User interest profile model."""

    __tablename__ = "interest_profiles"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), unique=True
    )
    interest_topics: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    negative_filters: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    interest_embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(1536), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="interest_profile")


class PlatformConfigDB(Base):
    """Platform configuration model."""

    __tablename__ = "platform_configs"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    platform: Mapped[SourcePlatform] = mapped_column(Enum(SourcePlatform), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    encrypted_credentials: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    settings: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    last_fetch_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="platform_configs")


class ContentItemDB(Base):
    """Content item database model."""

    __tablename__ = "content_items"
    __table_args__ = (
        # Unique constraint to prevent duplicate content from same source
        UniqueConstraint('user_id', 'source_platform', 'source_id', name='uq_user_platform_source'),
    )

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    source_platform: Mapped[SourcePlatform] = mapped_column(
        Enum(SourcePlatform), nullable=False, index=True
    )
    source_id: Mapped[str] = mapped_column(String(500), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)

    # Content
    author: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    channel: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    media_type: Mapped[MediaType] = mapped_column(Enum(MediaType), nullable=False)
    media_urls: Mapped[List[str]] = mapped_column(ARRAY(Text), default=list)

    # Timestamps
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Enrichment
    topics: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    lang: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(1536), nullable=True)

    # Metadata
    metadata_: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, default=dict)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="content_items")


class ClusterDB(Base):
    """Content cluster database model."""

    __tablename__ = "clusters"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )

    # Cluster metadata
    topic: Mapped[str] = mapped_column(String(500), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    keywords: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)

    # Content references
    item_ids: Mapped[List[UUID]] = mapped_column(ARRAY(UUID(as_uuid=True)), default=list)

    # Scoring
    relevance_score: Mapped[float] = mapped_column(Float, nullable=False)
    diversity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Cross-platform
    platforms_represented: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    perspective_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="clusters")



class ActionableSignalDB(Base):
    """Actionable signal database model - primary product entity.

    This is the core table for the signal-to-action workflow. Each row represents
    a business-actionable opportunity or risk detected from content.

    Design notes:
    - Indexed on action_score for fast queue retrieval
    - Indexed on status for workflow queries
    - Indexed on created_at for time-based queries
    - Indexed on expires_at for SLA monitoring
    - Foreign key to users with cascade delete
    - source_item_ids stored as array for multi-item signals
    """

    __tablename__ = "actionable_signals"

    # Primary key
    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Ownership
    user_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Link to inference layer (Phase 1 architecture)
    signal_inference_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("signal_inferences.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Link to SignalInferenceDB for calibrated inference results"
    )
    normalized_observation_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("normalized_observations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Link to NormalizedObservationDB for source content"
    )

    # Signal classification
    signal_type: Mapped[SignalType] = mapped_column(
        Enum(SignalType),
        nullable=False,
        index=True,
    )

    # Source tracking
    source_item_ids: Mapped[List[UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)),
        nullable=False,
        default=list,
    )
    source_platform: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )
    source_url: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    source_author: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    # Signal content
    title: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )
    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    context: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # Multi-dimensional scoring (0-1 scale)
    urgency_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    impact_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    confidence_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    action_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        index=True,  # Critical for queue sorting
    )

    # Recommended actions
    recommended_action: Mapped[ActionType] = mapped_column(
        Enum(ActionType),
        nullable=False,
    )
    suggested_channel: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )
    suggested_tone: Mapped[ResponseTone] = mapped_column(
        Enum(ResponseTone),
        nullable=False,
    )

    # Generated assets
    draft_response: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    draft_post: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    draft_dm: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    positioning_angle: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # Workflow management
    status: Mapped[SignalStatus] = mapped_column(
        Enum(SignalStatus),
        nullable=False,
        default=SignalStatus.NEW,
        index=True,  # Critical for status filtering
    )
    assigned_to: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Team collaboration (competitive_analysis.md §5.5)
    team_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="UUID of the team this signal is scoped to",
    )
    assigned_role: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="TeamRole value of the user who last assigned this signal",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,  # For time-based queries
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,  # For SLA monitoring
    )
    acted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Learning loop
    outcome_feedback: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
    )

    # Additional data
    metadata_: Mapped[Dict[str, Any]] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
    )

    # Relationships
    user: Mapped["User"] = relationship(
        back_populates="actionable_signals",
        foreign_keys=[user_id],
    )
    assigned_user: Mapped[Optional["User"]] = relationship(
        foreign_keys=[assigned_to],
    )





# ============================================================================
# NEW DOMAIN MODELS - Phase 1 Architecture
# ============================================================================


class RawObservationDB(Base):
    """Raw observation database model - Layer 1.

    Stores unprocessed platform data with minimal transformation.
    """

    __tablename__ = "raw_observations"
    __table_args__ = (
        UniqueConstraint('user_id', 'source_platform', 'source_id', name='uq_raw_user_platform_source'),
    )

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    source_platform: Mapped[SourcePlatform] = mapped_column(
        Enum(SourcePlatform), nullable=False, index=True
    )
    source_id: Mapped[str] = mapped_column(String(500), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)

    # Content
    author: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    author_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    channel: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    channel_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    media_type: Mapped[MediaType] = mapped_column(Enum(MediaType), nullable=False)
    media_urls: Mapped[List[str]] = mapped_column(ARRAY(Text), default=list)

    # Timestamps
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Platform-specific metadata
    platform_metadata_: Mapped[Dict[str, Any]] = mapped_column(
        "platform_metadata", JSON, default=dict
    )

    # Relationships
    user: Mapped["User"] = relationship()


class NormalizedObservationDB(Base):
    """Normalized observation database model - Layer 2.

    Stores normalized, enriched content ready for inference.
    """

    __tablename__ = "normalized_observations"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    raw_observation_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("raw_observations.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    source_platform: Mapped[SourcePlatform] = mapped_column(
        Enum(SourcePlatform), nullable=False, index=True
    )
    source_id: Mapped[str] = mapped_column(String(500), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)

    # Normalized content
    author: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    channel: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    original_language: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    translated_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    media_type: Mapped[MediaType] = mapped_column(Enum(MediaType), nullable=False)
    media_urls: Mapped[List[str]] = mapped_column(ARRAY(Text), default=list)

    # Timestamps
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    normalized_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )

    # Enrichment (stored as JSON for flexibility)
    entities: Mapped[List[Dict[str, Any]]] = mapped_column(ARRAY(JSON), default=list)
    topics: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    keywords: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    sentiment: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Thread context (stored as JSON)
    thread_context_: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "thread_context", JSON, nullable=True
    )

    # Quality scores
    quality: Mapped[str] = mapped_column(String(50), nullable=False, default="unknown")
    quality_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    completeness_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Engagement features
    engagement_velocity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    virality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Embeddings
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(1536), nullable=True)

    # Metadata
    normalization_metadata_: Mapped[Dict[str, Any]] = mapped_column(
        "normalization_metadata", JSON, default=dict
    )

    # Relationships
    user: Mapped["User"] = relationship()
    raw_observation: Mapped["RawObservationDB"] = relationship()


class SignalInferenceDB(Base):
    """Signal inference database model - Layer 3.

    Stores ML/LLM interpretation results with calibrated confidence.
    """

    __tablename__ = "signal_inferences"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    normalized_observation_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("normalized_observations.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )

    # Inference results (stored as JSON for flexibility)
    predictions_: Mapped[List[Dict[str, Any]]] = mapped_column(
        "predictions", ARRAY(JSON), default=list
    )
    top_prediction_: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "top_prediction", JSON, nullable=True
    )

    # Abstention
    abstained: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    abstention_reason: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    abstention_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Rationale
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evidence_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Calibration metrics (stored as JSON)
    calibration_metrics_: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "calibration_metrics", JSON, nullable=True
    )

    # Model provenance
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    inference_method: Mapped[str] = mapped_column(String(100), nullable=False)

    # Timestamps
    inferred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )

    # Metadata
    inference_metadata_: Mapped[Dict[str, Any]] = mapped_column(
        "inference_metadata", JSON, default=dict
    )

    # Relationships
    user: Mapped["User"] = relationship()
    normalized_observation: Mapped["NormalizedObservationDB"] = relationship()


class SignalFeedbackDB(Base):
    """Signal-classification feedback record.

    Stores human corrections to model predictions.  Used by
    ``FeedbackStore`` to drive online calibration updates via
    ``ConfidenceCalibrator.update()``.

    Schema notes:
        * Additive-only — no existing columns are modified.
        * ``signal_id`` is a soft FK to ``actionable_signals.id`` declared
          with ``ON DELETE CASCADE`` so rows are pruned when the parent
          signal is deleted.
    """

    __tablename__ = "signal_feedback"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    signal_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("actionable_signals.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    predicted_type: Mapped[str] = mapped_column(String(50), nullable=False)
    true_type: Mapped[str] = mapped_column(String(50), nullable=False)
    predicted_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    user_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
