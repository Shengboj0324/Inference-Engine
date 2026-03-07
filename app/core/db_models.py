"""SQLAlchemy database models."""

from datetime import datetime
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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="platform_configs")


class ContentItemDB(Base):
    """Content item database model."""

    __tablename__ = "content_items"

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
    published_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

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
        DateTime, default=datetime.utcnow, index=True
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

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True,  # For time-based queries
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True,
        index=True,  # For SLA monitoring
    )
    acted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
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

