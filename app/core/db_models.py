"""SQLAlchemy database models."""

from datetime import datetime
from typing import Any, Dict, List
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


class InterestProfile(Base):
    """User interest profile model."""

    __tablename__ = "interest_profiles"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), unique=True
    )
    interest_topics: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    negative_filters: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    interest_embedding: Mapped[List[float] | None] = mapped_column(
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
    encrypted_credentials: Mapped[str | None] = mapped_column(Text, nullable=True)
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
    author: Mapped[str | None] = mapped_column(String(255), nullable=True)
    channel: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    media_type: Mapped[MediaType] = mapped_column(Enum(MediaType), nullable=False)
    media_urls: Mapped[List[str]] = mapped_column(ARRAY(Text), default=list)

    # Timestamps
    published_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Enrichment
    topics: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    lang: Mapped[str | None] = mapped_column(String(10), nullable=True)
    embedding: Mapped[List[float] | None] = mapped_column(Vector(1536), nullable=True)

    # Metadata
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

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
    diversity_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Cross-platform
    platforms_represented: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    perspective_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="clusters")

