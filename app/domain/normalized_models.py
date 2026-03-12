"""Normalized observation models - Layer 2 of domain architecture.

Normalized observations represent unified, cross-platform content with:
- Language normalization and translation
- Entity extraction (competitors, products, locations)
- Thread context and conversation structure
- Quality and completeness scores
- Engagement features
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from app.core.models import MediaType, SourcePlatform


class ContentQuality(str, Enum):
    """Content quality assessment."""
    
    HIGH = "high"  # Clear, complete, actionable
    MEDIUM = "medium"  # Usable but may need context
    LOW = "low"  # Noisy, incomplete, or unclear
    SPAM = "spam"  # Likely spam or low-value
    UNKNOWN = "unknown"  # Not yet assessed


class SentimentPolarity(str, Enum):
    """Sentiment polarity."""
    
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class EntityMention(BaseModel):
    """Extracted entity mention."""
    
    entity_type: str = Field(..., description="competitor, product, location, person, etc.")
    entity_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    span_start: Optional[int] = None
    span_end: Optional[int] = None
    canonical_id: Optional[str] = None  # Link to knowledge base


class ThreadContext(BaseModel):
    """Thread/conversation context."""
    
    thread_id: str
    is_root: bool = False
    parent_id: Optional[UUID] = None
    depth: int = 0
    total_thread_messages: int = 1
    user_is_participant: bool = False  # Whether our user/brand is in thread


class NormalizedObservation(BaseModel):
    """Normalized observation - unified cross-platform representation.
    
    This is Layer 2 - content has been normalized, enriched, and validated.
    Ready for inference pipeline.
    """

    # Identity (links back to raw)
    id: UUID = Field(default_factory=uuid4)
    raw_observation_id: UUID
    user_id: UUID
    source_platform: SourcePlatform
    source_id: str
    source_url: str
    
    # Normalized content
    author: Optional[str] = None
    channel: Optional[str] = None
    title: str
    normalized_text: Optional[str] = Field(
        None,
        description="Cleaned, normalized text (lowercased, whitespace normalized)"
    )
    original_language: Optional[str] = None
    translated_text: Optional[str] = Field(
        None,
        description="English translation if original_language != 'en'"
    )
    media_type: MediaType
    media_urls: List[str] = Field(default_factory=list)
    
    # Timestamps
    published_at: datetime
    fetched_at: datetime
    normalized_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Enrichment
    entities: List[EntityMention] = Field(
        default_factory=list,
        description="Extracted entities: competitors, products, locations"
    )
    topics: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    sentiment: SentimentPolarity = SentimentPolarity.UNKNOWN
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    
    # Thread context
    thread_context: Optional[ThreadContext] = None
    
    # Quality assessment
    quality: ContentQuality = ContentQuality.UNKNOWN
    quality_score: float = Field(0.0, ge=0.0, le=1.0)
    completeness_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="How complete is the information (0=missing context, 1=complete)"
    )
    
    # Engagement features (normalized across platforms)
    engagement_velocity: Optional[float] = Field(
        None,
        description="Engagement rate per hour since publication"
    )
    virality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Embeddings
    embedding: Optional[List[float]] = Field(
        None,
        description="Semantic embedding vector (1536-dim for OpenAI)"
    )
    
    # Metadata
    normalization_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Normalization process metadata: model versions, confidence scores"
    )
    
    @field_validator('quality_score', 'completeness_score')
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Ensure scores are in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {v}")
        return v
    
    class Config:
        """Pydantic config."""
        
        validate_assignment = True  # Allow updates during enrichment pipeline

