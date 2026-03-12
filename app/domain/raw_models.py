"""Raw observation models - Layer 1 of domain architecture.

Raw observations represent unprocessed platform data with minimal transformation.
These models preserve platform-specific fields and metadata.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from app.core.models import MediaType, SourcePlatform


class RawObservation(BaseModel):
    """Raw observation from a platform source.
    
    This is the first layer - direct representation of fetched content
    with minimal processing. Platform-specific fields are preserved in metadata.
    """

    # Identity
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    source_platform: SourcePlatform
    source_id: str = Field(..., description="Platform-specific content ID")
    source_url: str
    
    # Content
    author: Optional[str] = None
    author_id: Optional[str] = None
    channel: Optional[str] = None
    channel_id: Optional[str] = None
    title: str
    raw_text: Optional[str] = None
    media_type: MediaType
    media_urls: List[str] = Field(default_factory=list)
    
    # Timestamps
    published_at: datetime
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Platform-specific metadata (preserved as-is)
    platform_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Platform-specific fields: upvotes, shares, hashtags, etc."
    )
    
    class Config:
        """Pydantic config."""
        
        json_schema_extra = {
            "example": {
                "source_platform": "reddit",
                "source_id": "abc123",
                "source_url": "https://reddit.com/r/technology/comments/abc123",
                "author": "user123",
                "channel": "r/technology",
                "title": "New AI breakthrough announced",
                "raw_text": "Researchers have developed...",
                "media_type": "text",
                "published_at": "2024-01-15T10:30:00Z",
                "platform_metadata": {
                    "upvotes": 1234,
                    "num_comments": 56,
                    "awards": ["gold", "helpful"]
                }
            }
        }


class RawThread(BaseModel):
    """Raw thread/conversation context.
    
    Represents a conversation thread with parent-child relationships.
    """

    thread_id: str
    platform: SourcePlatform
    root_observation_id: UUID
    parent_observation_id: Optional[UUID] = None
    child_observation_ids: List[UUID] = Field(default_factory=list)
    depth: int = 0
    
    # Thread metadata
    total_participants: int = 0
    total_messages: int = 0
    thread_metadata: Dict[str, Any] = Field(default_factory=dict)


class RawEngagementMetrics(BaseModel):
    """Raw engagement metrics from platform.
    
    Preserves platform-specific engagement signals.
    """

    observation_id: UUID
    platform: SourcePlatform
    
    # Common metrics (when available)
    views: Optional[int] = None
    likes: Optional[int] = None
    shares: Optional[int] = None
    comments: Optional[int] = None
    saves: Optional[int] = None
    
    # Platform-specific metrics
    platform_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Platform-specific: upvotes, retweets, reactions, etc."
    )
    
    # Temporal data
    measured_at: datetime = Field(default_factory=datetime.utcnow)


class RawAuthorProfile(BaseModel):
    """Raw author profile data.
    
    Preserves author information from platform.
    """

    author_id: str
    platform: SourcePlatform
    username: str
    display_name: Optional[str] = None
    
    # Profile metrics
    follower_count: Optional[int] = None
    following_count: Optional[int] = None
    post_count: Optional[int] = None
    verified: bool = False
    
    # Profile metadata
    bio: Optional[str] = None
    profile_url: Optional[str] = None
    profile_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Temporal
    profile_fetched_at: datetime = Field(default_factory=datetime.utcnow)

