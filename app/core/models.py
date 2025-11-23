"""Core Pydantic models for the application."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MediaType(str, Enum):
    """Content media type enumeration."""

    TEXT = "text"
    VIDEO = "video"
    IMAGE = "image"
    MIXED = "mixed"


class SourcePlatform(str, Enum):
    """Supported source platforms."""

    # Social Media
    REDDIT = "reddit"
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    WECHAT = "wechat"

    # News Sources
    RSS = "rss"
    NEWSAPI = "newsapi"
    NYTIMES = "nytimes"
    WSJ = "wsj"
    ABC_NEWS = "abc_news"
    ABC_NEWS_AU = "abc_news_au"
    GOOGLE_NEWS = "google_news"
    APPLE_NEWS = "apple_news"


class ContentItem(BaseModel):
    """Unified content item model across all platforms."""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    source_platform: SourcePlatform
    source_id: str
    source_url: str

    # Content metadata
    author: Optional[str] = None
    channel: Optional[str] = None
    title: str
    raw_text: Optional[str] = None
    media_type: MediaType
    media_urls: List[str] = Field(default_factory=list)

    # Timestamps
    published_at: datetime
    fetched_at: datetime = Field(default_factory=datetime.utcnow)

    # Enrichment
    topics: List[str] = Field(default_factory=list)
    lang: Optional[str] = None
    embedding: Optional[List[float]] = None

    # Platform-specific metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

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
                "topics": ["AI", "technology", "research"],
                "lang": "en",
            }
        }


class UserInterestProfile(BaseModel):
    """User interest and preference profile."""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    interest_topics: List[str] = Field(default_factory=list)
    negative_filters: List[str] = Field(default_factory=list)
    interest_embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PlatformConfig(BaseModel):
    """Platform-specific configuration for a user."""

    platform: SourcePlatform
    enabled: bool = True
    credentials: Dict[str, Any] = Field(default_factory=dict)
    settings: Dict[str, Any] = Field(default_factory=dict)


class UserProfile(BaseModel):
    """Complete user profile with all configurations."""

    id: UUID = Field(default_factory=uuid4)
    email: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Interest profile
    interest_profile: Optional[UserInterestProfile] = None

    # Platform configurations
    platform_configs: List[PlatformConfig] = Field(default_factory=list)


class Cluster(BaseModel):
    """Content cluster representing a storyline or topic."""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Cluster metadata
    topic: str
    summary: str
    keywords: List[str] = Field(default_factory=list)

    # Content items in this cluster
    item_ids: List[UUID] = Field(default_factory=list)
    items: List[ContentItem] = Field(default_factory=list)

    # Scoring
    relevance_score: float
    diversity_score: Optional[float] = None

    # Cross-platform analysis
    platforms_represented: List[SourcePlatform] = Field(default_factory=list)
    perspective_summary: Optional[str] = None


class DigestRequest(BaseModel):
    """Request parameters for generating a daily digest."""

    topics: Optional[List[str]] = None
    since: Optional[datetime] = None
    max_clusters: int = Field(default=20, ge=1, le=100)
    platforms: Optional[List[SourcePlatform]] = None
    include_videos: bool = True


class DigestResponse(BaseModel):
    """Daily digest response."""

    generated_at: datetime = Field(default_factory=datetime.utcnow)
    period_start: datetime
    period_end: datetime
    clusters: List[Cluster]
    total_items: int
    summary: Optional[str] = None

