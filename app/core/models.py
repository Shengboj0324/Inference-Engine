"""Core Pydantic models for the application."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class MediaType(str, Enum):
    """Content media type enumeration."""

    TEXT = "text"
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"
    PDF = "pdf"
    MIXED = "mixed"


class PlatformAuthStatus(str, Enum):
    """OAuth connection status for a user-platform credential.

    Surfaced in the WebSocket push payload so the frontend can prompt
    the user to re-authenticate when a token expires mid-session.

    Values
    ------
    CONNECTED      — Valid access token; ingestion can proceed.
    EXPIRED        — Token TTL has elapsed; refresh is needed.
    REVOKED        — User revoked app access on the platform side.
    NOT_CONNECTED  — No OAuth flow has been completed for this platform.
    """

    CONNECTED = "connected"
    EXPIRED = "expired"
    REVOKED = "revoked"
    NOT_CONNECTED = "not_connected"


class ContentType(str, Enum):
    """Semantic content type used by platform connectors.

    Distinct from MediaType (which describes the technical format).
    ContentType describes what the content *is* at a business level.
    """

    ARTICLE = "article"
    VIDEO = "video"
    IMAGE = "image"
    TEXT = "text"
    LINK = "link"


class SourcePlatform(str, Enum):
    """Supported source platforms.

    Grouped by source family per the Source Intelligence Layer design:
      - Social streams  : REDDIT, YOUTUBE, TIKTOK, FACEBOOK, INSTAGRAM, WECHAT
      - News/editorial  : RSS, NEWSAPI, NYTIMES, WSJ, ABC_NEWS, GOOGLE_NEWS, APPLE_NEWS
      - Developer/release: GITHUB_RELEASES, GITHUB_REPO_EVENTS, GITHUB_DISCUSSIONS,
                           CHANGELOG, DOCS_MONITOR
      - Research        : ARXIV, OPENREVIEW, SEMANTIC_SCHOLAR
      - Media/audio     : PODCAST_RSS, TRANSCRIPT_FEED, YOUTUBE_TRANSCRIPT
    """

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

    # Developer / Release streams  (Phase 1 — Source Intelligence Layer)
    GITHUB_RELEASES = "github_releases"
    GITHUB_REPO_EVENTS = "github_repo_events"
    GITHUB_DISCUSSIONS = "github_discussions"
    CHANGELOG = "changelog"
    DOCS_MONITOR = "docs_monitor"

    # Research streams  (Phase 1)
    ARXIV = "arxiv"
    OPENREVIEW = "openreview"
    SEMANTIC_SCHOLAR = "semantic_scholar"

    # Media / Audio streams  (Phase 1)
    PODCAST_RSS = "podcast_rss"
    TRANSCRIPT_FEED = "transcript_feed"
    YOUTUBE_TRANSCRIPT = "youtube_transcript"


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

    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
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
        },
    )


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

    model_config = ConfigDict(validate_assignment=True)


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



class TeamRole(str, Enum):
    """Role-based access levels for team signal collaboration.

    Implements docs/competitive_analysis.md §5.5 — Team Collaboration.

    Roles are ordered by privilege (VIEWER < ANALYST < MANAGER).  The
    ordering is used by ``has_role_at_least()`` to enforce access gates.

    - ``VIEWER``  — read-only access; can view signals but not act on them.
    - ``ANALYST`` — can dismiss signals; cannot assign or manage team members.
    - ``MANAGER`` — full access including signal assignment and team digests.
    """

    VIEWER = "viewer"
    ANALYST = "analyst"
    MANAGER = "manager"

    @classmethod
    def privilege_level(cls, role: "TeamRole") -> int:
        """Return a numeric privilege level for comparison (higher = more access).

        Args:
            role: The ``TeamRole`` to evaluate.

        Returns:
            Integer privilege level: VIEWER=1, ANALYST=2, MANAGER=3.
        """
        _levels = {cls.VIEWER: 1, cls.ANALYST: 2, cls.MANAGER: 3}
        return _levels[role]

    @classmethod
    def has_role_at_least(cls, user_role: "TeamRole", required: "TeamRole") -> bool:
        """Return ``True`` if ``user_role`` meets or exceeds ``required``.

        Args:
            user_role: The role held by the requesting user.
            required: The minimum role required to access a resource.

        Returns:
            ``True`` if ``user_role`` privilege ≥ ``required`` privilege.
        """
        return cls.privilege_level(user_role) >= cls.privilege_level(required)
