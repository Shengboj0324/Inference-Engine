"""Output format models and preferences."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class OutputFormat(str, Enum):
    """Supported output formats."""

    # Text formats
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    PDF = "pdf"

    # Rich media
    IMAGE = "image"  # Infographic
    VIDEO = "video"  # AI-generated video
    AUDIO = "audio"  # Podcast/audio summary
    SLIDESHOW = "slideshow"  # Presentation

    # Interactive
    INTERACTIVE_HTML = "interactive_html"
    DASHBOARD = "dashboard"
    NEWSLETTER = "newsletter"

    # Social media optimized
    TWITTER_THREAD = "twitter_thread"
    LINKEDIN_POST = "linkedin_post"
    INSTAGRAM_STORY = "instagram_story"


class TextStyle(str, Enum):
    """Text output styles."""

    PROFESSIONAL = "professional"
    CASUAL = "casual"
    ACADEMIC = "academic"
    JOURNALISTIC = "journalistic"
    TECHNICAL = "technical"
    ELI5 = "eli5"  # Explain like I'm 5
    EXECUTIVE = "executive"  # Executive summary
    DETAILED = "detailed"
    BULLET_POINTS = "bullet_points"
    NARRATIVE = "narrative"


class TonePreference(str, Enum):
    """Tone preferences for output."""

    NEUTRAL = "neutral"
    OPTIMISTIC = "optimistic"
    CRITICAL = "critical"
    HUMOROUS = "humorous"
    SERIOUS = "serious"
    INSPIRATIONAL = "inspirational"


class LengthPreference(str, Enum):
    """Length preferences."""

    BRIEF = "brief"  # 1-2 paragraphs
    MEDIUM = "medium"  # 3-5 paragraphs
    DETAILED = "detailed"  # 6-10 paragraphs
    COMPREHENSIVE = "comprehensive"  # 10+ paragraphs


class VisualizationType(str, Enum):
    """Types of visualizations."""

    TIMELINE = "timeline"
    NETWORK_GRAPH = "network_graph"
    WORD_CLOUD = "word_cloud"
    SENTIMENT_CHART = "sentiment_chart"
    TOPIC_DISTRIBUTION = "topic_distribution"
    ENGAGEMENT_METRICS = "engagement_metrics"
    COMPARISON_TABLE = "comparison_table"
    INFOGRAPHIC = "infographic"


class OutputPreferences(BaseModel):
    """User preferences for output customization."""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    name: str = "Default"
    description: Optional[str] = None

    # Format preferences
    primary_format: OutputFormat = OutputFormat.MARKDOWN
    fallback_formats: List[OutputFormat] = Field(
        default_factory=lambda: [OutputFormat.PLAIN_TEXT]
    )

    # Text preferences
    text_style: TextStyle = TextStyle.PROFESSIONAL
    tone: TonePreference = TonePreference.NEUTRAL
    length: LengthPreference = LengthPreference.MEDIUM

    # Content preferences
    include_sources: bool = True
    include_timestamps: bool = True
    include_media: bool = True
    include_links: bool = True
    include_quotes: bool = True
    include_statistics: bool = True

    # Visualization preferences
    include_visualizations: bool = True
    visualization_types: List[VisualizationType] = Field(
        default_factory=lambda: [
            VisualizationType.TIMELINE,
            VisualizationType.TOPIC_DISTRIBUTION,
        ]
    )

    # Language preferences
    language: str = "en"
    translate_quotes: bool = False

    # Accessibility
    include_alt_text: bool = True
    high_contrast: bool = False
    large_text: bool = False

    # Advanced
    custom_template: Optional[str] = None
    custom_css: Optional[str] = None
    custom_prompts: Dict[str, str] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class OutputMetadata(BaseModel):
    """Metadata for generated output."""

    format: OutputFormat
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generation_time_ms: int
    word_count: Optional[int] = None
    character_count: Optional[int] = None
    file_size_bytes: Optional[int] = None
    media_urls: List[str] = Field(default_factory=list)
    quality_score: Optional[float] = None
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None


class GeneratedOutput(BaseModel):
    """Generated output content."""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    digest_id: Optional[UUID] = None
    preferences_id: UUID

    # Content
    format: OutputFormat
    content: str  # Main content (text, HTML, JSON, etc.)
    media_files: Dict[str, bytes] = Field(default_factory=dict)  # filename -> bytes
    attachments: List[str] = Field(default_factory=list)  # URLs or file paths

    # Metadata
    metadata: OutputMetadata
    title: Optional[str] = None
    summary: Optional[str] = None

    # Storage
    storage_url: Optional[str] = None  # S3/MinIO URL
    expires_at: Optional[datetime] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)


class OutputRequest(BaseModel):
    """Request for generating output."""

    user_id: UUID
    digest_id: Optional[UUID] = None
    cluster_ids: Optional[List[UUID]] = None
    content_item_ids: Optional[List[UUID]] = None

    preferences_id: Optional[UUID] = None
    override_preferences: Optional[Dict[str, Any]] = None

    # Custom instructions
    custom_prompt: Optional[str] = None
    focus_topics: Optional[List[str]] = None
    exclude_topics: Optional[List[str]] = None

    # Delivery
    deliver_via: List[str] = Field(default_factory=lambda: ["api"])  # api, email, slack, etc.

