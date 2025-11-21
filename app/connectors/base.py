"""Base connector interface and utilities."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel

from app.core.models import ContentItem, SourcePlatform


class ConnectorConfig(BaseModel):
    """Base configuration for platform connectors."""

    platform: SourcePlatform
    credentials: Dict[str, Any]
    settings: Dict[str, Any] = {}


class RateLimitInfo(BaseModel):
    """Rate limit information."""

    limit: int
    remaining: int
    reset_at: datetime


class FetchResult(BaseModel):
    """Result of a content fetch operation."""

    items: List[ContentItem]
    cursor: Optional[str] = None
    rate_limit: Optional[RateLimitInfo] = None
    errors: List[str] = []


class BaseConnector(ABC):
    """Abstract base class for all platform connectors."""

    def __init__(self, config: ConnectorConfig, user_id: UUID):
        """Initialize connector with configuration and user context.

        Args:
            config: Platform-specific configuration including credentials
            user_id: User ID for data segregation
        """
        self.config = config
        self.user_id = user_id
        self.platform = config.platform

    @abstractmethod
    async def validate_credentials(self) -> bool:
        """Validate that the provided credentials are valid.

        Returns:
            True if credentials are valid, False otherwise
        """
        pass

    @abstractmethod
    async def fetch_content(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> FetchResult:
        """Fetch content from the platform.

        Args:
            since: Fetch content published after this timestamp
            cursor: Pagination cursor from previous fetch
            max_items: Maximum number of items to fetch

        Returns:
            FetchResult containing items and pagination info
        """
        pass

    @abstractmethod
    async def get_user_feeds(self) -> List[str]:
        """Get list of feeds/channels/subreddits the user follows.

        Returns:
            List of feed identifiers
        """
        pass

    async def test_connection(self) -> Dict[str, Any]:
        """Test the connection and return diagnostic information.

        Returns:
            Dictionary with connection status and details
        """
        try:
            is_valid = await self.validate_credentials()
            if is_valid:
                feeds = await self.get_user_feeds()
                return {
                    "status": "success",
                    "platform": self.platform.value,
                    "feeds_count": len(feeds),
                    "sample_feeds": feeds[:5] if feeds else [],
                }
            else:
                return {
                    "status": "error",
                    "platform": self.platform.value,
                    "error": "Invalid credentials",
                }
        except Exception as e:
            return {
                "status": "error",
                "platform": self.platform.value,
                "error": str(e),
            }

    def _create_content_item(
        self,
        source_id: str,
        source_url: str,
        title: str,
        **kwargs: Any,
    ) -> ContentItem:
        """Helper to create a ContentItem with common fields.

        Args:
            source_id: Platform-specific content ID
            source_url: Canonical URL to the content
            title: Content title
            **kwargs: Additional ContentItem fields

        Returns:
            ContentItem instance
        """
        return ContentItem(
            user_id=self.user_id,
            source_platform=self.platform,
            source_id=source_id,
            source_url=source_url,
            title=title,
            **kwargs,
        )


class ConnectorError(Exception):
    """Base exception for connector errors."""

    pass


class AuthenticationError(ConnectorError):
    """Raised when authentication fails."""

    pass


class RateLimitError(ConnectorError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, reset_at: Optional[datetime] = None):
        super().__init__(message)
        self.reset_at = reset_at


class PlatformError(ConnectorError):
    """Raised when the platform returns an error."""

    pass

