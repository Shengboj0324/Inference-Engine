"""Base connector interface and utilities with comprehensive error handling.

This module provides the base connector interface for all platform integrations with:
- Retry logic with exponential backoff
- Circuit breaker pattern
- Rate limit handling
- Comprehensive error handling
- Connection health monitoring
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel

from app.core.models import ContentItem, SourcePlatform
from app.core.errors import APIError, RateLimitError, AuthenticationError
from app.core.retry import CircuitBreaker, retry_with_backoff

logger = logging.getLogger(__name__)


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
    """Abstract base class for all platform connectors with error handling.

    Features:
    - Automatic retry with exponential backoff
    - Circuit breaker for fault tolerance
    - Rate limit handling
    - Connection health monitoring
    - Comprehensive error logging
    """

    def __init__(self, config: ConnectorConfig, user_id: UUID):
        """Initialize connector with configuration and user context.

        Args:
            config: Platform-specific configuration including credentials
            user_id: User ID for data segregation
        """
        self.config = config
        self.user_id = user_id
        self.platform = config.platform

        # Initialize circuit breaker for this connector
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exception=APIError
        )

        # Track connection health
        self.last_successful_fetch: Optional[datetime] = None
        self.consecutive_failures = 0
        self.total_requests = 0
        self.failed_requests = 0

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
                    "health": self.get_health_status(),
                }
            else:
                return {
                    "status": "error",
                    "platform": self.platform.value,
                    "error": "Invalid credentials",
                }
        except Exception as e:
            logger.error(f"Connection test failed for {self.platform.value}: {e}")
            return {
                "status": "error",
                "platform": self.platform.value,
                "error": str(e),
            }

    def get_health_status(self) -> Dict[str, Any]:
        """Get connector health status.

        Returns:
            Dictionary with health metrics
        """
        success_rate = 0.0
        if self.total_requests > 0:
            success_rate = (self.total_requests - self.failed_requests) / self.total_requests

        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "consecutive_failures": self.consecutive_failures,
            "last_successful_fetch": self.last_successful_fetch.isoformat() if self.last_successful_fetch else None,
            "circuit_breaker_state": self.circuit_breaker.state.value,
        }

    def _record_success(self) -> None:
        """Record successful request."""
        self.total_requests += 1
        self.consecutive_failures = 0
        self.last_successful_fetch = datetime.utcnow()
        logger.debug(f"{self.platform.value}: Request successful")

    def _record_failure(self, error: Exception) -> None:
        """Record failed request.

        Args:
            error: Exception that caused the failure
        """
        self.total_requests += 1
        self.failed_requests += 1
        self.consecutive_failures += 1
        logger.warning(
            f"{self.platform.value}: Request failed (consecutive: {self.consecutive_failures})",
            extra={"error": str(error)}
        )

    @retry_with_backoff(
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        retry_on=(APIError, RateLimitError)
    )
    async def fetch_content_with_retry(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> FetchResult:
        """Fetch content with automatic retry and error handling.

        This method wraps fetch_content with retry logic and circuit breaker.

        Args:
            since: Fetch content published after this timestamp
            cursor: Pagination cursor from previous fetch
            max_items: Maximum number of items to fetch

        Returns:
            FetchResult containing items and pagination info

        Raises:
            APIError: If fetch fails after retries
            RateLimitError: If rate limited
            AuthenticationError: If credentials are invalid
        """
        try:
            result = await self.circuit_breaker.call_async(
                self.fetch_content,
                since=since,
                cursor=cursor,
                max_items=max_items
            )
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise

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

