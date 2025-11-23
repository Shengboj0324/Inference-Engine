"""Base scraper interface with compliance and anti-detection capabilities."""

import asyncio
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ScraperType(str, Enum):
    """Types of scrapers."""

    PLAYWRIGHT = "playwright"
    SELENIUM = "selenium"
    REQUESTS = "requests"
    API = "api"


class ProxyType(str, Enum):
    """Proxy types."""

    HTTP = "http"
    HTTPS = "https"
    SOCKS5 = "socks5"
    RESIDENTIAL = "residential"
    DATACENTER = "datacenter"


class ComplianceLevel(str, Enum):
    """Compliance levels for scraping."""

    STRICT = "strict"  # Only official APIs, no scraping
    MODERATE = "moderate"  # Scraping with robots.txt compliance
    AGGRESSIVE = "aggressive"  # Full scraping with anti-detection


@dataclass
class ProxyConfig:
    """Proxy configuration."""

    host: str
    port: int
    proxy_type: ProxyType
    username: Optional[str] = None
    password: Optional[str] = None
    country: Optional[str] = None
    rotation_interval: int = 300  # seconds


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    requests_per_second: float = 1.0
    requests_per_minute: int = 30
    requests_per_hour: int = 1000
    burst_size: int = 5
    backoff_factor: float = 2.0
    max_retries: int = 3


class ScraperConfig(BaseModel):
    """Configuration for web scraper."""

    scraper_type: ScraperType = ScraperType.PLAYWRIGHT
    compliance_level: ComplianceLevel = ComplianceLevel.MODERATE
    user_agent: Optional[str] = None
    headless: bool = True
    javascript_enabled: bool = True
    cookies_enabled: bool = True
    images_enabled: bool = False
    timeout: int = 30
    page_load_timeout: int = 60
    retry_attempts: int = 3
    retry_delay: int = 5
    use_proxy: bool = False
    proxy_rotation: bool = False
    fingerprint_randomization: bool = True
    respect_robots_txt: bool = True
    max_concurrent_requests: int = 5


class ScrapedContent(BaseModel):
    """Scraped content result."""

    url: str
    html: Optional[str] = None
    text: Optional[str] = None
    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    media_urls: List[str] = Field(default_factory=list)
    links: List[str] = Field(default_factory=list)
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
    success: bool = True
    error: Optional[str] = None
    response_time_ms: int = 0
    status_code: Optional[int] = None


class ScraperException(Exception):
    """Base exception for scraper errors."""

    pass


class RateLimitException(ScraperException):
    """Rate limit exceeded."""

    pass


class BlockedException(ScraperException):
    """Scraper was blocked or detected."""

    pass


class ComplianceException(ScraperException):
    """Compliance violation detected."""

    pass


class BaseScraper(ABC):
    """Base class for all scrapers with anti-detection and compliance."""

    def __init__(
        self,
        config: ScraperConfig,
        rate_limit_config: Optional[RateLimitConfig] = None,
        proxy_config: Optional[ProxyConfig] = None,
    ):
        """Initialize scraper.

        Args:
            config: Scraper configuration
            rate_limit_config: Rate limiting configuration
            proxy_config: Proxy configuration
        """
        self.config = config
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.proxy_config = proxy_config
        self._request_times: List[datetime] = []
        self._last_request_time: Optional[datetime] = None

    @abstractmethod
    async def scrape(self, url: str, **kwargs) -> ScrapedContent:
        """Scrape content from URL.

        Args:
            url: URL to scrape
            **kwargs: Additional scraper-specific arguments

        Returns:
            Scraped content

        Raises:
            ScraperException: If scraping fails
        """
        pass

    @abstractmethod
    async def close(self):
        """Clean up resources."""
        pass

    async def _check_rate_limit(self):
        """Check and enforce rate limits."""
        now = datetime.utcnow()

        # Clean old request times
        cutoff = now - timedelta(hours=1)
        self._request_times = [t for t in self._request_times if t > cutoff]

        # Check per-second rate
        if self._last_request_time:
            time_since_last = (now - self._last_request_time).total_seconds()
            min_interval = 1.0 / self.rate_limit_config.requests_per_second
            if time_since_last < min_interval:
                await asyncio.sleep(min_interval - time_since_last)

        # Check per-minute rate
        minute_ago = now - timedelta(minutes=1)
        recent_requests = sum(1 for t in self._request_times if t > minute_ago)
        if recent_requests >= self.rate_limit_config.requests_per_minute:
            raise RateLimitException("Per-minute rate limit exceeded")

        # Check per-hour rate
        if len(self._request_times) >= self.rate_limit_config.requests_per_hour:
            raise RateLimitException("Per-hour rate limit exceeded")

        self._last_request_time = now
        self._request_times.append(now)

