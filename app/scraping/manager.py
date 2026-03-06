"""Production-grade scraping manager with circuit breakers and retry logic."""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.scraping.base import (
    BaseScraper,
    BlockedException,
    ProxyConfig,
    RateLimitConfig,
    RateLimitException,
    ScrapedContent,
    ScraperConfig,
    ScraperException,
)
from app.scraping.playwright_scraper import PlaywrightScraper

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreaker:
    """Circuit breaker for scraping operations."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = ScraperException,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to track
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise ScraperException("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

    def _should_attempt_reset(self) -> bool:
        """Check if should attempt to reset circuit."""
        if self.last_failure_time is None:
            return True

        return (
            datetime.utcnow() - self.last_failure_time
        ).total_seconds() >= self.recovery_timeout


class ScrapingManager:
    """Production-grade scraping manager with advanced features."""

    def __init__(
        self,
        max_concurrent_scrapers: int = 10,
        default_config: Optional[ScraperConfig] = None,
        proxy_pool: Optional[List[ProxyConfig]] = None,
    ):
        """Initialize scraping manager.

        Args:
            max_concurrent_scrapers: Maximum concurrent scraping operations
            default_config: Default scraper configuration
            proxy_pool: Pool of proxy configurations
        """
        self.max_concurrent_scrapers = max_concurrent_scrapers
        self.default_config = default_config or ScraperConfig()
        self.proxy_pool = proxy_pool or []
        self._semaphore = asyncio.Semaphore(max_concurrent_scrapers)
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._active_scrapers: Dict[str, BaseScraper] = {}
        self._proxy_index = 0

        # Memory leak protection
        self._max_circuit_breakers = 1000  # Prevent unbounded growth
        self._last_cleanup = datetime.utcnow()
        self._cleanup_interval = timedelta(minutes=10)

    def _get_circuit_breaker(self, domain: str) -> CircuitBreaker:
        """Get or create circuit breaker for domain.

        Args:
            domain: Domain name

        Returns:
            Circuit breaker for domain
        """
        # Periodic cleanup
        self._cleanup_circuit_breakers()

        if domain not in self._circuit_breakers:
            # Enforce max size limit
            if len(self._circuit_breakers) >= self._max_circuit_breakers:
                # Remove oldest closed circuit breakers
                closed_breakers = [
                    (d, cb) for d, cb in self._circuit_breakers.items()
                    if cb.state == CircuitState.CLOSED
                ]
                if closed_breakers:
                    # Remove 20% of closed breakers
                    to_remove = max(1, len(closed_breakers) // 5)
                    for domain_to_remove, _ in closed_breakers[:to_remove]:
                        del self._circuit_breakers[domain_to_remove]
                    logger.warning(
                        f"Circuit breaker limit reached, removed {to_remove} closed breakers"
                    )

            self._circuit_breakers[domain] = CircuitBreaker()

        return self._circuit_breakers[domain]

    def _cleanup_circuit_breakers(self) -> None:
        """Clean up old circuit breakers to prevent memory leaks."""
        now = datetime.utcnow()

        # Only cleanup periodically
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now

        # Remove circuit breakers that have been closed for a long time
        domains_to_remove = []
        for domain, cb in self._circuit_breakers.items():
            if cb.state == CircuitState.CLOSED and cb.last_failure_time:
                # Remove if no failures in last hour
                if now - cb.last_failure_time > timedelta(hours=1):
                    domains_to_remove.append(domain)

        for domain in domains_to_remove:
            del self._circuit_breakers[domain]

        if domains_to_remove:
            logger.info(
                f"Cleaned up {len(domains_to_remove)} inactive circuit breakers. "
                f"Remaining: {len(self._circuit_breakers)}"
            )

    def _get_next_proxy(self) -> Optional[ProxyConfig]:
        """Get next proxy from pool using round-robin."""
        if not self.proxy_pool:
            return None

        proxy = self.proxy_pool[self._proxy_index]
        self._proxy_index = (self._proxy_index + 1) % len(self.proxy_pool)
        return proxy

    async def scrape_with_retry(
        self,
        url: str,
        config: Optional[ScraperConfig] = None,
        max_retries: int = 3,
        **kwargs,
    ) -> ScrapedContent:
        """Scrape URL with automatic retry and circuit breaker.

        Args:
            url: URL to scrape
            config: Scraper configuration (uses default if None)
            max_retries: Maximum retry attempts
            **kwargs: Additional scraper arguments

        Returns:
            Scraped content

        Raises:
            ScraperException: If scraping fails after retries
        """
        from urllib.parse import urlparse

        domain = urlparse(url).netloc
        circuit_breaker = self._get_circuit_breaker(domain)
        scraper_config = config or self.default_config

        async def _scrape():
            async with self._semaphore:
                # Get proxy if enabled
                proxy = None
                if scraper_config.use_proxy:
                    proxy = self._get_next_proxy()

                # Create scraper
                scraper = PlaywrightScraper(
                    scraper_config,
                    rate_limit_config=RateLimitConfig(),
                    proxy_config=proxy,
                )

                try:
                    result = await scraper.scrape(url, **kwargs)
                    return result
                finally:
                    await scraper.close()

        # Retry with exponential backoff
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(
                    (RateLimitException, BlockedException)
                ),
                stop=stop_after_attempt(max_retries),
                wait=wait_exponential(multiplier=1, min=4, max=60),
                reraise=True,
            ):
                with attempt:
                    result = await circuit_breaker.call(_scrape)
                    return result
        except RetryError as e:
            logger.error(f"Failed to scrape {url} after {max_retries} retries: {e}")
            raise ScraperException(f"Scraping failed after retries: {e}")

    async def scrape_batch(
        self,
        urls: List[str],
        config: Optional[ScraperConfig] = None,
        **kwargs,
    ) -> List[ScrapedContent]:
        """Scrape multiple URLs concurrently.

        Args:
            urls: List of URLs to scrape
            config: Scraper configuration
            **kwargs: Additional scraper arguments

        Returns:
            List of scraped content
        """
        tasks = [self.scrape_with_retry(url, config, **kwargs) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed ScrapedContent
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    ScrapedContent(
                        url=urls[i], success=False, error=str(result)
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    async def close_all(self):
        """Close all active scrapers."""
        for scraper in self._active_scrapers.values():
            await scraper.close()
        self._active_scrapers.clear()

