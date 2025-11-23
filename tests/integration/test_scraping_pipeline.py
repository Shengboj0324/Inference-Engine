"""Integration tests for scraping pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.scraping.base import ScraperConfig, ComplianceLevel
from app.scraping.manager import ScrapingManager
from app.scraping.playwright_scraper import PlaywrightScraper


@pytest.fixture
async def scraping_manager():
    """Create scraping manager for tests."""
    config = ScraperConfig(
        headless=True,
        respect_robots_txt=True,
        compliance_level=ComplianceLevel.MODERATE,
    )
    manager = ScrapingManager(default_config=config)
    yield manager
    await manager.close_all()


@pytest.mark.asyncio
async def test_scrape_with_retry_success(scraping_manager):
    """Test successful scraping with retry logic."""
    url = "https://example.com"

    with patch.object(PlaywrightScraper, "scrape") as mock_scrape:
        mock_scrape.return_value = MagicMock(
            url=url,
            title="Example Domain",
            text="This domain is for use in illustrative examples",
            success=True,
            status_code=200,
        )

        result = await scraping_manager.scrape_with_retry(url)

        assert result.success is True
        assert result.url == url
        assert result.status_code == 200


@pytest.mark.asyncio
async def test_scrape_with_retry_failure(scraping_manager):
    """Test scraping failure and retry logic."""
    url = "https://example.com"

    with patch.object(PlaywrightScraper, "scrape") as mock_scrape:
        from app.scraping.base import BlockedException

        mock_scrape.side_effect = BlockedException("Blocked by server")

        with pytest.raises(Exception):
            await scraping_manager.scrape_with_retry(url, max_retries=2)

        # Should have retried
        assert mock_scrape.call_count == 2


@pytest.mark.asyncio
async def test_scrape_batch(scraping_manager):
    """Test batch scraping."""
    urls = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
    ]

    with patch.object(PlaywrightScraper, "scrape") as mock_scrape:
        mock_scrape.return_value = MagicMock(
            success=True,
            status_code=200,
        )

        results = await scraping_manager.scrape_batch(urls)

        assert len(results) == 3
        assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_circuit_breaker_opens_on_failures():
    """Test circuit breaker opens after threshold failures."""
    from app.scraping.manager import CircuitBreaker
    from app.scraping.base import ScraperException

    circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)

    async def failing_function():
        raise ScraperException("Test failure")

    # First 3 failures should be tracked
    for _ in range(3):
        with pytest.raises(ScraperException):
            await circuit_breaker.call(failing_function)

    # Circuit should now be open
    assert circuit_breaker.state.value == "open"

    # Next call should fail immediately without calling function
    with pytest.raises(ScraperException, match="Circuit breaker is OPEN"):
        await circuit_breaker.call(failing_function)


@pytest.mark.asyncio
async def test_robots_txt_compliance():
    """Test robots.txt compliance checking."""
    from app.scraping.robots import RobotsTxtChecker

    checker = RobotsTxtChecker()

    # Test with a URL that should be allowed
    can_fetch = await checker.can_fetch("https://example.com/page", "TestBot")

    # Should default to True if no robots.txt
    assert can_fetch is True

    await checker.close()


@pytest.mark.asyncio
async def test_fingerprint_randomization():
    """Test browser fingerprint randomization."""
    from app.scraping.fingerprint import BrowserFingerprint

    fingerprint = BrowserFingerprint()

    # Generate multiple fingerprints
    options1 = await fingerprint.generate_context_options(randomize=True)
    options2 = await fingerprint.generate_context_options(randomize=True)

    # Should be different (with high probability)
    assert options1["user_agent"] != options2["user_agent"] or options1["viewport"] != options2["viewport"]


@pytest.mark.asyncio
async def test_rate_limiting():
    """Test rate limiting in scraper."""
    from app.scraping.base import RateLimitConfig, RateLimitException

    config = ScraperConfig()
    rate_config = RateLimitConfig(
        requests_per_second=2.0,
        requests_per_minute=5,
    )

    scraper = PlaywrightScraper(config, rate_limit_config=rate_config)

    # Simulate rapid requests
    for i in range(6):
        try:
            await scraper._check_rate_limit()
        except RateLimitException:
            # Should hit rate limit on 6th request
            assert i == 5
            break

    await scraper.close()


@pytest.mark.asyncio
async def test_proxy_rotation():
    """Test proxy rotation in scraping manager."""
    from app.scraping.base import ProxyConfig, ProxyType

    proxies = [
        ProxyConfig(host="proxy1.example.com", port=8080, proxy_type=ProxyType.HTTP),
        ProxyConfig(host="proxy2.example.com", port=8080, proxy_type=ProxyType.HTTP),
        ProxyConfig(host="proxy3.example.com", port=8080, proxy_type=ProxyType.HTTP),
    ]

    manager = ScrapingManager(proxy_pool=proxies)

    # Get proxies in round-robin
    proxy1 = manager._get_next_proxy()
    proxy2 = manager._get_next_proxy()
    proxy3 = manager._get_next_proxy()
    proxy4 = manager._get_next_proxy()

    assert proxy1.host == "proxy1.example.com"
    assert proxy2.host == "proxy2.example.com"
    assert proxy3.host == "proxy3.example.com"
    assert proxy4.host == "proxy1.example.com"  # Should wrap around

    await manager.close_all()

