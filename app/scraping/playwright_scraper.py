"""Production-grade Playwright scraper with anti-detection."""

import asyncio
import random
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

from app.scraping.base import (
    BaseScraper,
    BlockedException,
    ComplianceException,
    ProxyConfig,
    RateLimitConfig,
    ScrapedContent,
    ScraperConfig,
    ScraperException,
)
from app.scraping.fingerprint import BrowserFingerprint
from app.scraping.robots import RobotsTxtChecker


class PlaywrightScraper(BaseScraper):
    """Advanced Playwright-based scraper with anti-detection."""

    def __init__(
        self,
        config: ScraperConfig,
        rate_limit_config: Optional[RateLimitConfig] = None,
        proxy_config: Optional[ProxyConfig] = None,
    ):
        """Initialize Playwright scraper."""
        super().__init__(config, rate_limit_config, proxy_config)
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._fingerprint = BrowserFingerprint()
        self._robots_checker = RobotsTxtChecker()

    async def _ensure_browser(self):
        """Ensure browser is initialized."""
        if self._browser is None:
            self._playwright = await async_playwright().start()

            # Browser launch options
            launch_options: Dict[str, Any] = {
                "headless": self.config.headless,
                "args": [
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-web-security",
                    "--disable-features=IsolateOrigins,site-per-process",
                ],
            }

            # Add proxy if configured
            if self.proxy_config:
                launch_options["proxy"] = {
                    "server": f"{self.proxy_config.proxy_type.value}://{self.proxy_config.host}:{self.proxy_config.port}",
                }
                if self.proxy_config.username:
                    launch_options["proxy"]["username"] = self.proxy_config.username
                    launch_options["proxy"]["password"] = self.proxy_config.password

            self._browser = await self._playwright.chromium.launch(**launch_options)

            # Create context with fingerprint
            context_options = await self._fingerprint.generate_context_options(
                randomize=self.config.fingerprint_randomization
            )
            self._context = await self._browser.new_context(**context_options)

            # Add stealth scripts
            await self._context.add_init_script(
                """
                // Override navigator.webdriver
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                
                // Override chrome property
                window.chrome = {
                    runtime: {}
                };
                
                // Override permissions
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
                
                // Override plugins
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                
                // Override languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en']
                });
            """
            )

    async def scrape(self, url: str, **kwargs) -> ScrapedContent:
        """Scrape content from URL using Playwright.

        Args:
            url: URL to scrape
            **kwargs: Additional options
                - wait_for_selector: CSS selector to wait for
                - wait_for_timeout: Time to wait after page load
                - scroll_to_bottom: Whether to scroll to bottom
                - extract_links: Whether to extract all links
                - screenshot: Whether to take screenshot

        Returns:
            Scraped content

        Raises:
            ScraperException: If scraping fails
        """
        # Check rate limits
        await self._check_rate_limit()

        # Check robots.txt compliance
        if self.config.respect_robots_txt:
            if not await self._robots_checker.can_fetch(url, self.config.user_agent):
                raise ComplianceException(f"robots.txt disallows scraping {url}")

        # Ensure browser is ready
        await self._ensure_browser()

        page: Optional[Page] = None
        start_time = asyncio.get_event_loop().time()

        try:
            # Create new page
            page = await self._context.new_page()

            # Set viewport to random size
            if self.config.fingerprint_randomization:
                width = random.randint(1280, 1920)
                height = random.randint(720, 1080)
                await page.set_viewport_size({"width": width, "height": height})

            # Navigate to URL
            response = await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=self.config.page_load_timeout * 1000,
            )

            if response is None:
                raise ScraperException(f"Failed to load {url}")

            status_code = response.status

            # Check for blocking
            if status_code in [403, 429]:
                raise BlockedException(f"Blocked with status {status_code}")

            # Wait for specific selector if provided
            if kwargs.get("wait_for_selector"):
                await page.wait_for_selector(
                    kwargs["wait_for_selector"], timeout=self.config.timeout * 1000
                )

            # Wait for additional timeout if specified
            if kwargs.get("wait_for_timeout"):
                await asyncio.sleep(kwargs["wait_for_timeout"])

            # Scroll to bottom if requested
            if kwargs.get("scroll_to_bottom", False):
                await self._scroll_to_bottom(page)

            # Extract content
            html = await page.content()
            text = await page.evaluate("() => document.body.innerText")
            title = await page.title()

            # Extract metadata
            metadata = await self._extract_metadata(page)

            # Extract media URLs
            media_urls = await self._extract_media_urls(page)

            # Extract links if requested
            links = []
            if kwargs.get("extract_links", False):
                links = await self._extract_links(page, url)

            # Take screenshot if requested
            if kwargs.get("screenshot", False):
                screenshot_path = kwargs.get("screenshot_path", f"/tmp/{urlparse(url).netloc}.png")
                await page.screenshot(path=screenshot_path, full_page=True)
                metadata["screenshot_path"] = screenshot_path

            end_time = asyncio.get_event_loop().time()
            response_time_ms = int((end_time - start_time) * 1000)

            return ScrapedContent(
                url=url,
                html=html,
                text=text,
                title=title,
                metadata=metadata,
                media_urls=media_urls,
                links=links,
                success=True,
                response_time_ms=response_time_ms,
                status_code=status_code,
            )

        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            response_time_ms = int((end_time - start_time) * 1000)

            return ScrapedContent(
                url=url,
                success=False,
                error=str(e),
                response_time_ms=response_time_ms,
            )

        finally:
            if page:
                await page.close()

    async def _scroll_to_bottom(self, page: Page):
        """Scroll to bottom of page to load dynamic content."""
        await page.evaluate(
            """
            async () => {
                await new Promise((resolve) => {
                    let totalHeight = 0;
                    const distance = 100;
                    const timer = setInterval(() => {
                        const scrollHeight = document.body.scrollHeight;
                        window.scrollBy(0, distance);
                        totalHeight += distance;

                        if(totalHeight >= scrollHeight){
                            clearInterval(timer);
                            resolve();
                        }
                    }, 100);
                });
            }
        """
        )

    async def _extract_metadata(self, page: Page) -> Dict[str, Any]:
        """Extract metadata from page."""
        metadata = await page.evaluate(
            """
            () => {
                const meta = {};

                // Open Graph tags
                document.querySelectorAll('meta[property^="og:"]').forEach(tag => {
                    meta[tag.getAttribute('property')] = tag.getAttribute('content');
                });

                // Twitter Card tags
                document.querySelectorAll('meta[name^="twitter:"]').forEach(tag => {
                    meta[tag.getAttribute('name')] = tag.getAttribute('content');
                });

                // Standard meta tags
                ['description', 'keywords', 'author'].forEach(name => {
                    const tag = document.querySelector(`meta[name="${name}"]`);
                    if (tag) meta[name] = tag.getAttribute('content');
                });

                return meta;
            }
        """
        )
        return metadata

    async def _extract_media_urls(self, page: Page) -> List[str]:
        """Extract media URLs from page."""
        media_urls = await page.evaluate(
            """
            () => {
                const urls = [];

                // Images
                document.querySelectorAll('img[src]').forEach(img => {
                    urls.push(img.src);
                });

                // Videos
                document.querySelectorAll('video[src], video source[src]').forEach(video => {
                    urls.push(video.src);
                });

                // Audio
                document.querySelectorAll('audio[src], audio source[src]').forEach(audio => {
                    urls.push(audio.src);
                });

                return [...new Set(urls)];
            }
        """
        )
        return media_urls

    async def _extract_links(self, page: Page, base_url: str) -> List[str]:
        """Extract all links from page."""
        links = await page.evaluate(
            """
            () => {
                return Array.from(document.querySelectorAll('a[href]'))
                    .map(a => a.href)
                    .filter(href => href && !href.startsWith('javascript:'));
            }
        """
        )
        return list(set(links))

    async def close(self):
        """Clean up browser resources."""
        if self._context:
            await self._context.close()
            self._context = None

        if self._browser:
            await self._browser.close()
            self._browser = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

