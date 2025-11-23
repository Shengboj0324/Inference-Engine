"""Robots.txt compliance checker."""

import asyncio
from typing import Dict, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import aiohttp


class RobotsTxtChecker:
    """Check robots.txt compliance for URLs."""

    def __init__(self, cache_ttl: int = 3600):
        """Initialize robots.txt checker.

        Args:
            cache_ttl: Cache TTL in seconds
        """
        self._cache: Dict[str, RobotFileParser] = {}
        self._cache_ttl = cache_ttl
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _fetch_robots_txt(self, base_url: str) -> Optional[str]:
        """Fetch robots.txt content.

        Args:
            base_url: Base URL of the site

        Returns:
            robots.txt content or None if not found
        """
        robots_url = urljoin(base_url, "/robots.txt")

        try:
            session = await self._get_session()
            async with session.get(robots_url, timeout=10) as response:
                if response.status == 200:
                    return await response.text()
        except Exception:
            pass

        return None

    async def can_fetch(
        self, url: str, user_agent: str = "*"
    ) -> bool:
        """Check if URL can be fetched according to robots.txt.

        Args:
            url: URL to check
            user_agent: User agent string

        Returns:
            True if URL can be fetched, False otherwise
        """
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        # Check cache
        if base_url not in self._cache:
            # Fetch robots.txt
            robots_content = await self._fetch_robots_txt(base_url)

            if robots_content:
                parser = RobotFileParser()
                parser.parse(robots_content.splitlines())
                self._cache[base_url] = parser
            else:
                # No robots.txt means everything is allowed
                return True

        parser = self._cache.get(base_url)
        if parser:
            return parser.can_fetch(user_agent, url)

        return True

    async def get_crawl_delay(
        self, url: str, user_agent: str = "*"
    ) -> Optional[float]:
        """Get crawl delay from robots.txt.

        Args:
            url: URL to check
            user_agent: User agent string

        Returns:
            Crawl delay in seconds or None
        """
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        if base_url not in self._cache:
            await self.can_fetch(url, user_agent)

        parser = self._cache.get(base_url)
        if parser:
            return parser.crawl_delay(user_agent)

        return None

    async def close(self):
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

