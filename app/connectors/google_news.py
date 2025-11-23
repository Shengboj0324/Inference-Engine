"""Google News RSS connector.

Google News provides RSS feeds with advanced search parameters.
Documentation: https://www.newscatcherapi.com/blog-posts/google-news-rss-search-parameters-the-missing-documentaiton
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus
from uuid import UUID

import aiohttp
import feedparser

from app.connectors.base import BaseConnector, ConnectorConfig, FetchResult
from app.core.errors import ConnectorError
from app.core.models import ContentItem, ContentType, SourcePlatform

logger = logging.getLogger(__name__)


class GoogleNewsConnector(BaseConnector):
    """Google News RSS connector with advanced search.
    
    Features:
    - Topic-based feeds
    - Keyword search with operators
    - Location-based news
    - Language filtering
    - Time-based filtering
    """

    BASE_URL = "https://news.google.com/rss"
    
    def __init__(self, config: ConnectorConfig, user_id: UUID):
        """Initialize Google News connector.
        
        Settings:
        - topics: List of topics (e.g., ['WORLD', 'TECHNOLOGY', 'BUSINESS'])
        - keywords: List of search keywords
        - location: Country code (e.g., 'US', 'GB')
        - language: Language code (e.g., 'en', 'es')
        """
        super().__init__(config, user_id)

    async def validate_credentials(self) -> bool:
        """Validate Google News access (no credentials needed)."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.BASE_URL}",
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Google News validation failed: {e}")
            return False

    async def fetch_content(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> FetchResult:
        """Fetch Google News articles."""
        items: List[ContentItem] = []
        errors: List[str] = []
        
        try:
            # Fetch from topics
            topics = self.config.settings.get("topics", [])
            for topic in topics:
                topic_items = await self._fetch_topic(topic, max_items=20)
                items.extend(topic_items)
            
            # Fetch from keyword searches
            keywords = self.config.settings.get("keywords", [])
            for keyword in keywords:
                search_items = await self._fetch_search(keyword, max_items=20)
                items.extend(search_items)
            
            # Filter by date if specified
            if since:
                items = [item for item in items if item.published_at >= since]
            
            return FetchResult(items=items[:max_items], errors=errors)
        
        except Exception as e:
            logger.error(f"Error fetching Google News content: {e}")
            errors.append(str(e))
            return FetchResult(items=items, errors=errors)

    async def _fetch_topic(self, topic: str, max_items: int = 20) -> List[ContentItem]:
        """Fetch articles for a specific topic."""
        items: List[ContentItem] = []
        
        location = self.config.settings.get("location", "US")
        language = self.config.settings.get("language", "en")
        
        url = f"{self.BASE_URL}/headlines/section/topic/{topic}?hl={language}&gl={location}&ceid={location}:{language}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch topic {topic}")
                        return items
                    
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    for entry in feed.entries[:max_items]:
                        items.append(self._parse_entry(entry, source=f"Topic: {topic}"))
        
        except Exception as e:
            logger.warning(f"Error fetching topic {topic}: {e}")
        
        return items

    async def _fetch_search(self, keyword: str, max_items: int = 20) -> List[ContentItem]:
        """Fetch articles matching keyword search."""
        items: List[ContentItem] = []
        
        location = self.config.settings.get("location", "US")
        language = self.config.settings.get("language", "en")
        
        # Build search URL with parameters
        query = quote_plus(keyword)
        url = f"{self.BASE_URL}/search?q={query}&hl={language}&gl={location}&ceid={location}:{language}"
        
        # Add time filter if specified
        when = self.config.settings.get("when")  # h (hour), d (day), w (week), m (month), y (year)
        if when:
            url += f"&when={when}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to search for {keyword}")
                        return items
                    
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    for entry in feed.entries[:max_items]:
                        items.append(self._parse_entry(entry, source=f"Search: {keyword}"))
        
        except Exception as e:
            logger.warning(f"Error searching for {keyword}: {e}")
        
        return items

    def _parse_entry(self, entry: Any, source: str = "Google News") -> ContentItem:
        """Parse RSS entry into ContentItem."""
        # Extract published date
        published_at = datetime.now()
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            from time import mktime
            published_at = datetime.fromtimestamp(mktime(entry.published_parsed))
        
        # Extract source from title (Google News format: "Title - Source")
        title = entry.get("title", "")
        author = "Unknown"
        if " - " in title:
            parts = title.rsplit(" - ", 1)
            title = parts[0]
            author = parts[1]
        
        return ContentItem(
            platform=SourcePlatform.GOOGLE_NEWS,
            platform_id=entry.get("id", entry.get("link", "")),
            content_type=ContentType.ARTICLE,
            title=title,
            text_content=entry.get("summary", ""),
            url=entry.get("link", ""),
            author=author,
            published_at=published_at,
            metadata={
                "source": source,
                "original_source": author,
            },
            user_id=self.user_id,
        )

    async def get_user_feeds(self) -> List[str]:
        """Get configured Google News sources."""
        feeds = []
        
        topics = self.config.settings.get("topics", [])
        for topic in topics:
            feeds.append(f"Topic: {topic}")
        
        keywords = self.config.settings.get("keywords", [])
        for keyword in keywords:
            feeds.append(f"Search: {keyword}")
        
        return feeds or ["Google News"]

