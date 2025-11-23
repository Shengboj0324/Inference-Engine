"""ABC News RSS connector.

ABC News provides RSS feeds for various sections.
Note: RSS feed URLs may vary by region (US, Australia, etc.)
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import aiohttp
import feedparser

from app.connectors.base import BaseConnector, ConnectorConfig, FetchResult
from app.core.errors import ConnectorError
from app.core.models import ContentItem, ContentType, SourcePlatform

logger = logging.getLogger(__name__)


class ABCNewsConnector(BaseConnector):
    """ABC News RSS connector.
    
    Supports both ABC News (US) and ABC News (Australia) feeds.
    """

    # ABC News US feeds (via third-party aggregators as ABC doesn't provide official RSS)
    US_FEEDS = {
        "top_stories": "https://abcnews.go.com/abcnews/topstories",
        "politics": "https://abcnews.go.com/abcnews/politicsheadlines",
        "international": "https://abcnews.go.com/abcnews/internationalheadlines",
        "technology": "https://abcnews.go.com/abcnews/technologyheadlines",
        "health": "https://abcnews.go.com/abcnews/healthheadlines",
    }
    
    # ABC News Australia feeds
    AU_FEEDS = {
        "news": "https://www.abc.net.au/news/feed/51120/rss.xml",
        "world": "https://www.abc.net.au/news/feed/51126/rss.xml",
        "business": "https://www.abc.net.au/news/feed/51124/rss.xml",
        "analysis": "https://www.abc.net.au/news/feed/51130/rss.xml",
        "sport": "https://www.abc.net.au/news/feed/51128/rss.xml",
        "science": "https://www.abc.net.au/news/feed/51136/rss.xml",
        "health": "https://www.abc.net.au/news/feed/51134/rss.xml",
        "arts": "https://www.abc.net.au/news/feed/51138/rss.xml",
    }
    
    def __init__(self, config: ConnectorConfig, user_id: UUID):
        """Initialize ABC News connector.
        
        Settings:
        - region: 'US' or 'AU' (default: 'US')
        - feeds: List of feed names
        """
        super().__init__(config, user_id)
        self.region = self.config.settings.get("region", "US")
        self.feeds_map = self.AU_FEEDS if self.region == "AU" else self.US_FEEDS

    async def validate_credentials(self) -> bool:
        """Validate ABC News RSS access."""
        try:
            # Test first available feed
            first_feed = list(self.feeds_map.values())[0]
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    first_feed,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"ABC News validation failed: {e}")
            return False

    async def fetch_content(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> FetchResult:
        """Fetch ABC News articles from RSS feeds."""
        items: List[ContentItem] = []
        errors: List[str] = []
        
        try:
            feeds = self.config.settings.get("feeds", list(self.feeds_map.keys())[:3])
            
            for feed_name in feeds:
                if feed_name not in self.feeds_map:
                    logger.warning(f"Unknown ABC News feed: {feed_name}")
                    continue
                
                feed_items = await self._fetch_feed(feed_name, max_items=30)
                items.extend(feed_items)
            
            # Filter by date if specified
            if since:
                items = [item for item in items if item.published_at >= since]
            
            return FetchResult(items=items[:max_items], errors=errors)
        
        except Exception as e:
            logger.error(f"Error fetching ABC News content: {e}")
            errors.append(str(e))
            return FetchResult(items=items, errors=errors)

    async def _fetch_feed(self, feed_name: str, max_items: int = 30) -> List[ContentItem]:
        """Fetch articles from a specific RSS feed."""
        items: List[ContentItem] = []
        url = self.feeds_map[feed_name]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch ABC News feed {feed_name}")
                        return items
                    
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    for entry in feed.entries[:max_items]:
                        items.append(self._parse_entry(entry, feed_name))
        
        except Exception as e:
            logger.warning(f"Error fetching ABC News feed {feed_name}: {e}")
        
        return items

    def _parse_entry(self, entry: Any, feed_name: str) -> ContentItem:
        """Parse RSS entry into ContentItem."""
        # Extract published date
        published_at = datetime.now()
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            from time import mktime
            published_at = datetime.fromtimestamp(mktime(entry.published_parsed))
        
        platform = SourcePlatform.ABC_NEWS_AU if self.region == "AU" else SourcePlatform.ABC_NEWS
        
        return ContentItem(
            platform=platform,
            platform_id=entry.get("id", entry.get("link", "")),
            content_type=ContentType.ARTICLE,
            title=entry.get("title", ""),
            text_content=entry.get("summary", ""),
            url=entry.get("link", ""),
            author=entry.get("author", f"ABC News {self.region}"),
            published_at=published_at,
            metadata={
                "feed": feed_name,
                "region": self.region,
            },
            user_id=self.user_id,
        )

    async def get_user_feeds(self) -> List[str]:
        """Get configured ABC News feeds."""
        feeds = self.config.settings.get("feeds", list(self.feeds_map.keys())[:3])
        return [f"ABC News {self.region} - {feed.title()}" for feed in feeds]

