"""Wall Street Journal RSS connector.

WSJ provides RSS feeds for various sections.
RSS Feeds: https://www.wsj.com/news/rss-news-and-feeds
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


class WSJConnector(BaseConnector):
    """Wall Street Journal RSS connector.
    
    Available feeds:
    - Opinion
    - World News
    - U.S. Business
    - Markets
    - Technology
    - Lifestyle
    """

    RSS_FEEDS = {
        "opinion": "https://feeds.a.dj.com/rss/RSSOpinion.xml",
        "world": "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
        "business": "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",
        "markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "technology": "https://feeds.a.dj.com/rss/RSSWSJD.xml",
        "lifestyle": "https://feeds.a.dj.com/rss/RSSLifestyle.xml",
        "real_estate": "https://feeds.a.dj.com/rss/RSSRealEstate.xml",
    }
    
    def __init__(self, config: ConnectorConfig, user_id: UUID):
        """Initialize WSJ connector.
        
        Settings:
        - feeds: List of feed names (e.g., ['opinion', 'technology', 'markets'])
        """
        super().__init__(config, user_id)

    async def validate_credentials(self) -> bool:
        """Validate WSJ RSS access."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.RSS_FEEDS["world"],
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"WSJ validation failed: {e}")
            return False

    async def fetch_content(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> FetchResult:
        """Fetch WSJ articles from RSS feeds."""
        items: List[ContentItem] = []
        errors: List[str] = []
        
        try:
            feeds = self.config.settings.get("feeds", ["world", "business", "technology"])
            
            for feed_name in feeds:
                if feed_name not in self.RSS_FEEDS:
                    logger.warning(f"Unknown WSJ feed: {feed_name}")
                    continue
                
                feed_items = await self._fetch_feed(feed_name, max_items=30)
                items.extend(feed_items)
            
            # Filter by date if specified
            if since:
                items = [item for item in items if item.published_at >= since]
            
            return FetchResult(items=items[:max_items], errors=errors)
        
        except Exception as e:
            logger.error(f"Error fetching WSJ content: {e}")
            errors.append(str(e))
            return FetchResult(items=items, errors=errors)

    async def _fetch_feed(self, feed_name: str, max_items: int = 30) -> List[ContentItem]:
        """Fetch articles from a specific RSS feed."""
        items: List[ContentItem] = []
        url = self.RSS_FEEDS[feed_name]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch WSJ feed {feed_name}")
                        return items
                    
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    for entry in feed.entries[:max_items]:
                        items.append(self._parse_entry(entry, feed_name))
        
        except Exception as e:
            logger.warning(f"Error fetching WSJ feed {feed_name}: {e}")
        
        return items

    def _parse_entry(self, entry: Any, feed_name: str) -> ContentItem:
        """Parse RSS entry into ContentItem."""
        # Extract published date
        published_at = datetime.now()
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            from time import mktime
            published_at = datetime.fromtimestamp(mktime(entry.published_parsed))
        
        return ContentItem(
            platform=SourcePlatform.WSJ,
            platform_id=entry.get("id", entry.get("link", "")),
            content_type=ContentType.ARTICLE,
            title=entry.get("title", ""),
            text_content=entry.get("summary", ""),
            url=entry.get("link", ""),
            author=entry.get("author", "WSJ"),
            published_at=published_at,
            metadata={
                "feed": feed_name,
                "categories": [tag.term for tag in entry.get("tags", [])],
            },
            user_id=self.user_id,
        )

    async def get_user_feeds(self) -> List[str]:
        """Get configured WSJ feeds."""
        feeds = self.config.settings.get("feeds", ["world", "business", "technology"])
        return [f"WSJ {feed.title()}" for feed in feeds]

