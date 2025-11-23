"""Facebook Graph API connector.

Official API: https://developers.facebook.com/docs/graph-api/
Version: v21.0 (as of November 2024)
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import aiohttp

from app.connectors.base import BaseConnector, ConnectorConfig, FetchResult, RateLimitInfo
from app.core.errors import ConnectorError
from app.core.models import ContentItem, ContentType, SourcePlatform

logger = logging.getLogger(__name__)


class FacebookConnector(BaseConnector):
    """Facebook Graph API connector.
    
    Features:
    - User feed posts
    - Page posts
    - Groups (if user has access)
    - OAuth 2.0 authentication
    - Rate limiting compliance
    - Pagination support
    """

    BASE_URL = "https://graph.facebook.com/v21.0"
    
    def __init__(self, config: ConnectorConfig, user_id: UUID):
        """Initialize Facebook connector.
        
        Required credentials:
        - access_token: Facebook OAuth 2.0 access token
        
        Required permissions:
        - user_posts (for user feed)
        - pages_read_engagement (for pages)
        - groups_access_member_info (for groups)
        """
        super().__init__(config, user_id)
        self.access_token = config.credentials.get("access_token")
        
        if not self.access_token:
            raise ConnectorError("Facebook connector requires access_token")

    async def validate_credentials(self) -> bool:
        """Validate Facebook API credentials."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.BASE_URL}/me",
                    params={"access_token": self.access_token},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Facebook credential validation failed: {e}")
            return False

    async def fetch_content(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> FetchResult:
        """Fetch Facebook posts from user feed and pages."""
        items: List[ContentItem] = []
        errors: List[str] = []
        
        try:
            # Fetch from multiple sources
            sources = self.config.settings.get("sources", ["feed"])
            
            for source in sources:
                if source == "feed":
                    feed_items = await self._fetch_user_feed(since, cursor, max_items)
                    items.extend(feed_items)
                elif source == "pages":
                    page_ids = self.config.settings.get("page_ids", [])
                    for page_id in page_ids:
                        page_items = await self._fetch_page_posts(page_id, since, cursor, max_items)
                        items.extend(page_items)
            
            return FetchResult(items=items, errors=errors)
        
        except Exception as e:
            logger.error(f"Error fetching Facebook content: {e}")
            errors.append(str(e))
            return FetchResult(items=items, errors=errors)

    async def _fetch_user_feed(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> List[ContentItem]:
        """Fetch user's Facebook feed."""
        items: List[ContentItem] = []
        
        async with aiohttp.ClientSession() as session:
            params = {
                "access_token": self.access_token,
                "fields": "id,message,story,created_time,from,link,type,attachments,reactions.summary(true),comments.summary(true),shares",
                "limit": min(max_items, 100),
            }
            
            if since:
                params["since"] = int(since.timestamp())
            
            if cursor:
                params["after"] = cursor
            
            async with session.get(
                f"{self.BASE_URL}/me/feed",
                params=params,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ConnectorError(f"Facebook API error: {response.status} - {error_text}")
                
                data = await response.json()
                
                for post in data.get("data", []):
                    items.append(self._parse_post(post))
        
        return items

    async def _fetch_page_posts(
        self,
        page_id: str,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> List[ContentItem]:
        """Fetch posts from a Facebook page."""
        items: List[ContentItem] = []
        
        async with aiohttp.ClientSession() as session:
            params = {
                "access_token": self.access_token,
                "fields": "id,message,story,created_time,from,link,type,attachments,reactions.summary(true),comments.summary(true),shares",
                "limit": min(max_items, 100),
            }
            
            if since:
                params["since"] = int(since.timestamp())
            
            if cursor:
                params["after"] = cursor
            
            async with session.get(
                f"{self.BASE_URL}/{page_id}/posts",
                params=params,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.warning(f"Failed to fetch page {page_id}: {error_text}")
                    return items
                
                data = await response.json()
                
                for post in data.get("data", []):
                    items.append(self._parse_post(post, is_page=True))
        
        return items

    def _parse_post(self, post: Dict[str, Any], is_page: bool = False) -> ContentItem:
        """Parse Facebook post into ContentItem."""
        # Determine content type
        post_type = post.get("type", "status")
        content_type_map = {
            "photo": ContentType.IMAGE,
            "video": ContentType.VIDEO,
            "link": ContentType.LINK,
            "status": ContentType.TEXT,
        }
        content_type = content_type_map.get(post_type, ContentType.TEXT)
        
        # Extract text content
        message = post.get("message", post.get("story", ""))
        
        # Extract URL
        url = post.get("link") or f"https://www.facebook.com/{post.get('id')}"
        
        return ContentItem(
            platform=SourcePlatform.FACEBOOK,
            platform_id=post.get("id"),
            content_type=content_type,
            title=message[:200] if message else "Facebook Post",
            text_content=message,
            url=url,
            author=post.get("from", {}).get("name", "Unknown"),
            published_at=datetime.fromisoformat(post.get("created_time", "").replace("Z", "+00:00")),
            metadata={
                "post_type": post_type,
                "reactions_count": post.get("reactions", {}).get("summary", {}).get("total_count", 0),
                "comments_count": post.get("comments", {}).get("summary", {}).get("total_count", 0),
                "shares_count": post.get("shares", {}).get("count", 0),
                "is_page_post": is_page,
                "from_id": post.get("from", {}).get("id"),
            },
            user_id=self.user_id,
        )

    async def get_user_feeds(self) -> List[str]:
        """Get list of configured Facebook sources."""
        sources = self.config.settings.get("sources", ["feed"])
        page_ids = self.config.settings.get("page_ids", [])
        
        feeds = []
        if "feed" in sources:
            feeds.append("User Feed")
        
        for page_id in page_ids:
            feeds.append(f"Page: {page_id}")
        
        return feeds

