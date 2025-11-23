"""WeChat Official Account API connector.

Official API: https://developers.weixin.qq.com/doc/offiaccount/en/Getting_Started/Overview.html
Note: Requires WeChat Official Account and API access approval
"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import aiohttp

from app.connectors.base import BaseConnector, ConnectorConfig, FetchResult
from app.core.errors import ConnectorError
from app.core.models import ContentItem, ContentType, SourcePlatform

logger = logging.getLogger(__name__)


class WeChatConnector(BaseConnector):
    """WeChat Official Account API connector.
    
    Features:
    - Fetch articles from subscribed official accounts
    - User message history
    - OAuth 2.0 authentication
    - Access token management
    
    Requirements:
    - WeChat Official Account (Service Account or Subscription Account)
    - API access approval from WeChat
    """

    BASE_URL = "https://api.weixin.qq.com/cgi-bin"
    
    def __init__(self, config: ConnectorConfig, user_id: UUID):
        """Initialize WeChat connector.
        
        Required credentials:
        - app_id: WeChat Official Account App ID
        - app_secret: WeChat Official Account App Secret
        - access_token: (optional) Pre-obtained access token
        """
        super().__init__(config, user_id)
        self.app_id = config.credentials.get("app_id")
        self.app_secret = config.credentials.get("app_secret")
        self.access_token = config.credentials.get("access_token")
        
        if not all([self.app_id, self.app_secret]):
            raise ConnectorError("WeChat connector requires app_id and app_secret")

    async def validate_credentials(self) -> bool:
        """Validate WeChat API credentials."""
        try:
            # Get or refresh access token
            if not self.access_token:
                await self._get_access_token()
            
            # Test token with a simple API call
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.BASE_URL}/get_api_domain_ip",
                    params={"access_token": self.access_token},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    data = await response.json()
                    return data.get("errcode", -1) == 0
        except Exception as e:
            logger.error(f"WeChat credential validation failed: {e}")
            return False

    async def _get_access_token(self) -> str:
        """Get WeChat API access token."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.BASE_URL}/token",
                params={
                    "grant_type": "client_credential",
                    "appid": self.app_id,
                    "secret": self.app_secret,
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                data = await response.json()
                
                if "access_token" not in data:
                    raise ConnectorError(f"Failed to get WeChat access token: {data}")
                
                self.access_token = data["access_token"]
                return self.access_token

    async def fetch_content(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> FetchResult:
        """Fetch WeChat articles."""
        items: List[ContentItem] = []
        errors: List[str] = []
        
        try:
            # Ensure we have a valid access token
            if not self.access_token:
                await self._get_access_token()
            
            # Fetch material list (published articles)
            offset = int(cursor) if cursor else 0
            count = min(max_items, 20)  # API limit
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.BASE_URL}/material/batchget_material",
                    params={"access_token": self.access_token},
                    json={
                        "type": "news",  # Article type
                        "offset": offset,
                        "count": count,
                    },
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    data = await response.json()
                    
                    if data.get("errcode", 0) != 0:
                        raise ConnectorError(f"WeChat API error: {data.get('errmsg')}")
                    
                    # Parse articles
                    for item_data in data.get("item", []):
                        content = item_data.get("content", {})
                        news_items = content.get("news_item", [])
                        
                        for news in news_items:
                            items.append(self._parse_article(news, item_data.get("update_time", 0)))
                    
                    # Determine if there are more items
                    total_count = data.get("total_count", 0)
                    next_cursor = str(offset + count) if offset + count < total_count else None
                    
                    return FetchResult(
                        items=items,
                        cursor=next_cursor,
                        errors=errors,
                    )
        
        except Exception as e:
            logger.error(f"Error fetching WeChat content: {e}")
            errors.append(str(e))
            return FetchResult(items=items, errors=errors)

    def _parse_article(self, news: Dict[str, Any], update_time: int) -> ContentItem:
        """Parse WeChat article into ContentItem."""
        return ContentItem(
            platform=SourcePlatform.WECHAT,
            platform_id=news.get("url", ""),
            content_type=ContentType.ARTICLE,
            title=news.get("title", ""),
            text_content=news.get("digest", ""),  # Summary/digest
            url=news.get("url", ""),
            media_url=news.get("thumb_url"),  # Cover image
            author=news.get("author", "Unknown"),
            published_at=datetime.fromtimestamp(update_time),
            metadata={
                "content_source_url": news.get("content_source_url"),  # Original article URL
                "show_cover_pic": news.get("show_cover_pic", 0),
            },
            user_id=self.user_id,
        )

    async def get_user_feeds(self) -> List[str]:
        """Get configured WeChat sources."""
        return [f"WeChat Official Account: {self.app_id}"]

