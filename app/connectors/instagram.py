"""Instagram Graph API connector.

Official API: https://developers.facebook.com/docs/instagram-api/
Note: Instagram Basic Display API deprecated Dec 2024, use Instagram Graph API
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import aiohttp

from app.connectors.auth import ConnectorAuthError, is_token_expired, safe_error_str
from app.connectors.base import BaseConnector, ConnectorConfig, FetchResult, RateLimitInfo
from app.core.errors import ConnectorError
from app.core.models import ContentItem, ContentType, SourcePlatform

logger = logging.getLogger(__name__)


class InstagramConnector(BaseConnector):
    """Instagram Graph API connector.
    
    Features:
    - User media (posts, reels, stories)
    - Hashtag search
    - Media insights (likes, comments, engagement)
    - OAuth 2.0 authentication
    - Rate limiting compliance
    
    Requirements:
    - Instagram Business or Creator account
    - Facebook Page connected to Instagram account
    """

    BASE_URL = "https://graph.facebook.com/v21.0"
    
    def __init__(self, config: ConnectorConfig, user_id: UUID):
        """Initialize Instagram connector.
        
        Required credentials:
        - access_token: Facebook/Instagram OAuth 2.0 access token
        - instagram_business_account_id: Instagram Business Account ID
        
        Required permissions:
        - instagram_basic
        - instagram_manage_insights
        - pages_read_engagement
        """
        super().__init__(config, user_id)
        self.access_token = config.credentials.get("access_token")
        self.ig_account_id = config.credentials.get("instagram_business_account_id")
        
        if not all([self.access_token, self.ig_account_id]):
            raise ConnectorError(
                "Instagram connector requires access_token and instagram_business_account_id"
            )

    async def validate_credentials(self) -> bool:
        """Validate Instagram API credentials."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.BASE_URL}/{self.ig_account_id}",
                    params={
                        "access_token": self.access_token,
                        "fields": "id,username",
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error("Instagram credential validation failed: %s", safe_error_str(e))
            return False

    async def fetch_content(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> FetchResult:
        """Fetch Instagram media."""
        # ── Pre-call token expiry check ──────────────────────────────────────
        if is_token_expired(self.config.credentials):
            raise ConnectorAuthError(
                "Instagram access token is expired or expires within the safety buffer",
                platform=SourcePlatform.INSTAGRAM.value,
                user_id=str(self.user_id),
                http_status=None,
            )
        # ─────────────────────────────────────────────────────────────────────

        items: List[ContentItem] = []
        errors: List[str] = []

        try:
            # Fetch user media
            media_items = await self._fetch_user_media(since, cursor, max_items)
            items.extend(media_items)
            
            # Optionally fetch hashtag media
            hashtags = self.config.settings.get("hashtags", [])
            if hashtags:
                for hashtag in hashtags[:5]:  # Limit to 5 hashtags to avoid rate limits
                    hashtag_items = await self._fetch_hashtag_media(hashtag, max_items=20)
                    items.extend(hashtag_items)
            
            return FetchResult(items=items, errors=errors)
        
        except ConnectorAuthError:
            raise
        except Exception as e:
            logger.error("Error fetching Instagram content: %s", safe_error_str(e))
            errors.append(safe_error_str(e))
            return FetchResult(items=items, errors=errors)

    async def _fetch_user_media(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> List[ContentItem]:
        """Fetch user's Instagram media."""
        items: List[ContentItem] = []
        
        async with aiohttp.ClientSession() as session:
            params = {
                "access_token": self.access_token,
                "fields": "id,caption,media_type,media_url,permalink,thumbnail_url,timestamp,username,like_count,comments_count",
                "limit": min(max_items, 100),
            }
            
            if since:
                params["since"] = int(since.timestamp())
            
            if cursor:
                params["after"] = cursor
            
            async with session.get(
                f"{self.BASE_URL}/{self.ig_account_id}/media",
                params=params,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status == 401:
                    raise ConnectorAuthError(
                        "Instagram API returned HTTP 401 — access token expired or revoked",
                        platform=SourcePlatform.INSTAGRAM.value,
                        user_id=str(self.user_id),
                        http_status=401,
                    )
                if response.status != 200:
                    error_text = await response.text()
                    raise ConnectorError(
                        f"Instagram API error: {response.status} - {error_text}"
                    )
                
                data = await response.json()
                
                for media in data.get("data", []):
                    items.append(self._parse_media(media))
        
        return items

    async def _fetch_hashtag_media(
        self,
        hashtag: str,
        max_items: int = 20,
    ) -> List[ContentItem]:
        """Fetch recent media for a hashtag."""
        items: List[ContentItem] = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # First, get hashtag ID
                async with session.get(
                    f"{self.BASE_URL}/ig_hashtag_search",
                    params={
                        "access_token": self.access_token,
                        "user_id": self.ig_account_id,
                        "q": hashtag.lstrip("#"),
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to search hashtag {hashtag}")
                        return items
                    
                    hashtag_data = await response.json()
                    hashtag_id = hashtag_data.get("data", [{}])[0].get("id")
                    
                    if not hashtag_id:
                        return items
                
                # Get recent media for hashtag
                async with session.get(
                    f"{self.BASE_URL}/{hashtag_id}/recent_media",
                    params={
                        "access_token": self.access_token,
                        "user_id": self.ig_account_id,
                        "fields": "id,caption,media_type,media_url,permalink,thumbnail_url,timestamp,username,like_count,comments_count",
                        "limit": min(max_items, 50),
                    },
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch media for hashtag {hashtag}")
                        return items
                    
                    data = await response.json()
                    
                    for media in data.get("data", []):
                        item = self._parse_media(media)
                        item.metadata["source_hashtag"] = hashtag
                        items.append(item)
        
        except Exception as e:
            logger.warning(f"Error fetching hashtag {hashtag}: {e}")
        
        return items

    def _parse_media(self, media: Dict[str, Any]) -> ContentItem:
        """Parse Instagram media into ContentItem."""
        media_type = media.get("media_type", "IMAGE")
        content_type_map = {
            "IMAGE": ContentType.IMAGE,
            "VIDEO": ContentType.VIDEO,
            "CAROUSEL_ALBUM": ContentType.IMAGE,
        }
        content_type = content_type_map.get(media_type, ContentType.IMAGE)
        
        caption = media.get("caption", "")
        
        return ContentItem(
            platform=SourcePlatform.INSTAGRAM,
            platform_id=media.get("id"),
            content_type=content_type,
            title=caption[:200] if caption else "Instagram Post",
            text_content=caption,
            url=media.get("permalink", ""),
            media_url=media.get("media_url") or media.get("thumbnail_url"),
            author=media.get("username", "Unknown"),
            published_at=datetime.fromisoformat(media.get("timestamp", "").replace("Z", "+00:00")),
            metadata={
                "media_type": media_type,
                "like_count": media.get("like_count", 0),
                "comments_count": media.get("comments_count", 0),
                "thumbnail_url": media.get("thumbnail_url"),
            },
            user_id=self.user_id,
        )

    async def get_user_feeds(self) -> List[str]:
        """Get list of configured Instagram sources."""
        feeds = [f"Instagram Account: {self.ig_account_id}"]
        
        hashtags = self.config.settings.get("hashtags", [])
        for hashtag in hashtags:
            feeds.append(f"Hashtag: {hashtag}")
        
        return feeds

