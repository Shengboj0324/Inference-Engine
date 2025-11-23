"""TikTok Research API connector.

Official API: https://developers.tiktok.com/products/research-api/
Requires: Academic/Research institution approval
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

import aiohttp

from app.connectors.base import BaseConnector, ConnectorConfig, FetchResult, RateLimitInfo
from app.core.errors import ConnectorError
from app.core.models import ContentItem, ContentType, SourcePlatform

logger = logging.getLogger(__name__)


class TikTokConnector(BaseConnector):
    """TikTok Research API connector.
    
    Features:
    - Video search by keywords, hashtags
    - User video retrieval
    - Video metadata (views, likes, shares, comments)
    - OAuth 2.0 authentication
    - Rate limiting compliance
    """

    BASE_URL = "https://open.tiktokapis.com/v2"
    
    def __init__(self, config: ConnectorConfig, user_id: UUID):
        """Initialize TikTok connector.
        
        Required credentials:
        - client_key: TikTok app client key
        - client_secret: TikTok app client secret
        - access_token: OAuth 2.0 access token
        """
        super().__init__(config, user_id)
        self.client_key = config.credentials.get("client_key")
        self.client_secret = config.credentials.get("client_secret")
        self.access_token = config.credentials.get("access_token")
        
        if not all([self.client_key, self.client_secret, self.access_token]):
            raise ConnectorError(
                "TikTok connector requires client_key, client_secret, and access_token"
            )

    async def validate_credentials(self) -> bool:
        """Validate TikTok API credentials."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json",
                }
                
                # Test with a simple query
                async with session.post(
                    f"{self.BASE_URL}/research/video/query/",
                    headers=headers,
                    json={
                        "query": {
                            "and": [
                                {"field_name": "region_code", "operation": "IN", "field_values": ["US"]}
                            ]
                        },
                        "max_count": 1,
                        "start_date": (datetime.now() - timedelta(days=1)).strftime("%Y%m%d"),
                        "end_date": datetime.now().strftime("%Y%m%d"),
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"TikTok credential validation failed: {e}")
            return False

    async def fetch_content(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> FetchResult:
        """Fetch TikTok videos based on configured search criteria."""
        items: List[ContentItem] = []
        errors: List[str] = []
        
        try:
            # Get search queries from settings
            search_queries = self.config.settings.get("search_queries", [])
            hashtags = self.config.settings.get("hashtags", [])
            
            if not search_queries and not hashtags:
                logger.warning("No search queries or hashtags configured for TikTok")
                return FetchResult(items=[], errors=["No search criteria configured"])
            
            # Build query
            query_conditions = []
            
            # Add keyword search
            if search_queries:
                for query in search_queries:
                    query_conditions.append({
                        "field_name": "keyword",
                        "operation": "IN",
                        "field_values": [query]
                    })
            
            # Add hashtag search
            if hashtags:
                query_conditions.append({
                    "field_name": "hashtag_name",
                    "operation": "IN",
                    "field_values": hashtags
                })
            
            # Date range
            start_date = since or (datetime.now() - timedelta(days=7))
            end_date = datetime.now()
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json",
                }
                
                payload = {
                    "query": {"or": query_conditions} if len(query_conditions) > 1 else {"and": query_conditions},
                    "start_date": start_date.strftime("%Y%m%d"),
                    "end_date": end_date.strftime("%Y%m%d"),
                    "max_count": min(max_items, 100),  # API limit
                }
                
                if cursor:
                    payload["cursor"] = cursor
                
                async with session.post(
                    f"{self.BASE_URL}/research/video/query/",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ConnectorError(f"TikTok API error: {response.status} - {error_text}")
                    
                    data = await response.json()
                    
                    # Parse videos
                    for video in data.get("data", {}).get("videos", []):
                        items.append(self._parse_video(video))
                    
                    # Get next cursor
                    next_cursor = data.get("data", {}).get("cursor")
                    has_more = data.get("data", {}).get("has_more", False)
                    
                    return FetchResult(
                        items=items,
                        cursor=next_cursor if has_more else None,
                        errors=errors,
                    )
        
        except Exception as e:
            logger.error(f"Error fetching TikTok content: {e}")
            errors.append(str(e))
            return FetchResult(items=items, errors=errors)

    def _parse_video(self, video: Dict[str, Any]) -> ContentItem:
        """Parse TikTok video data into ContentItem."""
        return ContentItem(
            platform=SourcePlatform.TIKTOK,
            platform_id=video.get("id"),
            content_type=ContentType.VIDEO,
            title=video.get("video_description", "")[:200],  # Use description as title
            text_content=video.get("video_description", ""),
            url=f"https://www.tiktok.com/@{video.get('username', 'unknown')}/video/{video.get('id')}",
            author=video.get("username"),
            published_at=datetime.fromtimestamp(video.get("create_time", 0)),
            metadata={
                "view_count": video.get("view_count", 0),
                "like_count": video.get("like_count", 0),
                "comment_count": video.get("comment_count", 0),
                "share_count": video.get("share_count", 0),
                "duration": video.get("duration", 0),
                "hashtags": video.get("hashtag_names", []),
                "music_id": video.get("music_id"),
                "effect_ids": video.get("effect_ids", []),
                "region_code": video.get("region_code"),
            },
            user_id=self.user_id,
        )

    async def get_user_feeds(self) -> List[str]:
        """Get configured search queries and hashtags."""
        queries = self.config.settings.get("search_queries", [])
        hashtags = self.config.settings.get("hashtags", [])
        return queries + [f"#{tag}" for tag in hashtags]

