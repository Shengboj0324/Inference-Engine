"""Connector registry for platform-specific connector instantiation."""

from typing import Dict, Type
from uuid import UUID

from app.connectors.abc_news import ABCNewsConnector
from app.connectors.apple_news import AppleNewsConnector
from app.connectors.base import BaseConnector, ConnectorConfig
from app.connectors.facebook import FacebookConnector
from app.connectors.google_news import GoogleNewsConnector
from app.connectors.instagram import InstagramConnector
from app.connectors.nytimes import NYTimesConnector
from app.connectors.reddit import RedditConnector
from app.connectors.rss import RSSConnector
from app.connectors.tiktok import TikTokConnector
from app.connectors.wechat import WeChatConnector
from app.connectors.wsj import WSJConnector
from app.connectors.youtube import YouTubeConnector
from app.core.errors import ConnectorError
from app.core.models import SourcePlatform


class ConnectorRegistry:
    """Registry for platform-specific connectors."""

    _connectors: Dict[SourcePlatform, Type[BaseConnector]] = {
        # Social Media Platforms
        SourcePlatform.REDDIT: RedditConnector,
        SourcePlatform.YOUTUBE: YouTubeConnector,
        SourcePlatform.TIKTOK: TikTokConnector,
        SourcePlatform.FACEBOOK: FacebookConnector,
        SourcePlatform.INSTAGRAM: InstagramConnector,
        SourcePlatform.WECHAT: WeChatConnector,
        # News Sources
        SourcePlatform.RSS: RSSConnector,
        SourcePlatform.NYTIMES: NYTimesConnector,
        SourcePlatform.WSJ: WSJConnector,
        SourcePlatform.ABC_NEWS: ABCNewsConnector,
        SourcePlatform.ABC_NEWS_AU: ABCNewsConnector,
        SourcePlatform.GOOGLE_NEWS: GoogleNewsConnector,
        SourcePlatform.APPLE_NEWS: AppleNewsConnector,
    }

    @classmethod
    def get_connector(
        cls,
        platform: SourcePlatform,
        config: ConnectorConfig,
        user_id: UUID,
    ) -> BaseConnector:
        """Get connector instance for platform.
        
        Args:
            platform: Source platform
            config: Connector configuration
            user_id: User ID
            
        Returns:
            Connector instance
            
        Raises:
            ConnectorError: If platform not supported
        """
        connector_class = cls._connectors.get(platform)
        
        if not connector_class:
            raise ConnectorError(f"Unsupported platform: {platform}")
        
        return connector_class(config, user_id)

    @classmethod
    def get_supported_platforms(cls) -> list[SourcePlatform]:
        """Get list of supported platforms."""
        return list(cls._connectors.keys())

    @classmethod
    def is_platform_supported(cls, platform: SourcePlatform) -> bool:
        """Check if platform is supported."""
        return platform in cls._connectors

    @classmethod
    def register_connector(
        cls,
        platform: SourcePlatform,
        connector_class: Type[BaseConnector],
    ) -> None:
        """Register a custom connector.
        
        Args:
            platform: Source platform
            connector_class: Connector class
        """
        cls._connectors[platform] = connector_class

    @classmethod
    def get_platform_info(cls) -> Dict[str, Dict[str, any]]:
        """Get information about all supported platforms.
        
        Returns:
            Dictionary mapping platform names to their info
        """
        return {
            # Social Media
            "reddit": {
                "name": "Reddit",
                "platform": SourcePlatform.REDDIT,
                "type": "social_media",
                "requires_oauth": True,
                "api_docs": "https://www.reddit.com/dev/api/",
            },
            "youtube": {
                "name": "YouTube",
                "platform": SourcePlatform.YOUTUBE,
                "type": "social_media",
                "requires_oauth": True,
                "api_docs": "https://developers.google.com/youtube/v3",
            },
            "tiktok": {
                "name": "TikTok",
                "platform": SourcePlatform.TIKTOK,
                "type": "social_media",
                "requires_oauth": True,
                "api_docs": "https://developers.tiktok.com/doc/research-api-overview",
            },
            "facebook": {
                "name": "Facebook",
                "platform": SourcePlatform.FACEBOOK,
                "type": "social_media",
                "requires_oauth": True,
                "api_docs": "https://developers.facebook.com/docs/graph-api/",
            },
            "instagram": {
                "name": "Instagram",
                "platform": SourcePlatform.INSTAGRAM,
                "type": "social_media",
                "requires_oauth": True,
                "api_docs": "https://developers.facebook.com/docs/instagram-api/",
            },
            "wechat": {
                "name": "WeChat",
                "platform": SourcePlatform.WECHAT,
                "type": "social_media",
                "requires_oauth": True,
                "api_docs": "https://developers.weixin.qq.com/doc/offiaccount/en/",
            },
            # News Sources
            "nytimes": {
                "name": "New York Times",
                "platform": SourcePlatform.NYTIMES,
                "type": "news",
                "requires_oauth": False,
                "api_docs": "https://developer.nytimes.com/",
            },
            "wsj": {
                "name": "Wall Street Journal",
                "platform": SourcePlatform.WSJ,
                "type": "news",
                "requires_oauth": False,
                "api_docs": "https://www.wsj.com/news/rss-news-and-feeds",
            },
            "abc_news": {
                "name": "ABC News (US)",
                "platform": SourcePlatform.ABC_NEWS,
                "type": "news",
                "requires_oauth": False,
                "api_docs": None,
            },
            "abc_news_au": {
                "name": "ABC News (Australia)",
                "platform": SourcePlatform.ABC_NEWS_AU,
                "type": "news",
                "requires_oauth": False,
                "api_docs": "https://www.abc.net.au/news/feeds/",
            },
            "google_news": {
                "name": "Google News",
                "platform": SourcePlatform.GOOGLE_NEWS,
                "type": "news",
                "requires_oauth": False,
                "api_docs": None,
            },
            "apple_news": {
                "name": "Apple News",
                "platform": SourcePlatform.APPLE_NEWS,
                "type": "news",
                "requires_oauth": False,
                "api_docs": None,
            },
            "rss": {
                "name": "RSS Feed",
                "platform": SourcePlatform.RSS,
                "type": "generic",
                "requires_oauth": False,
                "api_docs": None,
            },
        }


# Singleton instance
connector_registry = ConnectorRegistry()

