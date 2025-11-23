"""Platform connectors for content ingestion."""

from app.connectors.abc_news import ABCNewsConnector
from app.connectors.apple_news import AppleNewsConnector
from app.connectors.base import BaseConnector, ConnectorConfig, FetchResult
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

__all__ = [
    "BaseConnector",
    "ConnectorConfig",
    "FetchResult",
    "RedditConnector",
    "YouTubeConnector",
    "TikTokConnector",
    "FacebookConnector",
    "InstagramConnector",
    "WeChatConnector",
    "RSSConnector",
    "NYTimesConnector",
    "WSJConnector",
    "ABCNewsConnector",
    "GoogleNewsConnector",
    "AppleNewsConnector",
]
