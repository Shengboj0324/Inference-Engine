"""Platform connectors for content ingestion.

Imports are **lazy** — platform SDKs (praw, google-api-python-client, etc.) are
only pulled in when the specific connector is first accessed.  This prevents
optional SDK import errors from crashing the entire application.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from app.connectors.base import BaseConnector, ConnectorConfig, FetchResult
    from app.connectors.abc_news import ABCNewsConnector
    from app.connectors.apple_news import AppleNewsConnector
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

_MODULE_MAP: dict[str, str] = {
    "BaseConnector":     "app.connectors.base",
    "ConnectorConfig":   "app.connectors.base",
    "FetchResult":       "app.connectors.base",
    "ABCNewsConnector":  "app.connectors.abc_news",
    "AppleNewsConnector":"app.connectors.apple_news",
    "FacebookConnector": "app.connectors.facebook",
    "GoogleNewsConnector":"app.connectors.google_news",
    "InstagramConnector":"app.connectors.instagram",
    "NYTimesConnector":  "app.connectors.nytimes",
    "RedditConnector":   "app.connectors.reddit",
    "RSSConnector":      "app.connectors.rss",
    "TikTokConnector":   "app.connectors.tiktok",
    "WeChatConnector":   "app.connectors.wechat",
    "WSJConnector":      "app.connectors.wsj",
    "YouTubeConnector":  "app.connectors.youtube",
}


def __getattr__(name: str):  # noqa: ANN001, ANN201
    if name in _MODULE_MAP:
        module = importlib.import_module(_MODULE_MAP[name])
        obj = getattr(module, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'app.connectors' has no attribute {name!r}")
