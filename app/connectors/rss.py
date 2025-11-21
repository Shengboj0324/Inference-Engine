"""RSS feed connector."""

from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

import feedparser
import httpx

from app.connectors.base import BaseConnector, ConnectorConfig, FetchResult, PlatformError
from app.core.models import ContentItem, MediaType


class RSSConnector(BaseConnector):
    """Connector for RSS/Atom feeds."""

    def __init__(self, config: ConnectorConfig, user_id: UUID):
        """Initialize RSS connector.

        Expected settings:
            - feed_urls: List of RSS feed URLs to monitor
        """
        super().__init__(config, user_id)
        self.feed_urls = config.settings.get("feed_urls", [])

    async def validate_credentials(self) -> bool:
        """RSS doesn't require credentials, validate feed URLs instead."""
        if not self.feed_urls:
            return False

        # Try to fetch first feed
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.feed_urls[0], timeout=10.0)
                return response.status_code == 200
        except Exception:
            return False

    async def get_user_feeds(self) -> List[str]:
        """Get list of configured RSS feeds."""
        return self.feed_urls

    async def fetch_content(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> FetchResult:
        """Fetch items from all configured RSS feeds."""
        items: List[ContentItem] = []
        errors: List[str] = []

        for feed_url in self.feed_urls:
            try:
                feed_items = await self._fetch_from_feed(feed_url, since, max_items)
                items.extend(feed_items)
            except Exception as e:
                errors.append(f"Error fetching from {feed_url}: {e}")

        return FetchResult(items=items, errors=errors)

    async def _fetch_from_feed(
        self,
        feed_url: str,
        since: Optional[datetime],
        max_items: int,
    ) -> List[ContentItem]:
        """Fetch items from a specific RSS feed."""
        items: List[ContentItem] = []

        try:
            # Fetch feed content
            async with httpx.AsyncClient() as client:
                response = await client.get(feed_url, timeout=30.0)
                response.raise_for_status()

            # Parse feed
            feed = feedparser.parse(response.text)

            if feed.bozo:
                raise PlatformError(f"Invalid feed format: {feed_url}")

            feed_title = feed.feed.get("title", feed_url)

            # Process entries
            for entry in feed.entries[:max_items]:
                # Parse published date
                published_at = self._parse_date(entry)

                # Filter by date if provided
                if since and published_at and published_at <= since:
                    continue

                # Determine media type
                media_type = MediaType.TEXT
                media_urls = []

                # Check for enclosures (podcasts, videos)
                if hasattr(entry, "enclosures") and entry.enclosures:
                    enclosure = entry.enclosures[0]
                    media_urls.append(enclosure.get("href", ""))
                    if "video" in enclosure.get("type", ""):
                        media_type = MediaType.VIDEO
                    elif "image" in enclosure.get("type", ""):
                        media_type = MediaType.IMAGE

                # Get content
                content = ""
                if hasattr(entry, "content"):
                    content = entry.content[0].value
                elif hasattr(entry, "summary"):
                    content = entry.summary
                elif hasattr(entry, "description"):
                    content = entry.description

                # Create content item
                item = self._create_content_item(
                    source_id=entry.get("id", entry.get("link", "")),
                    source_url=entry.get("link", ""),
                    title=entry.get("title", "Untitled"),
                    raw_text=content,
                    author=entry.get("author", None),
                    channel=feed_title,
                    media_type=media_type,
                    media_urls=media_urls,
                    published_at=published_at or datetime.now(timezone.utc),
                    metadata={
                        "feed_url": feed_url,
                        "feed_title": feed_title,
                        "tags": [tag.term for tag in entry.get("tags", [])],
                    },
                )
                items.append(item)

        except httpx.HTTPError as e:
            raise PlatformError(f"HTTP error fetching feed {feed_url}: {e}")
        except Exception as e:
            raise PlatformError(f"Error parsing feed {feed_url}: {e}")

        return items

    def _parse_date(self, entry: feedparser.FeedParserDict) -> Optional[datetime]:
        """Parse date from RSS entry."""
        # Try different date fields
        for field in ["published_parsed", "updated_parsed", "created_parsed"]:
            if hasattr(entry, field):
                time_struct = getattr(entry, field)
                if time_struct:
                    return datetime(*time_struct[:6], tzinfo=timezone.utc)

        return None

