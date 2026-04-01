"""Podcast RSS connector.

Specialised RSS/Atom feed connector for podcast feeds.  In addition to the
standard item fields consumed by ``RSSConnector``, this connector also parses
iTunes/podcast-specific tags:

- ``<itunes:duration>``   → metadata[``duration_seconds``]
- ``<itunes:author>``     → metadata[``podcast_author``]
- ``<enclosure url="…">`` → metadata[``audio_url``] (MP3/AAC)
- ``<itunes:episode>``    → metadata[``episode_number``]
- ``<itunes:season>``     → metadata[``season_number``]

Each episode becomes one ``ContentItem`` with ``MediaType.AUDIO``.  The
``raw_text`` field contains the episode description/show-notes (which are
surprisingly information-dense for AI podcasts).  A separate
``TranscriptFeedConnector`` or ``WhisperTranscriber`` (Phase 2) will convert
``audio_url`` to a full transcript.

Configuration (``ConnectorConfig.settings``)::

    feed_urls: List[str]         # Podcast RSS feed URLs
    min_duration_seconds: int    # skip episodes shorter than this; default 0
    include_show_notes: bool     # include HTML show notes; default True
"""

import logging
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import feedparser
import httpx

from app.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    FetchResult,
    PlatformError,
    RateLimitError,
)
from app.core.models import ContentItem, MediaType

logger = logging.getLogger(__name__)


class PodcastRSSConnector(BaseConnector):
    """Fetches podcast episode metadata + show notes from RSS/Atom feeds."""

    def __init__(self, config: ConnectorConfig, user_id: UUID) -> None:
        super().__init__(config, user_id)
        s: Dict[str, Any] = config.settings or {}
        self._feed_urls: List[str] = s.get("feed_urls", [])
        self._min_duration_s: int = int(s.get("min_duration_seconds", 0))
        self._include_show_notes: bool = bool(s.get("include_show_notes", True))
        if not isinstance(self._feed_urls, list):
            raise TypeError(f"'feed_urls' must be a list, got {type(self._feed_urls)!r}")

    async def validate_credentials(self) -> bool:
        """Podcast RSS feeds are public; test first URL reachability."""
        if not self._feed_urls:
            return False
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(self._feed_urls[0])
            return resp.status_code == 200
        except Exception:
            return False

    async def get_user_feeds(self) -> List[str]:
        return list(self._feed_urls)

    async def fetch_content(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> FetchResult:
        if not self._feed_urls:
            raise ValueError("PodcastRSSConnector: 'feed_urls' setting is empty")

        t0 = time.perf_counter()
        items: List[ContentItem] = []
        errors: List[str] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for feed_url in self._feed_urls:
                if len(items) >= max_items:
                    break
                try:
                    feed_items = await self._fetch_feed(client, feed_url, since, max_items - len(items))
                    items.extend(feed_items)
                except RateLimitError:
                    raise
                except Exception as exc:
                    errors.append(f"{feed_url}: {exc}")
                    logger.warning("PodcastRSSConnector: error fetching %s: %s", feed_url, exc)

        logger.info(
            "PodcastRSSConnector.fetch_content: feeds=%d items=%d latency_ms=%.1f",
            len(self._feed_urls), len(items), (time.perf_counter() - t0) * 1000,
        )
        return FetchResult(items=items, errors=errors)

    async def _fetch_feed(
        self,
        client: httpx.AsyncClient,
        feed_url: str,
        since: Optional[datetime],
        remaining: int,
    ) -> List[ContentItem]:
        resp = await client.get(feed_url, follow_redirects=True)
        if resp.status_code == 429:
            raise RateLimitError(f"Podcast feed rate-limited: {feed_url}")
        if resp.status_code != 200:
            raise PlatformError(f"HTTP {resp.status_code} fetching {feed_url}")

        feed = feedparser.parse(resp.text)
        podcast_title = feed.feed.get("title", feed_url)
        items: List[ContentItem] = []

        for entry in feed.entries[:remaining]:
            pub_at = self._parse_date(entry)
            if since and pub_at <= since.replace(tzinfo=timezone.utc):
                continue

            audio_url, duration_s = self._parse_enclosure(entry)
            if duration_s < self._min_duration_s:
                continue

            show_notes = ""
            if self._include_show_notes:
                show_notes = entry.get("summary") or entry.get("description") or ""
                show_notes = re.sub(r"<[^>]+>", " ", show_notes).strip()  # strip HTML

            items.append(self._create_content_item(
                source_id=entry.get("id") or entry.get("link", ""),
                source_url=entry.get("link", feed_url),
                title=f"[Podcast] {podcast_title}: {entry.get('title', '')}",
                raw_text=show_notes,
                media_type=MediaType.AUDIO,
                published_at=pub_at,
                metadata={
                    "podcast_title": podcast_title,
                    "feed_url": feed_url,
                    "audio_url": audio_url,
                    "duration_seconds": duration_s,
                    "podcast_author": entry.get("itunes_author") or feed.feed.get("itunes_author", ""),
                    "episode_number": entry.get("itunes_episode"),
                    "season_number": entry.get("itunes_season"),
                    "transcript_pending": bool(audio_url),
                },
            ))
        return items

    @staticmethod
    def _parse_date(entry: Any) -> datetime:
        for field in ("published_parsed", "updated_parsed"):
            val = entry.get(field)
            if val:
                import calendar
                return datetime.utcfromtimestamp(calendar.timegm(val)).replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc)

    @staticmethod
    def _parse_enclosure(entry: Any) -> tuple:
        """Return (audio_url, duration_seconds)."""
        audio_url = ""
        duration_s = 0
        for enc in entry.get("enclosures", []):
            mime = enc.get("type", "")
            if "audio" in mime or enc.get("href", "").endswith((".mp3", ".m4a", ".ogg")):
                audio_url = enc.get("href", "") or enc.get("url", "")
                break
        raw_dur = entry.get("itunes_duration", "0")
        if isinstance(raw_dur, str) and ":" in raw_dur:
            parts = raw_dur.split(":")
            try:
                multipliers = [1, 60, 3600]
                duration_s = sum(int(p) * m for p, m in zip(reversed(parts), multipliers))
            except ValueError:
                duration_s = 0
        else:
            try:
                duration_s = int(raw_dur)
            except (ValueError, TypeError):
                duration_s = 0
        return audio_url, duration_s

