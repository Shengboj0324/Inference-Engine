"""Transcript Feeds connector.

Monitors RSS/JSON feeds that provide **pre-built text transcripts** alongside
their audio episodes.  Examples include Lex Fridman's transcript feed, Dwarkesh
Patel's Substack, and any custom transcript RSS feed.

If a feed item contains a ``transcript_url`` in its extensions, the connector
fetches the transcript text directly.  Otherwise ``raw_text`` contains the
episode description as a fallback.

This connector complements ``PodcastRSSConnector`` by handling sources that
already provide text, avoiding the need for Whisper transcription (Phase 2).

Each episode becomes one ``ContentItem`` with ``MediaType.TEXT`` (since the
primary value is the transcript text, not the audio itself).

Configuration (``ConnectorConfig.settings``)::

    feed_urls: List[str]           # RSS/Atom feeds with transcript extensions
    fetch_transcript_text: bool    # actually HTTP-GET transcript URLs; default True
    transcript_max_chars: int      # truncate transcript; default 50 000 chars
    transcript_url_field: str      # RSS extension field name; default "transcript"
"""

import logging
import re
import time
from datetime import datetime, timezone
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

_DEFAULT_TRANSCRIPT_FIELD = "transcript"


class TranscriptFeedConnector(BaseConnector):
    """Fetches episodes with pre-built transcripts from RSS feeds."""

    def __init__(self, config: ConnectorConfig, user_id: UUID) -> None:
        super().__init__(config, user_id)
        s: Dict[str, Any] = config.settings or {}
        self._feed_urls: List[str] = s.get("feed_urls", [])
        self._fetch_text: bool = bool(s.get("fetch_transcript_text", True))
        self._max_chars: int = int(s.get("transcript_max_chars", 50_000))
        self._transcript_field: str = s.get("transcript_url_field", _DEFAULT_TRANSCRIPT_FIELD)
        if not isinstance(self._feed_urls, list):
            raise TypeError(f"'feed_urls' must be a list, got {type(self._feed_urls)!r}")
        if self._max_chars <= 0:
            raise ValueError(f"'transcript_max_chars' must be positive, got {self._max_chars!r}")

    async def validate_credentials(self) -> bool:
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
            raise ValueError("TranscriptFeedConnector: 'feed_urls' setting is empty")

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
                    logger.warning("TranscriptFeedConnector: error fetching %s: %s", feed_url, exc)

        logger.info(
            "TranscriptFeedConnector.fetch_content: feeds=%d items=%d latency_ms=%.1f",
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
            raise RateLimitError(f"Transcript feed rate-limited: {feed_url}")
        if resp.status_code != 200:
            raise PlatformError(f"HTTP {resp.status_code} fetching {feed_url}")

        feed = feedparser.parse(resp.text)
        podcast_title = feed.feed.get("title", feed_url)
        items: List[ContentItem] = []

        for entry in feed.entries[:remaining]:
            pub_at = self._parse_date(entry)
            if since and pub_at <= since.replace(tzinfo=timezone.utc):
                continue

            transcript_url = self._find_transcript_url(entry)
            transcript_text = ""
            has_transcript = False

            if transcript_url and self._fetch_text:
                try:
                    tr = await client.get(transcript_url, timeout=20.0, follow_redirects=True)
                    if tr.status_code == 200:
                        raw = tr.text
                        transcript_text = re.sub(r"<[^>]+>", " ", raw).strip()[: self._max_chars]
                        has_transcript = True
                except Exception as exc:
                    logger.debug("TranscriptFeedConnector: transcript fetch failed %s: %s", transcript_url, exc)

            if not transcript_text:
                fallback = entry.get("summary") or entry.get("description") or ""
                transcript_text = re.sub(r"<[^>]+>", " ", fallback).strip()

            items.append(self._create_content_item(
                source_id=entry.get("id") or entry.get("link", ""),
                source_url=entry.get("link", feed_url),
                title=f"[Transcript] {podcast_title}: {entry.get('title', '')}",
                raw_text=transcript_text,
                media_type=MediaType.TEXT,
                published_at=pub_at,
                metadata={
                    "podcast_title": podcast_title,
                    "feed_url": feed_url,
                    "transcript_url": transcript_url,
                    "has_full_transcript": has_transcript,
                    "transcript_chars": len(transcript_text),
                },
            ))
        return items

    def _find_transcript_url(self, entry: Any) -> str:
        """Look for transcript URL in RSS extensions or <link rel='transcript'>."""
        url = entry.get(self._transcript_field, "")
        if url:
            return url
        for link in entry.get("links", []):
            if link.get("rel") == "transcript" or link.get("type", "").startswith("text/"):
                return link.get("href", "")
        return ""

    @staticmethod
    def _parse_date(entry: Any) -> datetime:
        import calendar
        for field in ("published_parsed", "updated_parsed"):
            val = entry.get(field)
            if val:
                return datetime.utcfromtimestamp(calendar.timegm(val)).replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc)

