"""YouTube Transcript connector.

Fetches captions/transcripts for YouTube videos by combining:

1. **YouTube Data API v3** ``search.list`` — to discover recent videos from
   configured channel IDs matching optional topic queries.
2. **``youtube-transcript-api``** (third-party; no API quota cost) — to pull
   the actual caption text.

Each video with an available transcript produces one ``ContentItem`` with:
- ``media_type``   : ``MediaType.TEXT``
- ``raw_text``     : full concatenated transcript (respecting ``max_chars``)
- ``source_url``   : ``https://youtu.be/{video_id}``
- ``metadata``     : video_id, channel_id, channel_title, duration, view_count,
                     transcript_language, caption_kind ("asr" | "manual")

If ``youtube-transcript-api`` is not installed the connector logs a warning and
falls back to returning the video description as ``raw_text``.

Configuration (``ConnectorConfig.settings``)::

    channel_ids: List[str]   # YouTube channel IDs to scan
    video_ids: List[str]     # Direct video IDs to transcribe
    query: str               # Optional: search string within the channel
    max_videos: int          # default 20
    max_chars: int           # transcript truncation; default 80 000
    preferred_language: str  # BCP-47 language code; default "en"
    api_key: str             # YouTube Data API v3 key (REQUIRED for discovery)
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

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

_YT_API_BASE = "https://www.googleapis.com/youtube/v3"


def _get_transcript_api():  # type: ignore[return]
    """Lazy import of youtube_transcript_api to avoid hard dependency."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        return YouTubeTranscriptApi
    except ImportError:
        return None


class YouTubeTranscriptConnector(BaseConnector):
    """Retrieves YouTube video transcripts via the YouTube Data API + caption extraction."""

    def __init__(self, config: ConnectorConfig, user_id: UUID) -> None:
        super().__init__(config, user_id)
        s: Dict[str, Any] = config.settings or {}
        self._channel_ids: List[str] = s.get("channel_ids", [])
        self._video_ids: List[str] = s.get("video_ids", [])
        self._query: Optional[str] = s.get("query")
        self._max_videos: int = int(s.get("max_videos", 20))
        self._max_chars: int = int(s.get("max_chars", 80_000))
        self._preferred_lang: str = s.get("preferred_language", "en")
        self._api_key: Optional[str] = s.get("api_key") or config.credentials.get("api_key")
        if not self._channel_ids and not self._video_ids:
            raise ValueError("YouTubeTranscriptConnector: need 'channel_ids' or 'video_ids'")
        if self._max_chars <= 0:
            raise ValueError(f"'max_chars' must be positive, got {self._max_chars!r}")

    async def validate_credentials(self) -> bool:
        if not self._api_key:
            return bool(self._video_ids)  # no API key OK if video_ids provided directly
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{_YT_API_BASE}/videos", params={"part": "id", "id": "dQw4w9WgXcQ", "key": self._api_key})
            return resp.status_code == 200
        except Exception:
            return False

    async def get_user_feeds(self) -> List[str]:
        return self._channel_ids + [f"video:{vid}" for vid in self._video_ids]

    async def fetch_content(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> FetchResult:
        t0 = time.perf_counter()
        items: List[ContentItem] = []
        errors: List[str] = []
        video_queue: List[Dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Discover videos from channels
            for channel_id in self._channel_ids:
                if len(video_queue) >= self._max_videos:
                    break
                try:
                    vids = await self._search_channel(client, channel_id, since, self._max_videos - len(video_queue))
                    video_queue.extend(vids)
                except RateLimitError:
                    raise
                except Exception as exc:
                    errors.append(f"channel={channel_id}: {exc}")
                    logger.warning("YouTubeTranscriptConnector: channel error %s: %s", channel_id, exc)

        # Add direct video IDs to queue
        for vid in self._video_ids:
            if len(video_queue) < self._max_videos:
                video_queue.append({"video_id": vid, "title": "", "channel_id": "", "channel_title": "", "published_at": datetime.now(timezone.utc)})

        # Fetch transcripts
        transcript_api = _get_transcript_api()
        for video_meta in video_queue[: max_items]:
            try:
                item = self._build_content_item(video_meta, transcript_api)
                if item:
                    items.append(item)
            except Exception as exc:
                errors.append(f"video={video_meta.get('video_id')}: {exc}")
                logger.debug("YouTubeTranscriptConnector: transcript error: %s", exc)

        logger.info(
            "YouTubeTranscriptConnector.fetch_content: videos_queued=%d items=%d latency_ms=%.1f",
            len(video_queue), len(items), (time.perf_counter() - t0) * 1000,
        )
        return FetchResult(items=items, errors=errors)

    async def _search_channel(
        self,
        client: httpx.AsyncClient,
        channel_id: str,
        since: Optional[datetime],
        remaining: int,
    ) -> List[Dict[str, Any]]:
        if not self._api_key:
            raise PlatformError("YouTubeTranscriptConnector: 'api_key' required for channel discovery")
        params: Dict[str, Any] = {
            "part": "snippet",
            "channelId": channel_id,
            "maxResults": min(remaining, 50),
            "order": "date",
            "type": "video",
            "key": self._api_key,
        }
        if self._query:
            params["q"] = self._query
        if since:
            params["publishedAfter"] = since.strftime("%Y-%m-%dT%H:%M:%SZ")
        resp = await client.get(f"{_YT_API_BASE}/search", params=params)
        if resp.status_code == 429 or (resp.status_code == 403 and "quota" in resp.text.lower()):
            raise RateLimitError(f"YouTube API quota exceeded for channel {channel_id}")
        if resp.status_code != 200:
            raise PlatformError(f"YouTube API error {resp.status_code}")
        data = resp.json()
        result = []
        for item in data.get("items", []):
            vid_id = (item.get("id") or {}).get("videoId", "")
            if not vid_id:
                continue
            snippet = item.get("snippet", {})
            pub_raw = snippet.get("publishedAt", "")
            pub_at = datetime.fromisoformat(pub_raw.replace("Z", "+00:00")) if pub_raw else datetime.now(timezone.utc)
            result.append({
                "video_id": vid_id,
                "title": snippet.get("title", ""),
                "channel_id": channel_id,
                "channel_title": snippet.get("channelTitle", ""),
                "published_at": pub_at,
            })
        return result

    def _build_content_item(
        self,
        video_meta: Dict[str, Any],
        transcript_api: Any,
    ) -> Optional[ContentItem]:
        video_id = video_meta["video_id"]
        transcript_text = ""
        caption_kind = "none"
        lang_used = ""

        if transcript_api is not None:
            try:
                transcript_list = transcript_api.list_transcripts(video_id)
                transcript = None
                try:
                    transcript = transcript_list.find_transcript([self._preferred_lang])
                except Exception:
                    for t in transcript_list:
                        transcript = t
                        break
                if transcript:
                    fetched = transcript.fetch()
                    transcript_text = " ".join(seg["text"] for seg in fetched)[: self._max_chars]
                    caption_kind = "manual" if not transcript.is_generated else "asr"
                    lang_used = transcript.language_code
            except Exception as exc:
                logger.debug("YouTubeTranscriptConnector: transcript unavailable for %s: %s", video_id, exc)
        else:
            logger.warning("youtube-transcript-api not installed; transcript unavailable for %s", video_id)

        if not transcript_text:
            return None  # skip videos with no transcript

        return self._create_content_item(
            source_id=video_id,
            source_url=f"https://youtu.be/{video_id}",
            title=f"[Transcript] {video_meta.get('title', video_id)}",
            raw_text=transcript_text,
            media_type=MediaType.TEXT,
            published_at=video_meta.get("published_at", datetime.now(timezone.utc)),
            metadata={
                "video_id": video_id,
                "channel_id": video_meta.get("channel_id", ""),
                "channel_title": video_meta.get("channel_title", ""),
                "transcript_language": lang_used,
                "caption_kind": caption_kind,
                "transcript_chars": len(transcript_text),
            },
        )

