"""OpenReview connector.

Fetches papers and reviews from OpenReview.net via the public REST API
(``https://api2.openreview.net``).  No authentication is required for
public venues.

Each paper note becomes one ``ContentItem`` with:
- ``title``      : paper title
- ``raw_text``   : abstract (``content.abstract.value``)
- ``source_url`` : ``https://openreview.net/forum?id={forum_id}``
- ``media_type`` : ``MediaType.TEXT``
- ``metadata``   : venue, authors, keywords, pdf_url, decision (if available)

Configuration (``ConnectorConfig.settings``)::

    venues: List[str]      # e.g. ["NeurIPS.cc/2024/Conference", "ICLR.cc/2025/Conference"]
    keywords: List[str]    # Optional content filter (matched against title/abstract)
    max_results: int       # default 50 per venue
    include_rejected: bool # include rejected submissions; default False
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

_OR_BASE = "https://api2.openreview.net"


class OpenReviewConnector(BaseConnector):
    """Fetches academic papers from OpenReview venues."""

    def __init__(self, config: ConnectorConfig, user_id: UUID) -> None:
        super().__init__(config, user_id)
        s: Dict[str, Any] = config.settings or {}
        self._venues: List[str] = s.get("venues", [])
        self._keywords: List[str] = [kw.lower() for kw in s.get("keywords", [])]
        self._max_results: int = int(s.get("max_results", 50))
        self._include_rejected: bool = bool(s.get("include_rejected", False))
        if not isinstance(self._venues, list):
            raise TypeError(f"'venues' must be a list, got {type(self._venues)!r}")

    async def validate_credentials(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{_OR_BASE}/venues", params={"limit": 1})
            return resp.status_code == 200
        except Exception:
            return False

    async def get_user_feeds(self) -> List[str]:
        return list(self._venues)

    async def fetch_content(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> FetchResult:
        if not self._venues:
            raise ValueError("OpenReviewConnector: 'venues' setting is empty")

        t0 = time.perf_counter()
        items: List[ContentItem] = []
        errors: List[str] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for venue in self._venues:
                if len(items) >= max_items:
                    break
                try:
                    v_items = await self._fetch_venue(client, venue, since, max_items - len(items))
                    items.extend(v_items)
                except RateLimitError:
                    raise
                except Exception as exc:
                    errors.append(f"{venue}: {exc}")
                    logger.warning("OpenReviewConnector: error on %s: %s", venue, exc)

        logger.info(
            "OpenReviewConnector.fetch_content: venues=%d items=%d latency_ms=%.1f",
            len(self._venues), len(items), (time.perf_counter() - t0) * 1000,
        )
        return FetchResult(items=items, errors=errors)

    async def _fetch_venue(
        self,
        client: httpx.AsyncClient,
        venue: str,
        since: Optional[datetime],
        remaining: int,
    ) -> List[ContentItem]:
        params: Dict[str, Any] = {
            "venueid": venue,
            "limit": min(self._max_results, remaining, 100),
            "offset": 0,
            "details": "replyCount,invitation",
        }
        resp = await client.get(f"{_OR_BASE}/notes", params=params)
        if resp.status_code == 429:
            raise RateLimitError(f"OpenReview rate-limited on {venue}")
        if resp.status_code != 200:
            raise PlatformError(f"OpenReview API error {resp.status_code} for {venue}")

        data = resp.json()
        notes = data.get("notes", [])
        items: List[ContentItem] = []

        for note in notes:
            content = note.get("content") or {}
            decision = self._extract_str(content.get("decision", ""))
            if not self._include_rejected and "reject" in decision.lower():
                continue

            title = self._extract_str(content.get("title", ""))
            abstract = self._extract_str(content.get("abstract", ""))
            if not title:
                continue
            if self._keywords and not self._keyword_match(title + " " + abstract):
                continue

            created_ms = note.get("cdate") or note.get("tcdate") or 0
            pub_at = datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc) if created_ms else datetime.now(timezone.utc)
            if since and pub_at <= since.replace(tzinfo=timezone.utc):
                continue

            forum_id = note.get("forum") or note.get("id", "")
            authors = content.get("authors", {})
            author_list: List[str] = (
                authors.get("value", []) if isinstance(authors, dict) else authors if isinstance(authors, list) else []
            )
            keywords_raw = content.get("keywords", {})
            kw_list: List[str] = (
                keywords_raw.get("value", []) if isinstance(keywords_raw, dict) else keywords_raw if isinstance(keywords_raw, list) else []
            )
            pdf_url = self._extract_str(content.get("pdf", ""))
            if pdf_url and not pdf_url.startswith("http"):
                pdf_url = f"https://openreview.net{pdf_url}"

            items.append(self._create_content_item(
                source_id=note.get("id", ""),
                source_url=f"https://openreview.net/forum?id={forum_id}",
                title=title,
                raw_text=abstract,
                media_type=MediaType.TEXT,
                published_at=pub_at,
                metadata={
                    "venue": venue,
                    "authors": author_list,
                    "keywords": kw_list,
                    "pdf_url": pdf_url,
                    "decision": decision,
                    "forum_id": forum_id,
                },
            ))
        return items

    @staticmethod
    def _extract_str(val: Any) -> str:
        """Extract string from OpenReview's nested value wrapper or plain str."""
        if isinstance(val, dict):
            return str(val.get("value", ""))
        return str(val) if val else ""

    def _keyword_match(self, text: str) -> bool:
        lower = text.lower()
        return any(kw in lower for kw in self._keywords)

