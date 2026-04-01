"""Documentation Monitor connector.

Monitors documentation page URLs for content changes using HTTP
``ETag`` and ``Last-Modified`` conditional-request semantics.

When a page changes, the new content is extracted, cleaned (strip HTML),
and emitted as a ``ContentItem`` with ``MediaType.TEXT``.

Use cases:
- Tracking API documentation pages (e.g. ``platform.openai.com/docs``)
- Monitoring pricing/policy pages for changes
- Watching model capability tables and feature pages

ETag state is persisted in-memory (surviving for the lifetime of the connector
instance).  For cross-restart persistence, callers should use
``get_etag_state()`` / ``restore_etag_state()`` to serialise/deserialise state
to their own storage.

Configuration (``ConnectorConfig.settings``)::

    page_urls: List[str]          # Documentation page URLs to monitor
    user_agent: str               # User-Agent header; default: "SMR-DocsMonitor/1.0"
    strip_nav: bool               # Heuristically strip nav/footer HTML; default True
    max_content_chars: int        # Truncate extracted content; default 20 000
    change_detection_mode: str    # "etag" | "hash" | "both"; default "both"
"""

import hashlib
import logging
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
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

_DEFAULT_USER_AGENT = "SMR-DocsMonitor/1.0"
_NAV_PATTERN = re.compile(
    r"<(nav|header|footer|aside|script|style)[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)


class DocsMonitorConnector(BaseConnector):
    """Monitors documentation URLs and emits ContentItems on content change."""

    def __init__(self, config: ConnectorConfig, user_id: UUID) -> None:
        super().__init__(config, user_id)
        s: Dict[str, Any] = config.settings or {}
        self._page_urls: List[str] = s.get("page_urls", [])
        self._user_agent: str = s.get("user_agent", _DEFAULT_USER_AGENT)
        self._strip_nav: bool = bool(s.get("strip_nav", True))
        self._max_chars: int = int(s.get("max_content_chars", 20_000))
        self._mode: str = s.get("change_detection_mode", "both")
        if not isinstance(self._page_urls, list):
            raise TypeError(f"'page_urls' must be a list, got {type(self._page_urls)!r}")
        if self._mode not in {"etag", "hash", "both"}:
            raise ValueError(f"'change_detection_mode' must be 'etag'|'hash'|'both', got {self._mode!r}")
        if self._max_chars <= 0:
            raise ValueError(f"'max_content_chars' must be positive, got {self._max_chars!r}")
        # Thread-safe ETag/hash state: url → (etag, content_hash)
        self._state: Dict[str, Tuple[str, str]] = {}
        self._lock: threading.Lock = threading.Lock()

    def get_etag_state(self) -> Dict[str, Tuple[str, str]]:
        """Return a copy of ETag/hash state for external persistence."""
        with self._lock:
            return dict(self._state)

    def restore_etag_state(self, state: Dict[str, Tuple[str, str]]) -> None:
        """Restore ETag/hash state (e.g. from Redis or disk)."""
        if not isinstance(state, dict):
            raise TypeError(f"'state' must be a dict, got {type(state)!r}")
        with self._lock:
            self._state.update(state)

    async def validate_credentials(self) -> bool:
        if not self._page_urls:
            return False
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.head(self._page_urls[0], headers={"User-Agent": self._user_agent})
            return resp.status_code < 500
        except Exception:
            return False

    async def get_user_feeds(self) -> List[str]:
        return list(self._page_urls)

    async def fetch_content(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> FetchResult:
        if not self._page_urls:
            raise ValueError("DocsMonitorConnector: 'page_urls' setting is empty")

        t0 = time.perf_counter()
        items: List[ContentItem] = []
        errors: List[str] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for url in self._page_urls:
                if len(items) >= max_items:
                    break
                try:
                    item = await self._check_url(client, url)
                    if item is not None:
                        items.append(item)
                except RateLimitError:
                    raise
                except Exception as exc:
                    errors.append(f"{url}: {exc}")
                    logger.warning("DocsMonitorConnector: error checking %s: %s", url, exc)

        logger.info(
            "DocsMonitorConnector.fetch_content: urls=%d changed=%d latency_ms=%.1f",
            len(self._page_urls), len(items), (time.perf_counter() - t0) * 1000,
        )
        return FetchResult(items=items, errors=errors)

    async def _check_url(self, client: httpx.AsyncClient, url: str) -> Optional[ContentItem]:
        with self._lock:
            prev_etag, prev_hash = self._state.get(url, ("", ""))

        request_headers = {"User-Agent": self._user_agent}
        if prev_etag and self._mode in ("etag", "both"):
            request_headers["If-None-Match"] = prev_etag

        resp = await client.get(url, headers=request_headers, follow_redirects=True)
        if resp.status_code == 304:
            logger.debug("DocsMonitorConnector: no change (304) for %s", url)
            return None
        if resp.status_code == 429:
            raise RateLimitError(f"Rate-limited on {url}")
        if resp.status_code != 200:
            raise PlatformError(f"HTTP {resp.status_code} fetching {url}")

        raw_html = resp.text
        content = self._extract_text(raw_html)
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        if self._mode in ("hash", "both") and content_hash == prev_hash:
            logger.debug("DocsMonitorConnector: content hash unchanged for %s", url)
            with self._lock:
                self._state[url] = (resp.headers.get("etag", prev_etag), content_hash)
            return None

        new_etag = resp.headers.get("etag", "")
        with self._lock:
            self._state[url] = (new_etag, content_hash)

        logger.debug("DocsMonitorConnector: change detected for %s (etag=%r)", url, new_etag)
        title = self._extract_title(raw_html) or url
        return self._create_content_item(
            source_id=f"{url}:{content_hash[:12]}",
            source_url=url,
            title=f"[DocsUpdate] {title}",
            raw_text=content[: self._max_chars],
            media_type=MediaType.TEXT,
            published_at=datetime.now(timezone.utc),
            metadata={
                "page_url": url,
                "etag": new_etag,
                "content_hash": content_hash,
                "content_chars": len(content),
                "change_detected": True,
            },
        )

    def _extract_text(self, html: str) -> str:
        """Strip navigation elements and convert HTML to plain text."""
        text = html
        if self._strip_nav:
            text = _NAV_PATTERN.sub(" ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _extract_title(html: str) -> str:
        m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        return m.group(1).strip() if m else ""

