"""arXiv connector.

Queries the arXiv Atom API (``http://export.arxiv.org/api/query``) for recent
papers matching configured search queries.  No API key is required.

Each paper becomes one ``ContentItem`` with:
- ``title``      : paper title
- ``raw_text``   : abstract text
- ``source_url`` : abstract page URL (``https://arxiv.org/abs/{arxiv_id}``)
- ``media_type`` : ``MediaType.TEXT``
- ``metadata``   : authors, categories, arxiv_id, pdf_url, doi, journal_ref

Configuration (``ConnectorConfig.settings``)::

    queries: List[str]        # arXiv search strings; e.g. ["ti:LLM AND cat:cs.AI"]
    categories: List[str]     # arXiv category filter; e.g. ["cs.AI", "cs.LG", "cs.CL"]
    max_results_per_query: int # default 25
    sort_by: str              # "submittedDate" | "relevance" | "lastUpdatedDate" (default: submittedDate)

arXiv API documentation: https://info.arxiv.org/help/api/user-manual
"""

import logging
import time
import xml.etree.ElementTree as ET
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

_ARXIV_BASE = "http://export.arxiv.org/api/query"
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}


class ArxivConnector(BaseConnector):
    """Fetches arXiv pre-prints matching one or more search queries.

    The arXiv Atom API is free and unauthenticated but imposes a 3-second
    delay between requests (documented best-practice).  This connector
    enforces a 3.1-second inter-request sleep when iterating over multiple
    queries.
    """

    _INTER_REQUEST_DELAY_S: float = 3.1  # arXiv API courtesy delay

    def __init__(self, config: ConnectorConfig, user_id: UUID) -> None:
        super().__init__(config, user_id)
        s: Dict[str, Any] = config.settings or {}
        self._queries: List[str] = s.get("queries", [])
        self._categories: List[str] = s.get("categories", [])
        self._max_per_query: int = int(s.get("max_results_per_query", 25))
        self._sort_by: str = s.get("sort_by", "submittedDate")
        if not isinstance(self._queries, list):
            raise TypeError(f"'queries' must be a list, got {type(self._queries)!r}")
        valid_sorts = {"submittedDate", "relevance", "lastUpdatedDate"}
        if self._sort_by not in valid_sorts:
            raise ValueError(f"'sort_by' must be one of {valid_sorts}, got {self._sort_by!r}")

    async def validate_credentials(self) -> bool:
        """arXiv API needs no credentials; always returns True if endpoint is reachable."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(_ARXIV_BASE, params={"search_query": "ti:test", "max_results": 1})
            return resp.status_code == 200
        except Exception:
            return False

    async def get_user_feeds(self) -> List[str]:
        return list(self._queries)

    async def fetch_content(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> FetchResult:
        if not self._queries:
            raise ValueError("ArxivConnector: 'queries' setting is empty")

        t0 = time.perf_counter()
        items: List[ContentItem] = []
        errors: List[str] = []
        seen_ids: set = set()

        async with httpx.AsyncClient(timeout=30.0) as client:
            for idx, query in enumerate(self._queries):
                if len(items) >= max_items:
                    break
                if idx > 0:
                    time.sleep(self._INTER_REQUEST_DELAY_S)
                try:
                    query_items = await self._fetch_query(client, query, since, max_items - len(items), seen_ids)
                    items.extend(query_items)
                except RateLimitError:
                    raise
                except Exception as exc:
                    errors.append(f"query={query!r}: {exc}")
                    logger.warning("ArxivConnector: error on query %r: %s", query, exc)

        logger.info(
            "ArxivConnector.fetch_content: queries=%d items=%d latency_ms=%.1f",
            len(self._queries), len(items), (time.perf_counter() - t0) * 1000,
        )
        return FetchResult(items=items, errors=errors)

    async def _fetch_query(
        self,
        client: httpx.AsyncClient,
        query: str,
        since: Optional[datetime],
        remaining: int,
        seen_ids: set,
    ) -> List[ContentItem]:
        cat_filter = " AND ".join(f"cat:{c}" for c in self._categories)
        full_query = f"({query}) AND ({cat_filter})" if self._categories else query
        params = {
            "search_query": full_query,
            "max_results": min(self._max_per_query, remaining, 100),
            "sortBy": self._sort_by,
            "sortOrder": "descending",
        }
        resp = await client.get(_ARXIV_BASE, params=params)
        if resp.status_code == 429:
            raise RateLimitError("arXiv API rate-limited")
        if resp.status_code != 200:
            raise PlatformError(f"arXiv API error {resp.status_code}")

        root = ET.fromstring(resp.text)
        items: List[ContentItem] = []
        for entry in root.findall("atom:entry", _NS):
            arxiv_id_raw = (entry.findtext("atom:id", "", _NS) or "").strip()
            arxiv_id = arxiv_id_raw.split("/abs/")[-1] if "/abs/" in arxiv_id_raw else arxiv_id_raw
            if arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)

            published_raw = entry.findtext("atom:published", "", _NS) or ""
            pub_at = datetime.fromisoformat(published_raw.replace("Z", "+00:00")) if published_raw else datetime.now(timezone.utc)
            if since and pub_at.replace(tzinfo=timezone.utc) <= since.replace(tzinfo=timezone.utc):
                continue

            title = (entry.findtext("atom:title", "", _NS) or "").strip().replace("\n", " ")
            abstract = (entry.findtext("atom:summary", "", _NS) or "").strip()
            authors = [a.findtext("atom:name", "", _NS) or "" for a in entry.findall("atom:author", _NS)]
            cats = [c.get("term", "") for c in entry.findall("atom:category", _NS)]
            pdf_url = next((l.get("href", "") for l in entry.findall("atom:link", _NS) if l.get("title") == "pdf"), "")
            doi = entry.findtext("arxiv:doi", "", _NS) or ""
            journal_ref = entry.findtext("arxiv:journal_ref", "", _NS) or ""

            items.append(self._create_content_item(
                source_id=arxiv_id,
                source_url=f"https://arxiv.org/abs/{arxiv_id}",
                title=title,
                raw_text=abstract,
                media_type=MediaType.TEXT,
                published_at=pub_at,
                metadata={
                    "arxiv_id": arxiv_id,
                    "authors": authors,
                    "categories": cats,
                    "pdf_url": pdf_url,
                    "doi": doi,
                    "journal_ref": journal_ref,
                    "query": query,
                },
            ))
        return items

