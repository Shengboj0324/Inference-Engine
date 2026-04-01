"""Semantic Scholar connector.

Queries the Semantic Scholar Academic Graph API to discover and track
research papers relevant to configured topics and author IDs.

API base: ``https://api.semanticscholar.org/graph/v1``

Two sub-modes:
1. **Keyword search** — ``GET /paper/search?query={q}&fields=...``
2. **Author papers**  — ``GET /author/{id}/papers?fields=...``

Each paper maps to one ``ContentItem`` with ``MediaType.TEXT`` (abstract as
``raw_text``).

Configuration (``ConnectorConfig.settings``)::

    queries: List[str]          # keyword search queries
    author_ids: List[str]       # Semantic Scholar author IDs
    min_citation_count: int     # filter; default 0
    fields_of_study: List[str]  # e.g. ["Computer Science", "Machine Learning"]
    max_results_per_query: int  # default 20
    api_key: str                # optional partner API key (higher rate limits)

Rate limits: 100 req/5 min unauthenticated; 1 req/s with API key.
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

_SS_BASE = "https://api.semanticscholar.org/graph/v1"
_PAPER_FIELDS = "paperId,title,abstract,year,publicationDate,authors,externalIds,fieldsOfStudy,citationCount,openAccessPdf,venue,publicationVenue"


class SemanticScholarConnector(BaseConnector):
    """Discovers AI/ML research papers via the Semantic Scholar Graph API."""

    def __init__(self, config: ConnectorConfig, user_id: UUID) -> None:
        super().__init__(config, user_id)
        s: Dict[str, Any] = config.settings or {}
        self._queries: List[str] = s.get("queries", [])
        self._author_ids: List[str] = s.get("author_ids", [])
        self._min_citations: int = int(s.get("min_citation_count", 0))
        self._fields_of_study: List[str] = s.get("fields_of_study", [])
        self._max_per_query: int = int(s.get("max_results_per_query", 20))
        self._api_key: Optional[str] = s.get("api_key") or config.credentials.get("api_key")
        if not self._queries and not self._author_ids:
            raise ValueError("SemanticScholarConnector: at least one of 'queries' or 'author_ids' is required")

    async def validate_credentials(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{_SS_BASE}/paper/search", params={"query": "test", "limit": 1}, headers=self._headers())
            return resp.status_code in (200, 400)  # 400 = bad query but API is reachable
        except Exception:
            return False

    async def get_user_feeds(self) -> List[str]:
        return self._queries + [f"author:{aid}" for aid in self._author_ids]

    async def fetch_content(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> FetchResult:
        t0 = time.perf_counter()
        items: List[ContentItem] = []
        errors: List[str] = []
        seen_ids: set = set()

        async with httpx.AsyncClient(timeout=30.0) as client:
            for query in self._queries:
                if len(items) >= max_items:
                    break
                try:
                    q_items = await self._search_papers(client, query, since, max_items - len(items), seen_ids)
                    items.extend(q_items)
                except RateLimitError:
                    raise
                except Exception as exc:
                    errors.append(f"query={query!r}: {exc}")
                    logger.warning("SemanticScholarConnector: error on query %r: %s", query, exc)

            for author_id in self._author_ids:
                if len(items) >= max_items:
                    break
                try:
                    a_items = await self._author_papers(client, author_id, since, max_items - len(items), seen_ids)
                    items.extend(a_items)
                except RateLimitError:
                    raise
                except Exception as exc:
                    errors.append(f"author_id={author_id!r}: {exc}")
                    logger.warning("SemanticScholarConnector: error on author %r: %s", author_id, exc)

        logger.info(
            "SemanticScholarConnector.fetch_content: queries=%d authors=%d items=%d latency_ms=%.1f",
            len(self._queries), len(self._author_ids), len(items), (time.perf_counter() - t0) * 1000,
        )
        return FetchResult(items=items, errors=errors)

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {}
        if self._api_key:
            h["x-api-key"] = self._api_key
        return h

    async def _search_papers(
        self, client: httpx.AsyncClient, query: str, since: Optional[datetime], remaining: int, seen_ids: set
    ) -> List[ContentItem]:
        params: Dict[str, Any] = {"query": query, "fields": _PAPER_FIELDS, "limit": min(self._max_per_query, remaining, 100)}
        if self._fields_of_study:
            params["fieldsOfStudy"] = ",".join(self._fields_of_study)
        resp = await client.get(f"{_SS_BASE}/paper/search", params=params, headers=self._headers())
        if resp.status_code == 429:
            raise RateLimitError("Semantic Scholar rate-limited")
        if resp.status_code != 200:
            raise PlatformError(f"Semantic Scholar API error {resp.status_code}")
        data = resp.json()
        return self._parse_papers(data.get("data", []), since, seen_ids, {"query": query})

    async def _author_papers(
        self, client: httpx.AsyncClient, author_id: str, since: Optional[datetime], remaining: int, seen_ids: set
    ) -> List[ContentItem]:
        resp = await client.get(
            f"{_SS_BASE}/author/{author_id}/papers",
            params={"fields": _PAPER_FIELDS, "limit": min(self._max_per_query, remaining, 100)},
            headers=self._headers(),
        )
        if resp.status_code == 429:
            raise RateLimitError("Semantic Scholar rate-limited")
        if resp.status_code == 404:
            raise PlatformError(f"Author not found: {author_id}")
        if resp.status_code != 200:
            raise PlatformError(f"Semantic Scholar API error {resp.status_code}")
        data = resp.json()
        return self._parse_papers(data.get("data", []), since, seen_ids, {"author_id": author_id})

    def _parse_papers(
        self, papers: List[Dict], since: Optional[datetime], seen_ids: set, extra_meta: Dict
    ) -> List[ContentItem]:
        items: List[ContentItem] = []
        for paper in papers:
            pid = paper.get("paperId", "")
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)
            if paper.get("citationCount", 0) < self._min_citations:
                continue
            pub_raw = paper.get("publicationDate") or str(paper.get("year", ""))
            try:
                pub_at = datetime.strptime(pub_raw[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc) if pub_raw else datetime.now(timezone.utc)
            except ValueError:
                pub_at = datetime.now(timezone.utc)
            if since and pub_at <= since.replace(tzinfo=timezone.utc):
                continue
            title = paper.get("title") or ""
            abstract = paper.get("abstract") or ""
            pdf_info = paper.get("openAccessPdf") or {}
            items.append(self._create_content_item(
                source_id=pid,
                source_url=f"https://www.semanticscholar.org/paper/{pid}",
                title=title,
                raw_text=abstract,
                media_type=MediaType.TEXT,
                published_at=pub_at,
                metadata={
                    "paper_id": pid,
                    "authors": [a.get("name", "") for a in (paper.get("authors") or [])],
                    "citation_count": paper.get("citationCount", 0),
                    "fields_of_study": paper.get("fieldsOfStudy") or [],
                    "venue": paper.get("venue") or "",
                    "pdf_url": pdf_info.get("url", ""),
                    "external_ids": paper.get("externalIds") or {},
                    **extra_meta,
                },
            ))
        return items

