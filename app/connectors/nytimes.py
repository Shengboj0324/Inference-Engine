"""New York Times API connector.

Official API: https://developer.nytimes.com/
APIs: Article Search, Archive, Top Stories, Most Popular
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

import aiohttp

from app.connectors.base import BaseConnector, ConnectorConfig, FetchResult, RateLimitInfo
from app.core.errors import ConnectorError
from app.core.models import ContentItem, ContentType, SourcePlatform

logger = logging.getLogger(__name__)


class NYTimesConnector(BaseConnector):
    """New York Times API connector.
    
    Features:
    - Article search with filters
    - Top stories by section
    - Most popular articles
    - Archive access
    - Rate limiting: 500 requests/day, 5 requests/minute
    """

    BASE_URL = "https://api.nytimes.com/svc"
    
    def __init__(self, config: ConnectorConfig, user_id: UUID):
        """Initialize NYTimes connector.
        
        Required credentials:
        - api_key: NYTimes API key (get from https://developer.nytimes.com/)
        """
        super().__init__(config, user_id)
        self.api_key = config.credentials.get("api_key")
        
        if not self.api_key:
            raise ConnectorError("NYTimes connector requires api_key")

    async def validate_credentials(self) -> bool:
        """Validate NYTimes API credentials."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.BASE_URL}/topstories/v2/home.json",
                    params={"api-key": self.api_key},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"NYTimes credential validation failed: {e}")
            return False

    async def fetch_content(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> FetchResult:
        """Fetch NYTimes articles."""
        items: List[ContentItem] = []
        errors: List[str] = []
        
        try:
            # Determine which API to use
            api_mode = self.config.settings.get("api_mode", "search")
            
            if api_mode == "search":
                items = await self._fetch_article_search(since, cursor, max_items)
            elif api_mode == "top_stories":
                items = await self._fetch_top_stories(max_items)
            elif api_mode == "most_popular":
                items = await self._fetch_most_popular(max_items)
            else:
                items = await self._fetch_article_search(since, cursor, max_items)
            
            return FetchResult(items=items, errors=errors)
        
        except Exception as e:
            logger.error(f"Error fetching NYTimes content: {e}")
            errors.append(str(e))
            return FetchResult(items=items, errors=errors)

    async def _fetch_article_search(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> List[ContentItem]:
        """Fetch articles using Article Search API."""
        items: List[ContentItem] = []
        
        async with aiohttp.ClientSession() as session:
            # Build query
            query = self.config.settings.get("query", "")
            sections = self.config.settings.get("sections", [])
            
            params = {
                "api-key": self.api_key,
                "page": int(cursor) if cursor else 0,
            }
            
            if query:
                params["q"] = query
            
            if sections:
                params["fq"] = f"section_name:({' '.join(sections)})"
            
            if since:
                params["begin_date"] = since.strftime("%Y%m%d")
            
            async with session.get(
                f"{self.BASE_URL}/search/v2/articlesearch.json",
                params=params,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status == 429:
                    raise ConnectorError("NYTimes API rate limit exceeded")
                
                if response.status != 200:
                    error_text = await response.text()
                    raise ConnectorError(f"NYTimes API error: {response.status} - {error_text}")
                
                data = await response.json()
                
                for doc in data.get("response", {}).get("docs", [])[:max_items]:
                    items.append(self._parse_article(doc))
        
        return items

    async def _fetch_top_stories(self, max_items: int = 100) -> List[ContentItem]:
        """Fetch top stories from specified section."""
        items: List[ContentItem] = []
        section = self.config.settings.get("section", "home")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.BASE_URL}/topstories/v2/{section}.json",
                params={"api-key": self.api_key},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ConnectorError(f"NYTimes API error: {response.status} - {error_text}")
                
                data = await response.json()
                
                for article in data.get("results", [])[:max_items]:
                    items.append(self._parse_top_story(article))
        
        return items

    async def _fetch_most_popular(self, max_items: int = 100) -> List[ContentItem]:
        """Fetch most popular articles."""
        items: List[ContentItem] = []
        period = self.config.settings.get("period", 7)  # 1, 7, or 30 days
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.BASE_URL}/mostpopular/v2/viewed/{period}.json",
                params={"api-key": self.api_key},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ConnectorError(f"NYTimes API error: {response.status} - {error_text}")
                
                data = await response.json()
                
                for article in data.get("results", [])[:max_items]:
                    items.append(self._parse_popular_article(article))
        
        return items

    def _parse_article(self, doc: Dict[str, Any]) -> ContentItem:
        """Parse article search result."""
        return ContentItem(
            platform=SourcePlatform.NYTIMES,
            platform_id=doc.get("_id"),
            content_type=ContentType.ARTICLE,
            title=doc.get("headline", {}).get("main", ""),
            text_content=doc.get("abstract", "") or doc.get("lead_paragraph", ""),
            url=doc.get("web_url", ""),
            author=", ".join([person.get("firstname", "") + " " + person.get("lastname", "") for person in doc.get("byline", {}).get("person", [])]),
            published_at=datetime.fromisoformat(doc.get("pub_date", "").replace("Z", "+00:00")),
            metadata={
                "section": doc.get("section_name"),
                "subsection": doc.get("subsection_name"),
                "keywords": [kw.get("value") for kw in doc.get("keywords", [])],
                "word_count": doc.get("word_count", 0),
                "document_type": doc.get("document_type"),
            },
            user_id=self.user_id,
        )

    def _parse_top_story(self, article: Dict[str, Any]) -> ContentItem:
        """Parse top story."""
        return ContentItem(
            platform=SourcePlatform.NYTIMES,
            platform_id=article.get("uri", "").split("/")[-1],
            content_type=ContentType.ARTICLE,
            title=article.get("title", ""),
            text_content=article.get("abstract", ""),
            url=article.get("url", ""),
            author=article.get("byline", ""),
            published_at=datetime.fromisoformat(article.get("published_date", "").replace("Z", "+00:00")),
            metadata={
                "section": article.get("section"),
                "subsection": article.get("subsection"),
                "multimedia": article.get("multimedia", []),
            },
            user_id=self.user_id,
        )

    def _parse_popular_article(self, article: Dict[str, Any]) -> ContentItem:
        """Parse most popular article."""
        return ContentItem(
            platform=SourcePlatform.NYTIMES,
            platform_id=str(article.get("id")),
            content_type=ContentType.ARTICLE,
            title=article.get("title", ""),
            text_content=article.get("abstract", ""),
            url=article.get("url", ""),
            author=article.get("byline", ""),
            published_at=datetime.fromisoformat(article.get("published_date", "").replace("Z", "+00:00")),
            metadata={
                "section": article.get("section"),
                "views": article.get("views", 0),
            },
            user_id=self.user_id,
        )

    async def get_user_feeds(self) -> List[str]:
        """Get configured NYTimes sources."""
        api_mode = self.config.settings.get("api_mode", "search")
        
        if api_mode == "search":
            query = self.config.settings.get("query", "")
            sections = self.config.settings.get("sections", [])
            return [f"Search: {query}"] + [f"Section: {s}" for s in sections]
        elif api_mode == "top_stories":
            section = self.config.settings.get("section", "home")
            return [f"Top Stories: {section}"]
        elif api_mode == "most_popular":
            period = self.config.settings.get("period", 7)
            return [f"Most Popular ({period} days)"]
        
        return ["NYTimes"]

