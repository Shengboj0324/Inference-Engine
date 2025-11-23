"""Apple News connector using web scraping.

Note: Apple News does not provide a public API.
This connector uses web scraping with compliance to robots.txt.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.connectors.base import BaseConnector, ConnectorConfig, FetchResult
from app.core.errors import ConnectorError
from app.core.models import ContentItem, ContentType, SourcePlatform
from app.scraping.manager import ScrapingManager
from app.scraping.base import ScraperConfig, ComplianceLevel

logger = logging.getLogger(__name__)


class AppleNewsConnector(BaseConnector):
    """Apple News connector using web scraping.
    
    Features:
    - Topic-based scraping
    - Compliance with robots.txt
    - Rate limiting
    - Anti-detection measures
    
    Note: This is a scraping-based connector. Use responsibly and in compliance with Apple's terms.
    """

    BASE_URL = "https://www.apple.com/newsroom"
    
    def __init__(self, config: ConnectorConfig, user_id: UUID):
        """Initialize Apple News connector.
        
        Settings:
        - topics: List of topics to scrape (e.g., ['technology', 'business'])
        - compliance_level: STRICT, MODERATE, or AGGRESSIVE
        """
        super().__init__(config, user_id)

        # Initialize scraping manager
        scraper_config = ScraperConfig(
            compliance_level=ComplianceLevel(
                self.config.settings.get("compliance_level", "MODERATE")
            ),
        )
        self.scraper = ScrapingManager(scraper_config)

    async def validate_credentials(self) -> bool:
        """Validate Apple News access (no credentials needed)."""
        try:
            # Check if we can access the newsroom
            result = await self.scraper.scrape_with_retry(self.BASE_URL)
            return result.success
        except Exception as e:
            logger.error(f"Apple News validation failed: {e}")
            return False

    async def fetch_content(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> FetchResult:
        """Fetch Apple News articles via web scraping."""
        items: List[ContentItem] = []
        errors: List[str] = []
        
        try:
            # Scrape newsroom page
            result = await self.scraper.scrape_with_retry(
                self.BASE_URL,
                extract_links=True,
            )

            if not result.success:
                errors.append(f"Failed to scrape Apple Newsroom: {result.error}")
                return FetchResult(items=items, errors=errors)

            # Parse articles from links
            article_links = [
                link for link in result.links
                if "/newsroom/" in link and link != self.BASE_URL
            ][:max_items]

            # Scrape individual articles
            for link in article_links:
                try:
                    article_result = await self.scraper.scrape_with_retry(link)

                    if article_result.success:
                        item = self._parse_article(article_result.metadata, link)
                        if item:
                            items.append(item)

                except Exception as e:
                    logger.warning(f"Error scraping article {link}: {e}")
                    continue
            
            # Filter by date if specified
            if since:
                items = [item for item in items if item.published_at >= since]
            
            return FetchResult(items=items, errors=errors)
        
        except Exception as e:
            logger.error(f"Error fetching Apple News content: {e}")
            errors.append(str(e))
            return FetchResult(items=items, errors=errors)

    def _parse_article(self, metadata: Dict[str, Any], url: str) -> Optional[ContentItem]:
        """Parse scraped article metadata into ContentItem."""
        try:
            title = metadata.get("og:title") or metadata.get("title", "")
            description = metadata.get("og:description") or metadata.get("description", "")
            
            if not title:
                return None
            
            # Try to extract date
            published_at = datetime.now()
            date_str = metadata.get("article:published_time") or metadata.get("date")
            if date_str:
                try:
                    published_at = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except:
                    pass
            
            return ContentItem(
                platform=SourcePlatform.APPLE_NEWS,
                platform_id=url,
                content_type=ContentType.ARTICLE,
                title=title,
                text_content=description,
                url=url,
                author=metadata.get("author", "Apple"),
                published_at=published_at,
                metadata={
                    "image": metadata.get("og:image"),
                    "type": metadata.get("og:type"),
                },
                user_id=self.user_id,
            )
        
        except Exception as e:
            logger.warning(f"Error parsing article metadata: {e}")
            return None

    async def get_user_feeds(self) -> List[str]:
        """Get configured Apple News sources."""
        return ["Apple Newsroom"]

