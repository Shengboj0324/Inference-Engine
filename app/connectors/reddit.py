"""Reddit connector using PRAW (Python Reddit API Wrapper)."""

from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

import praw
from praw.exceptions import PRAWException
from praw.models import Subreddit

from app.connectors.base import (
    AuthenticationError,
    BaseConnector,
    ConnectorConfig,
    FetchResult,
    PlatformError,
    RateLimitError,
    RateLimitInfo,
)
from app.core.models import ContentItem, MediaType, SourcePlatform


class RedditConnector(BaseConnector):
    """Connector for Reddit using official API."""

    def __init__(self, config: ConnectorConfig, user_id: UUID):
        """Initialize Reddit connector.

        Expected credentials:
            - client_id: Reddit app client ID
            - client_secret: Reddit app client secret
            - refresh_token: User's OAuth refresh token
            - user_agent: Application user agent
        """
        super().__init__(config, user_id)
        self.reddit: Optional[praw.Reddit] = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize PRAW Reddit client."""
        creds = self.config.credentials
        try:
            self.reddit = praw.Reddit(
                client_id=creds.get("client_id"),
                client_secret=creds.get("client_secret"),
                refresh_token=creds.get("refresh_token"),
                user_agent=creds.get("user_agent", "SocialMediaRadar/0.1"),
            )
        except Exception as e:
            raise AuthenticationError(f"Failed to initialize Reddit client: {e}")

    async def validate_credentials(self) -> bool:
        """Validate Reddit credentials by attempting to get user info."""
        try:
            if not self.reddit:
                return False
            # Try to access user info
            _ = self.reddit.user.me()
            return True
        except PRAWException:
            return False

    async def get_user_feeds(self) -> List[str]:
        """Get list of subreddits the user is subscribed to."""
        if not self.reddit:
            raise AuthenticationError("Reddit client not initialized")

        try:
            subreddits = []
            for subreddit in self.reddit.user.subreddits(limit=None):
                subreddits.append(subreddit.display_name)
            return subreddits
        except PRAWException as e:
            raise PlatformError(f"Failed to fetch subreddits: {e}")

    async def fetch_content(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> FetchResult:
        """Fetch posts from user's subscribed subreddits.

        Args:
            since: Fetch posts created after this timestamp
            cursor: Not used for Reddit (uses timestamp filtering)
            max_items: Maximum posts to fetch per subreddit

        Returns:
            FetchResult with posts from subscribed subreddits
        """
        if not self.reddit:
            raise AuthenticationError("Reddit client not initialized")

        items: List[ContentItem] = []
        errors: List[str] = []

        try:
            # Get user's subscribed subreddits
            subreddits = await self.get_user_feeds()

            # Limit subreddits if too many
            max_subreddits = self.config.settings.get("max_subreddits", 50)
            subreddits = subreddits[:max_subreddits]

            # Fetch from each subreddit
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    items.extend(
                        await self._fetch_from_subreddit(subreddit, since, max_items)
                    )
                except Exception as e:
                    errors.append(f"Error fetching from r/{subreddit_name}: {e}")

            return FetchResult(
                items=items,
                errors=errors,
            )

        except PRAWException as e:
            raise PlatformError(f"Reddit API error: {e}")

    async def _fetch_from_subreddit(
        self,
        subreddit: Subreddit,
        since: Optional[datetime],
        max_items: int,
    ) -> List[ContentItem]:
        """Fetch posts from a specific subreddit."""
        items: List[ContentItem] = []
        since_timestamp = since.timestamp() if since else 0

        # Fetch new posts
        for submission in subreddit.new(limit=max_items):
            # Filter by timestamp if provided
            if submission.created_utc <= since_timestamp:
                continue

            # Determine media type
            media_type = MediaType.TEXT
            media_urls = []

            if submission.is_video:
                media_type = MediaType.VIDEO
                if hasattr(submission, "media") and submission.media:
                    media_urls.append(submission.url)
            elif submission.url and any(
                submission.url.endswith(ext) for ext in [".jpg", ".png", ".gif"]
            ):
                media_type = MediaType.IMAGE
                media_urls.append(submission.url)
            elif submission.url != submission.permalink:
                media_type = MediaType.MIXED
                media_urls.append(submission.url)

            # Create content item
            item = self._create_content_item(
                source_id=submission.id,
                source_url=f"https://reddit.com{submission.permalink}",
                title=submission.title,
                raw_text=submission.selftext if submission.selftext else None,
                author=str(submission.author) if submission.author else None,
                channel=f"r/{subreddit.display_name}",
                media_type=media_type,
                media_urls=media_urls,
                published_at=datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
                metadata={
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "upvote_ratio": submission.upvote_ratio,
                    "subreddit": subreddit.display_name,
                    "flair": submission.link_flair_text,
                },
            )
            items.append(item)

        return items

