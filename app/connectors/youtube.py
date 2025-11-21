"""YouTube connector using Google API."""

from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.connectors.base import (
    AuthenticationError,
    BaseConnector,
    ConnectorConfig,
    FetchResult,
    PlatformError,
)
from app.core.models import ContentItem, MediaType


class YouTubeConnector(BaseConnector):
    """Connector for YouTube using official Data API v3."""

    def __init__(self, config: ConnectorConfig, user_id: UUID):
        """Initialize YouTube connector.

        Expected credentials:
            - access_token: OAuth access token
            - refresh_token: OAuth refresh token
            - token_uri: Token refresh URI
            - client_id: OAuth client ID
            - client_secret: OAuth client secret
        """
        super().__init__(config, user_id)
        self.youtube = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize YouTube API client."""
        creds_dict = self.config.credentials
        try:
            credentials = Credentials(
                token=creds_dict.get("access_token"),
                refresh_token=creds_dict.get("refresh_token"),
                token_uri=creds_dict.get("token_uri", "https://oauth2.googleapis.com/token"),
                client_id=creds_dict.get("client_id"),
                client_secret=creds_dict.get("client_secret"),
            )
            self.youtube = build("youtube", "v3", credentials=credentials)
        except Exception as e:
            raise AuthenticationError(f"Failed to initialize YouTube client: {e}")

    async def validate_credentials(self) -> bool:
        """Validate YouTube credentials."""
        try:
            if not self.youtube:
                return False
            # Try to get channel info
            request = self.youtube.channels().list(part="snippet", mine=True)
            request.execute()
            return True
        except HttpError:
            return False

    async def get_user_feeds(self) -> List[str]:
        """Get list of channels the user is subscribed to."""
        if not self.youtube:
            raise AuthenticationError("YouTube client not initialized")

        channels = []
        try:
            request = self.youtube.subscriptions().list(
                part="snippet", mine=True, maxResults=50
            )

            while request:
                response = request.execute()
                for item in response.get("items", []):
                    channel_id = item["snippet"]["resourceId"]["channelId"]
                    channel_title = item["snippet"]["title"]
                    channels.append(f"{channel_title}|{channel_id}")

                request = self.youtube.subscriptions().list_next(request, response)

            return channels
        except HttpError as e:
            raise PlatformError(f"Failed to fetch subscriptions: {e}")

    async def fetch_content(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> FetchResult:
        """Fetch videos from subscribed channels."""
        if not self.youtube:
            raise AuthenticationError("YouTube client not initialized")

        items: List[ContentItem] = []
        errors: List[str] = []

        try:
            # Get subscribed channels
            channels = await self.get_user_feeds()

            # Limit channels
            max_channels = self.config.settings.get("max_channels", 30)
            channels = channels[:max_channels]

            for channel_info in channels:
                try:
                    channel_title, channel_id = channel_info.split("|")
                    channel_items = await self._fetch_from_channel(
                        channel_id, channel_title, since, max_items // len(channels)
                    )
                    items.extend(channel_items)
                except Exception as e:
                    errors.append(f"Error fetching from {channel_info}: {e}")

            return FetchResult(items=items, errors=errors)

        except HttpError as e:
            raise PlatformError(f"YouTube API error: {e}")

    async def _fetch_from_channel(
        self,
        channel_id: str,
        channel_title: str,
        since: Optional[datetime],
        max_items: int,
    ) -> List[ContentItem]:
        """Fetch videos from a specific channel."""
        items: List[ContentItem] = []

        try:
            # Get uploads playlist
            channel_response = (
                self.youtube.channels()
                .list(part="contentDetails", id=channel_id)
                .execute()
            )

            if not channel_response.get("items"):
                return items

            uploads_playlist_id = channel_response["items"][0]["contentDetails"][
                "relatedPlaylists"
            ]["uploads"]

            # Fetch videos from uploads playlist
            request = self.youtube.playlistItems().list(
                part="snippet,contentDetails",
                playlistId=uploads_playlist_id,
                maxResults=min(max_items, 50),
            )

            response = request.execute()

            for item in response.get("items", []):
                snippet = item["snippet"]
                video_id = snippet["resourceId"]["videoId"]

                # Parse published date
                published_at = datetime.fromisoformat(
                    snippet["publishedAt"].replace("Z", "+00:00")
                )

                # Filter by date if provided
                if since and published_at <= since:
                    continue

                # Create content item
                content_item = self._create_content_item(
                    source_id=video_id,
                    source_url=f"https://www.youtube.com/watch?v={video_id}",
                    title=snippet["title"],
                    raw_text=snippet.get("description", ""),
                    author=channel_title,
                    channel=channel_title,
                    media_type=MediaType.VIDEO,
                    media_urls=[f"https://www.youtube.com/watch?v={video_id}"],
                    published_at=published_at,
                    metadata={
                        "channel_id": channel_id,
                        "video_id": video_id,
                        "thumbnails": snippet.get("thumbnails", {}),
                    },
                )
                items.append(content_item)

        except HttpError as e:
            raise PlatformError(f"Error fetching from channel {channel_id}: {e}")

        return items

