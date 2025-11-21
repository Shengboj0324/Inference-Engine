# Platform Connectors Guide

This document describes how to configure and use each platform connector in Social Media Radar.

## Overview

All connectors follow the same pattern:
1. User provides their own OAuth tokens or API keys
2. Connector validates credentials
3. System fetches content using official APIs only
4. Content is normalized into the unified `ContentItem` schema

## Connector Status

| Platform | Status | API Type | Auth Method |
|----------|--------|----------|-------------|
| Reddit | ✅ Implemented | Official API | OAuth 2.0 |
| YouTube | ✅ Implemented | Data API v3 | OAuth 2.0 |
| RSS | ✅ Implemented | RSS/Atom | None |
| TikTok | 🚧 Planned | Display API | OAuth 2.0 |
| Facebook | 🚧 Planned | Graph API | OAuth 2.0 |
| Instagram | 🚧 Planned | Graph API | OAuth 2.0 |
| NewsAPI | 🚧 Planned | REST API | API Key |
| NYT | 🚧 Planned | Article Search | API Key |

## Reddit Connector

### Prerequisites
1. Create a Reddit app at https://www.reddit.com/prefs/apps
2. Note your `client_id` and `client_secret`
3. Obtain a refresh token using OAuth flow

### Configuration
```json
{
  "platform": "reddit",
  "credentials": {
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "refresh_token": "your_refresh_token",
    "user_agent": "SocialMediaRadar/0.1 by YourUsername"
  },
  "settings": {
    "max_subreddits": 50,
    "max_posts_per_subreddit": 100
  }
}
```

### What It Fetches
- New posts from subscribed subreddits
- Post metadata (score, comments, upvote ratio)
- Post content (title, selftext, media URLs)

### Rate Limits
- 60 requests per minute per OAuth client
- Automatically handled by PRAW

## YouTube Connector

### Prerequisites
1. Create a Google Cloud project
2. Enable YouTube Data API v3
3. Create OAuth 2.0 credentials
4. Obtain user's access and refresh tokens

### Configuration
```json
{
  "platform": "youtube",
  "credentials": {
    "client_id": "your_client_id.apps.googleusercontent.com",
    "client_secret": "your_client_secret",
    "access_token": "ya29.xxx",
    "refresh_token": "1//xxx",
    "token_uri": "https://oauth2.googleapis.com/token"
  },
  "settings": {
    "max_channels": 30,
    "max_videos_per_channel": 50
  }
}
```

### What It Fetches
- Latest videos from subscribed channels
- Video metadata (title, description, thumbnails)
- Channel information
- Available captions (for transcription)

### Rate Limits
- 10,000 quota units per day
- Each request costs different units (list=1, search=100)
- Automatically tracked and respected

## RSS Connector

### Prerequisites
- None! RSS is open and doesn't require authentication

### Configuration
```json
{
  "platform": "rss",
  "credentials": {},
  "settings": {
    "feed_urls": [
      "https://abcnews.go.com/abcnews/topstories",
      "https://feeds.bbci.co.uk/news/world/rss.xml",
      "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"
    ]
  }
}
```

### What It Fetches
- Latest entries from each feed
- Entry metadata (title, author, published date)
- Content (description, full content if available)
- Enclosures (podcast audio, video links)

### Rate Limits
- Respectful polling (recommended: every 15-30 minutes)
- No official limits, but be courteous

## Adding New Connectors

To add a new platform connector:

1. Create a new file in `app/connectors/` (e.g., `tiktok.py`)
2. Implement the `BaseConnector` interface:
   ```python
   from app.connectors.base import BaseConnector
   
   class TikTokConnector(BaseConnector):
       async def validate_credentials(self) -> bool:
           # Validate API credentials
           pass
       
       async def get_user_feeds(self) -> List[str]:
           # Get list of creators user follows
           pass
       
       async def fetch_content(self, since, cursor, max_items) -> FetchResult:
           # Fetch videos from followed creators
           pass
   ```

3. Register the connector in the connector factory
4. Add tests in `tests/unit/connectors/`
5. Document the connector in this file

## Best Practices

### Credential Security
- Never log credentials
- Encrypt credentials at rest in database
- Use environment variables for development
- Rotate tokens regularly

### Rate Limiting
- Respect platform rate limits
- Implement exponential backoff on errors
- Cache responses when appropriate
- Batch requests when possible

### Error Handling
- Gracefully handle API errors
- Log errors for debugging
- Continue processing other sources on failure
- Notify user of persistent failures

### Content Quality
- Validate content before storing
- Handle missing fields gracefully
- Normalize timestamps to UTC
- Sanitize HTML/markdown in content

