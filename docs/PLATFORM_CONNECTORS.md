# Platform Connectors Guide

This document provides comprehensive information about all supported platform connectors in Social Media Radar.

## Overview

Social Media Radar supports **13 platform connectors** across social media and news sources:

### Social Media Platforms (6)
- Reddit
- YouTube
- TikTok
- Facebook
- Instagram
- WeChat

### News Sources (7)
- New York Times
- Wall Street Journal
- ABC News (US & Australia)
- Google News
- Apple News
- RSS (Generic)

---

## Social Media Connectors

### 1. Reddit Connector

**Platform**: Reddit  
**API**: Reddit API (PRAW)  
**Authentication**: OAuth 2.0  
**Documentation**: https://www.reddit.com/dev/api/

#### Required Credentials
```json
{
  "client_id": "your_reddit_client_id",
  "client_secret": "your_reddit_client_secret",
  "username": "your_reddit_username",
  "password": "your_reddit_password"
}
```

#### Settings
```json
{
  "subreddits": ["python", "machinelearning", "technology"]
}
```

#### Features
- Fetch posts from subscribed subreddits
- Filter by hot, new, top, rising
- Extract comments and metadata
- Rate limiting compliance

---

### 2. YouTube Connector

**Platform**: YouTube  
**API**: YouTube Data API v3  
**Authentication**: OAuth 2.0  
**Documentation**: https://developers.google.com/youtube/v3

#### Required Credentials
```json
{
  "api_key": "your_youtube_api_key",
  "oauth_token": "optional_oauth_token"
}
```

#### Settings
```json
{
  "channels": ["channel_id_1", "channel_id_2"],
  "search_queries": ["AI news", "tech reviews"]
}
```

#### Features
- Fetch videos from subscribed channels
- Search videos by keywords
- Extract video metadata, transcripts
- Quota management (10,000 units/day)

---

### 3. TikTok Connector

**Platform**: TikTok  
**API**: TikTok Research API  
**Authentication**: OAuth 2.0  
**Documentation**: https://developers.tiktok.com/doc/research-api-overview

#### Required Credentials
```json
{
  "client_key": "your_tiktok_client_key",
  "client_secret": "your_tiktok_client_secret",
  "access_token": "your_access_token"
}
```

#### Settings
```json
{
  "search_queries": ["AI", "technology"],
  "hashtags": ["#tech", "#innovation"],
  "max_count": 100
}
```

#### Features
- Video search by keywords and hashtags
- Metadata extraction (views, likes, shares, comments)
- Music and effect tracking
- Region-based filtering

**Note**: Requires academic/research approval from TikTok

---

### 4. Facebook Connector

**Platform**: Facebook  
**API**: Graph API v21.0  
**Authentication**: OAuth 2.0  
**Documentation**: https://developers.facebook.com/docs/graph-api/

#### Required Credentials
```json
{
  "access_token": "your_facebook_access_token"
}
```

#### Settings
```json
{
  "sources": ["feed", "pages"],
  "page_ids": ["page_id_1", "page_id_2"]
}
```

#### Features
- User feed posts
- Page posts
- Reactions, comments, shares
- Rate limiting compliance

---

### 5. Instagram Connector

**Platform**: Instagram  
**API**: Instagram Graph API  
**Authentication**: OAuth 2.0  
**Documentation**: https://developers.facebook.com/docs/instagram-api/

#### Required Credentials
```json
{
  "access_token": "your_instagram_access_token",
  "instagram_business_account_id": "your_ig_account_id"
}
```

#### Settings
```json
{
  "hashtags": ["#technology", "#innovation"]
}
```

#### Features
- User media (posts, reels, stories)
- Hashtag search
- Media insights (likes, comments)
- Requires Instagram Business/Creator account

**Note**: Instagram Basic Display API deprecated Dec 2024

---

### 6. WeChat Connector

**Platform**: WeChat  
**API**: WeChat Official Account API  
**Authentication**: OAuth 2.0  
**Documentation**: https://developers.weixin.qq.com/doc/offiaccount/en/

#### Required Credentials
```json
{
  "app_id": "your_wechat_app_id",
  "app_secret": "your_wechat_app_secret"
}
```

#### Features
- Fetch articles from official accounts
- User message history
- Access token management
- Requires WeChat Official Account approval

---

## News Source Connectors

### 7. New York Times Connector

**Platform**: New York Times  
**API**: NYTimes API  
**Authentication**: API Key  
**Documentation**: https://developer.nytimes.com/

#### Required Credentials
```json
{
  "api_key": "your_nytimes_api_key"
}
```

#### Settings
```json
{
  "api_mode": "search",
  "query": "artificial intelligence",
  "sections": ["Technology", "Business"],
  "period": 7
}
```

#### API Modes
- `search`: Article Search API
- `top_stories`: Top Stories API
- `most_popular`: Most Popular API

#### Features
- Article search with filters
- Top stories by section
- Most popular articles
- Rate limiting: 500 requests/day, 5 requests/minute

---

### 8. Wall Street Journal Connector

**Platform**: Wall Street Journal  
**API**: RSS Feeds  
**Authentication**: None  
**Documentation**: https://www.wsj.com/news/rss-news-and-feeds

#### Settings
```json
{
  "feeds": ["opinion", "world", "business", "technology", "markets"]
}
```

#### Available Feeds
- Opinion
- World News
- U.S. Business
- Markets
- Technology
- Lifestyle
- Real Estate

---

### 9. ABC News Connector

**Platform**: ABC News (US & Australia)  
**API**: RSS Feeds  
**Authentication**: None

#### Settings
```json
{
  "region": "US",
  "feeds": ["top_stories", "politics", "technology"]
}
```

#### US Feeds
- Top Stories, Politics, International, Technology, Health

#### Australia Feeds
- News, World, Business, Analysis, Sport, Science, Health, Arts

---

### 10. Google News Connector

**Platform**: Google News  
**API**: RSS Feeds  
**Authentication**: None  
**Documentation**: https://news.google.com/

#### Settings
```json
{
  "topics": ["WORLD", "TECHNOLOGY", "BUSINESS"],
  "keywords": ["AI", "climate change"],
  "location": "US",
  "language": "en",
  "when": "d"
}
```

#### Features
- Topic-based feeds
- Keyword search with operators
- Location-based news
- Language filtering
- Time-based filtering (h, d, w, m, y)

---

### 11. Apple News Connector

**Platform**: Apple News  
**API**: Web Scraping (No official API)  
**Authentication**: None

#### Settings
```json
{
  "topics": ["technology", "business"],
  "compliance_level": "MODERATE"
}
```

#### Features
- Scrapes Apple Newsroom
- Compliance with robots.txt
- Rate limiting
- Anti-detection measures

**Note**: Uses web scraping. Use responsibly.

---

### 12. RSS Connector

**Platform**: Generic RSS/Atom Feeds  
**API**: RSS/Atom  
**Authentication**: None

#### Settings
```json
{
  "feed_urls": [
    "https://example.com/feed.xml",
    "https://another.com/rss"
  ]
}
```

#### Features
- Generic RSS/Atom feed parser
- Supports any valid RSS feed
- Automatic feed type detection

---

## Usage Example

```python
from app.connectors.registry import connector_registry
from app.connectors.base import ConnectorConfig
from app.core.models import SourcePlatform
from uuid import uuid4

# Create connector config
config = ConnectorConfig(
    platform=SourcePlatform.NYTIMES,
    credentials={"api_key": "your_api_key"},
    settings={"api_mode": "search", "query": "AI"}
)

# Get connector instance
connector = connector_registry.get_connector(
    platform=SourcePlatform.NYTIMES,
    config=config,
    user_id=uuid4()
)

# Fetch content
result = await connector.fetch_content(max_items=50)
print(f"Fetched {len(result.items)} items")
```

---

## Rate Limits & Quotas

| Platform | Rate Limit | Notes |
|----------|------------|-------|
| Reddit | 60 requests/minute | Per OAuth app |
| YouTube | 10,000 units/day | Quota-based |
| TikTok | Varies | Research API limits |
| Facebook | Dynamic | Based on app usage |
| Instagram | Dynamic | Based on app usage |
| WeChat | 2000 requests/day | Per official account |
| NYTimes | 500/day, 5/min | Per API key |
| Others | No official limits | Use responsibly |

---

## Best Practices

1. **Always validate credentials** before fetching content
2. **Respect rate limits** to avoid being blocked
3. **Use appropriate compliance levels** for scraping
4. **Store credentials securely** using encryption
5. **Handle errors gracefully** with retry logic
6. **Monitor API quotas** and usage
7. **Follow platform terms of service**

---

## Adding New Connectors

To add a new connector:

1. Create connector class in `app/connectors/your_platform.py`
2. Inherit from `BaseConnector`
3. Implement required methods:
   - `validate_credentials()`
   - `fetch_content()`
   - `get_user_feeds()`
4. Register in `app/connectors/registry.py`
5. Add platform to `SourcePlatform` enum
6. Update documentation

See `app/connectors/base.py` for the interface definition.

