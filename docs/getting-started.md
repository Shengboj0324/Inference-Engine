# Getting Started with Social Media Radar

This guide will help you get Social Media Radar up and running in under 10 minutes.

## Prerequisites

Before you begin, ensure you have:

- **Docker & Docker Compose** (recommended) OR Python 3.11+
- **API Keys** for LLM providers (OpenAI or Anthropic)
- **Platform Credentials** for sources you want to monitor (optional for initial setup)

## Quick Start with Docker

### 1. Clone and Configure

```bash
# Clone the repository
git clone https://github.com/yourusername/social-media-radar.git
cd social-media-radar

# Copy environment template
cp .env.example .env
```

### 2. Edit Configuration

Open `.env` and set at minimum:

```bash
# Required: LLM API Key (choose one)
OPENAI_API_KEY=sk-your-key-here
# OR
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Optional: Change default passwords
SECRET_KEY=your-random-secret-key
ENCRYPTION_KEY=your-32-byte-base64-key
```

### 3. Start Services

```bash
# Start all services
docker-compose up -d

# Wait for services to be healthy (30-60 seconds)
docker-compose ps

# Initialize database
docker-compose exec api alembic upgrade head
```

### 4. Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# View API documentation
open http://localhost:8000/docs
```

You should see:
```json
{"status": "healthy"}
```

## Configure Your First Source

### Option 1: RSS Feed (Easiest)

RSS feeds don't require authentication:

```bash
curl -X POST http://localhost:8000/api/v1/sources \
  -H "Content-Type: application/json" \
  -d '{
    "platform": "rss",
    "credentials": {},
    "settings": {
      "feed_urls": [
        "https://abcnews.go.com/abcnews/topstories",
        "https://feeds.bbci.co.uk/news/world/rss.xml"
      ]
    }
  }'
```

### Option 2: Reddit

1. **Create Reddit App**
   - Go to https://www.reddit.com/prefs/apps
   - Click "create another app"
   - Choose "script" type
   - Note your `client_id` and `client_secret`

2. **Get Refresh Token**
   ```bash
   # Use PRAW's OAuth helper or manual OAuth flow
   # See docs/connectors.md for detailed instructions
   ```

3. **Configure Source**
   ```bash
   curl -X POST http://localhost:8000/api/v1/sources \
     -H "Content-Type: application/json" \
     -d '{
       "platform": "reddit",
       "credentials": {
         "client_id": "your_client_id",
         "client_secret": "your_client_secret",
         "refresh_token": "your_refresh_token"
       }
     }'
   ```

### Option 3: YouTube

1. **Create Google Cloud Project**
   - Go to https://console.cloud.google.com
   - Create new project
   - Enable YouTube Data API v3
   - Create OAuth 2.0 credentials

2. **Get User Tokens**
   ```bash
   # Use Google's OAuth flow
   # See docs/connectors.md for detailed instructions
   ```

3. **Configure Source**
   ```bash
   curl -X POST http://localhost:8000/api/v1/sources \
     -H "Content-Type: application/json" \
     -d '{
       "platform": "youtube",
       "credentials": {
         "client_id": "your_client_id",
         "client_secret": "your_client_secret",
         "access_token": "your_access_token",
         "refresh_token": "your_refresh_token"
       }
     }'
   ```

## Get Your First Digest

Once you have at least one source configured:

```bash
# Trigger content fetch (normally runs automatically every 15 minutes)
docker-compose exec celery-worker celery -A app.ingestion.celery_app call app.ingestion.tasks.fetch_all_sources

# Wait a few minutes for processing

# Get your digest
curl http://localhost:8000/api/v1/digest/latest?hours=24&max_clusters=10
```

## Next Steps

### Customize Your Interests

Create an interest profile to get more relevant content:

```bash
curl -X POST http://localhost:8000/api/v1/profile/interests \
  -H "Content-Type: application/json" \
  -d '{
    "interest_topics": ["AI", "technology", "science", "space"],
    "negative_filters": ["sports", "celebrity"]
  }'
```

### Set Up MCP Integration

If you use Claude Desktop or another MCP-compatible client:

1. Add to your MCP configuration:
   ```json
   {
     "mcpServers": {
       "social-media-radar": {
         "command": "docker",
         "args": ["exec", "radar-api", "python", "-m", "app.mcp_server.server"]
       }
     }
   }
   ```

2. Restart your MCP client

3. Ask your AI assistant:
   > "Get my daily digest from Social Media Radar"

### Explore the API

Visit http://localhost:8000/docs for interactive API documentation.

Key endpoints:
- `GET /api/v1/sources` - List configured sources
- `GET /api/v1/digest/latest` - Get latest digest
- `POST /api/v1/search` - Search content
- `GET /api/v1/digest/history` - View past digests

## Troubleshooting

### Services won't start

```bash
# Check logs
docker-compose logs

# Restart services
docker-compose down
docker-compose up -d
```

### Database connection errors

```bash
# Ensure PostgreSQL is healthy
docker-compose ps postgres

# Check database logs
docker-compose logs postgres

# Recreate database
docker-compose down -v
docker-compose up -d
```

### No content appearing

1. Verify sources are configured: `curl http://localhost:8000/api/v1/sources`
2. Check Celery worker logs: `docker-compose logs celery-worker`
3. Manually trigger fetch: `docker-compose exec celery-worker celery call ...`

### API errors

1. Check API logs: `docker-compose logs api`
2. Verify environment variables: `docker-compose exec api env | grep API`
3. Test health endpoint: `curl http://localhost:8000/health`

## Getting Help

- 📖 [Full Documentation](../README.md)
- 🐛 [Report Issues](https://github.com/yourusername/social-media-radar/issues)
- 💬 [Discussions](https://github.com/yourusername/social-media-radar/discussions)

## What's Next?

- [Configure more sources](connectors.md)
- [Deploy to production](deployment.md)
- [Understand the architecture](architecture.md)
- [Set up MCP integration](mcp.md)

