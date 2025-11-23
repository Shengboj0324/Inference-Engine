# Social Media Radar 🎯

> Your personal multi-channel intelligence desk: always listening, always ranking, delivering tight daily briefs with the content that matters.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

Social Media Radar is a **compliance-first, multi-channel intelligence aggregation system** that helps you stay informed across diverse content sources without drowning in information overload.

### Key Features

- 🔌 **13 Platform Connectors**: Reddit, YouTube, TikTok, Facebook, Instagram, WeChat, NYTimes, WSJ, ABC News, Google News, Apple News, RSS
- 🕷️ **Advanced Scraping**: Production-grade web scraping with anti-detection, proxy rotation, and compliance
- 🔐 **Privacy-First**: Your data, your tokens, your control
- 🤖 **AI-Powered**: Smart clustering, summarization, and relevance ranking
- 🎨 **Multi-Format Output**: 14 output formats including text, images, videos, infographics, podcasts
- 📊 **Cross-Platform Analysis**: See how different sources cover the same story
- 🎯 **Personalized**: Tailored to your interests and output preferences
- 🔧 **MCP Integration**: Works with Model Context Protocol clients
- 📦 **Production-Ready**: Kubernetes deployment, monitoring, CI/CD
- 🛡️ **Security Hardened**: Encryption, rate limiting, input validation

### How It Works

```
Your Sources → Ingestion → Normalization → AI Analysis → Daily Digest
(Reddit, YT)   (OAuth)     (Unified DB)    (Cluster+LLM)  (Ranked Brief)
```

1. **Connect** your accounts (Reddit, YouTube, RSS feeds, etc.)
2. **Configure** your interests and preferences
3. **Receive** daily digests with clustered, summarized content
4. **Explore** via API, MCP tools, or CLI

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- API keys for LLM providers (OpenAI, Anthropic, or local models)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/social-media-radar.git
   cd social-media-radar
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Start services**
   ```bash
   docker-compose up -d
   ```

4. **Run migrations**
   ```bash
   docker-compose exec api alembic upgrade head
   ```

5. **Access the API**
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## Architecture

Social Media Radar is built on five core layers:

### 1. Ingestion & Connectors

**Social Media Platforms (6)**:
- Reddit (PRAW + OAuth 2.0)
- YouTube (Data API v3 + OAuth 2.0)
- TikTok (Research API + OAuth 2.0)
- Facebook (Graph API v21.0 + OAuth 2.0)
- Instagram (Graph API + OAuth 2.0)
- WeChat (Official Account API + OAuth 2.0)

**News Sources (7)**:
- New York Times (Official API)
- Wall Street Journal (RSS Feeds)
- ABC News US & Australia (RSS Feeds)
- Google News (RSS + Advanced Search)
- Apple News (Web Scraping with Compliance)
- Generic RSS/Atom Feeds

**Features**:
- Platform-specific connectors using official APIs
- OAuth 2.0 authentication for social platforms
- Scheduled fetching with Celery

### 2. Normalization & Storage
- Unified `ContentItem` schema
- PostgreSQL with pgvector for embeddings
- MinIO/S3 for media storage

### 3. Intelligence & Relevance
- User interest profiling
- Embedding-based similarity
- Content clustering (HDBSCAN)
- Multi-signal ranking

### 4. Summarization & Generation
- Multi-document summarization
- Cross-platform perspective analysis
- Daily digest generation
- Custom content generation

### 5. Delivery
- REST API (FastAPI)
- MCP server for AI assistants
- CLI tools
- Future: Email, Slack, webhooks

## Usage

### Configure a Source

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

### Get Daily Digest

```bash
curl http://localhost:8000/api/v1/digest/latest?hours=24&max_clusters=20
```

### Search Content

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence policy",
    "limit": 50
  }'
```

## MCP Integration

Social Media Radar exposes MCP tools for use with compatible AI assistants:

- `get_daily_digest` - Get personalized daily briefing
- `search_content` - Search your content backlog
- `configure_source` - Add/update content sources
- `list_sources` - View configured sources
- `get_cluster_detail` - Deep dive into a topic cluster

## Development

### Local Setup

```bash
# Install dependencies
poetry install

# Start PostgreSQL and Redis
docker-compose up -d postgres redis

# Run migrations
poetry run alembic upgrade head

# Start API server
poetry run uvicorn app.api.main:app --reload

# Start Celery worker
poetry run celery -A app.ingestion.celery_app worker --loglevel=info
```

### Running Tests

```bash
poetry run pytest
poetry run pytest --cov=app tests/
```

### Code Quality

```bash
# Format code
poetry run black app tests

# Lint
poetry run ruff app tests

# Type check
poetry run mypy app
```

## Documentation

- [Architecture](docs/architecture.md) - System design and components
- [Connectors](docs/connectors.md) - Platform integration guide
- [Deployment](docs/deployment.md) - Production deployment guide
- [API Reference](http://localhost:8000/docs) - Interactive API docs

## Roadmap

### Phase 1: MVP (Current)
- [x] Core architecture and data models
- [x] Reddit, YouTube, RSS connectors
- [x] Basic clustering and ranking
- [x] FastAPI backend
- [x] MCP server foundation
- [ ] LLM summarization
- [ ] User authentication

### Phase 2: Enhanced Intelligence
- [ ] TikTok, Facebook, Instagram connectors
- [ ] Video transcription (Whisper)
- [ ] Advanced personalization
- [ ] Feedback loops
- [ ] Multi-persona support

### Phase 3: Scale & Polish
- [ ] Real-time updates
- [ ] Email/Slack delivery
- [ ] Mobile app
- [ ] Advanced analytics
- [ ] Community connectors

## Contributing

Contributions are welcome! Please read our contributing guidelines and code of conduct.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Compliance & Ethics

Social Media Radar is designed with compliance and ethics as core principles:

- ✅ Uses only official APIs and authorized access
- ✅ Respects platform Terms of Service
- ✅ No paywall bypassing or scraping
- ✅ User-owned credentials and data
- ✅ Privacy-preserving architecture
- ✅ Transparent data handling
- ✅ GDPR-compliant export/delete

## Support

- 📖 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/yourusername/social-media-radar/issues)
- 💬 [Discussions](https://github.com/yourusername/social-media-radar/discussions)

## Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [SQLAlchemy](https://www.sqlalchemy.org/) - Database ORM
- [Celery](https://docs.celeryq.dev/) - Distributed task queue
- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity search
- [OpenAI](https://openai.com/) - LLM and embeddings
- [PRAW](https://praw.readthedocs.io/) - Reddit API wrapper

---

**Note**: This is an early-stage project. APIs and features may change. Use in production at your own risk.