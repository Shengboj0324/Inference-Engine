# Social Media Radar - System Architecture

## Overview

Social Media Radar is a compliance-first, multi-channel intelligence aggregation system that provides personalized daily briefings from diverse content sources. The system operates as a **user-authorized aggregator**, not a scraper, ensuring all data access is properly licensed and authenticated.

## Design Principles

1. **Compliance-First**: Only use official APIs, RSS feeds, or properly licensed sources
2. **User-Owned Auth**: Users provide their own OAuth tokens, API keys, and subscriptions
3. **Privacy-Preserving**: Per-user data segregation, encrypted credentials, audit trails
4. **Pluggable Architecture**: Easy to add new sources without modifying core logic
5. **LLM-Agnostic**: Support both hosted (OpenAI/Anthropic) and self-hosted models

## System Layers

### 1. Ingestion & Connectors Layer
- **Purpose**: Pull content from each platform using user-provided authentication
- **Components**:
  - Connector Framework (base classes and interfaces)
  - Platform-specific connectors (Reddit, YouTube, TikTok, etc.)
  - Scheduler for periodic fetching
  - Rate limiting and error handling

### 2. Normalization & Storage Layer
- **Purpose**: Unify diverse content into a single schema
- **Components**:
  - ContentItem unified model
  - Postgres for metadata and relationships
  - pgvector for embeddings
  - MinIO/S3 for media and transcripts

### 3. Intelligence & Relevance Layer
- **Purpose**: Determine what content matters to each user
- **Components**:
  - User interest profiling
  - Relevance scoring engine
  - Content clustering (HDBSCAN/hierarchical)
  - Deduplication and diversity enforcement

### 4. Summarization & Generation Layer
- **Purpose**: Create actionable insights from raw content
- **Components**:
  - Multi-document summarization
  - Cross-platform perspective analysis
  - Daily digest generation
  - Custom content generation (ELI5, contrarian views, etc.)

### 5. Delivery Layer
- **Purpose**: Expose intelligence through multiple interfaces
- **Components**:
  - FastAPI REST API
  - MCP (Model Context Protocol) server
  - CLI tools
  - Future: Email/Slack/webhook delivery

## Data Flow

```
User Profile + Auth
       ↓
Scheduled Ingestion Jobs
       ↓
Raw Content → Queue (Redis Streams)
       ↓
Normalization & Enrichment
  - Language detection
  - NLP (NER, keywords, topics)
  - Embedding generation
  - Transcript extraction (ASR if needed)
       ↓
ContentItem Storage (Postgres + pgvector)
       ↓
Relevance Filtering & Ranking
  - User interest matching
  - Recency weighting
  - Engagement signals
       ↓
Clustering & Storyline Detection
       ↓
LLM Summarization
       ↓
Daily Digest / API Response
```

## Core Data Models

### ContentItem
```python
- id: UUID
- source_platform: str (reddit, youtube, tiktok, etc.)
- source_id: str (native platform ID)
- author: str
- channel/feed: str
- title: str
- raw_text: str
- media_type: enum (text, video, image, mixed)
- media_urls: list[str]
- published_at: datetime
- fetched_at: datetime
- topics: list[str]
- lang: str
- embedding: vector(1536)
- metadata: jsonb
```

### UserProfile
```python
- id: UUID
- interest_topics: list[str]
- negative_filters: list[str]
- interest_embedding: vector(1536)
- platform_configs: jsonb
```

### Cluster
```python
- id: UUID
- user_id: UUID
- created_at: datetime
- topic: str
- summary: str
- items: list[ContentItem]
- relevance_score: float
```

## Technology Stack

### Backend
- **Language**: Python 3.11+
- **API Framework**: FastAPI
- **Task Queue**: Celery + Redis
- **Scheduler**: Celery Beat
- **Database**: PostgreSQL 15+ with pgvector
- **Vector Store**: pgvector (primary), Qdrant (optional)
- **Object Storage**: MinIO (local) / S3 (production)
- **Message Queue**: Redis Streams

### AI/ML
- **Embeddings**: OpenAI text-embedding-3-large (default), BGE-m3 (OSS)
- **LLM**: OpenAI/Anthropic (default), Llama 3.x via vLLM (OSS)
- **NLP**: spaCy, transformers
- **ASR**: OpenAI Whisper (for video transcription)
- **Clustering**: scikit-learn (HDBSCAN, hierarchical)

### Infrastructure
- **Containerization**: Docker + Docker Compose
- **Orchestration**: Kubernetes (optional, for scale)
- **Monitoring**: Prometheus + Grafana
- **Logging**: Structured logging with Python logging

## Security & Privacy

### Authentication & Authorization
- OAuth 2.0 for platform integrations
- API key management with encryption at rest
- Per-user token isolation
- No shared credentials

### Data Protection
- Encrypted credential storage (libsodium/KMS)
- Per-user data segregation
- Audit logging for all data access
- GDPR-compliant export/delete endpoints

### Compliance
- No ToS violations
- No paywall bypassing
- Fair use for transcription
- Rate limit compliance
- User consent for all data collection

