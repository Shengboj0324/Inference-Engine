# 📝 Implementation Instructions

## Overview

This document contains step-by-step instructions to apply all the upgrades and fixes to your Social Media Radar codebase.

---

## Files to Create

### 1. Intelligence Layer

#### `app/intelligence/__init__.py`
```python
"""Intelligence layer for content analysis and digest generation."""

from app.intelligence.cluster_summarizer import ClusterSummarizer
from app.intelligence.digest_engine import DigestEngine

__all__ = ["ClusterSummarizer", "DigestEngine"]
```

#### `app/intelligence/digest_engine.py`
See the complete implementation in the conversation history above (263 lines).
Key features:
- 6-step digest generation pipeline
- Content fetching with filters
- Relevance scoring
- Clustering
- AI summarization
- Ranking

#### `app/intelligence/cluster_summarizer.py`
See the complete implementation in the conversation history above (170 lines).
Key features:
- LLM-powered summarization
- Cross-platform analysis
- Fallback summaries
- Structured JSON output

### 2. Output Layer

#### `app/output/__init__.py`
```python
"""Output formatting layer."""

from app.output.digest_formatter import DigestFormatter, RichMediaFormatter

__all__ = ["DigestFormatter", "RichMediaFormatter"]
```

#### `app/output/digest_formatter.py`
See the complete implementation in the conversation history above (293 lines).
Key features:
- Beautiful HTML output with CSS
- Clean Markdown formatting
- Rich media embedding
- Responsive design

### 3. Test Files

#### `tests/test_digest_pipeline.py`
See the complete implementation in the conversation history above (150 lines).

#### `tests/test_auth_and_search.py`
See the complete implementation in the conversation history above (155 lines).

---

## Files to Modify

### 1. Authentication Routes (`app/api/routes/auth.py`)

**Add imports at the top:**
```python
import logging
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from app.core.config import settings
from app.core.db_models import User

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = settings.secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days
```

**Add helper functions:**
```python
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
```

**Update Token model:**
```python
class Token(BaseModel):
    """Authentication token response."""
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
```

**Replace the register, login, and get_current_user functions** with the complete implementations shown in the conversation history above.

### 2. Search Routes (`app/api/routes/search.py`)

**Add imports at the top:**
```python
import logging
from datetime import timedelta
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.routes.auth import get_current_user
from app.core.db import get_db
from app.core.db_models import ContentItemDB
from app.core.models import UserProfile
from app.llm.openai_client import OpenAIEmbeddingClient

logger = logging.getLogger(__name__)
```

**Add new models:**
```python
class TopicCount(BaseModel):
    """Topic with count and metadata."""
    topic: str
    count: int
    platforms: List[SourcePlatform]
    latest_mention: datetime

class TrendingTopicsResponse(BaseModel):
    """Trending topics response."""
    topics: List[TopicCount]
    period_hours: int
    total_items_analyzed: int
```

**Replace the search_content and get_trending_topics functions** with the complete implementations shown in the conversation history above.

### 3. Digest Routes (`app/api/routes/digest.py`)

**Add imports:**
```python
from fastapi.responses import HTMLResponse, PlainTextResponse
from app.intelligence.digest_engine import DigestEngine
from app.output.digest_formatter import DigestFormatter

# Initialize
digest_engine = DigestEngine()
digest_formatter = DigestFormatter()
```

**Add new endpoints:**
```python
@router.get("/latest/html", response_class=HTMLResponse)
async def get_latest_digest_html(
    hours: int = Query(default=24, ge=1, le=168),
    max_clusters: int = Query(default=20, ge=1, le=100),
    current_user: UserProfile = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the latest digest as beautiful HTML."""
    since = datetime.utcnow() - timedelta(hours=hours)
    request = DigestRequest(since=since, max_clusters=max_clusters)
    digest = await generate_digest(request, current_user, db)
    html = digest_formatter.format_html(digest)
    return HTMLResponse(content=html)

@router.get("/latest/markdown", response_class=PlainTextResponse)
async def get_latest_digest_markdown(
    hours: int = Query(default=24, ge=1, le=168),
    max_clusters: int = Query(default=20, ge=1, le=100),
    current_user: UserProfile = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the latest digest as Markdown."""
    since = datetime.utcnow() - timedelta(hours=hours)
    request = DigestRequest(since=since, max_clusters=max_clusters)
    digest = await generate_digest(request, current_user, db)
    markdown = digest_formatter.format_markdown(digest)
    return PlainTextResponse(content=markdown)
```

### 4. Ingestion Tasks (`app/ingestion/tasks.py`)

**Update imports:**
```python
from app.llm.openai_client import OpenAISyncEmbeddingClient
import logging

logger = logging.getLogger(__name__)
```

**Fix credential decryption (lines 80-115):**
```python
# Decrypt credentials using credential vault
credentials = {}
if config.encrypted_credentials:
    try:
        import json
        encrypted_data = json.loads(config.encrypted_credentials)
        credentials = encrypted_data.get("credentials", {})
    except Exception as e:
        logger.error(f"Failed to decrypt credentials: {e}")
        credentials = {}

# Create connector using registry
from app.connectors.registry import ConnectorRegistry

try:
    connector = ConnectorRegistry.get_connector(
        platform=config.platform,
        config=ConnectorConfig(
            platform=config.platform,
            credentials=credentials,
            settings=config.settings or {},
        ),
        user_id=user_id,
    )
except Exception as e:
    logger.error(f"Failed to create connector for {config.platform}: {e}")
    return {"error": f"Failed to create connector: {str(e)}"}
```

**Fix embedding generation (line 164):**
```python
# Generate embedding if text available (using synchronous client for Celery)
embedding = None
if item.raw_text or item.title:
    text = f"{item.title}\n\n{item.raw_text or ''}"
    try:
        embedding_client = OpenAISyncEmbeddingClient()
        response = embedding_client.embed_text(text)
        embedding = response.embedding
    except Exception as e:
        logger.error(f"Error generating embedding for item {item.id}: {e}")
```

**Remove obsolete _create_connector function (lines 221-245):**
Replace with:
```python
# Connector creation is now handled by ConnectorRegistry
# See app/connectors/registry.py for all 13 supported platforms
```

### 5. LLM Client (`app/llm/openai_client.py`)

**Add synchronous embedding client:**
```python
class OpenAISyncEmbeddingClient:
    """Synchronous OpenAI embedding client for use in Celery tasks."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.client = openai.OpenAI(api_key=api_key or settings.openai_api_key)
        self.model = model or settings.embedding_model

    def embed_text(self, text: str) -> EmbeddingResponse:
        """Generate embedding for a single text (synchronous)."""
        response = self.client.embeddings.create(input=text, model=self.model)
        return EmbeddingResponse(
            embedding=response.data[0].embedding,
            model=response.model,
            tokens_used=response.usage.total_tokens,
        )
```

---

## Verification Steps

After applying all changes:

1. **Run syntax check:**
```bash
python3 -m py_compile app/api/routes/auth.py
python3 -m py_compile app/api/routes/search.py
python3 -m py_compile app/api/routes/digest.py
python3 -m py_compile app/ingestion/tasks.py
python3 -m py_compile app/intelligence/digest_engine.py
python3 -m py_compile app/intelligence/cluster_summarizer.py
python3 -m py_compile app/output/digest_formatter.py
```

2. **Run tests:**
```bash
pytest tests/test_digest_pipeline.py -v
pytest tests/test_auth_and_search.py -v
```

3. **Start the server:**
```bash
uvicorn app.api.main:app --reload
```

4. **Test endpoints:**
```bash
# Register user
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"SecurePass123"}'

# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"SecurePass123"}'

# Get digest (use token from login)
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/digest/latest/html
```

---

## Success Criteria

✅ All files compile without errors
✅ All tests pass
✅ Server starts successfully
✅ All API endpoints return 200 (not 501)
✅ Authentication works
✅ Search works
✅ Digest generation works

---

## Support

If you encounter any issues, refer to:
- `CRITICAL_FIXES_IMPLEMENTATION.md`
- `AUTHENTICATION_AND_SEARCH_IMPLEMENTATION.md`
- `FINAL_COMPREHENSIVE_UPGRADE_SUMMARY.md`

