"""Daily digest routes."""

import logging
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import HTMLResponse, PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import get_current_user
from app.core.db import get_db
from app.core.models import DigestRequest, DigestResponse, SourcePlatform, UserProfile
from app.intelligence.digest_engine import DigestEngine
from app.output.digest_formatter import DigestFormatter

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency providers — instances created on first request, not at import.
# This prevents OpenAIError crashes when the module is loaded without an
# OPENAI_API_KEY set in the environment (e.g. during testing or local dev
# without an LLM key configured).
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_digest_engine() -> DigestEngine:
    """Singleton DigestEngine; constructed on first HTTP request."""
    return DigestEngine()


@lru_cache(maxsize=1)
def _get_digest_formatter() -> DigestFormatter:
    """Singleton DigestFormatter; constructed on first HTTP request."""
    return DigestFormatter()


def get_digest_engine() -> DigestEngine:
    """FastAPI dependency: return the cached DigestEngine instance."""
    return _get_digest_engine()


def get_digest_formatter() -> DigestFormatter:
    """FastAPI dependency: return the cached DigestFormatter instance."""
    return _get_digest_formatter()


@router.post("/generate", response_model=DigestResponse)
async def generate_digest(
    request: DigestRequest,
    current_user: UserProfile = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    engine: DigestEngine = Depends(get_digest_engine),
):
    """Generate a personalized daily digest.

    Args:
        request: Digest generation parameters
        current_user: Authenticated user
        db: Database session
        engine: Injected DigestEngine instance

    Returns:
        Generated digest with clusters and summaries
    """
    try:
        digest = await engine.generate_digest(
            user_id=current_user.id,
            request=request,
            db=db,
        )
        return digest

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate digest: {str(e)}",
        )


@router.get("/latest", response_model=DigestResponse)
async def get_latest_digest(
    hours: int = Query(default=24, ge=1, le=168),
    max_clusters: int = Query(default=20, ge=1, le=100),
    current_user: UserProfile = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    engine: DigestEngine = Depends(get_digest_engine),
):
    """Get the latest digest for the user."""
    since = datetime.utcnow() - timedelta(hours=hours)
    request = DigestRequest(since=since, max_clusters=max_clusters)
    return await generate_digest(request, current_user, db, engine)


@router.get("/latest/html", response_class=HTMLResponse)
async def get_latest_digest_html(
    hours: int = Query(default=24, ge=1, le=168),
    max_clusters: int = Query(default=20, ge=1, le=100),
    current_user: UserProfile = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    engine: DigestEngine = Depends(get_digest_engine),
    formatter: DigestFormatter = Depends(get_digest_formatter),
):
    """Get the latest digest as HTML."""
    since = datetime.utcnow() - timedelta(hours=hours)
    request = DigestRequest(since=since, max_clusters=max_clusters)
    digest = await generate_digest(request, current_user, db, engine)
    html = formatter.format_html(digest)
    return HTMLResponse(content=html)


@router.get("/latest/markdown", response_class=PlainTextResponse)
async def get_latest_digest_markdown(
    hours: int = Query(default=24, ge=1, le=168),
    max_clusters: int = Query(default=20, ge=1, le=100),
    current_user: UserProfile = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    engine: DigestEngine = Depends(get_digest_engine),
    formatter: DigestFormatter = Depends(get_digest_formatter),
):
    """Get the latest digest as Markdown."""
    since = datetime.utcnow() - timedelta(hours=hours)
    request = DigestRequest(since=since, max_clusters=max_clusters)
    digest = await generate_digest(request, current_user, db, engine)
    markdown = formatter.format_markdown(digest)
    return PlainTextResponse(content=markdown)


@router.get("/history")
async def get_digest_history(
    limit: int = Query(default=10, ge=1, le=100),
    current_user: UserProfile = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get historical digests for the user.

    Args:
        limit: Number of historical digests to return
        current_user: Authenticated user
        db: Database session

    Returns:
        List of historical digests
    """
    try:
        from app.core.db_models import DigestDB
        from sqlalchemy import select, desc

        # Query stored digests from database
        result = await db.execute(
            select(DigestDB)
            .where(DigestDB.user_id == current_user.id)
            .order_by(desc(DigestDB.created_at))
            .limit(limit)
        )
        digests = result.scalars().all()

        # Convert to response format
        return [
            {
                "id": str(digest.id),
                "title": digest.title or "Daily Digest",
                "created_at": digest.created_at.isoformat(),
                "summary": digest.summary,
                "item_count": len(digest.items) if digest.items else 0,
            }
            for digest in digests
        ]

    except Exception as e:
        logger.error(f"Failed to fetch digest history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve digest history",
        )

