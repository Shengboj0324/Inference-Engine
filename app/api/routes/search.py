"""Content search routes."""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import get_current_user
from app.core.db import get_db
from app.core.db_models import ContentItemDB
from app.core.models import ContentItem, SourcePlatform, UserProfile
from app.llm.openai_client import OpenAIEmbeddingClient

logger = logging.getLogger(__name__)

router = APIRouter()


class SearchRequest(BaseModel):
    """Content search request."""

    query: str
    platforms: Optional[List[SourcePlatform]] = None
    since: Optional[datetime] = None
    limit: int = 50


class SearchResponse(BaseModel):
    """Content search response."""

    items: List[ContentItem]
    total: int
    query: str


@router.post("/", response_model=SearchResponse)
async def search_content(
    request: SearchRequest,
    current_user: UserProfile = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Search through user's content backlog using vector similarity.

    Args:
        request: Search parameters
        current_user: Authenticated user
        db: Database session

    Returns:
        Matching content items ranked by relevance
    """
    try:
        # Generate embedding for search query
        embedding_client = OpenAIEmbeddingClient()
        embedding_response = await embedding_client.embed_text(request.query)
        query_embedding = embedding_response.embedding

        # Build base query
        query = select(ContentItemDB).where(ContentItemDB.user_id == current_user.id)

        # Apply platform filter
        if request.platforms:
            query = query.where(ContentItemDB.source_platform.in_(request.platforms))

        # Apply time filter
        if request.since:
            query = query.where(ContentItemDB.published_at >= request.since)

        # Add vector similarity search using pgvector
        # Calculate cosine similarity: 1 - (embedding <=> query_embedding)
        if query_embedding:
            # Use pgvector's cosine distance operator (<=>)
            # Lower distance = higher similarity
            query = query.order_by(
                ContentItemDB.embedding.cosine_distance(query_embedding)
            ).limit(request.limit)
        else:
            # Fallback to recency if no embedding
            query = query.order_by(desc(ContentItemDB.published_at)).limit(request.limit)

        # Execute query
        result = await db.execute(query)
        db_items = result.scalars().all()

        # Convert to ContentItem models
        items = []
        for db_item in db_items:
            items.append(ContentItem(
                id=db_item.id,
                user_id=db_item.user_id,
                source_platform=db_item.source_platform,
                source_id=db_item.source_id,
                source_url=db_item.source_url,
                author=db_item.author,
                title=db_item.title,
                raw_text=db_item.raw_text,
                published_at=db_item.published_at,
                fetched_at=db_item.fetched_at,
                media_urls=db_item.media_urls or [],
                topics=db_item.topics or [],
                engagement_score=db_item.engagement_score,
            ))

        logger.info(f"Search query '{request.query}' returned {len(items)} results for user {current_user.id}")

        return SearchResponse(
            items=items,
            total=len(items),
            query=request.query,
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


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


@router.get("/topics", response_model=TrendingTopicsResponse)
async def get_trending_topics(
    hours: int = Query(default=24, ge=1, le=168),
    limit: int = Query(default=20, ge=1, le=100),
    current_user: UserProfile = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get trending topics from recent content.

    Args:
        hours: Number of hours to analyze
        limit: Maximum number of topics to return
        current_user: Authenticated user
        db: Database session

    Returns:
        List of trending topics with counts and metadata
    """
    try:
        # Calculate time window
        since = datetime.utcnow() - timedelta(hours=hours)

        # Query recent content items with topics
        query = select(ContentItemDB).where(
            ContentItemDB.user_id == current_user.id,
            ContentItemDB.published_at >= since,
            ContentItemDB.topics.isnot(None),
        )

        result = await db.execute(query)
        items = result.scalars().all()

        # Count topics across all items
        topic_data = {}
        for item in items:
            if not item.topics:
                continue

            for topic in item.topics:
                if topic not in topic_data:
                    topic_data[topic] = {
                        "count": 0,
                        "platforms": set(),
                        "latest_mention": item.published_at,
                    }

                topic_data[topic]["count"] += 1
                topic_data[topic]["platforms"].add(item.source_platform)

                # Update latest mention
                if item.published_at > topic_data[topic]["latest_mention"]:
                    topic_data[topic]["latest_mention"] = item.published_at

        # Sort by count (descending) and convert to response format
        sorted_topics = sorted(
            topic_data.items(),
            key=lambda x: (x[1]["count"], x[1]["latest_mention"]),
            reverse=True,
        )[:limit]

        # Build response
        topics = []
        for topic, data in sorted_topics:
            topics.append(TopicCount(
                topic=topic,
                count=data["count"],
                platforms=list(data["platforms"]),
                latest_mention=data["latest_mention"],
            ))

        logger.info(f"Trending topics analysis: {len(topics)} topics from {len(items)} items")

        return TrendingTopicsResponse(
            topics=topics,
            period_hours=hours,
            total_items_analyzed=len(items),
        )

    except Exception as e:
        logger.error(f"Trending topics error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trending topics: {str(e)}",
        )

