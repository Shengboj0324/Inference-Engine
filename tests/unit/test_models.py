"""Tests for core data models."""

from datetime import datetime
from uuid import uuid4

import pytest

from app.core.models import (
    ContentItem,
    MediaType,
    SourcePlatform,
    UserInterestProfile,
    Cluster,
)


def test_content_item_creation():
    """Test creating a ContentItem."""
    user_id = uuid4()
    item = ContentItem(
        user_id=user_id,
        source_platform=SourcePlatform.REDDIT,
        source_id="test123",
        source_url="https://reddit.com/r/test/comments/test123",
        title="Test Post",
        media_type=MediaType.TEXT,
        published_at=datetime.utcnow(),
    )

    assert item.user_id == user_id
    assert item.source_platform == SourcePlatform.REDDIT
    assert item.title == "Test Post"
    assert item.media_type == MediaType.TEXT


def test_content_item_with_topics():
    """Test ContentItem with topics."""
    item = ContentItem(
        user_id=uuid4(),
        source_platform=SourcePlatform.YOUTUBE,
        source_id="video123",
        source_url="https://youtube.com/watch?v=video123",
        title="AI Tutorial",
        media_type=MediaType.VIDEO,
        published_at=datetime.utcnow(),
        topics=["AI", "machine learning", "tutorial"],
    )

    assert len(item.topics) == 3
    assert "AI" in item.topics


def test_user_interest_profile():
    """Test UserInterestProfile creation."""
    user_id = uuid4()
    profile = UserInterestProfile(
        user_id=user_id,
        interest_topics=["AI", "technology", "science"],
        negative_filters=["sports", "celebrity"],
    )

    assert profile.user_id == user_id
    assert len(profile.interest_topics) == 3
    assert len(profile.negative_filters) == 2


def test_cluster_creation():
    """Test Cluster creation."""
    user_id = uuid4()
    items = [
        ContentItem(
            user_id=user_id,
            source_platform=SourcePlatform.REDDIT,
            source_id=f"post{i}",
            source_url=f"https://reddit.com/post{i}",
            title=f"Post {i}",
            media_type=MediaType.TEXT,
            published_at=datetime.utcnow(),
        )
        for i in range(3)
    ]

    cluster = Cluster(
        user_id=user_id,
        topic="AI News",
        summary="Latest developments in AI",
        items=items,
        item_ids=[item.id for item in items],
        relevance_score=0.85,
        platforms_represented=[SourcePlatform.REDDIT],
    )

    assert cluster.topic == "AI News"
    assert len(cluster.items) == 3
    assert cluster.relevance_score == 0.85
    assert SourcePlatform.REDDIT in cluster.platforms_represented

