"""Tests for ranking and scoring algorithms."""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from app.core.models import ContentItem, MediaType, SourcePlatform, UserInterestProfile
from app.core.ranking import RelevanceScorer


def test_recency_score():
    """Test recency scoring."""
    scorer = RelevanceScorer()

    # Recent item (1 hour ago)
    recent_item = ContentItem(
        user_id=uuid4(),
        source_platform=SourcePlatform.REDDIT,
        source_id="recent",
        source_url="https://reddit.com/recent",
        title="Recent Post",
        media_type=MediaType.TEXT,
        published_at=datetime.utcnow() - timedelta(hours=1),
    )

    # Old item (7 days ago)
    old_item = ContentItem(
        user_id=uuid4(),
        source_platform=SourcePlatform.REDDIT,
        source_id="old",
        source_url="https://reddit.com/old",
        title="Old Post",
        media_type=MediaType.TEXT,
        published_at=datetime.utcnow() - timedelta(days=7),
    )

    recent_score = scorer._recency_score(recent_item.published_at)
    old_score = scorer._recency_score(old_item.published_at)

    assert recent_score > old_score
    assert 0 <= recent_score <= 1
    assert 0 <= old_score <= 1


def test_topic_match_score():
    """Test topic matching."""
    scorer = RelevanceScorer()

    item_topics = ["AI", "machine learning", "technology"]
    interest_topics = ["AI", "technology", "science"]

    score = scorer._topic_match_score(item_topics, interest_topics)

    # Should have 2/3 overlap (AI and technology)
    assert score > 0
    assert score <= 1


def test_engagement_score_reddit():
    """Test engagement scoring for Reddit content."""
    scorer = RelevanceScorer()

    metadata = {"score": 1000, "num_comments": 50, "upvote_ratio": 0.95}

    score = scorer._engagement_score(metadata)

    assert 0 <= score <= 1
    assert score > 0  # Should have some engagement


def test_relevance_scorer_with_profile():
    """Test complete relevance scoring with user profile."""
    user_id = uuid4()

    profile = UserInterestProfile(
        user_id=user_id,
        interest_topics=["AI", "technology"],
    )

    scorer = RelevanceScorer(interest_profile=profile)

    item = ContentItem(
        user_id=user_id,
        source_platform=SourcePlatform.REDDIT,
        source_id="test",
        source_url="https://reddit.com/test",
        title="AI Breakthrough",
        media_type=MediaType.TEXT,
        published_at=datetime.utcnow() - timedelta(hours=2),
        topics=["AI", "research"],
        metadata={"score": 500},
    )

    score = scorer.score_item(item)

    assert 0 <= score <= 1
    assert score > 0.3  # Should be reasonably relevant

