"""End-to-end tests for complete pipeline."""

import pytest
from datetime import datetime
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.models import (
    ContentItem,
    DigestRequest,
    MediaType,
    SourcePlatform,
    UserInterestProfile,
)
from app.core.ranking import ContentClusterer, RelevanceScorer
from app.output.manager import OutputManager
from app.output.models import OutputFormat, OutputPreferences, OutputRequest


@pytest.mark.asyncio
async def test_complete_digest_pipeline():
    """Test complete pipeline from content ingestion to output generation."""
    user_id = uuid4()

    # Step 1: Create user profile
    user_profile = UserInterestProfile(
        user_id=user_id,
        topics=["artificial intelligence", "machine learning", "technology"],
        keywords=["AI", "ML", "neural networks", "deep learning"],
        preferred_sources=[SourcePlatform.REDDIT, SourcePlatform.YOUTUBE],
    )

    # Step 2: Simulate content items
    content_items = [
        ContentItem(
            user_id=user_id,
            source_platform=SourcePlatform.REDDIT,
            source_id="post1",
            source_url="https://reddit.com/r/MachineLearning/post1",
            title="New Breakthrough in Neural Network Architecture",
            raw_text="Researchers have developed a new neural network architecture...",
            media_type=MediaType.TEXT,
            published_at=datetime.utcnow(),
        ),
        ContentItem(
            user_id=user_id,
            source_platform=SourcePlatform.YOUTUBE,
            source_id="video1",
            source_url="https://youtube.com/watch?v=video1",
            title="Explaining Transformers in Deep Learning",
            raw_text="This video explains how transformer models work...",
            media_type=MediaType.VIDEO,
            published_at=datetime.utcnow(),
        ),
        ContentItem(
            user_id=user_id,
            source_platform=SourcePlatform.RSS,
            source_id="article1",
            source_url="https://techcrunch.com/article1",
            title="AI Startup Raises $100M",
            raw_text="A new AI startup focused on enterprise solutions...",
            media_type=MediaType.TEXT,
            published_at=datetime.utcnow(),
        ),
    ]

    # Step 3: Score relevance
    with patch("app.core.ranking.OpenAILLMClient") as mock_llm:
        # Mock embeddings
        mock_llm.return_value.get_embedding = AsyncMock(
            return_value=[0.1] * 1536  # Mock embedding vector
        )

        scorer = RelevanceScorer(llm_client=mock_llm.return_value)

        # Score items
        for item in content_items:
            score = await scorer.score_item(item, user_profile)
            item.relevance_score = score

        # All items should have relevance scores
        assert all(item.relevance_score is not None for item in content_items)
        assert all(0 <= item.relevance_score <= 1 for item in content_items)

    # Step 4: Cluster content
    with patch("app.core.ranking.OpenAILLMClient") as mock_llm:
        mock_llm.return_value.get_embedding = AsyncMock(
            return_value=[0.1] * 1536
        )
        mock_llm.return_value.generate = AsyncMock(
            return_value=MagicMock(
                content="AI and Machine Learning Advances",
                model="gpt-4",
            )
        )

        clusterer = ContentClusterer(llm_client=mock_llm.return_value)

        clusters = await clusterer.cluster_items(content_items, user_id)

        # Should have created clusters
        assert len(clusters) > 0
        assert all(cluster.topic for cluster in clusters)
        assert all(cluster.summary for cluster in clusters)

    # Step 5: Generate output
    with patch("app.output.generators.text_generator.OpenAILLMClient") as mock_llm:
        mock_llm.return_value.generate = AsyncMock(
            return_value=MagicMock(
                content="# Daily Intelligence Digest\n\n## AI and Machine Learning\n\nToday's top stories...",
                model="gpt-4",
                usage={"total_tokens": 500},
            )
        )

        output_manager = OutputManager(llm_client=mock_llm.return_value)

        preferences = OutputPreferences(
            user_id=user_id,
            name="Default",
            primary_format=OutputFormat.MARKDOWN,
        )

        request = OutputRequest(
            user_id=user_id,
            preferences_id=preferences.id,
        )

        output = await output_manager.generate_output(
            request=request,
            preferences=preferences,
            clusters=clusters,
            items=content_items,
        )

        # Verify output
        assert output.success is True
        assert output.format == OutputFormat.MARKDOWN
        assert len(output.content) > 0
        assert "Daily Intelligence Digest" in output.content


@pytest.mark.asyncio
async def test_error_recovery_in_pipeline():
    """Test error recovery and graceful degradation."""
    user_id = uuid4()

    content_items = [
        ContentItem(
            user_id=user_id,
            source_platform=SourcePlatform.REDDIT,
            source_id="post1",
            source_url="https://reddit.com/r/test/post1",
            title="Test Post",
            raw_text="Test content",
            media_type=MediaType.TEXT,
            published_at=datetime.utcnow(),
        ),
    ]

    # Test LLM failure with fallback
    with patch("app.output.generators.text_generator.OpenAILLMClient") as mock_llm:
        # First call fails, second succeeds
        mock_llm.return_value.generate = AsyncMock(
            side_effect=[
                Exception("API timeout"),
                MagicMock(
                    content="Fallback summary",
                    model="gpt-4",
                    usage={"total_tokens": 100},
                ),
            ]
        )

        output_manager = OutputManager(llm_client=mock_llm.return_value)

        preferences = OutputPreferences(
            user_id=user_id,
            name="Test",
            primary_format=OutputFormat.MARKDOWN,
        )

        request = OutputRequest(
            user_id=user_id,
            preferences_id=preferences.id,
        )

        # Should retry and succeed
        with patch("app.core.ranking.OpenAILLMClient") as mock_cluster_llm:
            mock_cluster_llm.return_value.get_embedding = AsyncMock(
                return_value=[0.1] * 1536
            )
            mock_cluster_llm.return_value.generate = AsyncMock(
                return_value=MagicMock(content="Test Topic", model="gpt-4")
            )

            clusterer = ContentClusterer(llm_client=mock_cluster_llm.return_value)
            clusters = await clusterer.cluster_items(content_items, user_id)

            # This should succeed on retry
            output = await output_manager.generate_output(
                request=request,
                preferences=preferences,
                clusters=clusters,
                items=content_items,
            )

            assert output.success is True


@pytest.mark.asyncio
async def test_multi_platform_aggregation():
    """Test aggregation from multiple platforms."""
    user_id = uuid4()

    # Content from different platforms
    platforms = [
        SourcePlatform.REDDIT,
        SourcePlatform.YOUTUBE,
        SourcePlatform.RSS,
        SourcePlatform.TWITTER,
    ]

    content_items = []
    for i, platform in enumerate(platforms):
        item = ContentItem(
            user_id=user_id,
            source_platform=platform,
            source_id=f"item{i}",
            source_url=f"https://example.com/item{i}",
            title=f"Content from {platform.value}",
            raw_text=f"This is content from {platform.value}",
            media_type=MediaType.TEXT,
            published_at=datetime.utcnow(),
        )
        content_items.append(item)

    # Cluster should identify multi-platform coverage
    with patch("app.core.ranking.OpenAILLMClient") as mock_llm:
        mock_llm.return_value.get_embedding = AsyncMock(
            return_value=[0.1] * 1536
        )
        mock_llm.return_value.generate = AsyncMock(
            return_value=MagicMock(
                content="Multi-Platform Topic",
                model="gpt-4",
            )
        )

        clusterer = ContentClusterer(llm_client=mock_llm.return_value)
        clusters = await clusterer.cluster_items(content_items, user_id)

        # Should have identified platforms
        for cluster in clusters:
            assert len(cluster.platforms_represented) > 0

