"""Integration tests for output generation pipeline."""

import pytest
from datetime import datetime
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.models import Cluster, ContentItem, MediaType, SourcePlatform
from app.output.models import (
    OutputFormat,
    OutputPreferences,
    OutputRequest,
    TextStyle,
    TonePreference,
    LengthPreference,
)
from app.output.manager import OutputManager
from app.output.generators.text_generator import MarkdownGenerator
from app.output.generators.visual_generator import InfographicGenerator


@pytest.fixture
def sample_clusters():
    """Create sample clusters for testing."""
    user_id = uuid4()

    items1 = [
        ContentItem(
            user_id=user_id,
            source_platform=SourcePlatform.REDDIT,
            source_id="post1",
            source_url="https://reddit.com/r/test/post1",
            title="AI Breakthrough in Language Models",
            raw_text="Researchers announce new breakthrough in AI...",
            media_type=MediaType.TEXT,
            published_at=datetime.utcnow(),
        ),
        ContentItem(
            user_id=user_id,
            source_platform=SourcePlatform.YOUTUBE,
            source_id="video1",
            source_url="https://youtube.com/watch?v=video1",
            title="Explaining the Latest AI Research",
            media_type=MediaType.VIDEO,
            published_at=datetime.utcnow(),
        ),
    ]

    cluster1 = Cluster(
        user_id=user_id,
        topic="AI Research Advances",
        summary="Multiple sources report significant advances in AI research",
        items=items1,
        item_ids=[item.id for item in items1],
        relevance_score=0.92,
        platforms_represented=[SourcePlatform.REDDIT, SourcePlatform.YOUTUBE],
    )

    return [cluster1]


@pytest.fixture
def sample_preferences():
    """Create sample output preferences."""
    return OutputPreferences(
        user_id=uuid4(),
        name="Test Preferences",
        primary_format=OutputFormat.MARKDOWN,
        text_style=TextStyle.PROFESSIONAL,
        tone=TonePreference.NEUTRAL,
        length=LengthPreference.MEDIUM,
        include_sources=True,
        include_visualizations=False,
    )


@pytest.mark.asyncio
async def test_markdown_generation(sample_clusters, sample_preferences):
    """Test Markdown output generation."""
    with patch("app.output.generators.text_generator.OpenAILLMClient") as mock_llm:
        # Mock LLM response
        mock_llm.return_value.generate = AsyncMock(
            return_value=MagicMock(
                content="# AI Research Advances\n\nThis is a test summary...",
                model="gpt-4",
                usage={"total_tokens": 150},
            )
        )

        manager = OutputManager(llm_client=mock_llm.return_value)

        request = OutputRequest(
            user_id=sample_preferences.user_id,
            preferences_id=sample_preferences.id,
        )

        output = await manager.generate_output(
            request=request,
            preferences=sample_preferences,
            clusters=sample_clusters,
            items=sample_clusters[0].items,
        )

        assert output.format == OutputFormat.MARKDOWN
        assert output.success is True
        assert len(output.content) > 0
        assert output.metadata.word_count > 0
        assert "AI Research Advances" in output.content


@pytest.mark.asyncio
async def test_infographic_generation(sample_clusters, sample_preferences):
    """Test infographic generation."""
    # Update preferences for image output
    sample_preferences.primary_format = OutputFormat.IMAGE

    manager = OutputManager()

    request = OutputRequest(
        user_id=sample_preferences.user_id,
        preferences_id=sample_preferences.id,
    )

    output = await manager.generate_output(
        request=request,
        preferences=sample_preferences,
        clusters=sample_clusters,
        items=sample_clusters[0].items,
    )

    assert output.format == OutputFormat.IMAGE
    assert output.success is True
    assert len(output.content) > 0  # Base64 encoded image
    assert "infographic.png" in output.media_files
    assert output.metadata.file_size_bytes > 0


@pytest.mark.asyncio
async def test_multi_format_generation(sample_clusters, sample_preferences):
    """Test generating multiple formats concurrently."""
    with patch("app.output.generators.text_generator.OpenAILLMClient") as mock_llm:
        mock_llm.return_value.generate = AsyncMock(
            return_value=MagicMock(
                content="# Test Summary",
                model="gpt-4",
                usage={"total_tokens": 100},
            )
        )

        manager = OutputManager(llm_client=mock_llm.return_value)

        request = OutputRequest(
            user_id=sample_preferences.user_id,
            preferences_id=sample_preferences.id,
        )

        formats = [OutputFormat.MARKDOWN, OutputFormat.IMAGE]

        outputs = await manager.generate_multi_format(
            request=request,
            preferences=sample_preferences,
            clusters=sample_clusters,
            items=sample_clusters[0].items,
            formats=formats,
        )

        assert len(outputs) == 2
        assert OutputFormat.MARKDOWN in outputs
        assert OutputFormat.IMAGE in outputs


@pytest.mark.asyncio
async def test_quality_check_and_retry(sample_clusters, sample_preferences):
    """Test quality checking with retry logic."""
    with patch("app.output.generators.text_generator.OpenAILLMClient") as mock_llm:
        # First attempt returns low quality
        # Second attempt returns high quality
        mock_llm.return_value.generate = AsyncMock(
            side_effect=[
                MagicMock(
                    content="Short",  # Low quality
                    model="gpt-4",
                    usage={"total_tokens": 10},
                ),
                MagicMock(
                    content="# High Quality Summary\n\nThis is a much better summary with more detail and proper structure...",
                    model="gpt-4",
                    usage={"total_tokens": 200},
                ),
            ]
        )

        manager = OutputManager(llm_client=mock_llm.return_value)

        request = OutputRequest(
            user_id=sample_preferences.user_id,
            preferences_id=sample_preferences.id,
        )

        output = await manager.generate_with_quality_check(
            request=request,
            preferences=sample_preferences,
            clusters=sample_clusters,
            items=sample_clusters[0].items,
            min_quality_score=0.7,
            max_retries=3,
        )

        # Should have retried and gotten better output
        assert output.metadata.word_count > 10


@pytest.mark.asyncio
async def test_fallback_format_on_failure(sample_clusters, sample_preferences):
    """Test fallback to alternative format on failure."""
    sample_preferences.fallback_formats = [OutputFormat.PLAIN_TEXT, OutputFormat.JSON]

    with patch("app.output.generators.text_generator.MarkdownGenerator.generate") as mock_gen:
        # Make primary format fail
        mock_gen.side_effect = Exception("Generation failed")

        manager = OutputManager()

        request = OutputRequest(
            user_id=sample_preferences.user_id,
            preferences_id=sample_preferences.id,
        )

        # Should fall back to alternative format
        # In this test, we'll just verify the fallback logic is triggered
        with pytest.raises(ValueError, match="All output generation attempts failed"):
            await manager.generate_output(
                request=request,
                preferences=sample_preferences,
                clusters=sample_clusters,
                items=sample_clusters[0].items,
            )


@pytest.mark.asyncio
async def test_custom_prompt_integration(sample_clusters, sample_preferences):
    """Test custom prompt integration."""
    with patch("app.output.generators.text_generator.OpenAILLMClient") as mock_llm:
        mock_llm.return_value.generate = AsyncMock(
            return_value=MagicMock(
                content="Custom formatted output",
                model="gpt-4",
                usage={"total_tokens": 100},
            )
        )

        manager = OutputManager(llm_client=mock_llm.return_value)

        request = OutputRequest(
            user_id=sample_preferences.user_id,
            preferences_id=sample_preferences.id,
            custom_prompt="Focus on technical details and include code examples",
            focus_topics=["machine learning", "neural networks"],
        )

        output = await manager.generate_output(
            request=request,
            preferences=sample_preferences,
            clusters=sample_clusters,
            items=sample_clusters[0].items,
        )

        # Verify custom prompt was used
        call_args = mock_llm.return_value.generate.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "Focus on technical details" in prompt
        assert "machine learning" in prompt

