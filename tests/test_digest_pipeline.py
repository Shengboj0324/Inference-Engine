"""Integration tests for the complete digest generation pipeline."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from app.core.models import (
    ContentItem,
    Cluster,
    DigestRequest,
    DigestResponse,
    SourcePlatform,
)
from app.intelligence.digest_engine import DigestEngine
from app.intelligence.cluster_summarizer import ClusterSummarizer
from app.output.digest_formatter import DigestFormatter
from app.llm.openai_client import OpenAILLMClient


class TestDigestPipeline:
    """Test the complete digest generation pipeline."""

    def test_digest_engine_initialization(self):
        """Test that DigestEngine initializes correctly."""
        engine = DigestEngine()
        assert engine is not None
        assert engine.llm_client is not None
        assert engine.cluster_summarizer is not None

    def test_cluster_summarizer_initialization(self):
        """Test that ClusterSummarizer initializes correctly."""
        llm_client = OpenAILLMClient()
        summarizer = ClusterSummarizer(llm_client)
        assert summarizer is not None
        assert summarizer.llm_client is not None

    def test_digest_formatter_html(self):
        """Test HTML formatting of digest."""
        formatter = DigestFormatter()
        
        # Create mock digest
        digest = DigestResponse(
            period_start=datetime.utcnow() - timedelta(hours=24),
            period_end=datetime.utcnow(),
            clusters=[],
            total_items=0,
            summary="Test summary",
        )
        
        html = formatter.format_html(digest)
        
        # Verify HTML structure
        assert "<!DOCTYPE html>" in html
        assert "Your Daily Digest" in html
        assert "Test summary" in html
        assert "</html>" in html

    def test_digest_formatter_markdown(self):
        """Test Markdown formatting of digest."""
        formatter = DigestFormatter()
        
        # Create mock digest
        digest = DigestResponse(
            period_start=datetime.utcnow() - timedelta(hours=24),
            period_end=datetime.utcnow(),
            clusters=[],
            total_items=0,
            summary="Test summary",
        )
        
        markdown = formatter.format_markdown(digest)
        
        # Verify Markdown structure
        assert "# 📰 Your Daily Digest" in markdown
        assert "Test summary" in markdown
        assert "**Total Items:**" in markdown

    def test_cluster_formatting_html(self):
        """Test HTML formatting of clusters."""
        formatter = DigestFormatter()
        
        # Create mock cluster
        item = ContentItem(
            id=uuid4(),
            user_id=uuid4(),
            source_platform=SourcePlatform.REDDIT,
            source_id="test123",
            source_url="https://reddit.com/r/test/123",
            author="test_user",
            title="Test Post Title",
            raw_text="This is a test post content.",
            published_at=datetime.utcnow(),
            fetched_at=datetime.utcnow(),
        )
        
        cluster = Cluster(
            user_id=uuid4(),
            topic="Test Topic",
            summary="This is a test cluster summary.",
            keywords=["test", "cluster", "summary"],
            item_ids=[item.id],
            items=[item],
            relevance_score=0.85,
            platforms_represented=[SourcePlatform.REDDIT],
        )
        
        digest = DigestResponse(
            period_start=datetime.utcnow() - timedelta(hours=24),
            period_end=datetime.utcnow(),
            clusters=[cluster],
            total_items=1,
            summary="Test digest with one cluster",
        )
        
        html = formatter.format_html(digest)
        
        # Verify cluster content
        assert "Test Topic" in html
        assert "Test Post Title" in html
        assert "test_user" in html
        assert "This is a test cluster summary" in html

    def test_cluster_formatting_markdown(self):
        """Test Markdown formatting of clusters."""
        formatter = DigestFormatter()
        
        # Create mock cluster
        item = ContentItem(
            id=uuid4(),
            user_id=uuid4(),
            source_platform=SourcePlatform.YOUTUBE,
            source_id="video123",
            source_url="https://youtube.com/watch?v=123",
            author="Test Channel",
            title="Test Video Title",
            raw_text="Video description here.",
            published_at=datetime.utcnow(),
            fetched_at=datetime.utcnow(),
        )
        
        cluster = Cluster(
            user_id=uuid4(),
            topic="Video Topic",
            summary="Summary of video content.",
            keywords=["video", "content", "test"],
            item_ids=[item.id],
            items=[item],
            relevance_score=0.92,
            platforms_represented=[SourcePlatform.YOUTUBE],
        )
        
        digest = DigestResponse(
            period_start=datetime.utcnow() - timedelta(hours=24),
            period_end=datetime.utcnow(),
            clusters=[cluster],
            total_items=1,
            summary="Test digest with video cluster",
        )
        
        markdown = formatter.format_markdown(digest)
        
        # Verify cluster content
        assert "Video Topic" in markdown
        assert "Test Video Title" in markdown
        assert "Test Channel" in markdown
        assert "Summary of video content" in markdown
        assert "youtube" in markdown.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

