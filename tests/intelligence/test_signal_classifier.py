"""Unit tests for signal classifier.

Tests the core signal classification logic including:
- Pattern matching
- LLM classification
- Signal creation
- Scoring integration
"""

import pytest
from datetime import datetime
from uuid import uuid4

from app.core.models import ContentItem, SourcePlatform, MediaType
from app.core.signal_models import SignalType, ActionType, ResponseTone
from app.intelligence.signal_classifier import SignalClassifier


@pytest.fixture
def classifier():
    """Create signal classifier without LLM for testing."""
    return SignalClassifier(use_llm=False, min_confidence=0.7)


@pytest.fixture
def lead_opportunity_item():
    """Create content item representing lead opportunity."""
    user_id = uuid4()
    return ContentItem(
        id=uuid4(),
        user_id=user_id,
        source_platform=SourcePlatform.REDDIT,
        source_id="reddit_123",
        source_url="https://reddit.com/r/saas/comments/123",
        title="Looking for alternatives to Slack",
        raw_text="We're a team of 20 and looking for alternatives to Slack. "
                 "Need something with better pricing and integrations. "
                 "Any recommendations?",
        author="tech_startup_ceo",
        media_type=MediaType.TEXT,
        published_at=datetime.utcnow(),
    )


@pytest.fixture
def competitor_weakness_item():
    """Create content item representing competitor weakness."""
    user_id = uuid4()
    return ContentItem(
        id=uuid4(),
        user_id=user_id,
        source_platform=SourcePlatform.REDDIT,
        source_id="reddit_456",
        source_url="https://reddit.com/r/saas/comments/456",
        title="Terrible customer support from Zendesk",
        raw_text="Been waiting 3 days for a response from Zendesk support. "
                 "This is ridiculous for a paid product. "
                 "Their support is terrible and pricing is too high.",
        author="frustrated_user",
        media_type=MediaType.TEXT,
        published_at=datetime.utcnow(),
    )


@pytest.fixture
def product_confusion_item():
    """Create content item representing product confusion."""
    user_id = uuid4()
    return ContentItem(
        id=uuid4(),
        user_id=user_id,
        source_platform=SourcePlatform.REDDIT,
        source_id="reddit_789",
        source_url="https://reddit.com/r/programming/comments/789",
        title="How do I configure SSO in Okta?",
        raw_text="I'm confused about how to setup SSO with Okta. "
                 "The documentation is unclear. Can anyone help?",
        author="developer123",
        media_type=MediaType.TEXT,
        published_at=datetime.utcnow(),
    )


class TestPatternMatching:
    """Test pattern matching functionality."""
    
    def test_lead_opportunity_pattern_match(self, classifier, lead_opportunity_item):
        """Test that lead opportunity patterns are detected."""
        matches = classifier._pattern_match(lead_opportunity_item)
        
        assert len(matches) > 0
        assert matches[0][0] == SignalType.LEAD_OPPORTUNITY
        assert matches[0][1] > 0.5  # Confidence should be reasonable
    
    def test_competitor_weakness_pattern_match(self, classifier, competitor_weakness_item):
        """Test that competitor weakness patterns are detected."""
        matches = classifier._pattern_match(competitor_weakness_item)
        
        assert len(matches) > 0
        assert SignalType.COMPETITOR_WEAKNESS in [m[0] for m in matches]
    
    def test_product_confusion_pattern_match(self, classifier, product_confusion_item):
        """Test that product confusion patterns are detected."""
        matches = classifier._pattern_match(product_confusion_item)
        
        assert len(matches) > 0
        assert SignalType.PRODUCT_CONFUSION in [m[0] for m in matches]
    
    def test_no_match_for_irrelevant_content(self, classifier):
        """Test that irrelevant content doesn't match."""
        user_id = uuid4()
        item = ContentItem(
            id=uuid4(),
            user_id=user_id,
            source_platform=SourcePlatform.REDDIT,
            source_id="reddit_999",
            source_url="https://reddit.com/r/food/comments/999",
            title="Just had a great lunch",
            raw_text="The weather is nice today. Had sushi for lunch.",
            author="random_user",
            media_type=MediaType.TEXT,
            published_at=datetime.utcnow(),
        )

        matches = classifier._pattern_match(item)
        assert len(matches) == 0


class TestSignalCreation:
    """Test signal creation from classified content."""
    
    @pytest.mark.asyncio
    async def test_create_lead_opportunity_signal(self, classifier, lead_opportunity_item):
        """Test creating signal from lead opportunity."""
        user_id = uuid4()
        
        signal = await classifier.classify_content(lead_opportunity_item, user_id)
        
        assert signal is not None
        assert signal.signal_type == SignalType.LEAD_OPPORTUNITY
        assert signal.user_id == user_id
        assert signal.source_platform == "reddit"
        assert signal.recommended_action == ActionType.REPLY_PUBLIC
        assert signal.suggested_tone == ResponseTone.HELPFUL
        assert 0.0 <= signal.action_score <= 1.0
        assert 0.0 <= signal.urgency_score <= 1.0
        assert 0.0 <= signal.impact_score <= 1.0
        assert signal.expires_at is not None
    
    @pytest.mark.asyncio
    async def test_create_competitor_weakness_signal(self, classifier, competitor_weakness_item):
        """Test creating signal from competitor weakness."""
        user_id = uuid4()
        
        signal = await classifier.classify_content(competitor_weakness_item, user_id)
        
        assert signal is not None
        assert signal.signal_type == SignalType.COMPETITOR_WEAKNESS
        assert signal.recommended_action == ActionType.CREATE_CONTENT
        assert signal.suggested_tone == ResponseTone.PROFESSIONAL


class TestSignalMetadata:
    """Test signal metadata generation."""
    
    @pytest.mark.asyncio
    async def test_signal_title_generation(self, classifier, lead_opportunity_item):
        """Test that signal title is generated correctly."""
        signal = await classifier.classify_content(lead_opportunity_item, uuid4())
        
        assert signal is not None
        assert len(signal.title) > 0
        assert len(signal.title) <= 200
        assert "Lead Opportunity" in signal.title
    
    @pytest.mark.asyncio
    async def test_signal_context_generation(self, classifier, lead_opportunity_item):
        """Test that business context is generated."""
        signal = await classifier.classify_content(lead_opportunity_item, uuid4())
        
        assert signal is not None
        assert len(signal.context) > 0
        assert "conversion" in signal.context.lower() or "opportunity" in signal.context.lower()

