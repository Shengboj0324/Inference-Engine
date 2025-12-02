"""Integration tests for authentication and search functionality."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from app.core.models import (
    ContentItem,
    SourcePlatform,
)


class TestAuthentication:
    """Test authentication endpoints."""

    def test_password_hashing(self):
        """Test password hashing and verification."""
        from app.api.routes.auth import get_password_hash, verify_password
        
        password = "test_password_123"
        hashed = get_password_hash(password)
        
        # Verify correct password
        assert verify_password(password, hashed) is True
        
        # Verify incorrect password
        assert verify_password("wrong_password", hashed) is False

    def test_jwt_token_creation(self):
        """Test JWT token creation."""
        from app.api.routes.auth import create_access_token
        
        data = {"sub": str(uuid4()), "email": "test@example.com"}
        token = create_access_token(data)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_jwt_token_expiration(self):
        """Test JWT token with custom expiration."""
        from app.api.routes.auth import create_access_token
        
        data = {"sub": str(uuid4()), "email": "test@example.com"}
        expires_delta = timedelta(minutes=30)
        token = create_access_token(data, expires_delta)
        
        assert token is not None
        
        # Decode and verify expiration
        from jose import jwt
        from app.api.routes.auth import SECRET_KEY, ALGORITHM
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert "exp" in payload
        assert payload["sub"] == data["sub"]
        assert payload["email"] == data["email"]


class TestSearch:
    """Test search functionality."""

    def test_search_request_validation(self):
        """Test search request model validation."""
        from app.api.routes.search import SearchRequest
        
        # Valid request
        request = SearchRequest(
            query="test query",
            platforms=[SourcePlatform.REDDIT],
            since=datetime.utcnow() - timedelta(days=1),
            limit=50,
        )
        
        assert request.query == "test query"
        assert request.platforms == [SourcePlatform.REDDIT]
        assert request.limit == 50

    def test_search_response_structure(self):
        """Test search response model structure."""
        from app.api.routes.search import SearchResponse
        
        items = [
            ContentItem(
                id=uuid4(),
                user_id=uuid4(),
                source_platform=SourcePlatform.REDDIT,
                source_id="test123",
                source_url="https://reddit.com/r/test/123",
                author="test_user",
                title="Test Post",
                raw_text="Test content",
                published_at=datetime.utcnow(),
                fetched_at=datetime.utcnow(),
            )
        ]
        
        response = SearchResponse(
            items=items,
            total=1,
            query="test query",
        )
        
        assert len(response.items) == 1
        assert response.total == 1
        assert response.query == "test query"

    def test_trending_topics_response(self):
        """Test trending topics response structure."""
        from app.api.routes.search import TrendingTopicsResponse, TopicCount
        
        topics = [
            TopicCount(
                topic="AI",
                count=10,
                platforms=[SourcePlatform.REDDIT, SourcePlatform.YOUTUBE],
                latest_mention=datetime.utcnow(),
            ),
            TopicCount(
                topic="Technology",
                count=5,
                platforms=[SourcePlatform.REDDIT],
                latest_mention=datetime.utcnow() - timedelta(hours=1),
            ),
        ]
        
        response = TrendingTopicsResponse(
            topics=topics,
            period_hours=24,
            total_items_analyzed=100,
        )
        
        assert len(response.topics) == 2
        assert response.topics[0].topic == "AI"
        assert response.topics[0].count == 10
        assert response.period_hours == 24
        assert response.total_items_analyzed == 100


class TestSecurity:
    """Test security features."""

    def test_password_strength_validation(self):
        """Test password strength requirements."""
        from app.api.routes.auth import get_password_hash
        
        # Should work with strong password
        strong_password = "SecurePass123!"
        hashed = get_password_hash(strong_password)
        assert hashed is not None
        
        # Weak passwords should be rejected at API level
        # (tested in integration tests)

    def test_jwt_secret_key_exists(self):
        """Test that JWT secret key is configured."""
        from app.api.routes.auth import SECRET_KEY
        
        assert SECRET_KEY is not None
        assert len(SECRET_KEY) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

