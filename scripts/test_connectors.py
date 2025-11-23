#!/usr/bin/env python3
"""Test script to validate all connector implementations."""

import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all connectors can be imported."""
    print("=" * 80)
    print("Testing Connector Imports")
    print("=" * 80)
    
    connectors = [
        ("Reddit", "app.connectors.reddit", "RedditConnector"),
        ("YouTube", "app.connectors.youtube", "YouTubeConnector"),
        ("TikTok", "app.connectors.tiktok", "TikTokConnector"),
        ("Facebook", "app.connectors.facebook", "FacebookConnector"),
        ("Instagram", "app.connectors.instagram", "InstagramConnector"),
        ("WeChat", "app.connectors.wechat", "WeChatConnector"),
        ("NYTimes", "app.connectors.nytimes", "NYTimesConnector"),
        ("WSJ", "app.connectors.wsj", "WSJConnector"),
        ("ABC News", "app.connectors.abc_news", "ABCNewsConnector"),
        ("Google News", "app.connectors.google_news", "GoogleNewsConnector"),
        ("Apple News", "app.connectors.apple_news", "AppleNewsConnector"),
        ("RSS", "app.connectors.rss", "RSSConnector"),
    ]
    
    passed = 0
    failed = 0
    
    for name, module_path, class_name in connectors:
        try:
            module = __import__(module_path, fromlist=[class_name])
            connector_class = getattr(module, class_name)
            print(f"✓ {name:20s} - {module_path}")
            passed += 1
        except Exception as e:
            print(f"✗ {name:20s} - {module_path}")
            print(f"  Error: {e}")
            traceback.print_exc()
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_registry():
    """Test connector registry."""
    print("\n" + "=" * 80)
    print("Testing Connector Registry")
    print("=" * 80)
    
    try:
        from app.connectors.registry import connector_registry
        from app.core.models import SourcePlatform
        
        platforms = connector_registry.get_supported_platforms()
        print(f"✓ Registry loaded successfully")
        print(f"  Supported platforms: {len(platforms)}")
        
        for platform in platforms:
            print(f"  - {platform.value}")
        
        # Test platform info
        info = connector_registry.get_platform_info()
        print(f"\n✓ Platform info loaded: {len(info)} platforms")
        
        return True
    except Exception as e:
        print(f"✗ Registry test failed: {e}")
        traceback.print_exc()
        return False


def test_models():
    """Test core models."""
    print("\n" + "=" * 80)
    print("Testing Core Models")
    print("=" * 80)
    
    try:
        from app.core.models import SourcePlatform, ContentType, ContentItem
        from uuid import uuid4
        from datetime import datetime
        
        # Test SourcePlatform enum
        platforms = list(SourcePlatform)
        print(f"✓ SourcePlatform enum: {len(platforms)} platforms")
        
        # Test ContentItem creation
        item = ContentItem(
            platform=SourcePlatform.REDDIT,
            platform_id="test123",
            content_type=ContentType.TEXT,
            title="Test Post",
            text_content="This is a test",
            url="https://reddit.com/test",
            author="testuser",
            published_at=datetime.now(),
            user_id=uuid4(),
        )
        print(f"✓ ContentItem created successfully")
        
        return True
    except Exception as e:
        print(f"✗ Models test failed: {e}")
        traceback.print_exc()
        return False


def test_base_connector():
    """Test base connector interface."""
    print("\n" + "=" * 80)
    print("Testing Base Connector")
    print("=" * 80)
    
    try:
        from app.connectors.base import BaseConnector, ConnectorConfig, FetchResult
        from app.core.models import SourcePlatform
        
        print(f"✓ BaseConnector imported successfully")
        print(f"✓ ConnectorConfig imported successfully")
        print(f"✓ FetchResult imported successfully")
        
        # Test ConnectorConfig creation
        config = ConnectorConfig(
            platform=SourcePlatform.RSS,
            credentials={},
            settings={"feed_urls": ["https://example.com/feed.xml"]},
        )
        print(f"✓ ConnectorConfig created successfully")
        
        return True
    except Exception as e:
        print(f"✗ Base connector test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Social Media Radar - Connector Test Suite")
    print("=" * 80 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Models", test_models()))
    results.append(("Base Connector", test_base_connector()))
    results.append(("Registry", test_registry()))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    failed = sum(1 for _, result in results if not result)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} - {name}")
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()

