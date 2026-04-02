"""Production-grade normalization engine for data standardization.

This module provides the NormalizationEngine class that standardizes content
from different platforms into a unified ContentItem schema with:
- Platform-specific data transformation
- Content validation and quality checks
- Media URL normalization
- Timestamp standardization
- Text cleaning and sanitization
- Metadata extraction and structuring
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlparse, parse_qs

from app.core.models import ContentItem, SourcePlatform
from app.core.errors import ValidationError

logger = logging.getLogger(__name__)


class NormalizationMetrics:
    """Metrics for normalization operations."""

    def __init__(self):
        """Initialize metrics."""
        self.total_normalized = 0
        self.validation_failures = 0
        self.transformations_applied = 0
        self.platform_counts: Dict[SourcePlatform, int] = {}

    def record_normalization(
        self,
        platform: SourcePlatform,
        transformations: int = 0,
        failed: bool = False,
    ):
        """Record normalization metrics."""
        if not failed:
            self.total_normalized += 1
            self.transformations_applied += transformations
            self.platform_counts[platform] = self.platform_counts.get(platform, 0) + 1
        else:
            self.validation_failures += 1

    def get_summary(self) -> Dict:
        """Get metrics summary."""
        return {
            "total_normalized": self.total_normalized,
            "validation_failures": self.validation_failures,
            "transformations_applied": self.transformations_applied,
            "platform_counts": {
                platform.value: count
                for platform, count in self.platform_counts.items()
            },
        }


class NormalizationEngine:
    """Production-grade normalization engine for content standardization.

    Features:
    - Platform-specific transformations
    - Content validation
    - Media URL normalization
    - Timestamp standardization
    - Text cleaning
    - Metadata structuring
    """

    def __init__(
        self,
        strict_validation: bool = True,
        clean_html: bool = True,
        normalize_urls: bool = True,
    ):
        """Initialize normalization engine.

        Args:
            strict_validation: Enable strict validation (reject invalid content)
            clean_html: Remove HTML tags from text
            normalize_urls: Normalize and clean URLs
        """
        self.strict_validation = strict_validation
        self.clean_html = clean_html
        self.normalize_urls = normalize_urls

        # Metrics
        self.metrics = NormalizationMetrics()

        # HTML tag regex
        self.html_tag_pattern = re.compile(r'<[^>]+>')

        # URL tracking parameters to remove
        self.tracking_params = {
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
            'fbclid', 'gclid', 'msclkid', '_ga', 'mc_cid', 'mc_eid',
        }

        logger.info(
            f"NormalizationEngine initialized: strict={strict_validation}, "
            f"clean_html={clean_html}, normalize_urls={normalize_urls}"
        )

    def normalize(self, item: ContentItem) -> ContentItem:
        """Normalize a content item.

        Args:
            item: Content item to normalize

        Returns:
            Normalized content item

        Raises:
            ValidationError: If validation fails in strict mode
        """
        transformations = 0

        # Validate item
        if self.strict_validation:
            self._validate_item(item)

        # Normalize title
        if item.title:
            item.title = self._clean_text(item.title)
            transformations += 1

        # Normalize raw text
        if item.raw_text:
            item.raw_text = self._clean_text(item.raw_text)
            transformations += 1

        # ── Prompt-injection / adversarial-content defense ────────────────────
        # ContentSanitizer runs on title and raw_text *after* basic cleaning so
        # that HTML-stripped text is matched against injection patterns, not
        # raw markup.  Any hits are stamped in metadata for audit logging.
        from app.core.content_sanitizer import get_sanitizer
        _sanitizer = get_sanitizer()
        if item.title:
            _title_result = _sanitizer.sanitize(item.title)
            if _title_result.modified:
                item.title = _title_result.text
                item.metadata["sanitized_title"] = True
                item.metadata["sanitized_title_hits"] = [
                    h.threat_class for h in _title_result.hits
                ]
                transformations += 1
        if item.raw_text:
            _text_result = _sanitizer.sanitize(item.raw_text)
            if _text_result.modified:
                item.raw_text = _text_result.text
                item.metadata["sanitized_text"] = True
                item.metadata["sanitized_text_hits"] = [
                    h.threat_class for h in _text_result.hits
                ]
                transformations += 1

        # Normalize URLs
        if self.normalize_urls:
            item.source_url = self._normalize_url(item.source_url)
            item.media_urls = [self._normalize_url(url) for url in item.media_urls]
            transformations += 1

        # Normalize timestamps
        item.published_at = self._normalize_timestamp(item.published_at)
        item.fetched_at = self._normalize_timestamp(item.fetched_at)
        transformations += 1

        # Apply platform-specific transformations
        item = self._apply_platform_transformations(item)
        transformations += 1

        # Normalize metadata
        item.metadata = self._normalize_metadata(item.metadata)
        transformations += 1

        # Record metrics
        self.metrics.record_normalization(item.source_platform, transformations)

        return item

    def _validate_item(self, item: ContentItem):
        """Validate content item.

        Args:
            item: Content item to validate

        Raises:
            ValidationError: If validation fails
        """
        # Required fields
        if not item.source_id:
            raise ValidationError("source_id is required")

        if not item.source_url:
            raise ValidationError("source_url is required")

        if not item.title:
            raise ValidationError("title is required")

        # URL validation
        try:
            parsed = urlparse(item.source_url)
            if not parsed.scheme or not parsed.netloc:
                raise ValidationError(f"Invalid source_url: {item.source_url}")
        except Exception as e:
            raise ValidationError(f"Invalid source_url: {e}") from e

        # Timestamp validation
        if item.published_at > datetime.now(timezone.utc):
            raise ValidationError("published_at cannot be in the future")

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return text

        # Remove HTML tags if enabled
        if self.clean_html:
            text = self.html_tag_pattern.sub('', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        # Remove null bytes
        text = text.replace('\x00', '')

        return text

    def _normalize_url(self, url: str) -> str:
        """Normalize and clean URL.

        Args:
            url: URL to normalize

        Returns:
            Normalized URL
        """
        if not url:
            return url

        try:
            parsed = urlparse(url)

            # Remove tracking parameters
            if parsed.query:
                query_params = parse_qs(parsed.query)
                cleaned_params = {
                    k: v for k, v in query_params.items()
                    if k.lower() not in self.tracking_params
                }

                # Rebuild query string
                if cleaned_params:
                    from urllib.parse import urlencode
                    query_string = urlencode(cleaned_params, doseq=True)
                    parsed = parsed._replace(query=query_string)
                else:
                    parsed = parsed._replace(query='')

            # Remove fragment (anchor)
            parsed = parsed._replace(fragment='')

            # Rebuild URL
            from urllib.parse import urlunparse
            normalized_url = urlunparse(parsed)

            return normalized_url

        except Exception as e:
            logger.warning(f"Failed to normalize URL {url}: {e}")
            return url

    def _normalize_timestamp(self, timestamp: datetime) -> datetime:
        """Normalize timestamp to UTC.

        Args:
            timestamp: Timestamp to normalize

        Returns:
            Normalized timestamp in UTC
        """
        if not timestamp:
            return timestamp

        # Convert to UTC if timezone-aware
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone(timezone.utc)
        else:
            # Assume UTC if naive
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        return timestamp

    def _apply_platform_transformations(self, item: ContentItem) -> ContentItem:
        """Apply platform-specific transformations.

        Args:
            item: Content item

        Returns:
            Transformed content item
        """
        platform = item.source_platform

        if platform == SourcePlatform.REDDIT:
            item = self._transform_reddit(item)
        elif platform == SourcePlatform.YOUTUBE:
            item = self._transform_youtube(item)
        elif platform == SourcePlatform.TIKTOK:
            item = self._transform_tiktok(item)
        elif platform in {SourcePlatform.FACEBOOK, SourcePlatform.INSTAGRAM}:
            item = self._transform_meta(item)
        elif platform in {SourcePlatform.NYTIMES, SourcePlatform.WSJ}:
            item = self._transform_news(item)

        return item

    def _transform_reddit(self, item: ContentItem) -> ContentItem:
        """Apply Reddit-specific transformations."""
        # Extract subreddit from channel if not set
        if not item.channel and item.metadata.get('subreddit'):
            item.channel = f"r/{item.metadata['subreddit']}"

        # Normalize score/upvotes
        if 'score' in item.metadata:
            item.metadata['engagement_score'] = item.metadata['score']

        return item

    def _transform_youtube(self, item: ContentItem) -> ContentItem:
        """Apply YouTube-specific transformations."""
        # Extract video ID from URL if not in metadata
        if 'video_id' not in item.metadata:
            video_id = self._extract_youtube_video_id(item.source_url)
            if video_id:
                item.metadata['video_id'] = video_id

        # Normalize view count
        if 'view_count' in item.metadata:
            item.metadata['engagement_score'] = item.metadata['view_count']

        return item

    def _transform_tiktok(self, item: ContentItem) -> ContentItem:
        """Apply TikTok-specific transformations."""
        # Normalize engagement metrics
        if 'like_count' in item.metadata:
            item.metadata['engagement_score'] = item.metadata['like_count']

        return item

    def _transform_meta(self, item: ContentItem) -> ContentItem:
        """Apply Facebook/Instagram-specific transformations."""
        # Normalize reactions/likes
        if 'reactions' in item.metadata:
            item.metadata['engagement_score'] = item.metadata['reactions']
        elif 'likes' in item.metadata:
            item.metadata['engagement_score'] = item.metadata['likes']

        return item

    def _transform_news(self, item: ContentItem) -> ContentItem:
        """Apply news-specific transformations."""
        # Extract author from byline if not set
        if not item.author and item.metadata.get('byline'):
            item.author = item.metadata['byline']

        return item

    def _extract_youtube_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL."""
        try:
            parsed = urlparse(url)

            # youtube.com/watch?v=VIDEO_ID
            if 'youtube.com' in parsed.netloc and parsed.path == '/watch':
                query_params = parse_qs(parsed.query)
                if 'v' in query_params:
                    return query_params['v'][0]

            # youtu.be/VIDEO_ID
            if 'youtu.be' in parsed.netloc:
                return parsed.path.lstrip('/')

        except Exception as e:
            logger.warning(f"Failed to extract YouTube video ID from {url}: {e}")

        return None

    def _normalize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize metadata structure.

        Args:
            metadata: Metadata dictionary

        Returns:
            Normalized metadata
        """
        if not metadata:
            return {}

        # Remove null values
        normalized = {k: v for k, v in metadata.items() if v is not None}

        # Ensure consistent types for common fields
        if 'engagement_score' in normalized:
            try:
                normalized['engagement_score'] = int(normalized['engagement_score'])
            except (ValueError, TypeError):
                del normalized['engagement_score']

        return normalized

    def get_metrics(self) -> Dict:
        """Get normalization metrics.

        Returns:
            Dictionary of metrics
        """
        return self.metrics.get_summary()
