"""Base output generator interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app.core.models import Cluster, ContentItem
from app.output.models import GeneratedOutput, OutputPreferences, OutputRequest


class BaseOutputGenerator(ABC):
    """Base class for output generators."""

    def __init__(self, preferences: OutputPreferences):
        """Initialize generator.

        Args:
            preferences: Output preferences
        """
        self.preferences = preferences

    @abstractmethod
    async def generate(
        self,
        request: OutputRequest,
        clusters: List[Cluster],
        items: List[ContentItem],
        **kwargs,
    ) -> GeneratedOutput:
        """Generate output from content.

        Args:
            request: Output request
            clusters: Content clusters
            items: Content items
            **kwargs: Additional generator-specific arguments

        Returns:
            Generated output
        """
        pass

    @abstractmethod
    async def validate_output(self, output: GeneratedOutput) -> bool:
        """Validate generated output quality.

        Args:
            output: Generated output

        Returns:
            True if output meets quality standards
        """
        pass

    def _apply_text_style(self, content: str) -> str:
        """Apply text style transformations.

        Args:
            content: Raw content

        Returns:
            Styled content
        """
        # Override in subclasses for style-specific transformations
        return content

    def _apply_tone(self, content: str) -> str:
        """Apply tone transformations.

        Args:
            content: Raw content

        Returns:
            Content with applied tone
        """
        # Override in subclasses for tone-specific transformations
        return content

    def _truncate_to_length(self, content: str) -> str:
        """Truncate content to preferred length.

        Args:
            content: Raw content

        Returns:
            Truncated content
        """
        from app.output.models import LengthPreference

        # Approximate word counts for each length
        length_limits = {
            LengthPreference.BRIEF: 200,
            LengthPreference.MEDIUM: 500,
            LengthPreference.DETAILED: 1000,
            LengthPreference.COMPREHENSIVE: 5000,
        }

        limit = length_limits.get(self.preferences.length, 500)
        words = content.split()

        if len(words) <= limit:
            return content

        # Truncate and add ellipsis
        truncated = " ".join(words[:limit])
        return f"{truncated}..."

    def _format_sources(self, items: List[ContentItem]) -> str:
        """Format source citations.

        Args:
            items: Content items

        Returns:
            Formatted sources
        """
        if not self.preferences.include_sources:
            return ""

        sources = []
        for i, item in enumerate(items, 1):
            source = f"{i}. {item.title} - {item.source_platform.value}"
            if self.preferences.include_links:
                source += f" ({item.source_url})"
            if self.preferences.include_timestamps:
                source += f" [{item.published_at.strftime('%Y-%m-%d %H:%M')}]"
            sources.append(source)

        return "\n\n## Sources\n\n" + "\n".join(sources)

    def _extract_quotes(self, items: List[ContentItem]) -> List[str]:
        """Extract notable quotes from content.

        Args:
            items: Content items

        Returns:
            List of quotes
        """
        if not self.preferences.include_quotes:
            return []

        quotes = []
        for item in items:
            if item.raw_text:
                # Simple quote extraction (can be enhanced with NLP)
                text = item.raw_text
                # Look for quoted text
                import re

                found_quotes = re.findall(r'"([^"]+)"', text)
                quotes.extend(found_quotes[:2])  # Max 2 quotes per item

        return quotes[:5]  # Max 5 total quotes

    def _calculate_quality_score(self, output: GeneratedOutput) -> float:
        """Calculate quality score for output.

        Args:
            output: Generated output

        Returns:
            Quality score (0-1)
        """
        score = 1.0

        # Check minimum length
        if output.metadata.word_count and output.metadata.word_count < 50:
            score -= 0.3

        # Check for broken links (placeholder)
        # In production, validate all URLs
        if self.preferences.include_links and not output.metadata.media_urls:
            score -= 0.1

        # Check for required elements
        if self.preferences.include_sources and "Sources" not in output.content:
            score -= 0.2

        return max(0.0, score)

