"""Output manager for orchestrating multi-format content generation."""

import asyncio
import logging
from typing import Dict, List, Optional, Type

from app.core.models import Cluster, ContentItem
from app.llm.client_base import BaseLLMClient
from app.llm.openai_client import OpenAILLMClient
from app.output.generators.base import BaseOutputGenerator
from app.output.generators.text_generator import MarkdownGenerator
from app.output.generators.visual_generator import InfographicGenerator, VideoGenerator
from app.output.models import (
    GeneratedOutput,
    OutputFormat,
    OutputPreferences,
    OutputRequest,
)

logger = logging.getLogger(__name__)


class OutputManager:
    """Manage multi-format output generation."""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        """Initialize output manager.

        Args:
            llm_client: LLM client for text generation
        """
        self.llm_client = llm_client or OpenAILLMClient()
        self._generators: Dict[OutputFormat, Type[BaseOutputGenerator]] = {
            OutputFormat.MARKDOWN: MarkdownGenerator,
            OutputFormat.IMAGE: InfographicGenerator,
            OutputFormat.VIDEO: VideoGenerator,
        }

    def register_generator(
        self, format: OutputFormat, generator_class: Type[BaseOutputGenerator]
    ):
        """Register a custom output generator.

        Args:
            format: Output format
            generator_class: Generator class
        """
        self._generators[format] = generator_class

    async def generate_output(
        self,
        request: OutputRequest,
        preferences: OutputPreferences,
        clusters: List[Cluster],
        items: List[ContentItem],
    ) -> GeneratedOutput:
        """Generate output in requested format.

        Args:
            request: Output request
            preferences: Output preferences
            clusters: Content clusters
            items: Content items

        Returns:
            Generated output

        Raises:
            ValueError: If format not supported
        """
        format = preferences.primary_format

        # Get generator class
        generator_class = self._generators.get(format)
        if not generator_class:
            raise ValueError(f"Unsupported output format: {format}")

        # Create generator instance
        if format in [OutputFormat.MARKDOWN, OutputFormat.HTML, OutputFormat.JSON]:
            generator = generator_class(preferences, self.llm_client)
        else:
            generator = generator_class(preferences)

        # Generate output
        try:
            output = await generator.generate(request, clusters, items)

            # Validate output
            is_valid = await generator.validate_output(output)
            if not is_valid:
                logger.warning(f"Generated output failed validation for format {format}")

                # Try fallback format
                if preferences.fallback_formats:
                    return await self._generate_fallback(
                        request, preferences, clusters, items
                    )

            return output

        except Exception as e:
            logger.error(f"Error generating output in format {format}: {e}")

            # Try fallback format
            if preferences.fallback_formats:
                return await self._generate_fallback(
                    request, preferences, clusters, items
                )

            raise

    async def _generate_fallback(
        self,
        request: OutputRequest,
        preferences: OutputPreferences,
        clusters: List[Cluster],
        items: List[ContentItem],
    ) -> GeneratedOutput:
        """Generate output using fallback format.

        Args:
            request: Output request
            preferences: Output preferences
            clusters: Content clusters
            items: Content items

        Returns:
            Generated output using fallback format
        """
        for fallback_format in preferences.fallback_formats:
            try:
                # Create temporary preferences with fallback format
                fallback_prefs = preferences.model_copy()
                fallback_prefs.primary_format = fallback_format

                generator_class = self._generators.get(fallback_format)
                if not generator_class:
                    continue

                if fallback_format in [OutputFormat.MARKDOWN, OutputFormat.HTML]:
                    generator = generator_class(fallback_prefs, self.llm_client)
                else:
                    generator = generator_class(fallback_prefs)

                output = await generator.generate(request, clusters, items)
                is_valid = await generator.validate_output(output)

                if is_valid:
                    logger.info(f"Successfully generated fallback output in {fallback_format}")
                    return output

            except Exception as e:
                logger.error(f"Error generating fallback output in {fallback_format}: {e}")
                continue

        raise ValueError("All output generation attempts failed")

    async def generate_multi_format(
        self,
        request: OutputRequest,
        preferences: OutputPreferences,
        clusters: List[Cluster],
        items: List[ContentItem],
        formats: List[OutputFormat],
    ) -> Dict[OutputFormat, GeneratedOutput]:
        """Generate output in multiple formats concurrently.

        Args:
            request: Output request
            preferences: Output preferences
            clusters: Content clusters
            items: Content items
            formats: List of formats to generate

        Returns:
            Dictionary mapping format to generated output
        """
        tasks = []
        for format in formats:
            # Create preferences for this format
            format_prefs = preferences.model_copy()
            format_prefs.primary_format = format

            task = self.generate_output(request, format_prefs, clusters, items)
            tasks.append((format, task))

        # Execute concurrently
        results = {}
        for format, task in tasks:
            try:
                output = await task
                results[format] = output
            except Exception as e:
                logger.error(f"Failed to generate {format}: {e}")

        return results

    async def generate_with_quality_check(
        self,
        request: OutputRequest,
        preferences: OutputPreferences,
        clusters: List[Cluster],
        items: List[ContentItem],
        min_quality_score: float = 0.7,
        max_retries: int = 3,
    ) -> GeneratedOutput:
        """Generate output with quality validation and retries.

        Args:
            request: Output request
            preferences: Output preferences
            clusters: Content clusters
            items: Content items
            min_quality_score: Minimum acceptable quality score
            max_retries: Maximum retry attempts

        Returns:
            High-quality generated output
        """
        for attempt in range(max_retries):
            output = await self.generate_output(request, preferences, clusters, items)

            if (
                output.metadata.quality_score
                and output.metadata.quality_score >= min_quality_score
            ):
                return output

            logger.warning(
                f"Output quality score {output.metadata.quality_score} below threshold "
                f"{min_quality_score}, retrying (attempt {attempt + 1}/{max_retries})"
            )

            # Adjust temperature for retry
            if hasattr(self.llm_client, "temperature"):
                self.llm_client.temperature = min(0.9, self.llm_client.temperature + 0.1)

        # Return best attempt
        logger.warning(f"Failed to meet quality threshold after {max_retries} attempts")
        return output

