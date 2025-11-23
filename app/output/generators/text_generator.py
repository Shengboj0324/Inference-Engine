"""Text-based output generators (Markdown, HTML, PDF)."""

import asyncio
import time
from typing import List

from app.core.models import Cluster, ContentItem
from app.llm.client_base import BaseLLMClient
from app.llm.openai_client import OpenAILLMClient
from app.output.generators.base import BaseOutputGenerator
from app.output.models import (
    GeneratedOutput,
    OutputFormat,
    OutputMetadata,
    OutputPreferences,
    OutputRequest,
    TextStyle,
)


class MarkdownGenerator(BaseOutputGenerator):
    """Generate Markdown formatted output."""

    def __init__(self, preferences: OutputPreferences, llm_client: BaseLLMClient):
        """Initialize Markdown generator.

        Args:
            preferences: Output preferences
            llm_client: LLM client for content generation
        """
        super().__init__(preferences)
        self.llm_client = llm_client

    async def generate(
        self,
        request: OutputRequest,
        clusters: List[Cluster],
        items: List[ContentItem],
        **kwargs,
    ) -> GeneratedOutput:
        """Generate Markdown output."""
        start_time = time.time()

        # Build prompt based on preferences
        prompt = self._build_prompt(clusters, items, request)

        # Generate content using LLM
        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4000,
        )

        content = response.content

        # Apply style and tone
        content = self._apply_text_style(content)
        content = self._apply_tone(content)

        # Add sources if requested
        if self.preferences.include_sources:
            content += "\n\n" + self._format_sources(items)

        # Calculate metadata
        word_count = len(content.split())
        char_count = len(content)
        generation_time_ms = int((time.time() - start_time) * 1000)

        metadata = OutputMetadata(
            format=OutputFormat.MARKDOWN,
            generation_time_ms=generation_time_ms,
            word_count=word_count,
            character_count=char_count,
            model_used=response.model,
            tokens_used=response.usage.get("total_tokens") if response.usage else None,
        )

        output = GeneratedOutput(
            user_id=request.user_id,
            digest_id=request.digest_id,
            preferences_id=request.preferences_id or self.preferences.id,
            format=OutputFormat.MARKDOWN,
            content=content,
            metadata=metadata,
            title=self._generate_title(clusters),
            summary=self._generate_summary(clusters),
        )

        # Calculate quality score
        metadata.quality_score = self._calculate_quality_score(output)

        return output

    def _build_prompt(
        self, clusters: List[Cluster], items: List[ContentItem], request: OutputRequest
    ) -> str:
        """Build LLM prompt based on preferences."""
        # Base prompt
        prompt_parts = []

        # Style instruction
        style_instructions = {
            TextStyle.PROFESSIONAL: "Write in a professional, formal tone suitable for business communication.",
            TextStyle.CASUAL: "Write in a casual, conversational tone as if talking to a friend.",
            TextStyle.ACADEMIC: "Write in an academic style with proper citations and scholarly language.",
            TextStyle.JOURNALISTIC: "Write in a journalistic style with clear, objective reporting.",
            TextStyle.TECHNICAL: "Write in a technical style with precise terminology and detailed explanations.",
            TextStyle.ELI5: "Explain everything in simple terms that a 5-year-old could understand.",
            TextStyle.EXECUTIVE: "Write a concise executive summary focusing on key insights and actionable items.",
            TextStyle.BULLET_POINTS: "Present information in clear, concise bullet points.",
            TextStyle.NARRATIVE: "Write as an engaging narrative story.",
        }

        prompt_parts.append(
            style_instructions.get(
                self.preferences.text_style,
                "Write in a clear, informative style.",
            )
        )

        # Tone instruction
        tone_instructions = {
            "optimistic": "Maintain an optimistic, positive tone.",
            "critical": "Maintain a critical, analytical tone.",
            "humorous": "Add appropriate humor where suitable.",
            "serious": "Maintain a serious, grave tone.",
            "inspirational": "Write in an inspirational, motivating tone.",
        }

        if self.preferences.tone.value != "neutral":
            prompt_parts.append(tone_instructions.get(self.preferences.tone.value, ""))

        # Length instruction
        length_instructions = {
            "brief": "Keep it brief - 1-2 paragraphs maximum.",
            "medium": "Write a medium-length summary of 3-5 paragraphs.",
            "detailed": "Provide a detailed analysis of 6-10 paragraphs.",
            "comprehensive": "Write a comprehensive, in-depth report.",
        }

        prompt_parts.append(length_instructions.get(self.preferences.length.value, ""))

        # Custom prompt
        if request.custom_prompt:
            prompt_parts.append(f"\nAdditional instructions: {request.custom_prompt}")

        # Content to summarize
        prompt_parts.append("\n\n# Content to Summarize\n")

        for i, cluster in enumerate(clusters, 1):
            prompt_parts.append(f"\n## Topic {i}: {cluster.topic}\n")
            prompt_parts.append(f"Summary: {cluster.summary}\n")
            prompt_parts.append(f"Relevance: {cluster.relevance_score:.2f}\n")
            prompt_parts.append(f"Number of items: {len(cluster.items)}\n")

            # Add sample items
            for item in cluster.items[:3]:  # Max 3 items per cluster
                prompt_parts.append(f"\n- {item.title}")
                if item.raw_text:
                    # Truncate text
                    text = item.raw_text[:500]
                    prompt_parts.append(f"  {text}...")

        # Focus topics
        if request.focus_topics:
            prompt_parts.append(
                f"\n\nFocus especially on these topics: {', '.join(request.focus_topics)}"
            )

        # Exclude topics
        if request.exclude_topics:
            prompt_parts.append(
                f"\nAvoid or minimize coverage of: {', '.join(request.exclude_topics)}"
            )

        prompt_parts.append(
            "\n\nGenerate a well-structured Markdown document following the above instructions."
        )

        return "\n".join(prompt_parts)

    def _generate_title(self, clusters: List[Cluster]) -> str:
        """Generate title for output."""
        if not clusters:
            return "Daily Intelligence Digest"

        # Use top cluster topic
        top_cluster = max(clusters, key=lambda c: c.relevance_score)
        return f"Intelligence Digest: {top_cluster.topic}"

    def _generate_summary(self, clusters: List[Cluster]) -> str:
        """Generate brief summary."""
        if not clusters:
            return "No content available."

        topics = [c.topic for c in clusters[:3]]
        return f"Covering {len(clusters)} topics including: {', '.join(topics)}"

    async def validate_output(self, output: GeneratedOutput) -> bool:
        """Validate Markdown output."""
        # Check minimum length
        if output.metadata.word_count and output.metadata.word_count < 50:
            return False

        # Check for valid Markdown (basic check)
        if not output.content.strip():
            return False

        # Check quality score
        if output.metadata.quality_score and output.metadata.quality_score < 0.5:
            return False

        return True

