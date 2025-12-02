"""Cluster summarization using LLM for intelligent content synthesis."""

import json
import logging
from typing import Any, Dict, List

from app.core.models import Cluster, ContentItem
from app.llm.client_base import BaseLLMClient

logger = logging.getLogger(__name__)


class ClusterSummarizer:
    """Generate intelligent summaries for content clusters using LLM."""

    def __init__(self, llm_client: BaseLLMClient):
        """Initialize cluster summarizer.

        Args:
            llm_client: LLM client for generating summaries
        """
        self.llm_client = llm_client

    async def summarize_cluster(self, cluster: Cluster) -> Dict[str, Any]:
        """Generate comprehensive summary for a content cluster.

        Args:
            cluster: Content cluster to summarize

        Returns:
            Dictionary with summary data including:
            - topic: Brief topic title
            - summary: Comprehensive summary
            - key_points: List of key points
            - platforms: Platforms represented
            - perspective_notes: Cross-platform perspective analysis
        """
        # Build prompt with cluster content
        prompt = self._build_cluster_prompt(cluster)

        try:
            # Generate summary using LLM
            response = await self.llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=800,
            )

            # Parse JSON response
            summary_data = json.loads(response.content)
            return summary_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            # Return fallback structure
            return self._create_fallback_summary(cluster)

        except Exception as e:
            logger.error(f"Error generating cluster summary: {e}")
            return self._create_fallback_summary(cluster)

    def _build_cluster_prompt(self, cluster: Cluster) -> str:
        """Build prompt for cluster summarization."""
        prompt_parts = []

        prompt_parts.append(
            "You are an expert analyst creating concise summaries of related content from multiple sources.\n"
        )
        prompt_parts.append(
            "Your task is to analyze the following content items that have been clustered together "
            "as discussing the same topic or event, and create a comprehensive summary.\n"
        )

        # Add content items
        prompt_parts.append("\nCONTENT ITEMS:\n")
        for i, item in enumerate(cluster.items[:10], 1):  # Limit to 10 items
            prompt_parts.append(f"\n{i}. [{item.source_platform.value}] {item.title}")
            if item.author:
                prompt_parts.append(f"   Author: {item.author}")
            if item.raw_text:
                # Truncate long text
                text = item.raw_text[:500]
                prompt_parts.append(f"   Content: {text}...")
            if item.media_urls:
                prompt_parts.append(f"   Media: {len(item.media_urls)} items")
            prompt_parts.append(f"   Published: {item.published_at.strftime('%Y-%m-%d %H:%M')}")

        # Add instructions
        prompt_parts.append("\n\nINSTRUCTIONS:")
        prompt_parts.append("1. Identify the main topic or event being discussed")
        prompt_parts.append("2. Synthesize the key facts and perspectives from all sources")
        prompt_parts.append(
            "3. Note any significant differences in how different platforms or sources are covering this"
        )
        prompt_parts.append("4. Keep the summary factual and objective")
        prompt_parts.append("5. Highlight the most important or newsworthy aspects")
        prompt_parts.append("6. If there are videos or images, mention what they show or discuss")

        # Add output format
        prompt_parts.append("\n\nOUTPUT FORMAT:")
        prompt_parts.append("Provide your response as a JSON object with the following structure:")
        prompt_parts.append('{')
        prompt_parts.append('  "topic": "Brief topic title (max 100 chars)",')
        prompt_parts.append('  "summary": "Comprehensive summary (200-400 words)",')
        prompt_parts.append('  "key_points": ["point 1", "point 2", "point 3"],')
        prompt_parts.append('  "platforms": ["platform1", "platform2"],')
        prompt_parts.append(
            '  "perspective_notes": "Any notable differences in coverage or perspective across sources"'
        )
        prompt_parts.append('}')

        return "\n".join(prompt_parts)

    def _create_fallback_summary(self, cluster: Cluster) -> Dict[str, Any]:
        """Create fallback summary when LLM fails."""
        platforms = list(set([item.source_platform.value for item in cluster.items]))

        # Extract key points from titles
        key_points = [item.title for item in cluster.items[:3]]

        return {
            "topic": cluster.topic,
            "summary": f"This cluster contains {len(cluster.items)} items from {', '.join(platforms)} "
            f"discussing {cluster.topic}. Key items include: {', '.join([item.title[:50] for item in cluster.items[:3]])}.",
            "key_points": key_points,
            "platforms": platforms,
            "perspective_notes": "Multiple platforms covering this topic with varying perspectives.",
        }


    async def generate_cross_platform_analysis(
        self, clusters: List[Cluster]
    ) -> str:
        """Generate cross-platform perspective analysis across multiple clusters.

        Args:
            clusters: List of content clusters

        Returns:
            Cross-platform analysis summary
        """
        if not clusters:
            return "No content available for cross-platform analysis."

        # Build analysis prompt
        prompt = self._build_cross_platform_prompt(clusters)

        try:
            response = await self.llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=600,
            )
            return response.content

        except Exception as e:
            logger.error(f"Error generating cross-platform analysis: {e}")
            return "Cross-platform analysis unavailable."

    def _build_cross_platform_prompt(self, clusters: List[Cluster]) -> str:
        """Build prompt for cross-platform analysis."""
        prompt_parts = []

        prompt_parts.append(
            "Analyze how different platforms are covering the following topics. "
            "Identify any notable differences in perspective, emphasis, or framing:\n"
        )

        for cluster in clusters[:5]:  # Top 5 clusters
            platforms = ", ".join([p.value for p in cluster.platforms_represented])
            prompt_parts.append(f"\n- {cluster.topic} (covered by: {platforms})")
            prompt_parts.append(f"  Summary: {cluster.summary[:200]}")

        prompt_parts.append(
            "\n\nProvide a brief analysis (3-5 sentences) of how different platforms "
            "are approaching these topics differently."
        )

        return "\n".join(prompt_parts)

