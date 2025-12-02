"""Core digest generation engine - the heart of Social Media Radar.

This module orchestrates the complete digest generation pipeline:
1. Fetch content items from database
2. Score items for relevance
3. Cluster similar items into storylines
4. Generate AI summaries for each cluster
5. Rank and organize clusters
6. Create final digest output
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db_models import ContentItemDB, User
from app.core.models import Cluster, ContentItem, DigestRequest, DigestResponse, SourcePlatform
from app.core.ranking import ContentClusterer, RelevanceScorer
from app.llm.openai_client import OpenAILLMClient
from app.intelligence.cluster_summarizer import ClusterSummarizer

logger = logging.getLogger(__name__)


class DigestEngine:
    """Core engine for generating personalized daily digests."""

    def __init__(self):
        """Initialize digest engine."""
        self.llm_client = OpenAILLMClient()
        self.cluster_summarizer = ClusterSummarizer(self.llm_client)

    async def generate_digest(
        self,
        user_id: UUID,
        request: DigestRequest,
        db: AsyncSession,
    ) -> DigestResponse:
        """Generate a complete personalized digest.

        Args:
            user_id: User ID
            request: Digest generation parameters
            db: Database session

        Returns:
            Complete digest with clusters and summaries
        """
        logger.info(f"Generating digest for user {user_id}")

        # 1. Fetch content items from database
        items = await self._fetch_content_items(user_id, request, db)
        logger.info(f"Fetched {len(items)} content items")

        if not items:
            return DigestResponse(
                period_start=request.since or datetime.utcnow() - timedelta(hours=24),
                period_end=datetime.utcnow(),
                clusters=[],
                total_items=0,
                summary="No content available for this time period.",
            )

        # 2. Score items for relevance
        scored_items = await self._score_items(user_id, items, db)
        logger.info(f"Scored {len(scored_items)} items")

        # 3. Cluster similar items
        clusters = await self._cluster_items(user_id, scored_items)
        logger.info(f"Created {len(clusters)} clusters")

        # 4. Generate AI summaries for each cluster
        clusters = await self._summarize_clusters(clusters)
        logger.info(f"Generated summaries for {len(clusters)} clusters")

        # 5. Rank clusters by relevance
        ranked_clusters = self._rank_clusters(clusters, request.max_clusters)
        logger.info(f"Ranked top {len(ranked_clusters)} clusters")

        # 6. Generate overall digest summary
        overall_summary = await self._generate_overall_summary(ranked_clusters)

        # 7. Create response
        response = DigestResponse(
            period_start=request.since or datetime.utcnow() - timedelta(hours=24),
            period_end=datetime.utcnow(),
            clusters=ranked_clusters,
            total_items=len(items),
            summary=overall_summary,
        )

        logger.info(f"Digest generation complete: {len(ranked_clusters)} clusters, {len(items)} items")
        return response

    async def _fetch_content_items(
        self,
        user_id: UUID,
        request: DigestRequest,
        db: AsyncSession,
    ) -> List[ContentItem]:
        """Fetch content items from database based on request parameters."""
        # Build query
        query = select(ContentItemDB).where(ContentItemDB.user_id == user_id)

        # Time filter
        if request.since:
            query = query.where(ContentItemDB.published_at >= request.since)

        # Platform filter
        if request.platforms:
            query = query.where(ContentItemDB.source_platform.in_(request.platforms))

        # Topic filter
        if request.topics:
            # Filter items that have at least one matching topic
            query = query.where(ContentItemDB.topics.overlap(request.topics))

        # Order by published date (newest first)
        query = query.order_by(ContentItemDB.published_at.desc())

        # Limit to reasonable number
        query = query.limit(1000)

        # Execute query
        result = await db.execute(query)
        db_items = result.scalars().all()

        # Convert to ContentItem models
        items = []
        for db_item in db_items:
            item = ContentItem(
                id=db_item.id,
                user_id=db_item.user_id,
                source_platform=db_item.source_platform,
                source_id=db_item.source_id,
                source_url=db_item.source_url,
                author=db_item.author,
                channel=db_item.channel,
                title=db_item.title,
                raw_text=db_item.raw_text,
                media_type=db_item.media_type,
                media_urls=db_item.media_urls or [],
                published_at=db_item.published_at,
                fetched_at=db_item.fetched_at,
                topics=db_item.topics or [],
                lang=db_item.lang,
                embedding=db_item.embedding,
                metadata=db_item.metadata or {},
            )
            items.append(item)

        return items

    async def _score_items(
        self,
        user_id: UUID,
        items: List[ContentItem],
        db: AsyncSession,
    ) -> List[ContentItem]:
        """Score items for relevance using user's interest profile."""
        # TODO: Fetch user's interest profile from database
        # For now, use default scorer
        scorer = RelevanceScorer(interest_profile=None)

        for item in items:
            item.relevance_score = scorer.score_item(item)

        return items

    async def _cluster_items(
        self,
        user_id: UUID,
        items: List[ContentItem],
    ) -> List[Cluster]:
        """Cluster similar items into storylines."""
        clusterer = ContentClusterer(
            min_cluster_size=2,
            min_similarity=0.7,
        )

        clusters = clusterer.cluster_items(items, user_id)
        return clusters

    async def _summarize_clusters(self, clusters: List[Cluster]) -> List[Cluster]:
        """Generate AI summaries for each cluster."""
        for cluster in clusters:
            try:
                # Generate summary using LLM
                summary_data = await self.cluster_summarizer.summarize_cluster(cluster)

                # Update cluster with summary data
                cluster.summary = summary_data.get("summary", "")
                cluster.topic = summary_data.get("topic", cluster.topic)
                cluster.keywords = summary_data.get("key_points", cluster.keywords)
                cluster.perspective_summary = summary_data.get("perspective_notes", None)

            except Exception as e:
                logger.error(f"Error summarizing cluster {cluster.id}: {e}")
                # Use fallback summary
                cluster.summary = self._generate_fallback_summary(cluster)

        return clusters

    def _generate_fallback_summary(self, cluster: Cluster) -> str:
        """Generate a simple fallback summary when LLM fails."""
        item_count = len(cluster.items)
        platforms = ", ".join([p.value for p in cluster.platforms_represented])

        summary = f"This cluster contains {item_count} items from {platforms} "
        summary += f"discussing {cluster.topic}. "

        if cluster.items:
            # Add first item title as context
            summary += f"Key item: {cluster.items[0].title}"

        return summary

    def _rank_clusters(self, clusters: List[Cluster], max_clusters: int) -> List[Cluster]:
        """Rank clusters by relevance and return top N."""
        # Sort by relevance score (descending)
        sorted_clusters = sorted(
            clusters,
            key=lambda c: c.relevance_score,
            reverse=True,
        )

        # Return top N
        return sorted_clusters[:max_clusters]

    async def _generate_overall_summary(self, clusters: List[Cluster]) -> str:
        """Generate an overall summary of the digest."""
        if not clusters:
            return "No significant topics found in this time period."

        # Build summary prompt
        cluster_summaries = []
        for i, cluster in enumerate(clusters[:5], 1):  # Top 5 clusters
            cluster_summaries.append(
                f"{i}. {cluster.topic}: {cluster.summary[:200]}"
            )

        prompt = f"""Create a brief executive summary (2-3 sentences) of the following top topics:

{chr(10).join(cluster_summaries)}

Focus on the most important developments and emerging trends."""

        try:
            response = await self.llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200,
            )
            return response.content

        except Exception as e:
            logger.error(f"Error generating overall summary: {e}")
            # Fallback summary
            return f"Your digest contains {len(clusters)} major topics covering recent developments across your sources."

