"""Content ranking and clustering algorithms."""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from uuid import UUID

from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from app.core.models import ContentItem, Cluster, UserInterestProfile


class RelevanceScorer:
    """Score content items based on user interests and other signals."""

    def __init__(self, interest_profile: Optional[UserInterestProfile] = None):
        """Initialize scorer with user interest profile.

        Args:
            interest_profile: User's interest profile with topics and embedding
        """
        self.interest_profile = interest_profile

    def score_item(self, item: ContentItem) -> float:
        """Calculate relevance score for a content item.

        Args:
            item: Content item to score

        Returns:
            Relevance score between 0 and 1
        """
        scores = []

        # Embedding similarity score
        if (
            self.interest_profile
            and self.interest_profile.interest_embedding
            and item.embedding
        ):
            embedding_score = self._embedding_similarity(
                item.embedding, self.interest_profile.interest_embedding
            )
            scores.append(("embedding", embedding_score, 0.4))

        # Topic match score
        if self.interest_profile and self.interest_profile.interest_topics:
            topic_score = self._topic_match_score(
                item.topics, self.interest_profile.interest_topics
            )
            scores.append(("topic", topic_score, 0.3))

        # Recency score
        recency_score = self._recency_score(item.published_at)
        scores.append(("recency", recency_score, 0.2))

        # Engagement score (from metadata)
        engagement_score = self._engagement_score(item.metadata)
        scores.append(("engagement", engagement_score, 0.1))

        # Weighted average
        if not scores:
            return 0.5

        total_weight = sum(weight for _, _, weight in scores)
        weighted_sum = sum(score * weight for _, score, weight in scores)

        return weighted_sum / total_weight

    def _embedding_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between embeddings."""
        vec1 = np.array(embedding1).reshape(1, -1)
        vec2 = np.array(embedding2).reshape(1, -1)
        similarity = cosine_similarity(vec1, vec2)[0][0]
        # Normalize to 0-1 range (cosine similarity is -1 to 1)
        return (similarity + 1) / 2

    def _topic_match_score(
        self, item_topics: List[str], interest_topics: List[str]
    ) -> float:
        """Calculate topic overlap score."""
        if not item_topics or not interest_topics:
            return 0.0

        item_set = set(t.lower() for t in item_topics)
        interest_set = set(t.lower() for t in interest_topics)

        overlap = len(item_set & interest_set)
        return min(overlap / len(interest_set), 1.0)

    def _recency_score(self, published_at: datetime) -> float:
        """Calculate recency score with exponential decay."""
        now = datetime.utcnow()
        age_hours = (now - published_at).total_seconds() / 3600

        # Exponential decay: score = e^(-age/24)
        # Content from last 24 hours gets high score
        decay_rate = 24.0  # hours
        return np.exp(-age_hours / decay_rate)

    def _engagement_score(self, metadata: dict) -> float:
        """Calculate engagement score from platform metrics."""
        # Different platforms have different metrics
        score = 0.0

        # Reddit
        if "score" in metadata:
            # Normalize Reddit score (log scale)
            score = min(np.log10(max(metadata["score"], 1)) / 4, 1.0)

        # YouTube
        elif "view_count" in metadata:
            # Normalize view count (log scale)
            score = min(np.log10(max(metadata["view_count"], 1)) / 6, 1.0)

        # Generic engagement
        elif "likes" in metadata or "shares" in metadata:
            total = metadata.get("likes", 0) + metadata.get("shares", 0) * 2
            score = min(np.log10(max(total, 1)) / 4, 1.0)

        return score


class ContentClusterer:
    """Cluster similar content items into storylines."""

    def __init__(
        self,
        min_cluster_size: int = 2,
        min_similarity: float = 0.7,
    ):
        """Initialize clusterer.

        Args:
            min_cluster_size: Minimum items per cluster
            min_similarity: Minimum similarity threshold
        """
        self.min_cluster_size = min_cluster_size
        self.min_similarity = min_similarity

    def cluster_items(
        self, items: List[ContentItem], user_id: UUID
    ) -> List[Cluster]:
        """Cluster content items by similarity.

        Args:
            items: List of content items with embeddings
            user_id: User ID for cluster ownership

        Returns:
            List of clusters
        """
        if not items:
            return []

        # Filter items with embeddings
        items_with_embeddings = [item for item in items if item.embedding]

        if len(items_with_embeddings) < self.min_cluster_size:
            # Not enough items to cluster, return single cluster
            return self._create_single_cluster(items, user_id)

        # Extract embeddings
        embeddings = np.array([item.embedding for item in items_with_embeddings])

        # Perform clustering
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=1,
            metric="cosine",
        )
        labels = clusterer.fit_predict(embeddings)

        # Group items by cluster
        clusters_dict = {}
        for item, label in zip(items_with_embeddings, labels):
            if label == -1:  # Noise
                continue
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(item)

        # Create cluster objects
        clusters = []
        for cluster_items in clusters_dict.values():
            if len(cluster_items) >= self.min_cluster_size:
                cluster = self._create_cluster(cluster_items, user_id)
                clusters.append(cluster)

        return clusters

    def _create_cluster(self, items: List[ContentItem], user_id: UUID) -> Cluster:
        """Create a cluster from items."""
        # Extract common topic
        all_topics = []
        for item in items:
            all_topics.extend(item.topics)

        # Find most common topic
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        main_topic = max(topic_counts.items(), key=lambda x: x[1])[0] if topic_counts else "General"

        # Get platforms represented
        platforms = list(set(item.source_platform for item in items))

        # Calculate average relevance (placeholder)
        relevance_score = 0.5

        return Cluster(
            user_id=user_id,
            topic=main_topic,
            summary="",  # To be filled by LLM
            keywords=list(topic_counts.keys())[:10],
            item_ids=[item.id for item in items],
            items=items,
            relevance_score=relevance_score,
            platforms_represented=platforms,
        )

    def _create_single_cluster(
        self, items: List[ContentItem], user_id: UUID
    ) -> List[Cluster]:
        """Create a single cluster from all items."""
        if not items:
            return []

        return [self._create_cluster(items, user_id)]

