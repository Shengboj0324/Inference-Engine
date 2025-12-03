"""Industrial-grade priority queue with Min-Heap for intelligent URL crawl frontier.

Replaces simple FIFO queues with priority-based scheduling for optimal crawling.
Prioritizes by freshness, relevance, urgency, and platform-specific signals.
"""

import heapq
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class PriorityLevel(str, Enum):
    """Priority levels for crawl items."""

    CRITICAL = "critical"  # Breaking news, trending topics
    HIGH = "high"  # Recent posts, high engagement
    MEDIUM = "medium"  # Normal content
    LOW = "low"  # Old content, low engagement
    DEFERRED = "deferred"  # Can wait


@dataclass(order=True)
class CrawlItem:
    """Item in the crawl frontier with priority."""

    # Priority score (lower = higher priority for min-heap)
    priority_score: float = field(compare=True)

    # Item data (not used in comparison)
    url: str = field(compare=False)
    item_id: str = field(default_factory=lambda: str(uuid4()), compare=False)
    priority_level: PriorityLevel = field(default=PriorityLevel.MEDIUM, compare=False)
    created_at: datetime = field(default_factory=datetime.utcnow, compare=False)
    scheduled_at: Optional[datetime] = field(default=None, compare=False)
    retry_count: int = field(default=0, compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)

    # Platform-specific data
    platform: Optional[str] = field(default=None, compare=False)
    content_type: Optional[str] = field(default=None, compare=False)
    estimated_freshness: float = field(default=0.5, compare=False)  # 0-1
    estimated_relevance: float = field(default=0.5, compare=False)  # 0-1
    engagement_score: float = field(default=0.0, compare=False)  # Platform metrics


class PriorityScorer:
    """Calculate priority scores for crawl items."""

    def __init__(
        self,
        freshness_weight: float = 0.4,
        relevance_weight: float = 0.3,
        engagement_weight: float = 0.2,
        urgency_weight: float = 0.1,
    ):
        """Initialize priority scorer.

        Args:
            freshness_weight: Weight for content freshness (0-1)
            relevance_weight: Weight for content relevance (0-1)
            engagement_weight: Weight for engagement metrics (0-1)
            urgency_weight: Weight for urgency signals (0-1)
        """
        self.freshness_weight = freshness_weight
        self.relevance_weight = relevance_weight
        self.engagement_weight = engagement_weight
        self.urgency_weight = urgency_weight

        # Normalize weights
        total = sum([freshness_weight, relevance_weight, engagement_weight, urgency_weight])
        self.freshness_weight /= total
        self.relevance_weight /= total
        self.engagement_weight /= total
        self.urgency_weight /= total

    def calculate_priority(self, item: CrawlItem) -> float:
        """Calculate priority score for item (lower = higher priority).

        Args:
            item: Crawl item to score

        Returns:
            Priority score (0-1, lower is better)
        """
        # Calculate component scores (all 0-1, lower is better)
        freshness_score = 1.0 - item.estimated_freshness
        relevance_score = 1.0 - item.estimated_relevance
        engagement_score = 1.0 - min(item.engagement_score, 1.0)

        # Calculate urgency based on priority level
        urgency_map = {
            PriorityLevel.CRITICAL: 0.0,
            PriorityLevel.HIGH: 0.2,
            PriorityLevel.MEDIUM: 0.5,
            PriorityLevel.LOW: 0.8,
            PriorityLevel.DEFERRED: 1.0,
        }
        urgency_score = urgency_map.get(item.priority_level, 0.5)

        # Weighted combination
        priority = (
            self.freshness_weight * freshness_score
            + self.relevance_weight * relevance_score
            + self.engagement_weight * engagement_score
            + self.urgency_weight * urgency_score
        )

        # Add small time-based component to break ties (older items get slight boost)
        age_seconds = (datetime.utcnow() - item.created_at).total_seconds()
        time_penalty = min(age_seconds / 3600.0, 1.0) * 0.01  # Max 1% penalty

        return priority + time_penalty


class PriorityQueue:
    """Industrial-grade priority queue using Min-Heap.

    Features:
    - Priority-based scheduling (not FIFO)
    - Automatic priority calculation
    - Deduplication
    - Retry handling
    - Statistics tracking
    - Platform-aware prioritization
    """

    def __init__(
        self,
        scorer: Optional[PriorityScorer] = None,
        max_size: int = 100000,
        enable_deduplication: bool = True,
    ):
        """Initialize priority queue.

        Args:
            scorer: Priority scorer (uses default if None)
            max_size: Maximum queue size
            enable_deduplication: Whether to deduplicate URLs
        """
        self.scorer = scorer or PriorityScorer()
        self.max_size = max_size
        self.enable_deduplication = enable_deduplication

        # Min-heap for priority queue
        self.heap: List[CrawlItem] = []

        # Deduplication tracking
        self.seen_urls: set[str] = set()
        self.item_map: Dict[str, CrawlItem] = {}

        # Statistics
        self.total_added = 0
        self.total_popped = 0
        self.total_duplicates = 0

    def push(self, item: CrawlItem, recalculate_priority: bool = True) -> bool:
        """Add item to priority queue.

        Args:
            item: Crawl item to add
            recalculate_priority: Whether to recalculate priority score

        Returns:
            True if added, False if duplicate or queue full
        """
        # Check deduplication
        if self.enable_deduplication and item.url in self.seen_urls:
            self.total_duplicates += 1
            logger.debug(f"Duplicate URL skipped: {item.url}")
            return False

        # Check queue size
        if len(self.heap) >= self.max_size:
            logger.warning(f"Queue full ({self.max_size}), dropping item: {item.url}")
            return False

        # Recalculate priority if requested
        if recalculate_priority:
            item.priority_score = self.scorer.calculate_priority(item)

        # Add to heap
        heapq.heappush(self.heap, item)

        # Track for deduplication
        if self.enable_deduplication:
            self.seen_urls.add(item.url)
            self.item_map[item.item_id] = item

        self.total_added += 1
        logger.debug(
            f"Added to queue: {item.url} (priority={item.priority_score:.4f}, "
            f"level={item.priority_level.value})"
        )

        return True

    def pop(self) -> Optional[CrawlItem]:
        """Remove and return highest priority item.

        Returns:
            Highest priority item, or None if queue empty
        """
        if not self.heap:
            return None

        item = heapq.heappop(self.heap)
        self.total_popped += 1

        # Remove from tracking
        if self.enable_deduplication:
            self.seen_urls.discard(item.url)
            self.item_map.pop(item.item_id, None)

        logger.debug(
            f"Popped from queue: {item.url} (priority={item.priority_score:.4f})"
        )

        return item

    def peek(self) -> Optional[CrawlItem]:
        """View highest priority item without removing.

        Returns:
            Highest priority item, or None if queue empty
        """
        return self.heap[0] if self.heap else None

    def size(self) -> int:
        """Get current queue size."""
        return len(self.heap)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self.heap) == 0

    def clear(self) -> None:
        """Clear all items from queue."""
        self.heap.clear()
        self.seen_urls.clear()
        self.item_map.clear()
        logger.info("Queue cleared")

    def update_priority(self, item_id: str, new_priority_level: PriorityLevel) -> bool:
        """Update priority of an item in the queue.

        Args:
            item_id: ID of item to update
            new_priority_level: New priority level

        Returns:
            True if updated, False if item not found
        """
        if item_id not in self.item_map:
            return False

        item = self.item_map[item_id]
        item.priority_level = new_priority_level
        item.priority_score = self.scorer.calculate_priority(item)

        # Rebuild heap to maintain heap property
        heapq.heapify(self.heap)

        logger.debug(f"Updated priority for {item.url}: {new_priority_level.value}")
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics.

        Returns:
            Dictionary with queue statistics
        """
        priority_distribution = {level.value: 0 for level in PriorityLevel}
        platform_distribution: Dict[str, int] = {}

        for item in self.heap:
            priority_distribution[item.priority_level.value] += 1
            if item.platform:
                platform_distribution[item.platform] = (
                    platform_distribution.get(item.platform, 0) + 1
                )

        return {
            "current_size": len(self.heap),
            "max_size": self.max_size,
            "total_added": self.total_added,
            "total_popped": self.total_popped,
            "total_duplicates": self.total_duplicates,
            "priority_distribution": priority_distribution,
            "platform_distribution": platform_distribution,
            "utilization": len(self.heap) / self.max_size if self.max_size > 0 else 0,
        }

    def get_top_n(self, n: int) -> List[CrawlItem]:
        """Get top N highest priority items without removing them.

        Args:
            n: Number of items to retrieve

        Returns:
            List of top N items
        """
        return heapq.nsmallest(n, self.heap)

