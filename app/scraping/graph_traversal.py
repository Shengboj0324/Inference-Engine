"""Industrial-grade graph traversal algorithms for social network navigation.

Implements BFS and DFS for discovering related content, trending topics, and niche discussions.
Optimized for high-volume crawling with cycle detection and depth limiting.
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class TraversalStrategy(str, Enum):
    """Graph traversal strategy."""

    BFS = "bfs"  # Breadth-First Search - wide scanning
    DFS = "dfs"  # Depth-First Search - deep diving
    HYBRID = "hybrid"  # Adaptive strategy


class NodeType(str, Enum):
    """Type of graph node."""

    USER = "user"
    POST = "post"
    COMMENT = "comment"
    SUBREDDIT = "subreddit"
    CHANNEL = "channel"
    VIDEO = "video"
    HASHTAG = "hashtag"
    TOPIC = "topic"


@dataclass
class GraphNode:
    """Represents a node in the social graph."""

    id: str
    node_type: NodeType
    url: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    depth: int = 0
    parent_id: Optional[str] = None
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    priority: float = 1.0  # Higher = more important
    visited: bool = False


@dataclass
class TraversalConfig:
    """Configuration for graph traversal."""

    strategy: TraversalStrategy = TraversalStrategy.BFS
    max_depth: int = 3
    max_nodes: int = 1000
    max_children_per_node: int = 50
    enable_cycle_detection: bool = True
    priority_threshold: float = 0.5
    timeout_seconds: int = 300
    concurrent_fetches: int = 5


class GraphTraverser:
    """Industrial-grade graph traversal engine for social networks.

    Features:
    - BFS for wide scanning of trending topics
    - DFS for deep-diving into niche discussions
    - Cycle detection to avoid infinite loops
    - Priority-based node selection
    - Concurrent fetching for performance
    - Adaptive strategy switching
    """

    def __init__(
        self,
        config: TraversalConfig,
        fetch_neighbors: Callable[[GraphNode], asyncio.Future[List[GraphNode]]],
    ):
        """Initialize graph traverser.

        Args:
            config: Traversal configuration
            fetch_neighbors: Async function to fetch neighboring nodes
        """
        self.config = config
        self.fetch_neighbors = fetch_neighbors

        # Traversal state
        self.visited: Set[str] = set()
        self.discovered: Dict[str, GraphNode] = {}
        self.queue: deque[GraphNode] = deque()
        self.stack: List[GraphNode] = []
        self.results: List[GraphNode] = []

        # Statistics
        self.nodes_visited = 0
        self.nodes_discovered = 0
        self.edges_traversed = 0
        self.start_time: Optional[datetime] = None

    async def traverse(self, start_nodes: List[GraphNode]) -> List[GraphNode]:
        """Execute graph traversal from starting nodes.

        Args:
            start_nodes: Initial nodes to start traversal

        Returns:
            List of discovered nodes
        """
        self.start_time = datetime.utcnow()
        logger.info(
            f"Starting {self.config.strategy.value} traversal from {len(start_nodes)} nodes"
        )

        # Initialize with start nodes
        for node in start_nodes:
            self._add_node(node)

        # Execute traversal based on strategy
        if self.config.strategy == TraversalStrategy.BFS:
            await self._traverse_bfs()
        elif self.config.strategy == TraversalStrategy.DFS:
            await self._traverse_dfs()
        else:
            await self._traverse_hybrid()

        elapsed = (datetime.utcnow() - self.start_time).total_seconds()
        logger.info(
            f"Traversal complete: {self.nodes_visited} visited, "
            f"{self.nodes_discovered} discovered, {elapsed:.2f}s"
        )

        return self.results

    async def _traverse_bfs(self) -> None:
        """Breadth-First Search traversal for wide scanning."""
        semaphore = asyncio.Semaphore(self.config.concurrent_fetches)

        while self.queue and not self._should_stop():
            # Get next node from queue (FIFO)
            node = self.queue.popleft()

            if node.id in self.visited:
                continue

            # Mark as visited
            self._visit_node(node)

            # Fetch neighbors concurrently
            async with semaphore:
                try:
                    neighbors = await asyncio.wait_for(
                        self.fetch_neighbors(node),
                        timeout=30.0,
                    )
                    self.edges_traversed += len(neighbors)

                    # Add neighbors to queue
                    for neighbor in neighbors[:self.config.max_children_per_node]:
                        if neighbor.priority >= self.config.priority_threshold:
                            self._add_node(neighbor)

                except asyncio.TimeoutError:
                    logger.warning(f"Timeout fetching neighbors for {node.id}")
                except Exception as e:
                    logger.error(f"Error fetching neighbors for {node.id}: {e}")

    async def _traverse_dfs(self) -> None:
        """Depth-First Search traversal for deep diving."""
        semaphore = asyncio.Semaphore(self.config.concurrent_fetches)

        while self.stack and not self._should_stop():
            # Get next node from stack (LIFO)
            node = self.stack.pop()

            if node.id in self.visited:
                continue

            # Mark as visited
            self._visit_node(node)

            # Fetch neighbors
            async with semaphore:
                try:
                    neighbors = await asyncio.wait_for(
                        self.fetch_neighbors(node),
                        timeout=30.0,
                    )
                    self.edges_traversed += len(neighbors)

                    # Add neighbors to stack (reverse order for left-to-right traversal)
                    for neighbor in reversed(neighbors[:self.config.max_children_per_node]):
                        if neighbor.priority >= self.config.priority_threshold:
                            self._add_node(neighbor)

                except asyncio.TimeoutError:
                    logger.warning(f"Timeout fetching neighbors for {node.id}")
                except Exception as e:
                    logger.error(f"Error fetching neighbors for {node.id}: {e}")

    async def _traverse_hybrid(self) -> None:
        """Hybrid traversal: BFS for high-priority, DFS for exploration."""
        # Start with BFS for trending topics
        initial_depth = 2
        temp_max_depth = self.config.max_depth
        self.config.max_depth = initial_depth

        await self._traverse_bfs()

        # Switch to DFS for deep diving into interesting nodes
        self.config.max_depth = temp_max_depth
        high_priority_nodes = [
            node for node in self.results
            if node.priority > 0.7 and node.depth == initial_depth
        ]

        # Add high-priority nodes to stack for DFS
        for node in high_priority_nodes:
            if node.id not in self.visited:
                self.stack.append(node)

        await self._traverse_dfs()

    def _add_node(self, node: GraphNode) -> None:
        """Add node to traversal queue/stack."""
        # Check cycle detection
        if self.config.enable_cycle_detection and node.id in self.discovered:
            return

        # Check depth limit
        if node.depth > self.config.max_depth:
            return

        # Add to discovered set
        self.discovered[node.id] = node
        self.nodes_discovered += 1

        # Add to appropriate data structure
        if self.config.strategy == TraversalStrategy.BFS:
            self.queue.append(node)
        else:
            self.stack.append(node)

    def _visit_node(self, node: GraphNode) -> None:
        """Mark node as visited and add to results."""
        self.visited.add(node.id)
        node.visited = True
        self.results.append(node)
        self.nodes_visited += 1

        logger.debug(
            f"Visited {node.node_type.value} node: {node.id} "
            f"(depth={node.depth}, priority={node.priority:.2f})"
        )

    def _should_stop(self) -> bool:
        """Check if traversal should stop."""
        # Check node limit
        if self.nodes_visited >= self.config.max_nodes:
            logger.info(f"Reached max nodes limit: {self.config.max_nodes}")
            return True

        # Check timeout
        if self.start_time:
            elapsed = (datetime.utcnow() - self.start_time).total_seconds()
            if elapsed >= self.config.timeout_seconds:
                logger.info(f"Reached timeout: {self.config.timeout_seconds}s")
                return True

        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get traversal statistics."""
        elapsed = 0.0
        if self.start_time:
            elapsed = (datetime.utcnow() - self.start_time).total_seconds()

        return {
            "nodes_visited": self.nodes_visited,
            "nodes_discovered": self.nodes_discovered,
            "edges_traversed": self.edges_traversed,
            "elapsed_seconds": elapsed,
            "nodes_per_second": self.nodes_visited / elapsed if elapsed > 0 else 0,
            "strategy": self.config.strategy.value,
            "max_depth_reached": max((n.depth for n in self.results), default=0),
        }

