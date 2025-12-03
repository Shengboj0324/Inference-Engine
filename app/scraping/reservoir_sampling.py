"""Industrial-grade reservoir sampling for infinite stream sampling.

Implements Algorithm R for selecting representative samples from streams where
the total size is unknown (e.g., live comments, real-time feeds, infinite scrolls).
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class SampleStatistics:
    """Statistics for reservoir sampling."""

    total_items_seen: int = 0
    samples_collected: int = 0
    reservoir_size: int = 0
    acceptance_rate: float = 0.0
    start_time: datetime = field(default_factory=datetime.utcnow)
    last_update: datetime = field(default_factory=datetime.utcnow)


class ReservoirSampler(Generic[T]):
    """Industrial-grade reservoir sampling for infinite streams.

    Features:
    - Algorithm R for uniform random sampling
    - Weighted sampling support
    - Time-decay for recency bias
    - Statistics tracking
    - Memory-efficient (fixed size)

    Use cases:
    - Sampling live stream comments (Twitch, YouTube Live)
    - Sampling infinite scroll feeds (Twitter, TikTok)
    - Sampling real-time events
    """

    def __init__(
        self,
        reservoir_size: int,
        enable_weighted: bool = False,
        time_decay_factor: float = 0.0,
        random_seed: Optional[int] = None,
    ):
        """Initialize reservoir sampler.

        Args:
            reservoir_size: Maximum number of samples to keep
            enable_weighted: Enable weighted sampling
            time_decay_factor: Decay factor for time-based weighting (0=no decay, 1=max decay)
            random_seed: Random seed for reproducibility
        """
        self.reservoir_size = reservoir_size
        self.enable_weighted = enable_weighted
        self.time_decay_factor = time_decay_factor

        # Initialize random generator
        if random_seed is not None:
            random.seed(random_seed)

        # Reservoir storage
        self.reservoir: List[T] = []
        self.weights: List[float] = []

        # Statistics
        self.stats = SampleStatistics(reservoir_size=reservoir_size)

    def add(self, item: T, weight: float = 1.0) -> bool:
        """Add item to reservoir using Algorithm R.

        Args:
            item: Item to add
            weight: Weight for weighted sampling (higher = more likely to keep)

        Returns:
            True if item was added to reservoir, False otherwise
        """
        self.stats.total_items_seen += 1
        self.stats.last_update = datetime.utcnow()

        # Apply time decay to weight
        if self.time_decay_factor > 0:
            elapsed = (datetime.utcnow() - self.stats.start_time).total_seconds()
            decay = 1.0 - (self.time_decay_factor * min(elapsed / 3600.0, 1.0))
            weight *= decay

        # Phase 1: Fill reservoir
        if len(self.reservoir) < self.reservoir_size:
            self.reservoir.append(item)
            if self.enable_weighted:
                self.weights.append(weight)
            self.stats.samples_collected += 1
            self._update_acceptance_rate()
            return True

        # Phase 2: Probabilistic replacement
        if self.enable_weighted:
            # Weighted reservoir sampling
            return self._weighted_add(item, weight)
        else:
            # Standard Algorithm R
            return self._uniform_add(item)

    def _uniform_add(self, item: T) -> bool:
        """Add item using uniform random sampling (Algorithm R).

        Args:
            item: Item to add

        Returns:
            True if item was added
        """
        # Generate random index in range [0, total_items_seen)
        j = random.randint(0, self.stats.total_items_seen - 1)

        # Replace if index is within reservoir
        if j < self.reservoir_size:
            self.reservoir[j] = item
            self._update_acceptance_rate()
            return True

        return False

    def _weighted_add(self, item: T, weight: float) -> bool:
        """Add item using weighted reservoir sampling.

        Args:
            item: Item to add
            weight: Item weight

        Returns:
            True if item was added
        """
        # Calculate key for weighted sampling: random^(1/weight)
        key = random.random() ** (1.0 / weight)

        # Find minimum key in reservoir
        if not self.weights:
            return False

        min_idx = min(range(len(self.weights)), key=lambda i: self.weights[i])
        min_key = self.weights[min_idx]

        # Replace if new key is larger
        if key > min_key:
            self.reservoir[min_idx] = item
            self.weights[min_idx] = key
            self._update_acceptance_rate()
            return True

        return False

    def get_sample(self) -> List[T]:
        """Get current reservoir sample.

        Returns:
            List of sampled items
        """
        return self.reservoir.copy()

    def get_random_item(self) -> Optional[T]:
        """Get a random item from the reservoir.

        Returns:
            Random item, or None if reservoir is empty
        """
        if not self.reservoir:
            return None
        return random.choice(self.reservoir)

    def clear(self) -> None:
        """Clear reservoir and reset statistics."""
        self.reservoir.clear()
        self.weights.clear()
        self.stats = SampleStatistics(reservoir_size=self.reservoir_size)
        logger.info("Reservoir cleared")

    def _update_acceptance_rate(self) -> None:
        """Update acceptance rate statistic."""
        if self.stats.total_items_seen > 0:
            self.stats.acceptance_rate = (
                len(self.reservoir) / self.stats.total_items_seen
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get sampling statistics.

        Returns:
            Dictionary with statistics
        """
        elapsed = (datetime.utcnow() - self.stats.start_time).total_seconds()

        return {
            "total_items_seen": self.stats.total_items_seen,
            "samples_collected": len(self.reservoir),
            "reservoir_size": self.reservoir_size,
            "acceptance_rate": self.stats.acceptance_rate,
            "utilization": len(self.reservoir) / self.reservoir_size if self.reservoir_size > 0 else 0,
            "items_per_second": self.stats.total_items_seen / elapsed if elapsed > 0 else 0,
            "elapsed_seconds": elapsed,
            "weighted_sampling": self.enable_weighted,
            "time_decay_enabled": self.time_decay_factor > 0,
        }

