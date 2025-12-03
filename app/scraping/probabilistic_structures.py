"""Industrial-grade probabilistic data structures for memory-efficient operations.

Implements:
- Bloom Filter: Fast duplicate detection with minimal memory
- Count-Min Sketch: Frequency estimation for streaming data
- HyperLogLog: Cardinality estimation (unique count) with high accuracy
"""

import hashlib
import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BloomFilter:
    """Space-efficient probabilistic data structure for membership testing.

    Features:
    - O(1) insertion and lookup
    - Configurable false positive rate
    - Memory efficient (bits, not bytes)
    - No false negatives

    Use cases:
    - Duplicate URL detection
    - Already-scraped content tracking
    - Cache existence checking
    """

    def __init__(
        self,
        expected_elements: int,
        false_positive_rate: float = 0.01,
    ):
        """Initialize Bloom filter.

        Args:
            expected_elements: Expected number of elements
            false_positive_rate: Desired false positive rate (0-1)
        """
        self.expected_elements = expected_elements
        self.false_positive_rate = false_positive_rate

        # Calculate optimal size and hash functions
        self.size = self._optimal_size(expected_elements, false_positive_rate)
        self.num_hashes = self._optimal_hash_count(self.size, expected_elements)

        # Bit array
        self.bit_array = [False] * self.size

        # Statistics
        self.elements_added = 0

    def add(self, item: str) -> None:
        """Add item to Bloom filter.

        Args:
            item: Item to add
        """
        for i in range(self.num_hashes):
            index = self._hash(item, i) % self.size
            self.bit_array[index] = True

        self.elements_added += 1

    def contains(self, item: str) -> bool:
        """Check if item might be in the set.

        Args:
            item: Item to check

        Returns:
            True if item might be in set (possible false positive)
            False if item is definitely not in set (no false negatives)
        """
        for i in range(self.num_hashes):
            index = self._hash(item, i) % self.size
            if not self.bit_array[index]:
                return False
        return True

    def _hash(self, item: str, seed: int) -> int:
        """Generate hash for item with seed.

        Args:
            item: Item to hash
            seed: Hash seed

        Returns:
            Hash value
        """
        # Use SHA256 with seed for good distribution
        hash_input = f"{item}:{seed}".encode("utf-8")
        hash_bytes = hashlib.sha256(hash_input).digest()
        return int.from_bytes(hash_bytes[:8], byteorder="big")

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        """Calculate optimal bit array size.

        Formula: m = -(n * ln(p)) / (ln(2)^2)

        Args:
            n: Expected number of elements
            p: False positive rate

        Returns:
            Optimal size in bits
        """
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(math.ceil(m))

    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        """Calculate optimal number of hash functions.

        Formula: k = (m/n) * ln(2)

        Args:
            m: Bit array size
            n: Expected number of elements

        Returns:
            Optimal number of hash functions
        """
        k = (m / n) * math.log(2)
        return max(1, int(math.ceil(k)))

    def get_statistics(self) -> Dict[str, Any]:
        """Get Bloom filter statistics."""
        bits_set = sum(self.bit_array)
        actual_fpr = (bits_set / self.size) ** self.num_hashes if self.size > 0 else 0

        return {
            "size_bits": self.size,
            "size_kb": self.size / 8 / 1024,
            "num_hashes": self.num_hashes,
            "elements_added": self.elements_added,
            "bits_set": bits_set,
            "utilization": bits_set / self.size if self.size > 0 else 0,
            "expected_fpr": self.false_positive_rate,
            "actual_fpr": actual_fpr,
        }


class CountMinSketch:
    """Probabilistic frequency counter for streaming data.

    Features:
    - O(1) update and query
    - Sublinear space complexity
    - Overestimates frequency (never underestimates)

    Use cases:
    - Hashtag frequency tracking
    - Topic trending detection
    - Word frequency in streams
    """

    def __init__(
        self,
        width: int = 1000,
        depth: int = 5,
    ):
        """Initialize Count-Min Sketch.

        Args:
            width: Width of each hash table
            depth: Number of hash tables (accuracy)
        """
        self.width = width
        self.depth = depth

        # Create 2D array of counters
        self.table = [[0] * width for _ in range(depth)]

        # Statistics
        self.total_count = 0

    def update(self, item: str, count: int = 1) -> None:
        """Update count for item.

        Args:
            item: Item to update
            count: Count to add (default 1)
        """
        for i in range(self.depth):
            index = self._hash(item, i) % self.width
            self.table[i][index] += count

        self.total_count += count

    def estimate(self, item: str) -> int:
        """Estimate frequency of item.

        Args:
            item: Item to estimate

        Returns:
            Estimated frequency (may overestimate, never underestimates)
        """
        estimates = []
        for i in range(self.depth):
            index = self._hash(item, i) % self.width
            estimates.append(self.table[i][index])

        # Return minimum estimate (reduces overestimation)
        return min(estimates)

    def _hash(self, item: str, seed: int) -> int:
        """Generate hash for item with seed."""
        hash_input = f"{item}:{seed}".encode("utf-8")
        hash_bytes = hashlib.sha256(hash_input).digest()
        return int.from_bytes(hash_bytes[:8], byteorder="big")

    def get_statistics(self) -> Dict[str, Any]:
        """Get Count-Min Sketch statistics."""
        return {
            "width": self.width,
            "depth": self.depth,
            "total_count": self.total_count,
            "memory_kb": (self.width * self.depth * 8) / 1024,  # Assuming 8 bytes per int
        }


class HyperLogLog:
    """Probabilistic cardinality estimator for counting unique elements.

    Features:
    - O(1) insertion
    - Sublinear space (typically few KB)
    - ~2% standard error with 1.5KB memory

    Use cases:
    - Unique user counting
    - Unique URL counting
    - Distinct hashtag counting
    """

    def __init__(self, precision: int = 14):
        """Initialize HyperLogLog.

        Args:
            precision: Precision parameter (4-16, higher = more accurate but more memory)
                      14 = ~1.5KB memory, ~0.8% error
        """
        if not 4 <= precision <= 16:
            raise ValueError("Precision must be between 4 and 16")

        self.precision = precision
        self.m = 1 << precision  # 2^precision registers
        self.registers = [0] * self.m

        # Alpha constant for bias correction
        if self.m >= 128:
            self.alpha = 0.7213 / (1 + 1.079 / self.m)
        elif self.m >= 64:
            self.alpha = 0.709
        elif self.m >= 32:
            self.alpha = 0.697
        else:
            self.alpha = 0.673

    def add(self, item: str) -> None:
        """Add item to HyperLogLog.

        Args:
            item: Item to add
        """
        # Hash the item
        hash_value = self._hash(item)

        # Get register index (first p bits)
        register_index = hash_value & ((1 << self.precision) - 1)

        # Get remaining bits
        remaining_bits = hash_value >> self.precision

        # Count leading zeros + 1
        leading_zeros = self._count_leading_zeros(remaining_bits) + 1

        # Update register with maximum
        self.registers[register_index] = max(
            self.registers[register_index],
            leading_zeros
        )

    def cardinality(self) -> int:
        """Estimate number of unique elements.

        Returns:
            Estimated cardinality
        """
        # Calculate raw estimate
        raw_estimate = self.alpha * (self.m ** 2) / sum(2 ** (-x) for x in self.registers)

        # Apply bias correction for small/large cardinalities
        if raw_estimate <= 2.5 * self.m:
            # Small range correction
            zeros = self.registers.count(0)
            if zeros != 0:
                return int(self.m * math.log(self.m / zeros))

        if raw_estimate <= (1 / 30) * (1 << 32):
            # No correction
            return int(raw_estimate)

        # Large range correction
        return int(-1 * (1 << 32) * math.log(1 - raw_estimate / (1 << 32)))

    def _hash(self, item: str) -> int:
        """Generate 64-bit hash for item."""
        hash_bytes = hashlib.sha256(item.encode("utf-8")).digest()
        return int.from_bytes(hash_bytes[:8], byteorder="big")

    def _count_leading_zeros(self, value: int) -> int:
        """Count leading zeros in binary representation."""
        if value == 0:
            return 64 - self.precision

        count = 0
        # Check 64 - precision bits
        for i in range(64 - self.precision - 1, -1, -1):
            if value & (1 << i):
                break
            count += 1

        return count

    def merge(self, other: "HyperLogLog") -> None:
        """Merge another HyperLogLog into this one.

        Args:
            other: Another HyperLogLog with same precision
        """
        if self.precision != other.precision:
            raise ValueError("Cannot merge HyperLogLogs with different precision")

        for i in range(self.m):
            self.registers[i] = max(self.registers[i], other.registers[i])

    def get_statistics(self) -> Dict[str, Any]:
        """Get HyperLogLog statistics."""
        return {
            "precision": self.precision,
            "num_registers": self.m,
            "memory_kb": (self.m * 1) / 1024,  # 1 byte per register
            "estimated_cardinality": self.cardinality(),
            "standard_error": 1.04 / math.sqrt(self.m),
        }

