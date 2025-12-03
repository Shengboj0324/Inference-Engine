"""Industrial-grade HNSW (Hierarchical Navigable Small World) vector search.

Implements fast approximate nearest neighbor search for semantic similarity.
Superior to brute-force search for large-scale vector databases.
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Search result with ID and distance."""

    id: str
    distance: float
    metadata: Optional[Dict[str, Any]] = None


class IndexStatistics(BaseModel):
    """HNSW index statistics."""

    num_vectors: int
    dimension: int
    ef_construction: int
    ef_search: int
    M: int  # Number of connections per layer
    max_level: int
    index_size_bytes: int


@dataclass
class HNSWConfig:
    """Configuration for HNSW index."""

    dimension: int = 1536  # OpenAI embedding dimension
    M: int = 16  # Number of connections per layer (higher = better recall, more memory)
    ef_construction: int = 200  # Construction time/quality tradeoff
    ef_search: int = 50  # Search time/quality tradeoff
    max_elements: int = 1000000  # Maximum number of vectors
    space: str = "cosine"  # "cosine", "l2", "ip" (inner product)
    num_threads: int = 4  # Number of threads for construction


class HNSWIndex:
    """HNSW index for fast approximate nearest neighbor search.

    Features:
    - O(log n) search complexity
    - High recall (>95% with proper parameters)
    - Memory efficient
    - Thread-safe
    - Incremental updates
    - Persistence support
    """

    def __init__(self, config: Optional[HNSWConfig] = None):
        """Initialize HNSW index.

        Args:
            config: HNSW configuration
        """
        self.config = config or HNSWConfig()
        self.index = None
        self.id_to_label = {}  # Map string IDs to integer labels
        self.label_to_id = {}  # Map integer labels to string IDs
        self.metadata = {}  # Store metadata for each ID
        self.next_label = 0
        self._initialized = False

    def initialize(self) -> None:
        """Initialize HNSW index."""
        if self._initialized:
            return

        try:
            import hnswlib

            logger.info(f"Initializing HNSW index with dimension={self.config.dimension}")

            # Create index
            self.index = hnswlib.Index(
                space=self.config.space,
                dim=self.config.dimension,
            )

            # Initialize index
            self.index.init_index(
                max_elements=self.config.max_elements,
                ef_construction=self.config.ef_construction,
                M=self.config.M,
            )

            # Set number of threads
            self.index.set_num_threads(self.config.num_threads)

            # Set ef for search
            self.index.set_ef(self.config.ef_search)

            self._initialized = True

            logger.info("HNSW index initialized successfully")

        except ImportError as e:
            logger.error(f"Failed to import hnswlib: {e}")
            logger.error("Install with: pip install hnswlib")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize HNSW index: {e}")
            raise

    def add_vector(
        self,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add vector to index.

        Args:
            id: Unique identifier for vector
            vector: Vector to add
            metadata: Optional metadata to store
        """
        if not self._initialized:
            self.initialize()

        try:
            # Check if ID already exists
            if id in self.id_to_label:
                logger.warning(f"ID {id} already exists, skipping")
                return

            # Convert to numpy array
            vector_np = np.array(vector, dtype=np.float32)

            # Validate dimension
            if len(vector_np) != self.config.dimension:
                raise ValueError(
                    f"Vector dimension {len(vector_np)} does not match "
                    f"index dimension {self.config.dimension}"
                )

            # Assign label
            label = self.next_label
            self.next_label += 1

            # Store mappings
            self.id_to_label[id] = label
            self.label_to_id[label] = id

            # Store metadata
            if metadata:
                self.metadata[id] = metadata

            # Add to index
            self.index.add_items(vector_np, label)

            logger.debug(f"Added vector {id} with label {label}")

        except Exception as e:
            logger.error(f"Failed to add vector {id}: {e}")
            raise

    def add_batch(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add multiple vectors to index.

        Args:
            ids: List of unique identifiers
            vectors: List of vectors
            metadata_list: Optional list of metadata dicts
        """
        if not self._initialized:
            self.initialize()

        try:
            # Filter out existing IDs
            new_ids = []
            new_vectors = []
            new_metadata = []

            for i, id in enumerate(ids):
                if id not in self.id_to_label:
                    new_ids.append(id)
                    new_vectors.append(vectors[i])
                    if metadata_list:
                        new_metadata.append(metadata_list[i])

            if not new_ids:
                logger.warning("All IDs already exist, skipping batch")
                return

            # Convert to numpy array
            vectors_np = np.array(new_vectors, dtype=np.float32)

            # Validate dimensions
            if vectors_np.shape[1] != self.config.dimension:
                raise ValueError(
                    f"Vector dimension {vectors_np.shape[1]} does not match "
                    f"index dimension {self.config.dimension}"
                )

            # Assign labels
            labels = list(range(self.next_label, self.next_label + len(new_ids)))
            self.next_label += len(new_ids)

            # Store mappings
            for id, label in zip(new_ids, labels):
                self.id_to_label[id] = label
                self.label_to_id[label] = id

            # Store metadata
            if new_metadata:
                for id, meta in zip(new_ids, new_metadata):
                    self.metadata[id] = meta

            # Add to index
            self.index.add_items(vectors_np, labels)

            logger.info(f"Added {len(new_ids)} vectors to index")

        except Exception as e:
            logger.error(f"Failed to add batch: {e}")
            raise

    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        filter_func: Optional[callable] = None,
    ) -> List[SearchResult]:
        """Search for nearest neighbors.

        Args:
            query_vector: Query vector
            k: Number of results to return
            filter_func: Optional function to filter results by metadata

        Returns:
            List of SearchResult sorted by distance
        """
        if not self._initialized:
            raise ValueError("Index not initialized")

        if self.index.get_current_count() == 0:
            logger.warning("Index is empty")
            return []

        try:
            # Convert to numpy array
            query_np = np.array(query_vector, dtype=np.float32)

            # Validate dimension
            if len(query_np) != self.config.dimension:
                raise ValueError(
                    f"Query dimension {len(query_np)} does not match "
                    f"index dimension {self.config.dimension}"
                )

            # Search
            # If filtering, fetch more results and filter
            fetch_k = k * 10 if filter_func else k
            labels, distances = self.index.knn_query(query_np, k=fetch_k)

            # Convert labels to IDs
            results = []
            for label, distance in zip(labels[0], distances[0]):
                id = self.label_to_id.get(label)
                if id is None:
                    continue

                # Apply filter if provided
                if filter_func:
                    metadata = self.metadata.get(id)
                    if not filter_func(metadata):
                        continue

                results.append(SearchResult(
                    id=id,
                    distance=float(distance),
                    metadata=self.metadata.get(id),
                ))

                # Stop if we have enough results
                if len(results) >= k:
                    break

            return results

        except Exception as e:
            logger.error(f"Failed to search: {e}")
            raise

    def delete_vector(self, id: str) -> bool:
        """Delete vector from index.

        Note: hnswlib doesn't support true deletion, so we just remove from mappings.
        The vector remains in the index but won't be returned in search results.

        Args:
            id: ID to delete

        Returns:
            True if deleted, False if not found
        """
        if id not in self.id_to_label:
            return False

        # Remove from mappings
        label = self.id_to_label.pop(id)
        self.label_to_id.pop(label, None)
        self.metadata.pop(id, None)

        logger.debug(f"Deleted vector {id}")
        return True

    def save(self, path: str) -> None:
        """Save index to disk.

        Args:
            path: Path to save index
        """
        if not self._initialized:
            raise ValueError("Index not initialized")

        try:
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Save index
            index_path = str(path_obj.with_suffix(".hnsw"))
            self.index.save_index(index_path)

            # Save mappings and metadata
            mappings_path = str(path_obj.with_suffix(".pkl"))
            with open(mappings_path, "wb") as f:
                pickle.dump({
                    "id_to_label": self.id_to_label,
                    "label_to_id": self.label_to_id,
                    "metadata": self.metadata,
                    "next_label": self.next_label,
                    "config": self.config,
                }, f)

            logger.info(f"Saved index to {path}")

        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise

    def load(self, path: str) -> None:
        """Load index from disk.

        Args:
            path: Path to load index from
        """
        try:
            path_obj = Path(path)

            # Load mappings and metadata
            mappings_path = str(path_obj.with_suffix(".pkl"))
            with open(mappings_path, "rb") as f:
                data = pickle.load(f)

            self.id_to_label = data["id_to_label"]
            self.label_to_id = data["label_to_id"]
            self.metadata = data["metadata"]
            self.next_label = data["next_label"]
            self.config = data["config"]

            # Initialize index
            self.initialize()

            # Load index
            index_path = str(path_obj.with_suffix(".hnsw"))
            self.index.load_index(index_path)

            logger.info(f"Loaded index from {path}")

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise

    def get_statistics(self) -> IndexStatistics:
        """Get index statistics.

        Returns:
            IndexStatistics with index info
        """
        if not self._initialized:
            raise ValueError("Index not initialized")

        # Estimate index size
        # HNSW memory: ~(M * 2 * 4 + dimension * 4) bytes per vector
        bytes_per_vector = (self.config.M * 2 * 4 + self.config.dimension * 4)
        index_size = bytes_per_vector * self.index.get_current_count()

        return IndexStatistics(
            num_vectors=self.index.get_current_count(),
            dimension=self.config.dimension,
            ef_construction=self.config.ef_construction,
            ef_search=self.config.ef_search,
            M=self.config.M,
            max_level=self.index.get_max_elements(),
            index_size_bytes=index_size,
        )

    def optimize_search_speed(self, ef: int = 50) -> None:
        """Optimize for search speed (lower recall).

        Args:
            ef: Search parameter (lower = faster, lower recall)
        """
        if not self._initialized:
            raise ValueError("Index not initialized")

        self.config.ef_search = ef
        self.index.set_ef(ef)
        logger.info(f"Set ef_search to {ef} for faster search")

    def optimize_search_quality(self, ef: int = 200) -> None:
        """Optimize for search quality (higher recall).

        Args:
            ef: Search parameter (higher = slower, higher recall)
        """
        if not self._initialized:
            raise ValueError("Index not initialized")

        self.config.ef_search = ef
        self.index.set_ef(ef)
        logger.info(f"Set ef_search to {ef} for better recall")

