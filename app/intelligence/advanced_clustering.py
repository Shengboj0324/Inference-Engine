"""Industrial-grade Advanced Clustering for topic discovery and content organization.

Implements:
- DBSCAN (Density-Based Spatial Clustering) for arbitrary-shaped clusters
- Leiden algorithm for community detection in networks
- Hierarchical clustering with dendrogram analysis
- Cluster quality metrics and validation
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ClusterResult(BaseModel):
    """Clustering result."""

    cluster_id: int
    item_ids: List[str]
    centroid: Optional[List[float]] = None
    size: int
    density: Optional[float] = None
    coherence: Optional[float] = None


class CommunityResult(BaseModel):
    """Community detection result."""

    community_id: int
    node_ids: List[str]
    size: int
    modularity: float
    internal_edges: int
    external_edges: int


@dataclass
class DBSCANConfig:
    """Configuration for DBSCAN."""

    eps: float = 0.5  # Maximum distance between two samples
    min_samples: int = 5  # Minimum samples in neighborhood
    metric: str = "cosine"  # Distance metric
    algorithm: str = "auto"  # Algorithm for nearest neighbors
    leaf_size: int = 30
    n_jobs: int = -1  # Use all CPUs


@dataclass
class LeidenConfig:
    """Configuration for Leiden algorithm."""

    resolution: float = 1.0  # Resolution parameter
    n_iterations: int = 2  # Number of iterations
    randomness: float = 0.001  # Randomness parameter
    seed: Optional[int] = 42


class DBSCANClustering:
    """DBSCAN clustering for arbitrary-shaped clusters.

    Features:
    - No need to specify number of clusters
    - Handles noise and outliers
    - Finds clusters of arbitrary shape
    - Density-based approach
    """

    def __init__(self, config: Optional[DBSCANConfig] = None):
        """Initialize DBSCAN clustering.

        Args:
            config: DBSCAN configuration
        """
        self.config = config or DBSCANConfig()
        self.model = None
        self.labels_ = None
        self.item_ids = []

    def fit(
        self,
        embeddings: List[List[float]],
        item_ids: List[str],
    ) -> List[ClusterResult]:
        """Fit DBSCAN on embeddings.

        Args:
            embeddings: List of embedding vectors
            item_ids: List of item IDs

        Returns:
            List of cluster results
        """
        try:
            from sklearn.cluster import DBSCAN

            logger.info(f"Fitting DBSCAN on {len(embeddings)} items")

            self.item_ids = item_ids

            # Convert to numpy array
            X = np.array(embeddings)

            # Fit DBSCAN
            self.model = DBSCAN(
                eps=self.config.eps,
                min_samples=self.config.min_samples,
                metric=self.config.metric,
                algorithm=self.config.algorithm,
                leaf_size=self.config.leaf_size,
                n_jobs=self.config.n_jobs,
            )

            self.labels_ = self.model.fit_predict(X)

            # Group items by cluster
            clusters = {}
            for item_id, label in zip(item_ids, self.labels_):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(item_id)

            # Create cluster results
            results = []
            for cluster_id, cluster_items in clusters.items():
                if cluster_id == -1:
                    # Noise cluster
                    continue

                # Get cluster embeddings
                cluster_indices = [i for i, label in enumerate(self.labels_) if label == cluster_id]
                cluster_embeddings = X[cluster_indices]

                # Compute centroid
                centroid = np.mean(cluster_embeddings, axis=0)

                # Compute density (average distance to centroid)
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                density = 1.0 / (np.mean(distances) + 1e-8)

                # Compute coherence (average pairwise similarity)
                if len(cluster_embeddings) > 1:
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarities = cosine_similarity(cluster_embeddings)
                    # Exclude diagonal
                    mask = ~np.eye(similarities.shape[0], dtype=bool)
                    coherence = similarities[mask].mean()
                else:
                    coherence = 1.0

                results.append(ClusterResult(
                    cluster_id=int(cluster_id),
                    item_ids=cluster_items,
                    centroid=centroid.tolist(),
                    size=len(cluster_items),
                    density=float(density),
                    coherence=float(coherence),
                ))

            # Sort by size
            results.sort(key=lambda x: x.size, reverse=True)

            logger.info(f"Found {len(results)} clusters (excluding noise)")

            return results

        except ImportError as e:
            logger.error(f"Failed to import sklearn: {e}")
            logger.error("Install with: pip install scikit-learn")
            raise
        except Exception as e:
            logger.error(f"Failed to fit DBSCAN: {e}")
            raise

    def get_cluster_statistics(self) -> Dict[str, Any]:
        """Get clustering statistics.

        Returns:
            Dictionary of statistics
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted")

        unique_labels = set(self.labels_)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        num_noise = np.sum(self.labels_ == -1)

        return {
            'num_clusters': num_clusters,
            'num_noise': int(num_noise),
            'noise_ratio': float(num_noise / len(self.labels_)),
            'cluster_sizes': {
                int(label): int(np.sum(self.labels_ == label))
                for label in unique_labels if label != -1
            },
        }


class LeidenCommunityDetection:
    """Leiden algorithm for community detection in networks.

    Features:
    - Fast and accurate community detection
    - Better than Louvain algorithm
    - Guarantees well-connected communities
    - Scalable to large networks
    """

    def __init__(self, config: Optional[LeidenConfig] = None):
        """Initialize Leiden community detection.

        Args:
            config: Leiden configuration
        """
        self.config = config or LeidenConfig()
        self.partition = None
        self.node_ids = []

    def detect_communities(
        self,
        edges: List[Tuple[str, str, float]],
        node_ids: Optional[List[str]] = None,
    ) -> List[CommunityResult]:
        """Detect communities in network.

        Args:
            edges: List of (source, target, weight) tuples
            node_ids: Optional list of all node IDs

        Returns:
            List of community results
        """
        try:
            import igraph as ig
            import leidenalg

            logger.info(f"Detecting communities in network with {len(edges)} edges")

            # Extract unique nodes
            if node_ids is None:
                nodes = set()
                for source, target, _ in edges:
                    nodes.add(source)
                    nodes.add(target)
                node_ids = sorted(nodes)

            self.node_ids = node_ids

            # Create node index mapping
            node_to_idx = {node: idx for idx, node in enumerate(node_ids)}

            # Build graph
            edge_list = []
            weights = []

            for source, target, weight in edges:
                if source in node_to_idx and target in node_to_idx:
                    edge_list.append((node_to_idx[source], node_to_idx[target]))
                    weights.append(weight)

            # Create igraph
            g = ig.Graph(n=len(node_ids), edges=edge_list, directed=False)
            g.es['weight'] = weights

            # Run Leiden algorithm
            self.partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                weights='weight',
                resolution_parameter=self.config.resolution,
                n_iterations=self.config.n_iterations,
                seed=self.config.seed,
            )

            # Group nodes by community
            communities = {}
            for node_idx, community_id in enumerate(self.partition.membership):
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(node_ids[node_idx])

            # Create community results
            results = []
            for community_id, community_nodes in communities.items():
                # Count internal and external edges
                internal_edges = 0
                external_edges = 0

                community_set = set(community_nodes)

                for source, target, _ in edges:
                    if source in community_set and target in community_set:
                        internal_edges += 1
                    elif source in community_set or target in community_set:
                        external_edges += 1

                results.append(CommunityResult(
                    community_id=int(community_id),
                    node_ids=community_nodes,
                    size=len(community_nodes),
                    modularity=float(self.partition.modularity),
                    internal_edges=internal_edges,
                    external_edges=external_edges,
                ))

            # Sort by size
            results.sort(key=lambda x: x.size, reverse=True)

            logger.info(f"Found {len(results)} communities with modularity {self.partition.modularity:.4f}")

            return results

        except ImportError as e:
            logger.error(f"Failed to import igraph or leidenalg: {e}")
            logger.error("Install with: pip install python-igraph leidenalg")
            raise
        except Exception as e:
            logger.error(f"Failed to detect communities: {e}")
            raise

    def get_community_statistics(self) -> Dict[str, Any]:
        """Get community detection statistics.

        Returns:
            Dictionary of statistics
        """
        if self.partition is None:
            raise ValueError("Communities not detected")

        community_sizes = {}
        for node_idx, community_id in enumerate(self.partition.membership):
            if community_id not in community_sizes:
                community_sizes[community_id] = 0
            community_sizes[community_id] += 1

        return {
            'num_communities': len(set(self.partition.membership)),
            'modularity': float(self.partition.modularity),
            'community_sizes': {int(k): int(v) for k, v in community_sizes.items()},
            'largest_community': max(community_sizes.values()),
            'smallest_community': min(community_sizes.values()),
            'average_community_size': np.mean(list(community_sizes.values())),
        }


class HierarchicalClustering:
    """Hierarchical clustering with dendrogram analysis.

    Features:
    - Agglomerative clustering
    - Multiple linkage methods
    - Dendrogram visualization support
    - Automatic cluster number selection
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        linkage: str = "ward",
        distance_threshold: Optional[float] = None,
    ):
        """Initialize hierarchical clustering.

        Args:
            n_clusters: Number of clusters (if None, use distance_threshold)
            linkage: Linkage method ("ward", "complete", "average", "single")
            distance_threshold: Distance threshold for automatic cluster selection
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.model = None
        self.labels_ = None
        self.item_ids = []

    def fit(
        self,
        embeddings: List[List[float]],
        item_ids: List[str],
    ) -> List[ClusterResult]:
        """Fit hierarchical clustering.

        Args:
            embeddings: List of embedding vectors
            item_ids: List of item IDs

        Returns:
            List of cluster results
        """
        try:
            from sklearn.cluster import AgglomerativeClustering

            logger.info(f"Fitting hierarchical clustering on {len(embeddings)} items")

            self.item_ids = item_ids

            # Convert to numpy array
            X = np.array(embeddings)

            # Fit hierarchical clustering
            self.model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage=self.linkage,
                distance_threshold=self.distance_threshold,
            )

            self.labels_ = self.model.fit_predict(X)

            # Group items by cluster
            clusters = {}
            for item_id, label in zip(item_ids, self.labels_):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(item_id)

            # Create cluster results
            results = []
            for cluster_id, cluster_items in clusters.items():
                # Get cluster embeddings
                cluster_indices = [i for i, label in enumerate(self.labels_) if label == cluster_id]
                cluster_embeddings = X[cluster_indices]

                # Compute centroid
                centroid = np.mean(cluster_embeddings, axis=0)

                # Compute coherence
                if len(cluster_embeddings) > 1:
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarities = cosine_similarity(cluster_embeddings)
                    mask = ~np.eye(similarities.shape[0], dtype=bool)
                    coherence = similarities[mask].mean()
                else:
                    coherence = 1.0

                results.append(ClusterResult(
                    cluster_id=int(cluster_id),
                    item_ids=cluster_items,
                    centroid=centroid.tolist(),
                    size=len(cluster_items),
                    coherence=float(coherence),
                ))

            # Sort by size
            results.sort(key=lambda x: x.size, reverse=True)

            logger.info(f"Found {len(results)} clusters")

            return results

        except ImportError as e:
            logger.error(f"Failed to import sklearn: {e}")
            logger.error("Install with: pip install scikit-learn")
            raise
        except Exception as e:
            logger.error(f"Failed to fit hierarchical clustering: {e}")
            raise

