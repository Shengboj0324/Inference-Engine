"""Tier 3.2 — memory consolidation / summarization.

Bounding per-user memory by *oldest-record eviction* alone is lossy: a whole
topic the user cared about months ago can be dropped just because it is old.
``MemoryConsolidator`` instead clusters a user's stored vectors and keeps one
**representative per cluster** (the medoid — the member closest to the cluster
centroid), so the breadth of distinct topics is retained at a fixed memory
budget.  An optional ``summarize_fn`` can replace a cluster's representative
text with an abstractive summary of its members (e.g. an LLM call); the default
is the extractive medoid, which needs no model.

Pure NumPy (k-means++ init + Lloyd's iterations); deterministic given a seed.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _kmeans(X: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """k-means with k-means++ init.  Returns ``(labels, centers)``."""
    rng = np.random.default_rng(seed)
    n = len(X)
    k = max(1, min(k, n))
    # k-means++ seeding
    centers = [X[int(rng.integers(n))]]
    for _ in range(1, k):
        d = np.min(np.stack([np.sum((X - c) ** 2, axis=1) for c in centers]), axis=0)
        total = float(d.sum())
        probs = (d / total) if total > 0 else np.full(n, 1.0 / n)
        centers.append(X[int(rng.choice(n, p=probs))])
    C = np.array(centers, dtype=np.float64)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dist = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)
        newC = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else C[j]
            for j in range(len(C))
        ])
        if np.allclose(newC, C):
            C = newC
            break
        C = newC
    return labels, C


class MemoryConsolidator:
    """Cluster embeddings and pick one representative (medoid) per cluster.

    Args:
        seed: RNG seed for reproducible clustering.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)

    def consolidate(self, embeddings, target_k: int) -> Tuple[List[int], List[int]]:
        """Return ``(representative_indices, labels)``.

        ``representative_indices`` are indices into ``embeddings`` to keep (one
        medoid per cluster); ``labels`` is the per-record cluster assignment.
        When there are already ``<= target_k`` records, everything is kept.
        """
        X = np.asarray(embeddings, dtype=np.float64)
        n = len(X)
        if target_k <= 0:
            raise ValueError("target_k must be positive")
        if n <= target_k:
            return list(range(n)), list(range(n))
        labels, C = _kmeans(X, target_k, seed=self.seed)
        reps: List[int] = []
        for j in range(len(C)):
            members = np.where(labels == j)[0]
            if members.size == 0:
                continue
            d = ((X[members] - C[j]) ** 2).sum(axis=1)
            reps.append(int(members[int(d.argmin())]))
        return sorted(reps), labels.tolist()
