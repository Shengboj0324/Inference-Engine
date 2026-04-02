"""Retrieval quality evaluation: Recall@k, MRR, nDCG@k.

``RetrievalEvaluator`` takes a labelled query set and a retrieval function,
then computes the standard IR metrics used in the CI ``eval-gate``:

- **Recall@k** — fraction of queries where at least one relevant chunk appears
  in the top-*k* results.
- **MRR** (Mean Reciprocal Rank) — mean of ``1/rank`` of the first relevant
  result, averaged over queries.
- **nDCG@k** (Normalised Discounted Cumulative Gain) — graded relevance-aware
  metric; uses binary relevance (1 = relevant, 0 = not) by default.

The evaluator is *retrieval-function-agnostic*: callers supply a callable that
maps a query string to a ranked list of chunk IDs.  This decouples the
evaluator from ``ChunkStore``'s implementation details and allows evaluation
of any retrieval backend (dense, sparse, hybrid).

Usage::

    from app.intelligence.retrieval.retrieval_evaluator import (
        RetrievalEvaluator,
        RetrievalQuery,
        RetrievalMetrics,
    )

    queries = [
        RetrievalQuery(
            query_id="q1",
            query_text="attention mechanism transformer",
            relevant_chunk_ids={"chunk-abc", "chunk-def"},
        ),
    ]

    def my_retriever(text: str) -> list[str]:
        return store.keyword_search(text, top_k=10)

    evaluator = RetrievalEvaluator()
    metrics = evaluator.evaluate(queries=queries, retriever=my_retriever, k=10)
    print(metrics.recall_at_k, metrics.mrr, metrics.ndcg_at_k)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class RetrievalQuery:
    """A single labelled query for retrieval evaluation.

    Attributes
    ----------
    query_id:           Unique identifier (for error reporting).
    query_text:         The query string passed to the retriever.
    relevant_chunk_ids: Set of chunk IDs that count as "relevant" for this
                        query.  At least one must be present in the top-*k*
                        results for the query to count as a hit.
    """

    query_id:           str
    query_text:         str
    relevant_chunk_ids: Set[str] = field(default_factory=set)


@dataclass
class RetrievalMetrics:
    """Aggregated retrieval quality metrics over a query set.

    Attributes
    ----------
    k:                  Cutoff used for all @k metrics.
    n_queries:          Total number of queries evaluated.
    n_queries_with_hits: Number of queries where ≥ 1 relevant chunk was in
                        the top-*k*.
    recall_at_k:        Fraction of queries with a hit in top-*k*.
    mrr:                Mean Reciprocal Rank (first relevant result).
    ndcg_at_k:          Mean nDCG@k (binary relevance).
    per_query:          Per-query breakdown ``{query_id: {recall, rr, ndcg}}``.
    """

    k:                  int
    n_queries:          int
    n_queries_with_hits: int
    recall_at_k:        float
    mrr:                float
    ndcg_at_k:          float
    per_query:          Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Convenience
    def passes_thresholds(
        self,
        min_recall: float = 0.70,
        min_mrr:    float = 0.50,
        min_ndcg:   float = 0.60,
    ) -> bool:
        """Return True when all metrics clear their deployment thresholds."""
        return (
            self.recall_at_k >= min_recall
            and self.mrr     >= min_mrr
            and self.ndcg_at_k >= min_ndcg
        )

    def __str__(self) -> str:
        return (
            f"RetrievalMetrics(k={self.k}, n={self.n_queries}, "
            f"Recall@{self.k}={self.recall_at_k:.4f}, "
            f"MRR={self.mrr:.4f}, "
            f"nDCG@{self.k}={self.ndcg_at_k:.4f})"
        )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RetrievalEvaluator:
    """Compute Recall@k, MRR, and nDCG@k for a retrieval function.

    The evaluator is stateless — call ``evaluate()`` as many times as needed
    with different retrievers or query sets.
    """

    def evaluate(
        self,
        queries:   List[RetrievalQuery],
        retriever: Callable[[str], List[str]],
        k:         int = 10,
    ) -> RetrievalMetrics:
        """Evaluate *retriever* over the labelled *queries*.

        Args:
            queries:   List of :class:`RetrievalQuery` objects.
            retriever: Callable ``(query_text: str) -> list[chunk_id]``.
                       The list must be ranked (index 0 = most relevant).
                       The evaluator truncates to the first *k* results.
            k:         Rank cutoff for Recall, nDCG, and the numerator of MRR.

        Returns:
            :class:`RetrievalMetrics` aggregated over *queries*.

        Raises:
            ValueError: If *queries* is empty or *k* ≤ 0.
        """
        if not queries:
            raise ValueError("'queries' must be non-empty")
        if k <= 0:
            raise ValueError(f"'k' must be a positive integer; got {k}")

        hits            = 0
        sum_rr          = 0.0
        sum_ndcg        = 0.0
        per_query: Dict[str, Dict[str, float]] = {}

        for q in queries:
            retrieved = list(retriever(q.query_text))[:k]
            recall, rr, ndcg = self._per_query_metrics(
                retrieved=retrieved,
                relevant=q.relevant_chunk_ids,
                k=k,
            )
            if recall > 0:
                hits += 1
            sum_rr   += rr
            sum_ndcg += ndcg
            per_query[q.query_id] = {"recall": recall, "rr": rr, "ndcg": ndcg}

        n = len(queries)
        return RetrievalMetrics(
            k=k,
            n_queries=n,
            n_queries_with_hits=hits,
            recall_at_k=hits / n,
            mrr=sum_rr / n,
            ndcg_at_k=sum_ndcg / n,
            per_query=per_query,
        )

    # ------------------------------------------------------------------
    # Per-query metric computation
    # ------------------------------------------------------------------

    @staticmethod
    def _per_query_metrics(
        retrieved: List[str],
        relevant:  Set[str],
        k:         int,
    ) -> tuple[float, float, float]:
        """Return (recall, reciprocal_rank, ndcg) for one query."""
        if not relevant:
            return 0.0, 0.0, 0.0

        recall = 0.0
        rr     = 0.0
        dcg    = 0.0

        for rank, chunk_id in enumerate(retrieved[:k], start=1):
            is_rel = chunk_id in relevant
            if is_rel:
                recall = 1.0
                if rr == 0.0:
                    rr = 1.0 / rank
                dcg += 1.0 / math.log2(rank + 1)

        # Ideal DCG: all relevant docs at top ranks (up to k)
        n_rel_at_k = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel_at_k))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        return recall, rr, ndcg

