"""Ranking evaluation metrics.

Blueprint §8 — Ranking metrics:
- NDCG@k  (Normalized Discounted Cumulative Gain)
- Precision@k
- Opportunity hit rate  (at least one relevant item in top-k)
- Median rank of acted / relevant signals

Pure-numpy implementation — no sklearn required.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RankingReport:
    ndcg: Dict[int, float]            # k -> NDCG@k
    precision: Dict[int, float]       # k -> P@k
    opportunity_hit_rate: Dict[int, float]  # k -> fraction of queries with ≥1 hit
    median_rank: float                # Median rank of first relevant item (1-indexed)
    total_queries: int


class RankingEvaluator:
    """Compute ranking metrics over a set of ranked result lists.

    Parameters
    ----------
    k_values : list of int
        Cutoff values at which to compute NDCG and precision.
    """

    def __init__(self, k_values: List[int] | None = None) -> None:
        self.k_values = sorted(k_values or [5, 10, 20])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        ranked_ids: Sequence[Sequence[str]],
        relevant_ids: Sequence[Set[str]],
        relevance_scores: Sequence[Dict[str, float]] | None = None,
    ) -> RankingReport:
        """Compute all ranking metrics.

        Parameters
        ----------
        ranked_ids : outer = query, inner = ranked signal IDs (best first)
        relevant_ids : set of relevant signal IDs per query
        relevance_scores : optional graded relevance per signal; if None,
                           binary relevance (0/1) is assumed.
        """
        if len(ranked_ids) != len(relevant_ids):
            raise ValueError("ranked_ids and relevant_ids must have equal length")

        n_queries = len(ranked_ids)
        if n_queries == 0:
            raise ValueError("Empty evaluation batch")

        ndcg_sums: Dict[int, float] = {k: 0.0 for k in self.k_values}
        prec_sums: Dict[int, float] = {k: 0.0 for k in self.k_values}
        hit_sums:  Dict[int, int]   = {k: 0   for k in self.k_values}
        first_ranks: List[float] = []

        for ranked, relevant, rel_scores in zip(
            ranked_ids,
            relevant_ids,
            relevance_scores or [{}] * n_queries,
        ):
            # ---- rank of first relevant item ---------------------------
            first_rank = self._first_rank(ranked, relevant)
            first_ranks.append(first_rank)

            # ---- NDCG, P@k, hit rate -----------------------------------
            for k in self.k_values:
                top_k = list(ranked)[:k]
                ndcg_sums[k] += self._ndcg_at_k(top_k, relevant, rel_scores)
                prec_sums[k] += self._precision_at_k(top_k, relevant)
                hit_sums[k]  += int(any(sid in relevant for sid in top_k))

        ndcg  = {k: ndcg_sums[k]  / n_queries for k in self.k_values}
        prec  = {k: prec_sums[k]  / n_queries for k in self.k_values}
        ohr   = {k: hit_sums[k]   / n_queries for k in self.k_values}
        median_rank = float(np.median(first_ranks)) if first_ranks else float("inf")

        report = RankingReport(
            ndcg=ndcg,
            precision=prec,
            opportunity_hit_rate=ohr,
            median_rank=median_rank,
            total_queries=n_queries,
        )

        for k in self.k_values:
            logger.info(
                "RankingEvaluator k=%d: NDCG=%.4f P=%.4f OHR=%.4f",
                k, ndcg[k], prec[k], ohr[k],
            )
        logger.info("RankingEvaluator median_rank=%.1f", median_rank)
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ndcg_at_k(
        self,
        top_k: List[str],
        relevant: Set[str],
        rel_scores: Dict[str, float],
    ) -> float:
        def dcg(items: List[str]) -> float:
            return sum(
                (rel_scores.get(sid, 1.0) if sid in relevant else 0.0)
                / math.log2(rank + 2)
                for rank, sid in enumerate(items)
            )

        actual_dcg = dcg(top_k)
        ideal_items = sorted(relevant, key=lambda s: -rel_scores.get(s, 1.0))
        ideal_dcg = dcg(ideal_items[: len(top_k)])
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    @staticmethod
    def _precision_at_k(top_k: List[str], relevant: Set[str]) -> float:
        if not top_k:
            return 0.0
        return sum(1 for sid in top_k if sid in relevant) / len(top_k)

    def gate(
        self,
        report: "RankingReport",
        ndcg_at_k: int = 10,
        ndcg_threshold: float = 0.60,
        opportunity_hit_rate_threshold: Optional[float] = None,
    ) -> None:
        """Raise ``ValueError`` when the report does not meet deployment thresholds.

        Args:
            report:                         A ``RankingReport`` from ``evaluate()``.
            ndcg_at_k:                      The ``k`` value to gate on (default 10).
                                            Must be present in ``report.ndcg``.
            ndcg_threshold:                 Minimum required NDCG@k (default 0.60).
            opportunity_hit_rate_threshold: Optional minimum required
                                            opportunity-hit-rate@k (default: not checked).

        Raises:
            TypeError:  If *report* is not a ``RankingReport``.
            ValueError: If *ndcg_at_k* is not a positive int.
            ValueError: If the report does not meet the thresholds.
            KeyError:   If *ndcg_at_k* is not in ``report.ndcg``.
        """
        if not isinstance(report, RankingReport):
            raise TypeError(
                f"'report' must be a RankingReport, got {type(report).__name__!r}"
            )
        if not isinstance(ndcg_at_k, int) or ndcg_at_k < 1:
            raise ValueError(f"'ndcg_at_k' must be a positive int, got {ndcg_at_k!r}")
        if not (0.0 <= ndcg_threshold <= 1.0):
            raise ValueError(
                f"'ndcg_threshold' must be in [0, 1], got {ndcg_threshold!r}"
            )
        if ndcg_at_k not in report.ndcg:
            raise KeyError(
                f"ndcg_at_k={ndcg_at_k} not found in report.ndcg; "
                f"available k values: {sorted(report.ndcg)}"
            )

        failures = []
        actual_ndcg = report.ndcg[ndcg_at_k]
        if actual_ndcg < ndcg_threshold:
            failures.append(
                f"NDCG@{ndcg_at_k}={actual_ndcg:.4f} < threshold={ndcg_threshold:.4f}"
            )
        if opportunity_hit_rate_threshold is not None and ndcg_at_k in report.opportunity_hit_rate:
            ohr = report.opportunity_hit_rate[ndcg_at_k]
            if ohr < opportunity_hit_rate_threshold:
                failures.append(
                    f"OHR@{ndcg_at_k}={ohr:.4f} < threshold={opportunity_hit_rate_threshold:.4f}"
                )
        if failures:
            msg = "RankingEvaluator.gate FAILED: " + "; ".join(failures)
            logger.error(msg)
            raise ValueError(msg)
        logger.info(
            "RankingEvaluator.gate PASSED: NDCG@%d=%.4f threshold=%.2f",
            ndcg_at_k, actual_ndcg, ndcg_threshold,
        )

    @staticmethod
    def _first_rank(ranked: Sequence[str], relevant: Set[str]) -> float:
        """Return 1-indexed rank of first relevant item (inf if none found)."""
        for rank, sid in enumerate(ranked, start=1):
            if sid in relevant:
                return float(rank)
        return float("inf")

