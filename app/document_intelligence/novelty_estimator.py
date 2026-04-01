"""Novelty estimator.

Estimates how novel or incremental a paper is on a [0, 1] scale where:
  0.0 = incremental improvement (e.g., "+0.3% on MMLU with larger dataset")
  1.0 = paradigm shift (e.g., "Attention Is All You Need")

Score composition (all normalized to [0, 1]):
  - Method novelty      (0.40 weight): presence of novel-technique language
  - Benchmark delta     (0.25 weight): % improvement over baselines
  - Claim breadth       (0.20 weight): number of distinct contribution claims
  - Venue prestige      (0.15 weight): conference tier proxy

Optional LLM override: if an LLM router is provided, it re-scores the paper
and the final score is a weighted average of heuristic (0.5) + LLM (0.5).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from app.document_intelligence.models import BenchmarkResult, ClaimEvidence, PaperParsed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Method novelty vocabulary
# ---------------------------------------------------------------------------
_NOVEL_METHOD = re.compile(
    r"\b(novel|new\s+approach|first\s+to|we\s+introduce|we\s+propose|"
    r"unprecedented|paradigm|breakthrough|rethink|reformulat|"
    r"without\s+(?:any\s+)?(?:labels|supervision|fine.tun)|"
    r"zero.shot|few.shot\s+learning|emergent|scaling\s+law)\b",
    re.IGNORECASE,
)
_INCREMENTAL = re.compile(
    r"\b(extend|improve|enhance|augment|further|additional|slight|"
    r"marginal|ablation|hyperparameter|tuning\s+of|variant\s+of)\b",
    re.IGNORECASE,
)

# Venue prestige tiers (higher = more prestigious)
_VENUE_TIER: Dict[str, float] = {
    "neurips": 1.0, "iclr": 1.0, "icml": 1.0, "nature": 1.0, "science": 1.0,
    "acl": 0.9, "emnlp": 0.9, "naacl": 0.9, "cvpr": 0.9, "iccv": 0.9,
    "aaai": 0.8, "ijcai": 0.8, "eccv": 0.8, "coling": 0.8,
    "arxiv": 0.5,  # preprint — no peer review
}

_LLM_NOVELTY_PROMPT = """\
Rate this paper's novelty on a scale of 0.0 to 1.0.

0.0 = Incremental improvement with minor gains on standard benchmarks.
0.5 = Meaningful contribution with solid results on multiple benchmarks.
1.0 = Paradigm-shifting new methodology or framework.

Return ONLY a JSON object: {{"novelty_score": float, "reasoning": str (1 sentence)}}

Paper title: {title}
Abstract: {abstract}
Top contribution: {top_claim}
"""


class NoveltyEstimator:
    """Estimates paper novelty using heuristics + optional LLM scoring.

    Args:
        llm_router:    Optional LLM for novelty scoring override.
        llm_weight:    Weight of LLM score in final average [0, 1].
    """

    def __init__(
        self,
        llm_router: Optional[Any] = None,
        llm_weight: float = 0.5,
    ) -> None:
        if not (0.0 <= llm_weight <= 1.0):
            raise ValueError(f"'llm_weight' must be in [0, 1], got {llm_weight!r}")
        self._router = llm_router
        self._llm_weight = llm_weight

    async def estimate(self, paper: PaperParsed) -> float:
        """Estimate novelty score for *paper*.

        Args:
            paper: ``PaperParsed`` with at minimum abstract and sections.

        Returns:
            Float in [0.0, 1.0].

        Raises:
            TypeError: If *paper* is not a ``PaperParsed``.
        """
        if not isinstance(paper, PaperParsed):
            raise TypeError(f"Expected PaperParsed, got {type(paper)!r}")

        heuristic = self._heuristic_score(paper)

        if self._router is not None:
            try:
                llm_score = await self._llm_score(paper)
                final = (1 - self._llm_weight) * heuristic + self._llm_weight * llm_score
                final = round(min(max(final, 0.0), 1.0), 3)
                logger.debug("NoveltyEstimator: heuristic=%.3f llm=%.3f final=%.3f", heuristic, llm_score, final)
                return final
            except Exception as exc:
                logger.warning("NoveltyEstimator: LLM failed (%s), using heuristic only", exc)

        return round(heuristic, 3)

    # ------------------------------------------------------------------
    # Heuristic scoring
    # ------------------------------------------------------------------

    def _heuristic_score(self, paper: PaperParsed) -> float:
        text = f"{paper.abstract} {paper.full_text()}"

        # 1. Method novelty (weight 0.40)
        novel_hits = len(_NOVEL_METHOD.findall(text))
        incr_hits = len(_INCREMENTAL.findall(text))
        method_score = min(novel_hits / (incr_hits + novel_hits + 1), 1.0)

        # 2. Benchmark delta (weight 0.25)
        benchmark_score = self._benchmark_delta_score(paper.benchmarks)

        # 3. Claim breadth (weight 0.20)
        contrib_claims = [c for c in paper.claims if c.claim_type == "contribution"]
        claim_score = min(len(contrib_claims) / 5.0, 1.0)

        # 4. Venue prestige (weight 0.15)
        venue_score = self._venue_score(paper.venue)

        total = (
            0.40 * method_score
            + 0.25 * benchmark_score
            + 0.20 * claim_score
            + 0.15 * venue_score
        )
        return min(max(total, 0.0), 1.0)

    @staticmethod
    def _benchmark_delta_score(benchmarks: List[BenchmarkResult]) -> float:
        """Estimate novelty contribution from benchmark improvements."""
        if not benchmarks:
            return 0.3  # neutral when no benchmarks extracted
        # Simple heuristic: more benchmarks with high values → higher novelty
        high_values = 0
        for b in benchmarks:
            m = re.search(r"(\d+\.?\d*)", b.value)
            if m:
                val = float(m.group(1))
                # If value looks like ≥ 90%, assume it's a strong result
                if val >= 90.0 or (b.metric in ("pass@1", "em") and val >= 70.0):
                    high_values += 1
        return min(high_values / max(len(benchmarks), 1), 1.0)

    @staticmethod
    def _venue_score(venue: str) -> float:
        if not venue:
            return 0.5
        venue_lower = venue.lower()
        for key, score in _VENUE_TIER.items():
            if key in venue_lower:
                return score
        return 0.4  # unknown venue = slightly below average

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    async def _llm_score(self, paper: PaperParsed) -> float:
        import json
        top_claim = (paper.claims[0].claim[:200] if paper.claims else "Not available")
        prompt = _LLM_NOVELTY_PROMPT.format(
            title=paper.title,
            abstract=paper.abstract[:800],
            top_claim=top_claim,
        )
        raw = await self._router.complete(prompt, max_tokens=100, temperature=0.1)
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip().rstrip("`")
        data = json.loads(raw)
        return float(min(max(data["novelty_score"], 0.0), 1.0))

