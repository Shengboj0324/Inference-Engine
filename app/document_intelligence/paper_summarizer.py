"""LLM-powered paper summarizer.

Produces a structured natural-language summary of an academic paper
covering: problem, method, key results, practical importance, and novelty.

When no LLM router is available, falls back to an extractive summary
built from the abstract and conclusions.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from app.document_intelligence.models import PaperParsed, SectionType

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """\
You are an AI research analyst writing for a senior machine learning \
researcher.  Summarize this paper concisely and accurately.

Paper: {title}
Authors: {authors}
Year: {year}
Abstract: {abstract}

Key results:
{benchmarks}

Top claims/contributions:
{claims}

Write a structured summary with these labeled sections (each 1-3 sentences):
1. PROBLEM: What problem does this paper solve?
2. METHOD: What is the key technical approach?
3. RESULTS: What are the headline quantitative results?
4. IMPORTANCE: Why does this matter to practitioners?
5. LIMITATIONS: What are the key limitations?

Do NOT include commentary or caveats beyond the paper itself.
"""

_EXTRACTIVE_MAX_CHARS = 600


class PaperSummarizer:
    """Generates structured summaries of parsed academic papers.

    Args:
        llm_router: LLM router; if None, extractive fallback is used.
        max_tokens: Maximum tokens for LLM summary.
        temperature: LLM temperature.
    """

    def __init__(
        self,
        llm_router: Optional[Any] = None,
        max_tokens: int = 600,
        temperature: float = 0.3,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(f"'max_tokens' must be positive, got {max_tokens!r}")
        if not (0.0 <= temperature <= 2.0):
            raise ValueError(f"'temperature' must be in [0, 2], got {temperature!r}")
        self._router = llm_router
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def summarize(self, paper: PaperParsed) -> str:
        """Generate a structured summary of *paper*.

        Args:
            paper: ``PaperParsed`` object (must have at minimum a title
                   and abstract).

        Returns:
            Multi-paragraph summary string.

        Raises:
            TypeError: If *paper* is not a ``PaperParsed``.
        """
        if not isinstance(paper, PaperParsed):
            raise TypeError(f"'paper' must be PaperParsed, got {type(paper)!r}")

        t0 = time.perf_counter()
        if self._router is not None:
            try:
                summary = await self._llm_summarize(paper)
                logger.info(
                    "PaperSummarizer: LLM summary for %r in %.1fms",
                    paper.paper_id, (time.perf_counter() - t0) * 1000,
                )
                return summary
            except Exception as exc:
                logger.warning("PaperSummarizer: LLM failed (%s), using extractive fallback", exc)

        summary = self._extractive_summary(paper)
        logger.info(
            "PaperSummarizer: extractive summary for %r in %.1fms",
            paper.paper_id, (time.perf_counter() - t0) * 1000,
        )
        return summary

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    async def _llm_summarize(self, paper: PaperParsed) -> str:
        benchmarks_text = "\n".join(
            f"- {b.benchmark_name}: {b.value} ({b.model_name})"
            for b in paper.benchmarks[:6]
        ) or "None extracted."

        claims_text = "\n".join(
            f"- [{c.claim_type}] {c.claim[:120]}"
            for c in paper.claims[:5]
        ) or "None extracted."

        prompt = _SUMMARY_PROMPT.format(
            title=paper.title,
            authors=", ".join(paper.authors[:4]) or "Unknown",
            year=paper.year or "Unknown",
            abstract=paper.abstract[:1200],
            benchmarks=benchmarks_text,
            claims=claims_text,
        )
        return await self._router.complete(prompt, max_tokens=self._max_tokens, temperature=self._temperature)

    # ------------------------------------------------------------------
    # Extractive fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _extractive_summary(paper: PaperParsed) -> str:
        parts: list[str] = []

        # Abstract
        if paper.abstract:
            parts.append(f"ABSTRACT: {paper.abstract[:_EXTRACTIVE_MAX_CHARS].strip()}")

        # Conclusion
        conclusion = paper.get_section(SectionType.CONCLUSION)
        if conclusion and conclusion.text.strip():
            parts.append(f"CONCLUSION: {conclusion.text[:_EXTRACTIVE_MAX_CHARS].strip()}")

        # Key benchmarks
        if paper.benchmarks:
            bm_lines = [f"{b.benchmark_name}: {b.value}" for b in paper.benchmarks[:5]]
            parts.append("RESULTS: " + "; ".join(bm_lines))

        # Top claims
        contribs = [c for c in paper.claims if c.claim_type == "contribution"][:3]
        if contribs:
            claim_lines = [c.claim[:100] for c in contribs]
            parts.append("CONTRIBUTIONS: " + " | ".join(claim_lines))

        if not parts:
            parts.append(f"Title: {paper.title}. No structured content extracted.")

        return "\n\n".join(parts)

