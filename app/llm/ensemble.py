"""Industrial-grade LLM ensemble for superior summarization quality."""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class EnsembleStrategy(str, Enum):
    """Ensemble combination strategies."""

    BEST_OF_N = "best_of_n"  # Generate N, pick best
    CONSENSUS = "consensus"  # Combine multiple outputs
    FALLBACK = "fallback"  # Try providers in order
    PARALLEL_VOTE = "parallel_vote"  # Multiple providers vote


class SummaryQuality(BaseModel):
    """Quality metrics for a summary."""

    coherence_score: float  # 0-1
    factuality_score: float  # 0-1
    completeness_score: float  # 0-1
    conciseness_score: float  # 0-1
    overall_score: float  # 0-1

    def __lt__(self, other: "SummaryQuality") -> bool:
        """Compare quality scores."""
        return self.overall_score < other.overall_score


class EnsembleSummary(BaseModel):
    """Summary with quality metrics."""

    content: str
    provider: LLMProvider
    quality: SummaryQuality
    tokens_used: int
    latency_ms: int


class LLMEnsemble:
    """Industrial-grade LLM ensemble for peak summarization quality.

    Features:
    - Multi-provider support (OpenAI, Anthropic, Google)
    - Quality-based selection
    - Automatic fallback
    - Cost optimization
    - Parallel generation
    """

    def __init__(
        self,
        strategy: EnsembleStrategy = EnsembleStrategy.BEST_OF_N,
        providers: Optional[List[LLMProvider]] = None,
        enable_quality_validation: bool = True,
    ):
        """Initialize LLM ensemble.

        Args:
            strategy: Ensemble strategy to use
            providers: List of providers to use (defaults to all available)
            enable_quality_validation: Enable quality scoring
        """
        self.strategy = strategy
        self.providers = providers or [LLMProvider.OPENAI, LLMProvider.ANTHROPIC]
        self.enable_quality_validation = enable_quality_validation

        # Initialize clients
        self._clients: Dict[LLMProvider, Any] = {}
        self._init_clients()

    def _init_clients(self):
        """Initialize LLM clients for each provider."""
        from app.llm.openai_client import OpenAILLMClient

        if LLMProvider.OPENAI in self.providers:
            self._clients[LLMProvider.OPENAI] = OpenAILLMClient()

        if LLMProvider.ANTHROPIC in self.providers:
            try:
                from app.llm.providers.anthropic_provider import AnthropicLLMClient
                self._clients[LLMProvider.ANTHROPIC] = AnthropicLLMClient()
            except (ImportError, ValueError) as e:
                logger.warning(f"Anthropic client not available: {e}")

        if LLMProvider.GOOGLE in self.providers:
            try:
                from app.llm.google_client import GoogleLLMClient
                self._clients[LLMProvider.GOOGLE] = GoogleLLMClient()
            except ImportError:
                logger.warning("Google client not available")

    async def generate_summary(
        self,
        prompt: str,
        max_tokens: int = 800,
        temperature: float = 0.3,  # Lower for factual content
    ) -> EnsembleSummary:
        """Generate summary using ensemble strategy.

        Args:
            prompt: Summarization prompt
            max_tokens: Maximum tokens in response
            temperature: Generation temperature (0.0-1.0)

        Returns:
            Best summary based on quality metrics
        """
        if self.strategy == EnsembleStrategy.BEST_OF_N:
            return await self._best_of_n(prompt, max_tokens, temperature)
        elif self.strategy == EnsembleStrategy.FALLBACK:
            return await self._fallback(prompt, max_tokens, temperature)
        elif self.strategy == EnsembleStrategy.PARALLEL_VOTE:
            return await self._parallel_vote(prompt, max_tokens, temperature)
        else:
            # Default to best of N
            return await self._best_of_n(prompt, max_tokens, temperature)

    async def _best_of_n(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> EnsembleSummary:
        """Generate N summaries and pick the best one.

        This is the highest quality strategy but most expensive.
        """
        # Generate summaries from all providers in parallel
        tasks = []
        for provider, client in self._clients.items():
            tasks.append(self._generate_with_provider(
                provider, client, prompt, max_tokens, temperature
            ))

        summaries = await asyncio.gather(*tasks, return_exceptions=True)
        valid_summaries = [s for s in summaries if isinstance(s, EnsembleSummary)]

        if not valid_summaries:
            raise RuntimeError("All providers failed to generate summary")

        best_summary = max(valid_summaries, key=lambda s: s.quality.overall_score)

        logger.info(
            f"Best summary from {best_summary.provider.value} "
            f"(quality: {best_summary.quality.overall_score:.2f})"
        )

        return best_summary

    async def _fallback(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> EnsembleSummary:
        """Try providers in order until one succeeds.

        This is the most cost-effective strategy.
        """
        for provider, client in self._clients.items():
            try:
                summary = await self._generate_with_provider(
                    provider, client, prompt, max_tokens, temperature
                )
                logger.info(f"Successfully generated summary with {provider.value}")
                return summary
            except Exception as e:
                logger.warning(f"Provider {provider.value} failed: {e}")
                continue

        raise RuntimeError("All providers failed")

    async def _parallel_vote(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> EnsembleSummary:
        """Generate from multiple providers and combine via voting.

        This provides good quality with moderate cost.
        """
        # Generate from all providers
        tasks = []
        for provider, client in self._clients.items():
            tasks.append(self._generate_with_provider(
                provider, client, prompt, max_tokens, temperature
            ))

        summaries = await asyncio.gather(*tasks, return_exceptions=True)
        valid_summaries = [s for s in summaries if isinstance(s, EnsembleSummary)]

        if not valid_summaries:
            raise RuntimeError("All providers failed")

        # Implement voting/consensus mechanism
        return self._consensus_vote(valid_summaries)

    def _consensus_vote(self, summaries: List[EnsembleSummary]) -> EnsembleSummary:
        """Select best summary using consensus voting.

        Voting criteria:
        1. Quality score (40% weight)
        2. Coherence consistency (30% weight)
        3. Length appropriateness (20% weight, proxied by tokens_used)
        4. Cost efficiency (10% weight, proxied by tokens_used)
        """
        if len(summaries) == 1:
            return summaries[0]

        # Calculate average coherence for consistency check.
        # (SummaryQuality has no sentiment_score; coherence_score is the closest proxy.)
        avg_coherence = sum(s.quality.coherence_score for s in summaries) / len(summaries)

        # Score each summary
        scored_summaries = []
        for summary in summaries:
            # Quality score (0-1, higher is better)
            quality_score = summary.quality.overall_score

            # Coherence consistency (0-1, higher is better)
            coherence_diff = abs(summary.quality.coherence_score - avg_coherence)
            coherence_score = 1.0 - min(coherence_diff, 1.0)

            # Length appropriateness (0-1, higher is better).
            # EnsembleSummary has no original_text; use tokens_used as a length
            # proxy, preferring 50–200 tokens as the "ideal" summary range.
            tokens = summary.tokens_used
            if 50 <= tokens <= 200:
                length_score = 1.0
            elif tokens < 50:
                length_score = tokens / 50.0
            elif tokens < 400:
                length_score = 0.8
            else:
                length_score = max(0.0, 1.0 - (tokens - 400) / 800.0)

            # Cost efficiency (0-1, higher is better) — fewer tokens = cheaper.
            max_tokens_used = max(s.tokens_used for s in summaries)
            cost_score = (
                1.0 - (summary.tokens_used / max_tokens_used)
                if max_tokens_used > 0
                else 1.0
            )

            # Weighted total score
            total_score = (
                quality_score * 0.4
                + coherence_score * 0.3
                + length_score * 0.2
                + cost_score * 0.1
            )

            scored_summaries.append((total_score, summary))

        # Return summary with highest score
        return max(scored_summaries, key=lambda x: x[0])[1]

    async def _generate_with_provider(
        self,
        provider: LLMProvider,
        client: Any,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> EnsembleSummary:
        """Generate summary with a specific provider."""
        import time

        start_time = time.time()

        try:
            # Generate summary
            response = await client.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Validate quality if enabled
            quality = SummaryQuality(
                coherence_score=1.0,
                factuality_score=1.0,
                completeness_score=1.0,
                conciseness_score=1.0,
                overall_score=1.0,
            )

            if self.enable_quality_validation:
                quality = await self._validate_quality(response.content, prompt)

            return EnsembleSummary(
                content=response.content,
                provider=provider,
                quality=quality,
                tokens_used=response.tokens_used,
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.error(f"Provider {provider.value} failed: {e}")
            raise

    async def _validate_quality(
        self,
        summary: str,
        original_prompt: str,
    ) -> SummaryQuality:
        """Validate summary quality using heuristics and AI.

        Quality dimensions:
        1. Coherence: Is the summary well-structured and readable?
        2. Factuality: Does it avoid hallucinations?
        3. Completeness: Does it cover key points?
        4. Conciseness: Is it appropriately brief?
        """
        # Heuristic scoring
        coherence_score = self._score_coherence(summary)
        conciseness_score = self._score_conciseness(summary)

        # For production, use AI-based validation
        # factuality_score = await self._score_factuality_ai(summary, original_prompt)
        # completeness_score = await self._score_completeness_ai(summary, original_prompt)

        # For now, use heuristics
        factuality_score = 0.9  # Placeholder
        completeness_score = 0.9  # Placeholder

        overall_score = (
            coherence_score * 0.25 +
            factuality_score * 0.35 +
            completeness_score * 0.25 +
            conciseness_score * 0.15
        )

        return SummaryQuality(
            coherence_score=coherence_score,
            factuality_score=factuality_score,
            completeness_score=completeness_score,
            conciseness_score=conciseness_score,
            overall_score=overall_score,
        )

    def _score_coherence(self, summary: str) -> float:
        """Score summary coherence using heuristics."""
        score = 1.0

        # Check for minimum length
        if len(summary) < 50:
            score -= 0.3

        # Check for sentence structure
        sentences = summary.split(". ")
        if len(sentences) < 2:
            score -= 0.2

        # Check for proper capitalization
        if not summary[0].isupper():
            score -= 0.1

        # Check for ending punctuation
        if not summary.rstrip().endswith((".", "!", "?")):
            score -= 0.1

        return max(score, 0.0)

    def _score_conciseness(self, summary: str) -> float:
        """Score summary conciseness."""
        word_count = len(summary.split())

        # Optimal range: 100-400 words
        if 100 <= word_count <= 400:
            return 1.0
        elif word_count < 100:
            return 0.7
        elif word_count < 600:
            return 0.8
        else:
            return 0.5

