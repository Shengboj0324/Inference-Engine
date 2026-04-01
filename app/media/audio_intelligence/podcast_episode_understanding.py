"""Podcast / video episode understanding pipeline orchestrator.

Wires together all audio intelligence components into a single callable
that produces a fully structured ``EpisodeUnderstanding`` from raw
transcript text (or audio segments).

Usage::

    pipeline = PodcastEpisodeUnderstandingPipeline()
    understanding = await pipeline.process(
        transcript_segments=segments,    # List[TranscriptSegment]
        episode_id="lex-400",
        title="Lex Fridman #400",
        duration_s=10800.0,
    )
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

from app.media.audio_intelligence.claim_extraction import ClaimExtractor
from app.media.audio_intelligence.diarization import Diarizer
from app.media.audio_intelligence.models import (
    DiarizedSegment,
    EpisodeUnderstanding,
    ExtractedClaim,
    ExtractedQuote,
    TopicSegment,
    TranscriptSegment,
)
from app.media.audio_intelligence.quote_extraction import QuoteExtractor
from app.media.audio_intelligence.topic_segmentation import TopicSegmenter

logger = logging.getLogger(__name__)

# Simple NER heuristic: capitalize-run detection
_ENTITY_PATTERN = re.compile(r"\b([A-Z][a-zA-Z\-]*(?:\s+[A-Z][a-zA-Z\-]*){0,3})\b")
_STOP_WORDS = frozenset({"I", "The", "A", "An", "In", "Of", "And", "Or", "But", "So", "We",
                         "It", "Is", "To", "For", "On", "At", "With", "This", "That", "My"})

_SUMMARY_PROMPT = """\
You are an AI research podcast analyst. Write a concise executive summary \
(3–5 sentences) of this episode suitable for a senior AI researcher.

Focus on: key announcements, research findings, opinions from prominent speakers, \
and actionable takeaways.

EPISODE: {title}
DURATION: {duration_min:.0f} minutes

TOP CLAIMS:
{claims_text}

KEY TOPICS:
{topics_text}
"""


class PodcastEpisodeUnderstandingPipeline:
    """Orchestrates the full audio intelligence pipeline.

    All component arguments accept ``None`` to use defaults.  Each component
    is constructed lazily if not provided.

    Args:
        diarizer:          ``Diarizer`` instance; if None, heuristic diarizer used.
        topic_segmenter:   ``TopicSegmenter`` instance.
        quote_extractor:   ``QuoteExtractor`` instance.
        claim_extractor:   ``ClaimExtractor`` instance.
        llm_router:        LLM router for summary generation.
        max_top_claims:    Maximum claims in ``EpisodeUnderstanding.top_claims``.
        max_key_quotes:    Maximum quotes in ``EpisodeUnderstanding.key_quotes``.
        follow_up_sources: Static list of suggested sources appended to every result.
    """

    def __init__(
        self,
        diarizer: Optional[Diarizer] = None,
        topic_segmenter: Optional[TopicSegmenter] = None,
        quote_extractor: Optional[QuoteExtractor] = None,
        claim_extractor: Optional[ClaimExtractor] = None,
        llm_router: Optional[Any] = None,
        max_top_claims: int = 5,
        max_key_quotes: int = 10,
        follow_up_sources: Optional[List[str]] = None,
    ) -> None:
        if max_top_claims <= 0:
            raise ValueError(f"'max_top_claims' must be positive, got {max_top_claims!r}")
        if max_key_quotes <= 0:
            raise ValueError(f"'max_key_quotes' must be positive, got {max_key_quotes!r}")

        self._diarizer = diarizer or Diarizer()
        self._topic_segmenter = topic_segmenter or TopicSegmenter(llm_router=llm_router)
        self._quote_extractor = quote_extractor or QuoteExtractor(llm_router=llm_router, max_quotes=max_key_quotes)
        self._claim_extractor = claim_extractor or ClaimExtractor(llm_router=llm_router)
        self._llm_router = llm_router
        self._max_top_claims = max_top_claims
        self._max_key_quotes = max_key_quotes
        self._follow_up_sources = follow_up_sources or []

    async def process(
        self,
        transcript_segments: List[TranscriptSegment],
        episode_id: str,
        title: str,
        duration_s: float,
    ) -> EpisodeUnderstanding:
        """Run the full audio intelligence pipeline.

        Args:
            transcript_segments: ASR-produced segments (corrected).
            episode_id:          Unique episode identifier.
            title:               Episode title.
            duration_s:          Episode duration in seconds.

        Returns:
            Fully populated ``EpisodeUnderstanding``.

        Raises:
            ValueError: If required arguments are missing or invalid.
            TypeError:  If ``transcript_segments`` is not a list.
        """
        if not isinstance(transcript_segments, list):
            raise TypeError(f"'transcript_segments' must be a list, got {type(transcript_segments)!r}")
        if not episode_id or not isinstance(episode_id, str):
            raise ValueError("'episode_id' must be a non-empty string")
        if not title or not isinstance(title, str):
            raise ValueError("'title' must be a non-empty string")
        if duration_s < 0:
            raise ValueError(f"'duration_s' must be >= 0, got {duration_s!r}")

        t0 = time.perf_counter()
        meta: Dict[str, Any] = {"pipeline": "PodcastEpisodeUnderstandingPipeline", "episode_id": episode_id}

        # Build full transcript
        transcript = " ".join(seg.text for seg in transcript_segments).strip()

        # 1. Diarization
        t_diar = time.perf_counter()
        diarized: List[DiarizedSegment] = self._diarizer.diarize(transcript_segments)
        meta["diarization_ms"] = round((time.perf_counter() - t_diar) * 1000, 1)

        # 2. Topic segmentation
        t_topic = time.perf_counter()
        topics: List[TopicSegment] = await self._topic_segmenter.segment(diarized)
        meta["topic_segmentation_ms"] = round((time.perf_counter() - t_topic) * 1000, 1)
        meta["num_topics"] = len(topics)

        # 3. Quote extraction
        t_quote = time.perf_counter()
        quotes: List[ExtractedQuote] = await self._quote_extractor.extract(diarized, topics)
        meta["quote_extraction_ms"] = round((time.perf_counter() - t_quote) * 1000, 1)

        # 4. Claim extraction
        t_claim = time.perf_counter()
        claims: List[ExtractedClaim] = await self._claim_extractor.extract(diarized, topics)
        meta["claim_extraction_ms"] = round((time.perf_counter() - t_claim) * 1000, 1)

        # 5. Entity extraction (simple heuristic NER)
        entities = self._extract_entities(transcript)

        # 6. LLM summary
        llm_summary = ""
        if self._llm_router is not None:
            try:
                llm_summary = await self._generate_summary(title, duration_s, claims, topics)
                meta["llm_summary"] = True
            except Exception as exc:
                logger.warning("PodcastEpisodeUnderstandingPipeline: summary failed (%s)", exc)
                meta["llm_summary"] = False

        meta["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        meta["num_diarized_segments"] = len(diarized)
        meta["num_claims"] = len(claims)
        meta["num_quotes"] = len(quotes)

        logger.info(
            "PodcastEpisodeUnderstandingPipeline: episode=%r topics=%d claims=%d quotes=%d total_ms=%.1f",
            episode_id, len(topics), len(claims), len(quotes), meta["total_ms"],
        )
        return EpisodeUnderstanding(
            episode_id=episode_id,
            title=title,
            duration_s=duration_s,
            transcript=transcript,
            segments=diarized,
            topics=topics,
            top_claims=claims[: self._max_top_claims],
            key_quotes=quotes[: self._max_key_quotes],
            entities=entities,
            follow_up_sources=self._follow_up_sources,
            llm_summary=llm_summary,
            processing_metadata=meta,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _generate_summary(
        self, title: str, duration_s: float, claims: List[ExtractedClaim], topics: List[TopicSegment]
    ) -> str:
        claims_text = "\n".join(f"- [{c.claim_type.value}] {c.text}" for c in claims[:5])
        topics_text = "\n".join(f"- [{t.label.value}] {t.title}: {t.summary[:120]}" for t in topics[:6])
        prompt = _SUMMARY_PROMPT.format(
            title=title,
            duration_min=duration_s / 60,
            claims_text=claims_text or "None extracted.",
            topics_text=topics_text or "None detected.",
        )
        return await self._llm_router.complete(prompt, max_tokens=500, temperature=0.3)

    @staticmethod
    def _extract_entities(transcript: str) -> List[str]:
        matches = _ENTITY_PATTERN.findall(transcript)
        seen: Dict[str, int] = {}
        for m in matches:
            if m not in _STOP_WORDS and len(m) > 1:
                seen[m] = seen.get(m, 0) + 1
        # Return entities mentioned ≥ 2 times, sorted by frequency
        return sorted((e for e, cnt in seen.items() if cnt >= 2), key=lambda e: -seen[e])[:50]

