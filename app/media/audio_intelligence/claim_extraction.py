"""Factual claim extraction from transcript segments.

For each ``TopicSegment``, extracts ``ExtractedClaim`` objects that capture
distinct factual or epistemic assertions made by speakers.

Extraction pipeline:
1. Split segment text into candidate sentences.
2. Classify each sentence into a ``ClaimType`` using keyword rules.
3. Filter by minimum confidence threshold.
4. Optionally run an LLM pass for structured claim normalization and
   evidence linking (evidence = verbatim sentence that supports the claim).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from app.media.audio_intelligence.models import (
    ClaimType,
    DiarizedSegment,
    ExtractedClaim,
    TopicSegment,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic classifiers
# ---------------------------------------------------------------------------
_ANNOUNCEMENT = re.compile(
    r"\b(we('re|\s+are)\s+(releasing|launching|announc|open.sourc)|"
    r"today\s+we|introducing|available\s+(now|today))\b",
    re.IGNORECASE,
)
_BENCHMARK = re.compile(
    r"\b(\d+[\.,]?\d*\s*%|achieves?|score[sd]?|outperform|sota|state.of.the.art|"
    r"humaneval|mmlu|hellaswag|leaderboard|ranks?\s+(first|second|#\d+))\b",
    re.IGNORECASE,
)
_PREDICTION = re.compile(
    r"\b(will|going\s+to|next\s+(year|month|quarter)|by\s+\d{4}|"
    r"within\s+(months?|years?))\b",
    re.IGNORECASE,
)
_SPECULATION = re.compile(
    r"\b(maybe|perhaps|might|could|possibly|i\s+imagine|what\s+if|suppose)\b",
    re.IGNORECASE,
)
_OPINION = re.compile(
    r"\b(i\s+(think|believe|feel)|in\s+my\s+(opinion|view)|personally|"
    r"to\s+me\b|my\s+take|the\s+way\s+i\s+see)\b",
    re.IGNORECASE,
)
_FACT_SIGNALS = re.compile(
    r"\b(was|were|has|have|is|are)\s+(released|trained|built|shown|proven|"
    r"demonstrated|published)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MIN_WORDS = 8

_LLM_CLAIM_PROMPT = """\
Extract structured claims from this transcript segment.

Return a JSON array. Each object: {{
  "text": str,           // normalized claim (not necessarily verbatim)
  "claim_type": str,     // {claim_types}
  "confidence": float,   // 0-1
  "evidence": str,       // verbatim sentence(s) from transcript supporting this claim
  "entities": [str],     // named entities in the claim
  "supported": bool      // is the claim supported by evidence in this segment?
}}

SEGMENT TEXT:
{segment_text}
"""


class ClaimExtractor:
    """Extracts factual and epistemic claims from transcript topic segments.

    Args:
        llm_router:       Optional LLM router for structured extraction.
        min_confidence:   Minimum confidence to include a claim.
        max_claims_per_segment: Cap per topic segment.
        use_llm:          Override; set False to force heuristic.
    """

    def __init__(
        self,
        llm_router: Optional[Any] = None,
        min_confidence: float = 0.4,
        max_claims_per_segment: int = 8,
        use_llm: Optional[bool] = None,
    ) -> None:
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(f"'min_confidence' must be in [0, 1], got {min_confidence!r}")
        if max_claims_per_segment <= 0:
            raise ValueError(f"'max_claims_per_segment' must be positive, got {max_claims_per_segment!r}")
        self._router = llm_router
        self._min_conf = min_confidence
        self._max_per_segment = max_claims_per_segment
        self._use_llm = use_llm if use_llm is not None else (llm_router is not None)

    async def extract(
        self,
        diarized: List[DiarizedSegment],
        topics: Optional[List[TopicSegment]] = None,
    ) -> List[ExtractedClaim]:
        """Extract claims from the full episode.

        Args:
            diarized: Diarized transcript segments.
            topics:   Topic segments for per-segment extraction context.
                      If empty, treats entire transcript as one segment.

        Returns:
            List of ``ExtractedClaim`` sorted by confidence descending.

        Raises:
            TypeError: If *diarized* is not a list.
        """
        if not isinstance(diarized, list):
            raise TypeError(f"'diarized' must be a list, got {type(diarized)!r}")
        if not diarized:
            return []

        t0 = time.perf_counter()
        topic_list = topics or []
        all_claims: List[ExtractedClaim] = []

        if topic_list:
            for topic in topic_list:
                seg_text, speaker_id = self._get_segment_text(diarized, topic.start_s, topic.end_s)
                claims = await self._extract_segment(seg_text, speaker_id, topic.start_s)
                all_claims.extend(claims[: self._max_per_segment])
        else:
            full_text = " ".join(d.segment.text for d in diarized)
            claims = await self._extract_segment(full_text, diarized[0].speaker_id if diarized else "unknown", 0.0)
            all_claims.extend(claims)

        filtered = [c for c in all_claims if c.confidence >= self._min_conf]
        filtered.sort(key=lambda c: c.confidence, reverse=True)
        logger.info(
            "ClaimExtractor: extracted %d claims (filtered from %d) in %.1fms",
            len(filtered), len(all_claims), (time.perf_counter() - t0) * 1000,
        )
        return filtered

    # ------------------------------------------------------------------
    # Per-segment extraction
    # ------------------------------------------------------------------

    async def _extract_segment(self, text: str, speaker_id: str, start_s: float) -> List[ExtractedClaim]:
        if not text.strip():
            return []
        if self._use_llm and self._router is not None:
            try:
                return await self._llm_extract(text, speaker_id, start_s)
            except Exception as exc:
                logger.warning("ClaimExtractor: LLM extraction failed (%s), using heuristic", exc)
        return self._heuristic_extract(text, speaker_id, start_s)

    async def _llm_extract(self, text: str, speaker_id: str, start_s: float) -> List[ExtractedClaim]:
        claim_types = ", ".join(ct.value for ct in ClaimType)
        prompt = _LLM_CLAIM_PROMPT.format(claim_types=claim_types, segment_text=text[:4000])
        raw = await self._router.complete(prompt, max_tokens=1500, temperature=0.1)
        raw_json = re.sub(r"```[a-z]*\n?", "", raw).strip().rstrip("`")
        items: list[dict] = json.loads(raw_json)
        claims: List[ExtractedClaim] = []
        for item in items:
            try:
                claims.append(ExtractedClaim(
                    text=str(item["text"])[:400],
                    claim_type=ClaimType(item.get("claim_type", "opinion")),
                    confidence=min(max(float(item.get("confidence", 0.7)), 0.0), 1.0),
                    evidence=str(item.get("evidence", ""))[:500],
                    speaker_id=speaker_id,
                    start_s=start_s,
                    entities=list(item.get("entities", [])),
                    supported=bool(item.get("supported", True)),
                ))
            except Exception:
                continue
        return claims

    def _heuristic_extract(self, text: str, speaker_id: str, start_s: float) -> List[ExtractedClaim]:
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if len(s.split()) >= _MIN_WORDS]
        claims: List[ExtractedClaim] = []
        for sentence in sentences:
            claim_type, confidence = self._classify_sentence(sentence)
            if confidence < self._min_conf:
                continue
            claims.append(ExtractedClaim(
                text=sentence[:400],
                claim_type=claim_type,
                confidence=confidence,
                evidence=sentence[:500],
                speaker_id=speaker_id,
                start_s=start_s,
                entities=[],
                supported=claim_type in (ClaimType.ANNOUNCEMENT, ClaimType.BENCHMARK, ClaimType.FACT),
            ))
        return claims

    @staticmethod
    def _classify_sentence(sentence: str) -> tuple[ClaimType, float]:
        if _ANNOUNCEMENT.search(sentence):
            return ClaimType.ANNOUNCEMENT, 0.80
        if _BENCHMARK.search(sentence):
            return ClaimType.BENCHMARK, 0.75
        if _FACT_SIGNALS.search(sentence):
            return ClaimType.FACT, 0.65
        if _PREDICTION.search(sentence):
            return ClaimType.PREDICTION, 0.60
        if _OPINION.search(sentence):
            return ClaimType.OPINION, 0.55
        if _SPECULATION.search(sentence):
            return ClaimType.SPECULATION, 0.50
        return ClaimType.OPINION, 0.30

    @staticmethod
    def _get_segment_text(diarized: List[DiarizedSegment], start_s: float, end_s: float) -> tuple[str, str]:
        parts: List[str] = []
        first_speaker = "unknown"
        for d in diarized:
            if d.segment.end_s < start_s:
                continue
            if d.segment.start_s > end_s:
                break
            parts.append(d.segment.text)
            if first_speaker == "unknown":
                first_speaker = d.speaker_id
        return " ".join(parts), first_speaker

