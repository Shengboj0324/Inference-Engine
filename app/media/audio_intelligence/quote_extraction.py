"""Notable quote extraction.

Extracts the most significant verbatim quotes from diarized transcript
segments.  A quote is flagged as notable when it meets ≥1 of:

- Contains a superlative claim ("best", "most important", "first ever")
- Expresses a clear opinion or prediction with high confidence signal words
- Contains AI/ML technical terminology that indicates a substantive statement
- Is sufficiently long (≥ 15 words) and not a filler/transition phrase

LLM re-ranking (optional): when an LLM router is provided, the top-30
heuristic candidates are re-ranked by importance and reduced to the
top-N most notable.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

from app.media.audio_intelligence.models import (
    DiarizedSegment,
    ExtractedQuote,
    TopicLabel,
    TopicSegment,
)

logger = logging.getLogger(__name__)

_SUPERLATIVE = re.compile(
    r"\b(best|worst|most|least|first|only|never|always|critical|crucial|"
    r"revolutionary|breakthrough|state.of.the.art|sota|unprecedented)\b",
    re.IGNORECASE,
)
_OPINION_SIGNAL = re.compile(
    r"\b(i\s+think|i\s+believe|in\s+my\s+view|my\s+take|i\s+predict|"
    r"this\s+is\s+the|the\s+key|the\s+most\s+important)\b",
    re.IGNORECASE,
)
_TECHNICAL_CLAIM = re.compile(
    r"\b(parameter[s]?|token[s]?|benchmark|latency|throughput|accuracy|"
    r"trillion|billion|context.window|fine.tun|rlhf|alignment|emergent)\b",
    re.IGNORECASE,
)
_FILLER = re.compile(
    r"^(um|uh|yeah|right|okay|so|like|you\s+know|i\s+mean|basically|literally)\b",
    re.IGNORECASE,
)

_MIN_WORDS = 12
_MIN_IMPORTANCE = 0.3

_RERANK_PROMPT = """\
You are an AI podcast analyst. From the candidate quotes below, select the \
top {n} most important and insightful for an AI researcher audience.

Return a JSON array of objects: [{{"index": int, "importance": float (0-1), \
"topic_label": str}}]

Valid topic_labels: {labels}

CANDIDATES:
{candidates}
"""


class QuoteExtractor:
    """Extracts and ranks notable quotes from diarized transcript.

    Args:
        llm_router:       Optional LLM for re-ranking.
        max_quotes:       Maximum quotes to return per episode.
        min_words:        Minimum word count for a quote candidate.
        min_importance:   Minimum heuristic importance to consider.
        rerank_top_k:     How many heuristic candidates to send to LLM.
    """

    def __init__(
        self,
        llm_router: Optional[Any] = None,
        max_quotes: int = 10,
        min_words: int = _MIN_WORDS,
        min_importance: float = _MIN_IMPORTANCE,
        rerank_top_k: int = 30,
    ) -> None:
        if max_quotes <= 0:
            raise ValueError(f"'max_quotes' must be positive, got {max_quotes!r}")
        if not (0.0 <= min_importance <= 1.0):
            raise ValueError(f"'min_importance' must be in [0, 1], got {min_importance!r}")
        if min_words < 1:
            raise ValueError(f"'min_words' must be >= 1, got {min_words!r}")
        self._router = llm_router
        self._max_quotes = max_quotes
        self._min_words = min_words
        self._min_importance = min_importance
        self._rerank_top_k = rerank_top_k

    async def extract(
        self,
        diarized: List[DiarizedSegment],
        topics: Optional[List[TopicSegment]] = None,
    ) -> List[ExtractedQuote]:
        """Extract notable quotes from diarized segments.

        Args:
            diarized: Diarized transcript segments.
            topics:   Optional topic segments for topic-label assignment.

        Returns:
            List of ``ExtractedQuote`` sorted by importance descending.

        Raises:
            TypeError: If *diarized* is not a list.
        """
        if not isinstance(diarized, list):
            raise TypeError(f"'diarized' must be a list, got {type(diarized)!r}")
        if not diarized:
            return []

        candidates = self._score_heuristic(diarized, topics or [])
        candidates.sort(key=lambda q: q.importance, reverse=True)
        top = candidates[: self._rerank_top_k]

        if self._router is not None and top:
            try:
                top = await self._rerank_with_llm(top)
            except Exception as exc:
                logger.warning("QuoteExtractor: LLM re-ranking failed (%s), using heuristic order", exc)

        return top[: self._max_quotes]

    # ------------------------------------------------------------------
    # Heuristic scoring
    # ------------------------------------------------------------------

    def _score_heuristic(
        self, diarized: List[DiarizedSegment], topics: List[TopicSegment]
    ) -> List[ExtractedQuote]:
        quotes: List[ExtractedQuote] = []
        for seg in diarized:
            text = seg.segment.text.strip()
            if not text:
                continue
            words = text.split()
            if len(words) < self._min_words:
                continue
            if _FILLER.match(text):
                continue

            importance = self._compute_importance(text)
            if importance < self._min_importance:
                continue

            topic_label = self._find_topic(seg.segment.start_s, topics)
            quotes.append(ExtractedQuote(
                text=text,
                speaker_id=seg.speaker_id,
                start_s=seg.segment.start_s,
                importance=importance,
                context="",
                topic_label=topic_label,
            ))
        return quotes

    @staticmethod
    def _compute_importance(text: str) -> float:
        score = 0.0
        words = text.split()
        # Length bonus (saturates at 80 words)
        score += min(len(words) / 80.0, 1.0) * 0.2
        # Pattern bonuses
        if _SUPERLATIVE.search(text):
            score += 0.35
        if _OPINION_SIGNAL.search(text):
            score += 0.25
        if _TECHNICAL_CLAIM.search(text):
            score += 0.30
        return round(min(score, 1.0), 3)

    @staticmethod
    def _find_topic(start_s: float, topics: List[TopicSegment]) -> TopicLabel:
        for t in topics:
            if t.start_s <= start_s <= t.end_s:
                return t.label
        return TopicLabel.OTHER

    # ------------------------------------------------------------------
    # LLM re-ranking
    # ------------------------------------------------------------------

    async def _rerank_with_llm(self, candidates: List[ExtractedQuote]) -> List[ExtractedQuote]:
        candidate_text = "\n".join(
            f"[{i}] ({c.speaker_id}, {c.start_s:.1f}s): {c.text[:200]}"
            for i, c in enumerate(candidates)
        )
        labels = ", ".join(lbl.value for lbl in TopicLabel)
        prompt = _RERANK_PROMPT.format(
            n=self._max_quotes,
            labels=labels,
            candidates=candidate_text,
        )
        raw = await self._router.complete(prompt, max_tokens=1000, temperature=0.1)
        raw_json = re.sub(r"```[a-z]*\n?", "", raw).strip().rstrip("`")
        ranked: list[dict] = json.loads(raw_json)
        reranked: List[ExtractedQuote] = []
        for item in ranked:
            idx = int(item["index"])
            if 0 <= idx < len(candidates):
                orig = candidates[idx]
                label = TopicLabel(item.get("topic_label", orig.topic_label.value))
                reranked.append(ExtractedQuote(
                    text=orig.text,
                    speaker_id=orig.speaker_id,
                    start_s=orig.start_s,
                    importance=float(item.get("importance", orig.importance)),
                    context=orig.context,
                    topic_label=label,
                ))
        return sorted(reranked, key=lambda q: q.importance, reverse=True)

