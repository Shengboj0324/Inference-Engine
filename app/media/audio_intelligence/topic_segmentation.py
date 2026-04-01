"""Topic segmentation.

Splits a diarized transcript into semantically coherent ``TopicSegment``
blocks using an LLM.  The LLM receives sliding windows of transcript text
and returns structured JSON with segment boundaries and topic labels.

Fallback: when the LLM call fails or is unavailable, a keyword-based
heuristic groups segments by matched ``TopicLabel`` vocabulary.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from app.media.audio_intelligence.models import (
    DiarizedSegment,
    TopicLabel,
    TopicSegment,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword → TopicLabel heuristic map
# ---------------------------------------------------------------------------
_KEYWORD_MAP: List[tuple[re.Pattern[str], TopicLabel]] = [
    (re.compile(r"\b(release[sd]?|launch(ed)?|announc(ed|ing)|version\s+\d|v\d+\.\d+)\b", re.I), TopicLabel.MODEL_RELEASE),
    (re.compile(r"\b(benchmark|score[sd]?|mmlu|hellaswag|humaneval|leaderboard|eval(uat)?)\b", re.I), TopicLabel.BENCHMARK),
    (re.compile(r"\b(paper|research|arxiv|methodology|results|ablation|dataset)\b", re.I), TopicLabel.RESEARCH),
    (re.compile(r"\b(fund(ing|ed)?|invest(ment|or)|series\s+[abcde]|valuation|billion|million)\b", re.I), TopicLabel.FUNDING),
    (re.compile(r"\b(regulation|policy|govern(ance|ment)|congress|eu\s+ai\s+act|law|legal)\b", re.I), TopicLabel.POLICY),
    (re.compile(r"\b(safe(ty|guard)|alignment|risk|misuse|harmful|bias|fairness)\b", re.I), TopicLabel.SAFETY),
    (re.compile(r"\b(i\s+think|in\s+my\s+opinion|personally|believe|feel\s+like)\b", re.I), TopicLabel.OPINION),
    (re.compile(r"\b(maybe|perhaps|might|could\s+be|possibly|speculate|imagine)\b", re.I), TopicLabel.SPECULATION),
]

_LLM_TOPIC_PROMPT_TEMPLATE = """\
You are an AI research analyst. Segment the following transcript excerpt \
into coherent topic sections and classify each section.

INSTRUCTIONS:
- Return a JSON array of segments (no markdown, no prose).
- Each object: {{"start_s": float, "end_s": float, "label": str, "title": str (≤80 chars), "summary": str (2-4 sentences), "key_entities": [str]}}
- Valid labels: {labels}
- Contiguous segments must not overlap; gaps are allowed.
- Order by start_s ascending.

TRANSCRIPT (format: [HH:MM:SS] SPEAKER: text):
{transcript_text}
"""


class TopicSegmenter:
    """LLM-powered topic segmenter with keyword-based fallback.

    Args:
        llm_router:    App LLM router for structured completion calls.
                       Pass ``None`` to force heuristic mode.
        window_chars:  Max characters per LLM context window chunk.
        overlap_chars: Overlap between adjacent chunks.
        min_segment_s: Minimum segment duration in seconds.
    """

    def __init__(
        self,
        llm_router: Optional[Any] = None,
        window_chars: int = 8_000,
        overlap_chars: int = 500,
        min_segment_s: float = 30.0,
    ) -> None:
        if window_chars <= 0:
            raise ValueError(f"'window_chars' must be positive, got {window_chars!r}")
        if overlap_chars < 0:
            raise ValueError(f"'overlap_chars' must be >= 0, got {overlap_chars!r}")
        if min_segment_s < 0:
            raise ValueError(f"'min_segment_s' must be >= 0, got {min_segment_s!r}")
        self._router = llm_router
        self._window_chars = window_chars
        self._overlap_chars = overlap_chars
        self._min_segment_s = min_segment_s

    async def segment(self, diarized: List[DiarizedSegment]) -> List[TopicSegment]:
        """Segment the episode into topic blocks.

        Args:
            diarized: Diarized segments ordered by start time.

        Returns:
            List of ``TopicSegment`` ordered by start time; may overlap
            by at most one second at chunk boundaries.

        Raises:
            TypeError: If *diarized* is not a list.
        """
        if not isinstance(diarized, list):
            raise TypeError(f"'diarized' must be a list, got {type(diarized)!r}")
        if not diarized:
            return []

        t0 = time.perf_counter()
        if self._router is not None:
            try:
                segments = await self._segment_with_llm(diarized)
                if segments:
                    logger.info("TopicSegmenter: LLM produced %d segments in %.1fms", len(segments), (time.perf_counter() - t0) * 1000)
                    return segments
            except Exception as exc:
                logger.warning("TopicSegmenter: LLM failed (%s), using heuristic", exc)

        segments = self._segment_heuristic(diarized)
        logger.info("TopicSegmenter: heuristic produced %d segments in %.1fms", len(segments), (time.perf_counter() - t0) * 1000)
        return segments

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    async def _segment_with_llm(self, diarized: List[DiarizedSegment]) -> List[TopicSegment]:
        transcript_text = self._format_transcript(diarized)
        labels = ", ".join(lbl.value for lbl in TopicLabel)
        prompt = _LLM_TOPIC_PROMPT_TEMPLATE.format(transcript_text=transcript_text[:self._window_chars], labels=labels)
        raw = await self._router.complete(prompt, max_tokens=2000, temperature=0.2)
        # Strip any markdown fences
        raw_json = re.sub(r"```[a-z]*\n?", "", raw).strip().rstrip("`")
        parsed: List[Dict[str, Any]] = json.loads(raw_json)
        return [
            TopicSegment(
                start_s=float(item["start_s"]),
                end_s=float(item["end_s"]),
                label=TopicLabel(item["label"]),
                title=str(item["title"])[:80],
                summary=str(item.get("summary", "")),
                key_entities=list(item.get("key_entities", [])),
                confidence=0.85,
            )
            for item in parsed
            if (float(item["end_s"]) - float(item["start_s"])) >= self._min_segment_s
        ]

    # ------------------------------------------------------------------
    # Heuristic path
    # ------------------------------------------------------------------

    def _segment_heuristic(self, diarized: List[DiarizedSegment]) -> List[TopicSegment]:
        """Group segments using keyword-label majority vote in sliding windows."""
        if not diarized:
            return []

        WINDOW = 10  # number of ASR segments per heuristic block
        result: List[TopicSegment] = []

        for i in range(0, len(diarized), max(1, WINDOW // 2)):
            block = diarized[i : i + WINDOW]
            if not block:
                continue
            start_s = block[0].segment.start_s
            end_s = block[-1].segment.end_s
            if (end_s - start_s) < self._min_segment_s and result:
                # Merge into previous
                prev = result[-1]
                result[-1] = TopicSegment(
                    start_s=prev.start_s,
                    end_s=end_s,
                    label=prev.label,
                    title=prev.title,
                    summary=prev.summary,
                    key_entities=prev.key_entities,
                    confidence=prev.confidence,
                )
                continue

            combined_text = " ".join(d.segment.text for d in block)
            label = self._classify_text(combined_text)
            title = self._extract_title(combined_text, label)
            result.append(TopicSegment(
                start_s=start_s,
                end_s=end_s,
                label=label,
                title=title,
                summary=combined_text[:300].strip(),
                confidence=0.5,
            ))

        return result

    @staticmethod
    def _classify_text(text: str) -> TopicLabel:
        votes: Dict[TopicLabel, int] = {}
        for pattern, label in _KEYWORD_MAP:
            matches = len(pattern.findall(text))
            if matches:
                votes[label] = votes.get(label, 0) + matches
        if not votes:
            return TopicLabel.OTHER
        return max(votes, key=lambda k: votes[k])

    @staticmethod
    def _extract_title(text: str, label: TopicLabel) -> str:
        """Generate a short title from the first sentence."""
        first = re.split(r"[.!?]", text)[0].strip()
        if len(first) > 75:
            first = first[:72] + "…"
        return first or label.value.replace("_", " ").title()

    @staticmethod
    def _format_transcript(diarized: List[DiarizedSegment]) -> str:
        lines = []
        for d in diarized:
            s = d.segment
            ts = time.strftime("%H:%M:%S", time.gmtime(s.start_s))
            lines.append(f"[{ts}] {d.speaker_id}: {s.text}")
        return "\n".join(lines)

