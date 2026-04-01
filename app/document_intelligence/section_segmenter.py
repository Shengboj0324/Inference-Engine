"""Academic section segmentation.

Splits raw paper text into ``DocumentSection`` objects by detecting
standard academic section headings using regex pattern matching with
fallback to line-length and capitalization heuristics.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from app.document_intelligence.models import DocumentSection, SectionType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heading → SectionType mapping (ordered: first match wins)
# ---------------------------------------------------------------------------
_SECTION_PATTERNS: List[Tuple[re.Pattern[str], SectionType]] = [
    (re.compile(r"^\s*(?:\d+\.?\s+)?abstract\s*$", re.I), SectionType.ABSTRACT),
    (re.compile(r"^\s*(?:\d+\.?\s+)?introduction\s*$", re.I), SectionType.INTRODUCTION),
    (re.compile(r"^\s*(?:\d+\.?\s+)?related\s+work\s*$", re.I), SectionType.RELATED_WORK),
    (re.compile(r"^\s*(?:\d+\.?\s+)?background\s*$", re.I), SectionType.BACKGROUND),
    (re.compile(r"^\s*(?:\d+\.?\s+)?preliminaries\s*$", re.I), SectionType.BACKGROUND),
    (re.compile(r"^\s*(?:\d+\.?\s+)?(method(?:ology)?|approach|model|framework)\s*$", re.I), SectionType.METHODOLOGY),
    (re.compile(r"^\s*(?:\d+\.?\s+)?(experiments?|evaluation|ablation)\s*$", re.I), SectionType.EXPERIMENTS),
    (re.compile(r"^\s*(?:\d+\.?\s+)?results?\s*$", re.I), SectionType.RESULTS),
    (re.compile(r"^\s*(?:\d+\.?\s+)?discussion\s*$", re.I), SectionType.DISCUSSION),
    (re.compile(r"^\s*(?:\d+\.?\s+)?conclusion\s*$", re.I), SectionType.CONCLUSION),
    (re.compile(r"^\s*(?:\d+\.?\s+)?references?\s*$", re.I), SectionType.REFERENCES),
    (re.compile(r"^\s*(?:\d+\.?\s+)?appendix\s*$", re.I), SectionType.APPENDIX),
    (re.compile(r"^\s*(?:\d+\.?\s+)?acknowledge?ment\s*$", re.I), SectionType.OTHER),
]

_LINE_SPLIT = re.compile(r"\n")
_SHORT_LINE = 60  # max chars for a potential heading line
_NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)?\s+[A-Z]")


class SectionSegmenter:
    """Segments raw paper text into typed ``DocumentSection`` objects.

    Args:
        min_section_chars: Minimum characters for a section to be kept.
        merge_short:       Merge sections shorter than *min_section_chars*
                           into the previous section.
    """

    def __init__(self, min_section_chars: int = 100, merge_short: bool = True) -> None:
        if min_section_chars < 0:
            raise ValueError(f"'min_section_chars' must be >= 0, got {min_section_chars!r}")
        self._min_chars = min_section_chars
        self._merge_short = merge_short

    def segment(self, text: str, heading_offsets: Optional[Dict[str, int]] = None) -> List[DocumentSection]:
        """Split *text* into typed document sections.

        Args:
            text:            Full paper text (from ``PDFIngestor``).
            heading_offsets: Optional pre-detected heading offsets
                             from ``PDFIngestor._detect_headings()``.

        Returns:
            List of ``DocumentSection`` in document order.

        Raises:
            TypeError: If *text* is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(f"'text' must be str, got {type(text)!r}")
        if not text.strip():
            return []

        if heading_offsets:
            sections = self._segment_by_offsets(text, heading_offsets)
        else:
            sections = self._segment_by_lines(text)

        if self._merge_short:
            sections = self._merge_short_sections(sections)

        logger.debug("SectionSegmenter: %d sections detected", len(sections))
        return sections

    # ------------------------------------------------------------------
    # Offset-based segmentation (fast, uses PDFIngestor heading map)
    # ------------------------------------------------------------------

    def _segment_by_offsets(self, text: str, heading_offsets: Dict[str, int]) -> List[DocumentSection]:
        sorted_headings = sorted(heading_offsets.items(), key=lambda kv: kv[1])
        sections: List[DocumentSection] = []
        for i, (heading, start) in enumerate(sorted_headings):
            end = sorted_headings[i + 1][1] if i + 1 < len(sorted_headings) else len(text)
            section_text = text[start:end]
            # Strip the heading line itself from section_text
            first_nl = section_text.find("\n")
            body = section_text[first_nl:].strip() if first_nl != -1 else section_text.strip()
            section_type = self._classify_heading(heading)
            sections.append(DocumentSection(
                section_type=section_type,
                heading=heading.strip(),
                text=body,
                order=i,
            ))
        return sections

    # ------------------------------------------------------------------
    # Line-based segmentation (fallback for plain-text input)
    # ------------------------------------------------------------------

    def _segment_by_lines(self, text: str) -> List[DocumentSection]:
        lines = _LINE_SPLIT.split(text)
        sections: List[DocumentSection] = []
        current_heading = "Document"
        current_type = SectionType.OTHER
        current_lines: List[str] = []
        order = 0

        for line in lines:
            stripped = line.strip()
            if self._is_heading(stripped):
                if current_lines:
                    body = "\n".join(current_lines).strip()
                    sections.append(DocumentSection(
                        section_type=current_type,
                        heading=current_heading,
                        text=body,
                        order=order,
                    ))
                    order += 1
                current_heading = stripped
                current_type = self._classify_heading(stripped)
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append(DocumentSection(
                section_type=current_type,
                heading=current_heading,
                text="\n".join(current_lines).strip(),
                order=order,
            ))
        return sections

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def _merge_short_sections(self, sections: List[DocumentSection]) -> List[DocumentSection]:
        if not sections:
            return []
        result: List[DocumentSection] = [sections[0]]
        for sec in sections[1:]:
            if len(sec.text) < self._min_chars and result:
                prev = result[-1]
                merged_text = f"{prev.text}\n\n{sec.heading}\n{sec.text}".strip()
                result[-1] = DocumentSection(
                    section_type=prev.section_type,
                    heading=prev.heading,
                    text=merged_text,
                    page_start=prev.page_start,
                    page_end=sec.page_end,
                    order=prev.order,
                )
            else:
                result.append(sec)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_heading(heading: str) -> SectionType:
        for pattern, section_type in _SECTION_PATTERNS:
            if pattern.match(heading):
                return section_type
        return SectionType.OTHER

    @staticmethod
    def _is_heading(line: str) -> bool:
        if not line or len(line) > _SHORT_LINE:
            return False
        if _NUMBERED_HEADING.match(line):
            return True
        for pattern, _ in _SECTION_PATTERNS:
            if pattern.match(line):
                return True
        return False

