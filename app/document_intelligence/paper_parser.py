"""Paper parser — assembles PaperParsed from raw text + sections.

Extracts structured metadata:
- Title (first non-empty line or regex match)
- Authors (regex-based author block parsing)
- Institutions (affiliation keywords heuristic)
- arXiv ID, DOI
- Year (4-digit year in abstract area or first page)
- Abstract text (SectionType.ABSTRACT section)

All extraction is heuristic-only; LLM refinement happens downstream
in ``PaperSummarizer``.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import List, Optional

from app.document_intelligence.models import DocumentSection, PaperParsed, SectionType

logger = logging.getLogger(__name__)

_ARXIV_ID = re.compile(r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b")
_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s]+)\b")
_YEAR = re.compile(r"\b(20\d{2}|19[89]\d)\b")
_AUTHOR_BLOCK = re.compile(
    r"^([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-zA-Z\-]+(?:\s*,\s*)?){2,}$",
    re.MULTILINE,
)
_INSTITUTION_KEYWORDS = re.compile(
    r"\b(university|institute|lab(?:oratory)?|college|school|research|inc\.?|"
    r"corp\.?|ltd\.?|google|openai|anthropic|microsoft|meta|deepmind)\b",
    re.IGNORECASE,
)


class PaperParser:
    """Parses raw document text and sections into a ``PaperParsed`` object.

    Args:
        arxiv_id:   Optional pre-known arXiv ID.
        pdf_url:    Optional pre-known PDF URL.
        venue:      Optional pre-known venue string.
    """

    def __init__(
        self,
        arxiv_id: str = "",
        pdf_url: str = "",
        venue: str = "",
    ) -> None:
        self._arxiv_id = arxiv_id
        self._pdf_url = pdf_url
        self._venue = venue

    def parse(self, full_text: str, sections: List[DocumentSection]) -> PaperParsed:
        """Build a ``PaperParsed`` from raw text and detected sections.

        Args:
            full_text: Full paper text (from ``PDFIngestor`` or plain text).
            sections:  Detected sections (from ``SectionSegmenter``).

        Returns:
            ``PaperParsed`` with all extractable metadata populated.

        Raises:
            TypeError: If arguments are wrong types.
        """
        if not isinstance(full_text, str):
            raise TypeError(f"'full_text' must be str, got {type(full_text)!r}")
        if not isinstance(sections, list):
            raise TypeError(f"'sections' must be a list, got {type(sections)!r}")

        first_page = full_text[:3000]

        title = self._extract_title(full_text)
        authors = self._extract_authors(first_page)
        institutions = self._extract_institutions(first_page)
        arxiv_id = self._arxiv_id or self._extract_arxiv_id(full_text)
        doi = self._extract_doi(full_text)
        year = self._extract_year(first_page)
        abstract = self._extract_abstract(sections, first_page)
        paper_id = arxiv_id or doi or hashlib.sha256(title.encode()).hexdigest()[:16]

        logger.debug(
            "PaperParser: title=%r authors=%d arxiv_id=%r year=%s",
            title[:60], len(authors), arxiv_id, year,
        )
        return PaperParsed(
            paper_id=paper_id,
            title=title,
            authors=authors,
            institutions=institutions,
            year=year,
            venue=self._venue,
            abstract=abstract,
            sections=sections,
            pdf_url=self._pdf_url,
            arxiv_id=arxiv_id,
            doi=doi,
        )

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_title(full_text: str) -> str:
        lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
        for line in lines[:10]:
            # Skip lines that look like author names or institutional headers
            if len(line) > 10 and not _INSTITUTION_KEYWORDS.search(line):
                if not _AUTHOR_BLOCK.match(line):
                    return line
        return lines[0] if lines else "Unknown Title"

    @staticmethod
    def _extract_authors(text: str) -> List[str]:
        """Extract author names heuristically from the first ~3000 chars."""
        authors: List[str] = []
        for match in _AUTHOR_BLOCK.finditer(text[:2000]):
            raw = match.group(0).strip()
            # Split by comma or newline
            parts = [p.strip() for p in re.split(r",|\n", raw) if p.strip()]
            for part in parts:
                # Filter out obvious non-names
                if 5 <= len(part) <= 60 and not _INSTITUTION_KEYWORDS.search(part):
                    if part not in authors:
                        authors.append(part)
            if len(authors) >= 20:
                break
        return authors[:20]

    @staticmethod
    def _extract_institutions(text: str) -> List[str]:
        institutions: List[str] = []
        for line in text.splitlines()[:40]:
            if _INSTITUTION_KEYWORDS.search(line) and len(line) < 120:
                cleaned = line.strip().strip("†‡*∗")
                if cleaned and cleaned not in institutions:
                    institutions.append(cleaned)
        return institutions[:10]

    @staticmethod
    def _extract_arxiv_id(text: str) -> str:
        m = _ARXIV_ID.search(text)
        return m.group(1) if m else ""

    @staticmethod
    def _extract_doi(text: str) -> str:
        m = _DOI.search(text)
        return m.group(1) if m else ""

    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        m = _YEAR.search(text)
        return int(m.group(1)) if m else None

    @staticmethod
    def _extract_abstract(sections: List[DocumentSection], first_page: str) -> str:
        for sec in sections:
            if sec.section_type == SectionType.ABSTRACT and sec.text.strip():
                return sec.text.strip()
        # Fallback: find "Abstract" block in raw text
        m = re.search(r"(?:^|\n)Abstract\s*\n(.+?)(?:\n\s*\n|\n\s*\d+\.)", first_page, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

