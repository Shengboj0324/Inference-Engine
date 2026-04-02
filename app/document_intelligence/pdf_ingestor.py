"""PDF text and structure extraction.

Backend priority:
1. ``pdfplumber``  — layout-aware extraction (best for tables/columns)
2. ``pypdf``       — standard text extraction
3. Stub            — returns empty pages (CI/testing)

Produces a list of ``PDFPage`` objects (page number + text) and a
``PDFDocument`` summary object that stores raw-text-per-page plus detected
section heading offsets.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_HEADING_PATTERN = re.compile(
    r"^(?:\d+\.?\s+)?(?:Abstract|Introduction|Background|Related\s+Work|"
    r"Preliminaries|Method(?:ology)?|Approach|Model|Experiments?|Results?|"
    r"Evaluation|Ablation|Discussion|Conclusion|References?|Appendix|"
    r"Acknowledgements?)\b",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class PDFPage:
    """Single extracted page from a PDF.

    Attributes:
        page_number: 1-indexed page number.
        text:        Extracted text with whitespace normalized.
        char_count:  Number of characters in *text*.
    """

    page_number: int
    text: str
    char_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "char_count", len(self.text))


@dataclass
class PDFDocument:
    """Extracted content from a PDF file.

    Attributes:
        source_path:    Path to the original PDF.
        pages:          Ordered list of ``PDFPage`` objects.
        full_text:      Concatenated text of all pages.
        page_count:     Total page count.
        heading_offsets: Dict of {heading_text: char_offset_in_full_text}.
        content_hash:   SHA-256 of full_text (for deduplication).
        extraction_backend: Which backend produced this extraction.
    """

    source_path: str
    pages: List[PDFPage] = field(default_factory=list)
    full_text: str = ""
    page_count: int = 0
    heading_offsets: Dict[str, int] = field(default_factory=dict)
    content_hash: str = ""
    extraction_backend: str = "stub"


class PDFIngestor:
    """Extracts text from PDF files with automatic backend selection.

    Args:
        max_pages:    Maximum pages to extract (0 = all).
        min_chars:    Minimum characters per page to include.
        backend:      Force backend: ``"pdfplumber"``, ``"pypdf"``, ``"stub"``.
    """

    #: Set to ``True`` to raise ``RuntimeError`` when only the stub backend
    #: is available.  Must be ``True`` in production to surface misconfigured
    #: environments rather than silently returning stub content.
    PRODUCTION_GUARD: bool = False

    def __init__(
        self,
        max_pages: int = 0,
        min_chars: int = 50,
        backend: Optional[str] = None,
        production_safe: bool = False,
    ) -> None:
        """
        Args:
            max_pages: Maximum pages to extract (0 = all pages).
            min_chars: Minimum characters per page to include.
            backend: Force a specific backend: ``"pdfplumber"``, ``"pypdf"``,
                     or ``"stub"``.  ``None`` = auto-select.
            production_safe: When ``True``, raises ``RuntimeError`` if only the
                             stub backend is available.  Set to ``True`` in all
                             production deployments; leave ``False`` for CI/tests.
        """
        if not isinstance(max_pages, int):
            raise TypeError(f"'max_pages' must be int, got {type(max_pages).__name__!r}")
        if max_pages < 0:
            raise ValueError(f"'max_pages' must be >= 0, got {max_pages!r}")
        if not isinstance(min_chars, int):
            raise TypeError(f"'min_chars' must be int, got {type(min_chars).__name__!r}")
        if min_chars < 0:
            raise ValueError(f"'min_chars' must be >= 0, got {min_chars!r}")
        if backend is not None and backend not in {"pdfplumber", "pypdf", "stub"}:
            raise ValueError(f"'backend' must be one of 'pdfplumber', 'pypdf', 'stub', got {backend!r}")
        if not isinstance(production_safe, bool):
            raise TypeError(f"'production_safe' must be bool, got {type(production_safe).__name__!r}")
        self._max_pages = max_pages
        self._min_chars = min_chars
        self._forced_backend = backend
        self._production_safe = production_safe

    def ingest(self, pdf_path: str) -> PDFDocument:
        """Extract text and structure from a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ``PDFDocument`` with extracted text and structure.

        Raises:
            FileNotFoundError: If *pdf_path* does not exist.
            ValueError: If path is empty.
        """
        if not pdf_path or not isinstance(pdf_path, str):
            raise ValueError("'pdf_path' must be a non-empty string")
        p = Path(pdf_path)
        if not p.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        t0 = time.perf_counter()
        backend = self._forced_backend or self._select_backend()

        if backend == "stub" and self._production_safe:
            raise RuntimeError(
                "PDFIngestor: no real PDF extraction backend is available "
                "(pdfplumber and pypdf are both missing) but production_safe=True "
                "prevents silent stub fallback.  Install 'pdfplumber' or 'pypdf' "
                "before deploying, or set production_safe=False for CI/testing."
            )

        # Centralised fail-closed enforcement via ProductionSafetyContract.
        # This fires regardless of production_safe so that the global
        # settings.production_strict_mode gate is always respected.
        if backend == "stub":
            from app.core.production_guard import get_guard
            get_guard().require_real_backend(
                capability="pdf_extraction",
                resolved_backend="stub",
            )

        logger.info("PDFIngestor: backend=%s production_safe=%s path=%s",
                    backend, self._production_safe, pdf_path)

        if backend == "pdfplumber":
            doc = self._ingest_pdfplumber(pdf_path)
        elif backend == "pypdf":
            doc = self._ingest_pypdf(pdf_path)
        else:
            logger.warning(
                "PDFIngestor: falling back to stub for %s — no extraction "
                "backend available (pdfplumber/pypdf not installed)", pdf_path
            )
            doc = self._ingest_stub(pdf_path)

        doc.extraction_backend = backend
        doc.content_hash = hashlib.sha256(doc.full_text.encode("utf-8")).hexdigest()
        doc.heading_offsets = self._detect_headings(doc.full_text)

        logger.info(
            "PDFIngestor: pages=%d chars=%d headings=%d latency_ms=%.1f",
            doc.page_count, len(doc.full_text), len(doc.heading_offsets),
            (time.perf_counter() - t0) * 1000,
        )
        return doc

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _ingest_pdfplumber(self, pdf_path: str) -> PDFDocument:
        import pdfplumber  # type: ignore[import]
        pages: List[PDFPage] = []
        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
            limit = self._max_pages or total
            for i, page in enumerate(pdf.pages[:limit]):
                text = (page.extract_text() or "").strip()
                text = self._normalize_text(text)
                if len(text) >= self._min_chars:
                    pages.append(PDFPage(page_number=i + 1, text=text))
        full_text = "\n\n".join(p.text for p in pages)
        return PDFDocument(source_path=pdf_path, pages=pages, full_text=full_text, page_count=total)

    def _ingest_pypdf(self, pdf_path: str) -> PDFDocument:
        import pypdf  # type: ignore[import]
        reader = pypdf.PdfReader(pdf_path)
        total = len(reader.pages)
        limit = self._max_pages or total
        pages: List[PDFPage] = []
        for i in range(min(limit, total)):
            text = self._normalize_text(reader.pages[i].extract_text() or "")
            if len(text) >= self._min_chars:
                pages.append(PDFPage(page_number=i + 1, text=text))
        full_text = "\n\n".join(p.text for p in pages)
        return PDFDocument(source_path=pdf_path, pages=pages, full_text=full_text, page_count=total)

    @staticmethod
    def _ingest_stub(pdf_path: str) -> PDFDocument:
        logger.debug("PDFIngestor: stub mode for %s", pdf_path)
        stub_text = "[PDF STUB — no extraction backend available]"
        page = PDFPage(page_number=1, text=stub_text)
        return PDFDocument(source_path=pdf_path, pages=[page], full_text=stub_text, page_count=1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_backend() -> str:
        for name in ("pdfplumber", "pypdf"):
            try:
                __import__(name.replace("pdf", "pdf" if name == "pypdf" else ""))
                return name
            except ImportError:
                pass
        try:
            import pdfplumber  # type: ignore[import]
            return "pdfplumber"
        except ImportError:
            pass
        try:
            import pypdf  # type: ignore[import]
            return "pypdf"
        except ImportError:
            pass
        return "stub"

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"-\s*\n\s*", "", text)  # de-hyphenate line breaks
        return text.strip()

    @staticmethod
    def _detect_headings(full_text: str) -> Dict[str, int]:
        offsets: Dict[str, int] = {}
        for match in _HEADING_PATTERN.finditer(full_text):
            heading = match.group(0).strip()
            if heading not in offsets:
                offsets[heading] = match.start()
        return offsets

