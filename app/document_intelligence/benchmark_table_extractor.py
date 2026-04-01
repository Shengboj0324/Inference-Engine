"""Benchmark result table extractor.

Extracts benchmark/metric rows from the Results and Experiments sections
of academic papers using regex + heuristic line parsing.

Detected patterns:
- ``Model Name | 91.2 | 87.3 | 94.5`` (pipe-delimited tables)
- ``GPT-4: MMLU 86.4, HumanEval 67.0`` (inline benchmark mentions)
- ``Table 1: [caption]`` followed by rows

Emits ``BenchmarkResult`` objects for each (benchmark, model, metric, value) tuple.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from app.document_intelligence.models import BenchmarkResult, DocumentSection, SectionType

logger = logging.getLogger(__name__)

# Known benchmark names (used for inline extraction)
_KNOWN_BENCHMARKS = re.compile(
    r"\b(MMLU|HumanEval|GSM8K|BIG-Bench|HellaSwag|ARC|TruthfulQA|MATH|"
    r"WinoGrande|LAMBADA|SuperGLUE|GLUE|SQuAD|CodeXGLUE|HumanEval\+|"
    r"MBPP|LiveCodeBench|GPQA|MUSR|IFEval|BBH|MT-Bench|AlpacaEval)\b",
    re.IGNORECASE,
)
_PERCENT_VALUE = re.compile(r"(\d{1,3}\.?\d{0,2})\s*%?")
_PIPE_TABLE_ROW = re.compile(r"\|([^|]+)\|")
_MODEL_NAMES = re.compile(
    r"\b(GPT-?4o?|Claude[-\s]?\d|Gemini[-\s]?\w+|LLaMA[-\s]?\d|Mistral[-\s]?\w+|"
    r"Falcon[-\s]?\d+B?|Phi[-\s]?\d|Yi[-\s]?\d+B?|Qwen\w*|DeepSeek\w*|"
    r"Command[-\s]?\w+|Palm[-\s]?2?|Grok[-\s]?\d?)\b",
    re.IGNORECASE,
)


class BenchmarkTableExtractor:
    """Extracts benchmark results from document sections.

    Args:
        target_sections: Section types to search in.
        max_results:     Maximum number of results to return.
    """

    def __init__(
        self,
        target_sections: Optional[List[SectionType]] = None,
        max_results: int = 100,
    ) -> None:
        if max_results <= 0:
            raise ValueError(f"'max_results' must be positive, got {max_results!r}")
        self._target_sections = target_sections or [SectionType.RESULTS, SectionType.EXPERIMENTS]
        self._max_results = max_results

    def extract(self, sections: List[DocumentSection], focal_model: str = "") -> List[BenchmarkResult]:
        """Extract benchmark rows from *sections*.

        Args:
            sections:     Parsed document sections.
            focal_model:  The paper's primary model name (used as default
                          ``model_name`` when not detectable from table).

        Returns:
            List of ``BenchmarkResult`` sorted by benchmark name.

        Raises:
            TypeError: If *sections* is not a list.
        """
        if not isinstance(sections, list):
            raise TypeError(f"'sections' must be a list, got {type(sections)!r}")

        results: List[BenchmarkResult] = []
        for section in sections:
            if section.section_type not in self._target_sections:
                continue
            results.extend(self._extract_from_text(section.text, focal_model, section.section_type))

        # Deduplicate by (benchmark, model, metric)
        seen: set = set()
        deduped: List[BenchmarkResult] = []
        for r in results:
            key = (r.benchmark_name.lower(), r.model_name.lower(), r.metric.lower(), r.value)
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        deduped.sort(key=lambda r: r.benchmark_name)
        logger.debug("BenchmarkTableExtractor: extracted %d results", len(deduped))
        return deduped[: self._max_results]

    def _extract_from_text(self, text: str, focal_model: str, source_section: SectionType) -> List[BenchmarkResult]:
        results: List[BenchmarkResult] = []
        results.extend(self._extract_inline(text, focal_model, source_section))
        results.extend(self._extract_pipe_table(text, focal_model, source_section))
        return results

    def _extract_inline(self, text: str, focal_model: str, source_section: SectionType) -> List[BenchmarkResult]:
        """Extract benchmark mentions like 'achieves 91.2% on MMLU'."""
        results: List[BenchmarkResult] = []
        for m in _KNOWN_BENCHMARKS.finditer(text):
            benchmark = m.group(0)
            # Look for a numeric value nearby (within ±150 chars)
            start = max(0, m.start() - 150)
            end = min(len(text), m.end() + 150)
            context = text[start:end]
            val_m = _PERCENT_VALUE.search(context)
            if not val_m:
                continue
            value = val_m.group(0).strip()
            # Try to find a model name in the same context
            model_m = _MODEL_NAMES.search(context)
            model_name = model_m.group(0) if model_m else focal_model
            results.append(BenchmarkResult(
                benchmark_name=benchmark,
                metric="accuracy",
                value=value,
                model_name=model_name,
                source_section=source_section,
            ))
        return results

    def _extract_pipe_table(self, text: str, focal_model: str, source_section: SectionType) -> List[BenchmarkResult]:
        """Extract rows from pipe-delimited Markdown-style tables."""
        results: List[BenchmarkResult] = []
        lines = text.splitlines()
        header_cols: List[str] = []
        in_table = False

        for line in lines:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if not cells:
                in_table = False
                header_cols = []
                continue
            # Detect header row (contains benchmark keywords or "Model")
            if not in_table and (any(_KNOWN_BENCHMARKS.search(c) for c in cells) or "model" in cells[0].lower()):
                header_cols = cells
                in_table = True
                continue
            # Skip separator rows (--- or ===)
            if all(re.match(r"^[-=:]+$", c) for c in cells if c):
                continue
            if in_table and header_cols and len(cells) >= 2:
                model_name = cells[0] if not _PERCENT_VALUE.fullmatch(cells[0]) else focal_model
                for i, col_header in enumerate(header_cols[1:], start=1):
                    if i < len(cells) and _KNOWN_BENCHMARKS.search(col_header):
                        val_m = _PERCENT_VALUE.search(cells[i])
                        if val_m:
                            results.append(BenchmarkResult(
                                benchmark_name=col_header,
                                metric="score",
                                value=val_m.group(0),
                                model_name=model_name or focal_model,
                                source_section=source_section,
                            ))
        return results

