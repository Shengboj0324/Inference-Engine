"""Shared Pydantic models for the document intelligence pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class SectionType(str, Enum):
    """Standard academic paper section taxonomy."""

    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    BACKGROUND = "background"
    METHODOLOGY = "methodology"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    APPENDIX = "appendix"
    OTHER = "other"


class DocumentSection(BaseModel, frozen=True):
    """A single section of a parsed document.

    Attributes:
        section_type: Taxonomy classification.
        heading:      Original heading text as it appears in the document.
        text:         Full section text (stripped of headers).
        page_start:   First page number (1-indexed); None if unknown.
        page_end:     Last page number (1-indexed); None if unknown.
        order:        Section order index (0-indexed within document).
    """

    section_type: SectionType
    heading: str
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    order: int = 0

    @field_validator("order")
    @classmethod
    def _non_negative_order(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"'order' must be >= 0, got {v!r}")
        return v


class BenchmarkResult(BaseModel, frozen=True):
    """A single benchmark metric row extracted from a results table.

    Attributes:
        benchmark_name:  Name of the benchmark (e.g. ``"MMLU"``, ``"HumanEval"``).
        metric:          Metric name (e.g. ``"accuracy"``, ``"pass@1"``).
        value:           Numeric value (float or formatted string).
        model_name:      Model being evaluated.
        comparison_models: Other models in the same table row, if any.
        higher_is_better: Interpretation hint.
        source_section:  Which section this was extracted from.
    """

    benchmark_name: str
    metric: str
    value: str  # Keep as string to preserve formatting (e.g. "91.2%", "4.2")
    model_name: str = ""
    comparison_models: List[str] = Field(default_factory=list)
    higher_is_better: bool = True
    source_section: SectionType = SectionType.RESULTS


class ClaimEvidence(BaseModel, frozen=True):
    """A claim paired with its supporting evidence from the paper.

    Attributes:
        claim:         The claim statement (what the paper asserts).
        evidence:      Verbatim or paraphrased evidence from the paper.
        claim_type:    ``"contribution"``, ``"limitation"``, or ``"future_work"``.
        supported:     True if evidence supports the claim within this paper.
        section:       Section where evidence was found.
        confidence:    Extraction confidence [0, 1].
    """

    claim: str
    evidence: str
    claim_type: str = "contribution"
    supported: bool = True
    section: SectionType = SectionType.OTHER
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)

    @field_validator("claim_type")
    @classmethod
    def _valid_claim_type(cls, v: str) -> str:
        allowed = {"contribution", "limitation", "future_work", "comparison"}
        if v not in allowed:
            raise ValueError(f"'claim_type' must be one of {allowed}, got {v!r}")
        return v


class CitationNode(BaseModel):
    """A node in the citation graph.

    Attributes:
        paper_id:     Unique identifier (arXiv ID, DOI, or title hash).
        title:        Paper title.
        authors:      Author list.
        year:         Publication year.
        venue:        Conference or journal name.
        citation_count: Times cited (0 if unknown).
        is_focal:     True if this is the paper being analyzed.
    """

    paper_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    citation_count: int = 0
    is_focal: bool = False


class CitationEdge(BaseModel, frozen=True):
    """A directed citation edge: source → target."""

    source_id: str
    target_id: str
    context: str = ""  # Sentence in which the citation appears


class PaperParsed(BaseModel):
    """Complete parsed representation of an academic paper.

    Attributes:
        paper_id:        arXiv ID, DOI, or SHA-256 hash of title.
        title:           Paper title.
        authors:         List of author names.
        institutions:    Author institutions / affiliations.
        year:            Publication year.
        venue:           Conference or journal.
        abstract:        Paper abstract.
        sections:        Ordered list of document sections.
        benchmarks:      Extracted benchmark results.
        claims:          Extracted claim-evidence pairs.
        keywords:        Author-supplied or auto-extracted keywords.
        pdf_url:         URL to PDF (if available).
        arxiv_id:        arXiv identifier (e.g. ``"2401.12345"``).
        doi:             DOI string.
        citation_count:  Citations from Semantic Scholar / CrossRef.
        novelty_score:   Computed by ``NoveltyEstimator`` [0, 1].
        llm_summary:     Structured summary from ``PaperSummarizer``.
        metadata:        Arbitrary extra metadata.
    """

    paper_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    institutions: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    abstract: str = ""
    sections: List[DocumentSection] = Field(default_factory=list)
    benchmarks: List[BenchmarkResult] = Field(default_factory=list)
    claims: List[ClaimEvidence] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    pdf_url: str = ""
    arxiv_id: str = ""
    doi: str = ""
    citation_count: int = 0
    novelty_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    llm_summary: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_section(self, section_type: SectionType) -> Optional[DocumentSection]:
        """Return the first section of *section_type*, or None."""
        for sec in self.sections:
            if sec.section_type == section_type:
                return sec
        return None

    def full_text(self) -> str:
        """Return concatenated text of all sections."""
        return "\n\n".join(f"# {sec.heading}\n{sec.text}" for sec in self.sections)

