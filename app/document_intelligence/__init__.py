"""Document Intelligence — Phase 2 paper understanding pipeline.

Understands academic papers (PDF or raw text) by:
1. Extracting text from PDFs (``PDFIngestor``)
2. Segmenting into canonical sections (``SectionSegmenter``)
3. Building a structured ``PaperParsed`` object (``PaperParser``)
4. Extracting benchmark results (``BenchmarkTableExtractor``)
5. Separating claims from evidence (``MethodVsClaimExtractor``)
6. Building a citation graph (``CitationGraph``)
7. Producing an LLM-powered structured summary (``PaperSummarizer``)
8. Estimating novelty (``NoveltyEstimator``)

Public exports
--------------
Models: SectionType, DocumentSection, BenchmarkResult, ClaimEvidence,
        CitationNode, CitationEdge, PaperParsed
Components: PDFIngestor, SectionSegmenter, PaperParser,
            BenchmarkTableExtractor, MethodVsClaimExtractor,
            CitationGraph, PaperSummarizer, NoveltyEstimator
"""

from app.document_intelligence.models import (
    BenchmarkResult,
    CitationEdge,
    CitationNode,
    ClaimEvidence,
    DocumentSection,
    PaperParsed,
    SectionType,
)
from app.document_intelligence.pdf_ingestor import PDFIngestor
from app.document_intelligence.section_segmenter import SectionSegmenter
from app.document_intelligence.paper_parser import PaperParser
from app.document_intelligence.benchmark_table_extractor import BenchmarkTableExtractor
from app.document_intelligence.method_vs_claim_extractor import MethodVsClaimExtractor
from app.document_intelligence.citation_graph import CitationGraph
from app.document_intelligence.paper_summarizer import PaperSummarizer
from app.document_intelligence.novelty_estimator import NoveltyEstimator

__all__ = [
    # Models
    "BenchmarkResult",
    "CitationEdge",
    "CitationNode",
    "ClaimEvidence",
    "DocumentSection",
    "PaperParsed",
    "SectionType",
    # Components
    "BenchmarkTableExtractor",
    "CitationGraph",
    "MethodVsClaimExtractor",
    "NoveltyEstimator",
    "PDFIngestor",
    "PaperParser",
    "PaperSummarizer",
    "SectionSegmenter",
]

