"""Grounded Synthesis — Phase 4.

Produces attribution-backed, contradiction-aware, uncertainty-annotated
summaries from multiple evidence sources.

Public exports
--------------
Enumerations:
    ClaimType, ContradictionSeverity, UncertaintySeverity

Models:
    EvidenceSource, AttributedClaim, ContradictionPair,
    UncertaintyAnnotation, GroundedSummary, SynthesisRequest

Components:
    SourceAttributor, ClaimVerifier, ContradictionDetector,
    UncertaintyAnnotator, GroundedSummaryBuilder, MultiSourceSynthesizer
"""

from app.summarization.models import (
    AttributedClaim,
    ClaimType,
    ContradictionPair,
    ContradictionSeverity,
    EvidenceSource,
    GroundedSummary,
    SynthesisRequest,
    UncertaintyAnnotation,
    UncertaintySeverity,
)
from app.summarization.source_attribution import SourceAttributor
from app.summarization.claim_verifier import ClaimVerifier
from app.summarization.contradiction_detector import ContradictionDetector
from app.summarization.uncertainty_annotator import UncertaintyAnnotator
from app.summarization.grounded_summary_builder import GroundedSummaryBuilder
from app.summarization.multi_source_synthesizer import MultiSourceSynthesizer

__all__ = [
    # Enumerations
    "ClaimType",
    "ContradictionSeverity",
    "UncertaintySeverity",
    # Models
    "AttributedClaim",
    "ContradictionPair",
    "EvidenceSource",
    "GroundedSummary",
    "SynthesisRequest",
    "UncertaintyAnnotation",
    # Components
    "ClaimVerifier",
    "ContradictionDetector",
    "GroundedSummaryBuilder",
    "MultiSourceSynthesizer",
    "SourceAttributor",
    "UncertaintyAnnotator",
]

