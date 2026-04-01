"""Method vs claim extractor.

Separates what a paper *does* (methodology, contributions) from what it
*claims* (results, comparisons, limitations) and what it *promises*
(future work).

Each ``ClaimEvidence`` object records:
- The normalized claim statement
- Verbatim evidence from the same section
- ``claim_type``: ``"contribution" | "limitation" | "future_work" | "comparison"``
- Whether the paper provides supporting evidence in scope

LLM path: structured extraction via JSON prompt.
Heuristic path: keyword-triggered sentence classification.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.document_intelligence.models import ClaimEvidence, DocumentSection, SectionType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic classifiers
# ---------------------------------------------------------------------------
_CONTRIBUTION = re.compile(
    r"\b(we\s+(propose|introduce|present|develop|design|show|demonstrate)|"
    r"our\s+(model|method|approach|system|framework)|"
    r"in\s+this\s+(paper|work)|we\s+release)\b",
    re.IGNORECASE,
)
_COMPARISON = re.compile(
    r"\b(outperform|surpass|exceed|superior|better\s+than|"
    r"compared\s+to|vs\.?|versus|baseline|state.of.the.art|sota)\b",
    re.IGNORECASE,
)
_LIMITATION = re.compile(
    r"\b(limitation[s]?|drawback[s]?|shortcoming[s]?|fail[s]?\s+to|"
    r"unable\s+to|not\s+applicable|constrained|restricted)\b",
    re.IGNORECASE,
)
_FUTURE = re.compile(
    r"\b(future\s+work|future\s+research|we\s+plan|will\s+(explore|investigate|"
    r"extend|improve)|leave[s]?\s+for\s+future|open\s+problem)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MIN_WORDS = 10

_LLM_PROMPT = """\
Extract claims and evidence from this academic paper section.

Return a JSON array. Each object:
{{
  "claim": str,        // normalized claim (what the paper asserts)
  "evidence": str,     // verbatim sentence(s) supporting the claim
  "claim_type": str,   // "contribution" | "limitation" | "future_work" | "comparison"
  "supported": bool    // is evidence provided in this section?
}}

SECTION ({section_type}):
{text}
"""

_EVIDENCE_SECTIONS = {SectionType.RESULTS, SectionType.EXPERIMENTS, SectionType.METHODOLOGY}
_CLAIM_SECTIONS = {SectionType.ABSTRACT, SectionType.INTRODUCTION, SectionType.CONCLUSION,
                   SectionType.DISCUSSION, SectionType.RESULTS, SectionType.EXPERIMENTS}


class MethodVsClaimExtractor:
    """Extracts claims and methodology statements from paper sections.

    Args:
        llm_router:          Optional LLM router for structured extraction.
        min_confidence:      Minimum confidence for heuristic claims.
        max_claims_per_section: Cap per section.
    """

    def __init__(
        self,
        llm_router: Optional[Any] = None,
        min_confidence: float = 0.4,
        max_claims_per_section: int = 10,
    ) -> None:
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(f"'min_confidence' must be in [0, 1], got {min_confidence!r}")
        if max_claims_per_section <= 0:
            raise ValueError(f"'max_claims_per_section' must be positive, got {max_claims_per_section!r}")
        self._router = llm_router
        self._min_conf = min_confidence
        self._max_per_section = max_claims_per_section

    async def extract(self, sections: List[DocumentSection]) -> List[ClaimEvidence]:
        """Extract claim-evidence pairs from all relevant sections.

        Args:
            sections: Parsed document sections.

        Returns:
            List of ``ClaimEvidence`` sorted by section order.

        Raises:
            TypeError: If *sections* is not a list.
        """
        if not isinstance(sections, list):
            raise TypeError(f"'sections' must be a list, got {type(sections)!r}")
        results: List[ClaimEvidence] = []
        for sec in sections:
            if sec.section_type not in _CLAIM_SECTIONS:
                continue
            claims = await self._extract_section(sec)
            results.extend(claims[: self._max_per_section])
        logger.debug("MethodVsClaimExtractor: %d claims from %d sections", len(results), len(sections))
        return results

    async def _extract_section(self, section: DocumentSection) -> List[ClaimEvidence]:
        if not section.text.strip():
            return []
        if self._router is not None:
            try:
                return await self._llm_extract(section)
            except Exception as exc:
                logger.warning("MethodVsClaimExtractor: LLM failed (%s), using heuristic", exc)
        return self._heuristic_extract(section)

    async def _llm_extract(self, section: DocumentSection) -> List[ClaimEvidence]:
        prompt = _LLM_PROMPT.format(
            section_type=section.section_type.value,
            text=section.text[:4000],
        )
        raw = await self._router.complete(prompt, max_tokens=1200, temperature=0.1)
        raw_json = re.sub(r"```[a-z]*\n?", "", raw).strip().rstrip("`")
        items: list[dict] = json.loads(raw_json)
        claims: List[ClaimEvidence] = []
        for item in items:
            try:
                claims.append(ClaimEvidence(
                    claim=str(item["claim"])[:400],
                    evidence=str(item.get("evidence", ""))[:500],
                    claim_type=item.get("claim_type", "contribution"),
                    supported=bool(item.get("supported", True)),
                    section=section.section_type,
                    confidence=0.85,
                ))
            except Exception:
                continue
        return claims

    def _heuristic_extract(self, section: DocumentSection) -> List[ClaimEvidence]:
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(section.text) if len(s.split()) >= _MIN_WORDS]
        results: List[ClaimEvidence] = []
        for sentence in sentences:
            claim_type, confidence = self._classify(sentence)
            if confidence < self._min_conf:
                continue
            supported = section.section_type in _EVIDENCE_SECTIONS
            results.append(ClaimEvidence(
                claim=sentence[:400],
                evidence=sentence[:500],
                claim_type=claim_type,
                supported=supported,
                section=section.section_type,
                confidence=confidence,
            ))
        return results

    @staticmethod
    def _classify(sentence: str) -> tuple[str, float]:
        # Check limitation and future_work first — they are more specific
        if _LIMITATION.search(sentence):
            return "limitation", 0.65
        if _FUTURE.search(sentence):
            return "future_work", 0.60
        if _COMPARISON.search(sentence):
            return "comparison", 0.70
        if _CONTRIBUTION.search(sentence):
            return "contribution", 0.75
        return "contribution", 0.30

