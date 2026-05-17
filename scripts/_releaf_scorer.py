"""Deterministic content scorer for the Releaf crisis-intelligence report.

Mirrors the structure of :mod:`scripts._neurobridge_scorer`.  Each row is
a substring / regex check against the rendered Markdown report; either
the contract matches or it does not.  Categories are weighted to match
the Releaf mock-user rubric (accuracy + memory + evidence reasoning +
coordination detection + actionability = 100 total).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple


@dataclass
class ScoreBreakdown:
    categories: Dict[str, Tuple[int, int, List[Tuple[str, int, int]]]] = \
        field(default_factory=dict)

    @property
    def total_earned(self) -> int:
        return sum(e for e, _t, _r in self.categories.values())

    @property
    def total_max(self) -> int:
        return sum(t for _e, t, _r in self.categories.values())


_Check = Tuple[str, int, Callable[[str], bool]]


def _has(*needles: str) -> Callable[[str], bool]:
    lowers = [n.lower() for n in needles]
    return lambda text: all(n in text.lower() for n in lowers)


def _any(*needles: str) -> Callable[[str], bool]:
    lowers = [n.lower() for n in needles]
    return lambda text: any(n in text.lower() for n in lowers)


def _rx(pattern: str) -> Callable[[str], bool]:
    rx = re.compile(pattern, re.IGNORECASE | re.DOTALL)
    return lambda text: rx.search(text) is not None


_ACCURACY: List[_Check] = [
    ("Address-selling claim labelled FAKE / debunked", 6,
     _has("[fake]", "releaf-sells-addresses")),
    ("Onboarding wording labelled REAL issue", 5,
     _has("[real]", "onboarding")),
    ("Slow image upload on iPhone XR labelled REAL", 5,
     _has("[real]", "iphone")),
    ("Generic AI search labelled REAL", 5,
     _has("[real]", "ai search")),
    ("Old vs current onboarding screenshots reconciled", 5,
     _any("contradictory screenshots", "[mixed]")),
    ("EcoWatchdog misquotation labelled FAKE", 4,
     _has("[fake]", "ecowatchdog")),
]

_MEMORY: List[_Check] = [
    ("Yesterday's debunked claim reused", 5,
     _has("releaf-sells-addresses")),
    ("Known iPhone-XR upload issue cross-referenced", 4,
     _any("iphone-xr-slow-upload", "iphone xr")),
    ("Known generic-AI-search issue cross-referenced", 4,
     _any("ai-search-generic", "generic answers")),
    ("Important actors remembered with credibility tags", 4,
     _has("credibility=high", "credibility=low")),
    ("Old-bug-on-outdated-version confusion remembered", 3,
     _any("v1.0.0", "fixed in v1.0.2")),
]

_EVIDENCE: List[_Check] = [
    ("Counter-evidence on multiple platforms cited", 5,
     _has("counter-evidence")),
    ("Source credibility distinguishes high vs low", 4,
     _has("credibility=high", "credibility=low")),
    ("Sarcasm flagged and not literal", 4,
     _has("sarcasm")),
    ("Misquotation directly contradicted by source quote", 4,
     _has("direct contradiction by source")),
    ("False-consensus claim countered with real signals", 3,
     _has("[fake]", "everyone is deleting")),
]

_COORDINATION: List[_Check] = [
    ("Near-duplicate amplification volume reported", 3,
     _rx(r"(~|approximately\s+)?300\s+(near-duplicate|posts)")),
    ("New-account window flagged (<7 days)", 3,
     _rx(r"<\s*7[- ]day")),
    ("Sarcasm cluster separated from real complaints", 2,
     _has("sarcasm")),
    ("Multilingual duplicate cluster detected", 2,
     _any("chinese", "spanish", "french")),
]

_ACTIONABILITY: List[_Check] = [
    ("Privacy one-pager / explainer action", 4,
     _any("privacy one-pager", "privacy explainer")),
    ("Onboarding rewrite action", 4,
     _any("rewrite onboarding", "onboarding location screen")),
    ("Misinformation protocol action", 3,
     _has("misinformation protocol")),
    ("Stakeholder reply queue with named partners", 4,
     _has("@campusclimatelab", "@ecowatchdog", "@bayareaecoclub")),
    ("Required explicit answers a/b/c/d all YES", 5,
     _rx(
         r"\(a\).*yes.*\(b\).*yes.*\(c\).*yes.*\(d\)"
     )),
]


def _score_category(checks: List[_Check], text: str
                    ) -> Tuple[int, int, List[Tuple[str, int, int]]]:
    rows: List[Tuple[str, int, int]] = []
    earned = 0
    total = 0
    for label, pts, predicate in checks:
        total += pts
        ok = bool(predicate(text))
        rows.append((label, pts if ok else 0, pts))
        if ok:
            earned += pts
    return earned, total, rows


def score_report(markdown: str) -> ScoreBreakdown:
    b = ScoreBreakdown()
    b.categories["1. Accuracy (30)"] = _score_category(_ACCURACY, markdown)
    b.categories["2. Memory (20)"] = _score_category(_MEMORY, markdown)
    b.categories["3. Evidence Reasoning (20)"] = _score_category(
        _EVIDENCE, markdown)
    b.categories["4. Coordinated Activity Detection (10)"] = _score_category(
        _COORDINATION, markdown)
    b.categories["5. Stakeholder & Operational Actionability (20)"] = \
        _score_category(_ACTIONABILITY, markdown)
    return b
