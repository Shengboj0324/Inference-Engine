"""Deterministic scorer for the NeuroBridge rubric.

Walks the rendered Markdown report and checks each rubric criterion
against substring / regex contracts derived directly from the rubric in
``neurobridge_mock_user_test.md``.  No LLM-as-judge, no fuzz: a row
either matches or it does not, and the breakdown is reproducible.
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
    ("Confirmed limited roster exposure identified", 5,
     _has("confirmed limited", "roster exposure")),
    ("184 affected rows", 4, _has("184")),
    ("Northview High identified as affected", 3, _has("northview high")),
    ("Full-platform breach rejected", 4,
     _has("full-platform breach", "not confirmed")),
    ("SSN / payment / essay / chat / counsellor claims rejected", 5,
     _has("ssn", "payment", "essay", "tutoring chat", "counsel")),
    ("Password-reset + duplicate-email separated from incident", 3,
     _has("password-reset", "duplicate welcome", "unrelated")),
    ("Altered screenshots identified", 3,
     _any("altered screenshot", "some screenshots were altered",
          "screenshot manipulation")),
    ("Admin-account compromise labelled as likely incident path", 3,
     _has("admin", "compromise")),
]

_MEMORY: List[_Check] = [
    ("Prior debunked SSN memory used", 4,
     _has("previously debunked", "ssn")),
    ("Duplicate welcome-email known issue tracked", 3,
     _has("duplicate welcome", "operational bug")),
    ("Password-reset issue tracked across stages", 3,
     _has("password-reset", "email queue latency")),
    ("Source credibility profiles remembered", 3,
     _has("credibility=high", "credibility=low")),
    ("Claim status updates as evidence changes", 4,
     _has("contradicted by internal findings")),
    ("Severity updated across stages", 3,
     _has("sev-2", "sev-1", "sev-0")),
]

_EVIDENCE: List[_Check] = [
    ("Screenshots distinguished from verified evidence", 4,
     _any("existence of a screenshot does not prove",
          "screenshots, not verified", "amplification, not independent")),
    ("Internal investigation weighed highest", 4,
     _has("internal investigation", "highest-confidence")),
    ("Source credibility levels distinguished", 3,
     _has("higher-credibility", "lower-credibility") if False
     else _any("credibility=high", "credibility=low",
               "high-credibility", "low-reliability")),
    ("Volume separated from proof", 3,
     _any("volume of repetition is not evidence",
          "volume of repeated claims is not evidence",
          "amplification raised panic but did not define verified scope")),
    ("Conflicting evidence handled", 3,
     _any("real incident must not be dismissed",
          "misinformation must not be accepted")),
    ("Unresolved questions identified", 3,
     _any("under investigation", "scope may change",
          "lateral movement")),
]

_COORDINATION: List[_Check] = [
    ("Near-identical repeated claims", 2,
     _rx(r"1[,.]?100\s+posts")),
    ("Account-age metadata used", 2,
     _rx(r"620\s+accounts.*14\s+days")),
    ("Repeated screenshot amplification", 2,
     _rx(r"480\s+posts")),
    ("Altered screenshots", 2,
     _rx(r"90\s+(modified|altered)\s+screenshots?")),
    ("Bot noise separated from real parent concern", 2,
     _has("legitimate", "not bot noise")),
]

_ACTIONABILITY: List[_Check] = [
    ("Affected stakeholder groups identified", 4,
     _has("affected students", "parents", "northview high",
          "school administrators")),
    ("Security actions", 4,
     _has("disable or secure", "rotate", "mfa", "preserve",
          "lateral movement")),
    ("Legal / compliance actions", 3,
     _has("notification obligations", "school contract",
          "evidence chain")),
    ("Communications actions", 3,
     _has("notify northview high first", "false-claim correction",
          "high-credibility")),
    ("Product / engineering actions", 3,
     _has("admin export", "export alerting", "unusual-login")),
    ("Support workflow actions", 2,
     _has("parent support script", "school-admin support script")),
    ("Prioritises Northview and affected families", 1,
     _has("highest communication priority")),
]


def _score_category(name: str, checks: List[_Check], text: str
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
    b.categories["1. Accuracy (30)"] = _score_category(
        "accuracy", _ACCURACY, markdown)
    b.categories["2. Memory (20)"] = _score_category(
        "memory", _MEMORY, markdown)
    b.categories["3. Evidence Reasoning (20)"] = _score_category(
        "evidence", _EVIDENCE, markdown)
    b.categories["4. Coordinated Activity Detection (10)"] = _score_category(
        "coordination", _COORDINATION, markdown)
    b.categories["5. Stakeholder & Operational Actionability (20)"] = \
        _score_category("actionability", _ACTIONABILITY, markdown)
    return b
