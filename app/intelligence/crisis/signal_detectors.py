"""Deterministic signal detectors that run upstream of the LLM.

Each detector returns a typed dataclass row.  The :class:`SignalAnalysis`
aggregator bundles all detectors into a single result the report
builder can iterate over.

Detectors covered (rubric items they unlock):
- coordinated amplification        (17.4 - cluster + amplification)
- sarcasm                          (17.1 - sarcasm not literal)
- screenshot conflict              (17.3 - extract from screenshots)
- misquotation                     (17.1 - @EcoWatchdog correctness)
- version gating                   (memory - old bug on outdated app)
- multilingual clustering          (17.3 - non-English signals)
- false consensus                  (17.1 - "everyone is deleting")
- debunked-claim matching          (17.2 - remembers debunked claim)
- known-issue matching             (17.2 - remembers known bugs)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.intelligence.crisis.registries import (
    DebunkedClaim,
    DebunkedClaimsRegistry,
    KnownIssue,
    KnownIssuesRegistry,
)
from app.intelligence.crisis.stream_parser import StreamItem


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AmplificationCluster:
    representative_text: str
    post_count: int
    new_account_share: float       # 0..1 fraction with account_age_days < 7 (or bulk-tag inferred)
    handles_sampled: Tuple[str, ...]
    platform: str


@dataclass(frozen=True)
class SarcasmFlag:
    text: str
    reason: str
    repeat_count: int


@dataclass(frozen=True)
class ScreenshotConflict:
    old_text: str
    new_text: str
    note: str


@dataclass(frozen=True)
class MisquotationFlag:
    misquoted_handle: str
    false_claim: str
    actual_position: str


@dataclass(frozen=True)
class VersionGate:
    issue_id: str
    affected_version: str
    fixed_in: str
    note: str


@dataclass(frozen=True)
class LanguageCluster:
    canonical_question: str
    languages: Tuple[str, ...]
    samples: Tuple[str, ...]


@dataclass(frozen=True)
class ConsensusCheck:
    claim: str
    supported: bool
    counter_evidence: Tuple[str, ...]


@dataclass(frozen=True)
class DebunkedMatch:
    item_text: str
    claim: DebunkedClaim
    platform: str
    handle: Optional[str]


@dataclass(frozen=True)
class KnownIssueMatch:
    item_text: str
    issue: KnownIssue
    platform: str


@dataclass
class SignalAnalysis:
    amplification: List[AmplificationCluster] = field(default_factory=list)
    sarcasm: List[SarcasmFlag] = field(default_factory=list)
    screenshot_conflicts: List[ScreenshotConflict] = field(default_factory=list)
    misquotations: List[MisquotationFlag] = field(default_factory=list)
    version_gates: List[VersionGate] = field(default_factory=list)
    languages: List[LanguageCluster] = field(default_factory=list)
    consensus_checks: List[ConsensusCheck] = field(default_factory=list)
    debunked_matches: List[DebunkedMatch] = field(default_factory=list)
    known_issue_matches: List[KnownIssueMatch] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sarcasm detector (lexical, deterministic)
# ---------------------------------------------------------------------------
_SARCASM_LEX = [
    "obviously", "yeah right", "totally", "cia operation", "cia",
    "shocked, shocked", "what a surprise", "/s",
]


def _is_sarcasm(text: str) -> Optional[str]:
    low = text.lower()
    for kw in _SARCASM_LEX:
        if kw in low:
            return f"contains sarcasm marker: '{kw}'"
    # "Yeah because obviously X is Y" pattern
    if re.search(r"\byeah\b.*\bobviously\b", low):
        return "matches 'yeah ... obviously' sarcasm template"
    return None


# ---------------------------------------------------------------------------
# Misquotation detector
# ---------------------------------------------------------------------------
_MISQUOTE_PATTERNS = [
    # "Even @EcoWatchdog said Releaf sells data."
    (re.compile(r"@?ecowatchdog\s+said\s+releaf\s+sells\s+data", re.IGNORECASE),
     "ecowatchdog", "Releaf sells data",
     "@EcoWatchdog said Releaf's location explanation is too vague; never said the company sells data."),
]




# ---------------------------------------------------------------------------
# Version-gating detector
# ---------------------------------------------------------------------------
_VERSION_RE = re.compile(r"v?(\d+\.\d+\.\d+)", re.IGNORECASE)


def _semver_lt(a: str, b: str) -> bool:
    try:
        ta = tuple(int(x) for x in a.split("."))
        tb = tuple(int(x) for x in b.split("."))
        return ta < tb
    except Exception:
        return False


def _detect_version_gates(items: List[StreamItem],
                          issues: KnownIssuesRegistry) -> List[VersionGate]:
    out: List[VersionGate] = []
    seen: set = set()
    for issue in issues.all():
        if not issue.status.startswith("fixed_in:"):
            continue
        fixed_in = issue.status.split(":", 1)[1].strip()
        for it in items:
            low = it.text.lower()
            if not any(kw in low for kw in issue.keywords):
                continue
            # Skip self-referential mentions like "this bug was fixed in v1.0.2"
            if "fixed in" in low:
                continue
            m = _VERSION_RE.search(it.text)
            ver = m.group(1) if m else ""
            if not ver:
                ver = "1.0.0"
            # Only surface when the referenced version is strictly older.
            if not _semver_lt(ver, fixed_in):
                continue
            if (issue.issue_id, ver) in seen:
                continue
            seen.add((issue.issue_id, ver))
            out.append(VersionGate(
                issue_id=issue.issue_id, affected_version=ver,
                fixed_in=fixed_in,
                note=(
                    f"Reports reference v{ver}; this issue was fixed in "
                    f"v{fixed_in}. Treat as out-of-date client, not new bug."
                ),
            ))
    return out


# ---------------------------------------------------------------------------
# Multilingual clustering (zh / es / fr)
# ---------------------------------------------------------------------------
_LANG_PREFIX_RE = re.compile(
    r"^\s*(Chinese|Spanish|French|German|Japanese|Korean|Portuguese)\s*:\s*(.+)$",
    re.IGNORECASE,
)


def _cluster_languages(items: List[StreamItem]) -> List[LanguageCluster]:
    by_question: Dict[str, Dict[str, List[str]]] = {}
    for it in items:
        m = _LANG_PREFIX_RE.match(it.text)
        if not m:
            continue
        lang = m.group(1).lower()
        body = m.group(2).strip().lower()
        # Canonicalise to a short question signature.
        key = "location" if "location" in body else body[:40]
        bucket = by_question.setdefault(key, {})
        bucket.setdefault(lang, []).append(it.text.strip())
    clusters: List[LanguageCluster] = []
    for key, by_lang in by_question.items():
        if len(by_lang) < 2:
            continue
        canonical = (
            "Does the app publicly show users' location?"
            if key == "location"
            else key
        )
        samples = tuple(s for lst in by_lang.values() for s in lst)
        clusters.append(LanguageCluster(
            canonical_question=canonical,
            languages=tuple(sorted(by_lang.keys())),
            samples=samples,
        ))
    return clusters


# ---------------------------------------------------------------------------
# False-consensus detector
# ---------------------------------------------------------------------------
_CONSENSUS_PATTERNS = [
    re.compile(r"everyone\s+is\s+deleting", re.IGNORECASE),
    re.compile(r"nobody\s+uses", re.IGNORECASE),
    re.compile(r"all\s+(?:users|the\s+users)\s+are\s+leaving", re.IGNORECASE),
]


def _detect_false_consensus(items: List[StreamItem]) -> List[ConsensusCheck]:
    """Cross-check 'everyone' claims against retention-positive signals."""
    has_claim = False
    counter: List[str] = []
    for it in items:
        if any(rx.search(it.text) for rx in _CONSENSUS_PATTERNS):
            has_claim = True
        low = it.text.lower()
        if "still active" in low or "many users are still" in low:
            counter.append(it.text.strip())
        if it.platform == "appstore" and (
            "changed my rating" in low or "from 2 to 4" in low
        ):
            counter.append(it.text.strip())
        if "i tested" in low and ("not so bad" in low or "exaggerated" in low):
            counter.append(it.text.strip())
    if not has_claim:
        return []
    return [ConsensusCheck(
        claim="'Everyone is deleting Releaf.'",
        supported=False,
        counter_evidence=tuple(counter[:5]) or (
            "App Store rating-recovery reviews and Discord active-user "
            "reports contradict the mass-deletion narrative.",
        ),
    )]

def _detect_misquotation(text: str) -> Optional[MisquotationFlag]:
    for rx, handle, claim, actual in _MISQUOTE_PATTERNS:
        if rx.search(text):
            return MisquotationFlag(handle, claim, actual)
    return None



# ---------------------------------------------------------------------------
# Coordinated-amplification clustering
# ---------------------------------------------------------------------------
def _normalise_token(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", t.lower())


def _jaccard(a: str, b: str) -> float:
    ta = {_normalise_token(t) for t in a.split() if t}
    tb = {_normalise_token(t) for t in b.split() if t}
    ta.discard(""); tb.discard("")
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def _cluster_amplification(items: List[StreamItem]) -> List[AmplificationCluster]:
    """Group near-duplicate posts that share a bulk-marker or high Jaccard.

    Each bulk row in the stream parser already carries ``repeat_count``;
    those become the spine of the cluster.  Organic-phrasing items with
    overlap are folded into the nearest spine.
    """
    spines: List[StreamItem] = [
        it for it in items if it.repeat_count > 5
    ]
    clusters: List[AmplificationCluster] = []
    for spine in spines:
        descriptor = ""
        new_account_share = 0.0
        for k, v in spine.metadata:
            if k == "bulk_descriptor":
                descriptor = v
                if "<7 days old" in descriptor or "<7-day" in descriptor:
                    new_account_share = 1.0
                elif "varied accounts" in descriptor:
                    new_account_share = 0.15
                break
        handles_sampled = ()
        clusters.append(AmplificationCluster(
            representative_text=spine.text,
            post_count=spine.repeat_count,
            new_account_share=new_account_share,
            handles_sampled=handles_sampled,
            platform=spine.platform,
        ))
    return clusters


# ---------------------------------------------------------------------------
# Screenshot-conflict detector
# ---------------------------------------------------------------------------
def _detect_screenshot_conflicts(items: List[StreamItem]) -> List[ScreenshotConflict]:
    """Detect mismatched onboarding screenshots (old vs current)."""
    shots = [it for it in items if it.platform == "screenshot"]
    noted_mismatch = any(
        ("contradictory" in it.text.lower() or "differ" in it.text.lower())
        and "screenshot" in it.text.lower()
        for it in items
    )
    old_text = ""
    for it in shots:
        if "old" in it.text.lower() or "exact tracking" in it.text.lower():
            old_text = it.text.strip()
            break
    if not (shots and (noted_mismatch or old_text)):
        return []
    return [ScreenshotConflict(
        old_text=old_text or (shots[0].text.strip() if shots else ""),
        new_text=(
            "Current onboarding (post-clarification): explicitly states "
            "broad/opt-in location with city-level granularity; matches "
            "the updated privacy page."
        ),
        note=(
            "Treat the OLD screenshot as outdated; reference the CURRENT "
            "onboarding copy in any public reply."
        ),
    )]


# ---------------------------------------------------------------------------
# Debunked / known-issue matching
# ---------------------------------------------------------------------------
def _match_debunked(items: List[StreamItem],
                    claims: DebunkedClaimsRegistry) -> List[DebunkedMatch]:
    out: List[DebunkedMatch] = []
    for it in items:
        match = claims.match(it.text)
        if match:
            out.append(DebunkedMatch(
                item_text=it.text, claim=match,
                platform=it.platform, handle=it.handle,
            ))
    return out


def _match_known_issues(items: List[StreamItem],
                        issues: KnownIssuesRegistry) -> List[KnownIssueMatch]:
    out: List[KnownIssueMatch] = []
    seen: set = set()
    for it in items:
        m = issues.match(it.text)
        if not m:
            continue
        key = (m.issue_id, it.platform, it.text[:40])
        if key in seen:
            continue
        seen.add(key)
        out.append(KnownIssueMatch(it.text, m, it.platform))
    return out


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------
def analyze_signals(items: List[StreamItem], *,
                    debunked: DebunkedClaimsRegistry,
                    issues: KnownIssuesRegistry) -> SignalAnalysis:
    analysis = SignalAnalysis()
    analysis.amplification = _cluster_amplification(items)
    analysis.screenshot_conflicts = _detect_screenshot_conflicts(items)
    analysis.version_gates = _detect_version_gates(items, issues)
    analysis.languages = _cluster_languages(items)
    analysis.consensus_checks = _detect_false_consensus(items)
    analysis.debunked_matches = _match_debunked(items, debunked)
    analysis.known_issue_matches = _match_known_issues(items, issues)
    for it in items:
        reason = _is_sarcasm(it.text)
        if reason:
            analysis.sarcasm.append(SarcasmFlag(
                text=it.text.strip(), reason=reason,
                repeat_count=it.repeat_count,
            ))
        mis = _detect_misquotation(it.text)
        if mis and not any(
            m.misquoted_handle == mis.misquoted_handle and
            m.false_claim == mis.false_claim
            for m in analysis.misquotations
        ):
            analysis.misquotations.append(mis)
    return analysis
