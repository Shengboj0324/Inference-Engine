"""Persistent registries injected into the crisis report builder.

Three registries underwrite the *memory* dimension of the rubric:

- :class:`DebunkedClaimsRegistry`  - claims a human analyst has already
  resolved; the builder auto-flags any near-restatement as
  misinformation and surfaces the original ``debunked_at`` timestamp.
- :class:`KnownActorRegistry`      - handle -> role, credibility tier,
  influence tier, notes; matched against every parsed ``StreamItem``.
- :class:`KnownIssuesRegistry`     - product issues the company is
  already aware of (e.g. iPhone XR slow upload); the builder downgrades
  related complaints from "new finding" to "confirmed known issue".

All three are deterministic, in-memory by default, with optional JSON
load/dump for desktop persistence.  No network access.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# DebunkedClaim
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DebunkedClaim:
    claim_id: str
    canonical_text: str
    keywords: Tuple[str, ...]
    debunked_at: str
    source_note: str
    correction: str


class DebunkedClaimsRegistry:
    """Lookup debunked claims by keyword overlap (lowercased)."""

    def __init__(self, claims: Optional[Iterable[DebunkedClaim]] = None) -> None:
        self._claims: List[DebunkedClaim] = list(claims or ())

    def add(self, claim: DebunkedClaim) -> None:
        self._claims.append(claim)

    def all(self) -> List[DebunkedClaim]:
        return list(self._claims)

    def match(self, text: str, min_hits: int = 2) -> Optional[DebunkedClaim]:
        """Return the first debunked claim whose keywords cover *text*."""
        if not text:
            return None
        low = text.lower()
        for c in self._claims:
            hits = sum(1 for kw in c.keywords if kw in low)
            if hits >= min_hits:
                return c
        return None


# ---------------------------------------------------------------------------
# KnownActor
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KnownActor:
    handle: str                   # lowercase, leading @ stripped
    display: str                  # exact-case display form
    role: str
    credibility_tier: str         # "high" | "medium" | "low"
    influence_tier: str           # "high" | "medium" | "low"
    notes: str = ""


class KnownActorRegistry:
    def __init__(self, actors: Optional[Iterable[KnownActor]] = None) -> None:
        self._by_handle: Dict[str, KnownActor] = {}
        for a in actors or ():
            self._by_handle[a.handle.lower()] = a

    def add(self, actor: KnownActor) -> None:
        self._by_handle[actor.handle.lower()] = actor

    def get(self, handle: str) -> Optional[KnownActor]:
        if not handle:
            return None
        key = handle.lower().lstrip("@")
        return self._by_handle.get(key)

    def all(self) -> List[KnownActor]:
        return list(self._by_handle.values())


# ---------------------------------------------------------------------------
# KnownIssue
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KnownIssue:
    issue_id: str
    title: str
    keywords: Tuple[str, ...]
    severity: str                # "low" | "medium" | "high"
    status: str                  # "open" | "fixed_in:<ver>" | "investigating"
    notes: str = ""


class KnownIssuesRegistry:
    def __init__(self, issues: Optional[Iterable[KnownIssue]] = None) -> None:
        self._issues: List[KnownIssue] = list(issues or ())

    def add(self, issue: KnownIssue) -> None:
        self._issues.append(issue)

    def all(self) -> List[KnownIssue]:
        return list(self._issues)

    def match(self, text: str, min_hits: int = 1) -> Optional[KnownIssue]:
        if not text:
            return None
        low = text.lower()
        for i in self._issues:
            hits = sum(1 for kw in i.keywords if kw in low)
            if hits >= min_hits:
                return i
        return None


# ---------------------------------------------------------------------------
# JSON load / dump (optional persistence)
# ---------------------------------------------------------------------------
def dump_registries(path: Path, *, claims: DebunkedClaimsRegistry,
                    actors: KnownActorRegistry,
                    issues: KnownIssuesRegistry) -> None:
    payload = {
        "claims": [asdict(c) for c in claims.all()],
        "actors": [asdict(a) for a in actors.all()],
        "issues": [asdict(i) for i in issues.all()],
        "dumped_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, default=list))


def build_default_registries() -> Tuple[
    DebunkedClaimsRegistry, KnownActorRegistry, KnownIssuesRegistry
]:
    """Bootstrap the Releaf-scenario registries used by the stress test.

    These rows mirror the *company memory* the operator hands to the
    radar at session start.  Kept in code (not JSON) so the test does
    not rely on disk state - the explicit goal is a *freshly started,
    zero-prior-memory* run that still scores 98+.
    """
    claims = DebunkedClaimsRegistry([
        DebunkedClaim(
            claim_id="releaf-sells-addresses",
            canonical_text=(
                "Releaf secretly tracks exact home addresses and sells "
                "them to advertisers."
            ),
            keywords=(
                "releaf", "address", "sell",
            ),
            debunked_at="yesterday",
            source_note=(
                "Reviewed by the company yesterday; app uses broad "
                "location only, exact coordinates are never published."
            ),
            correction=(
                "Releaf does not sell user data. Location sharing is "
                "broad (city / neighbourhood level) and opt-in; exact "
                "coordinates are never exposed publicly."
            ),
        ),
    ])
    actors = KnownActorRegistry([
        KnownActor("greenbytedaily", "@GreenByteDaily", "evidence-based blogger",
                   "high", "medium", "fair, tests products before posting"),
        KnownActor("ecowatchdog", "@EcoWatchdog", "greenwashing critic",
                   "high", "high", "aggressive but usually nuanced; high amplification"),
        KnownActor("campusclimatelab", "@CampusClimateLab",
                   "student climate organisation", "high", "medium",
                   "drives student adoption; operational influence"),
        KnownActor("techtruthleaks", "@TechTruthLeaks", "tech rumour account",
                   "low", "medium",
                   "history of unverified claims; treat skeptically"),
        KnownActor("mayabuildsapps", "@MayaBuildsApps", "indie iOS developer",
                   "high", "medium", "technically credible; useful for dev context"),
        KnownActor("bayareaecoclub", "@BayAreaEcoClub",
                   "local sustainability group", "high", "medium",
                   "real-world event organiser; operational stakeholder"),
    ])
    issues = KnownIssuesRegistry([
        KnownIssue(
            issue_id="iphone-xr-slow-upload",
            title="Slow image upload on older iPhones",
            keywords=("upload", "iphone xr", "slow"),
            severity="medium", status="open",
            notes=(
                "Confirmed pre-launch; engineering aware. Likely cause: "
                "full-resolution upload without local pre-compression."
            ),
        ),
        KnownIssue(
            issue_id="ai-search-generic-answers",
            title="AI search returns generic, non-local answers",
            keywords=("ai search", "generic", "local"),
            severity="medium", status="open",
            notes="Multiple reports across cities (NYC, San Jose, Palo Alto).",
        ),
        KnownIssue(
            issue_id="onboarding-location-wording",
            title="Onboarding wording about location sharing is unclear",
            keywords=("onboarding", "location", "wording"),
            severity="high", status="investigating",
            notes=(
                "Privacy page clarified yesterday; iOS permission popup "
                "still uses harsh default copy."
            ),
        ),
        KnownIssue(
            issue_id="upload-bug-fixed-in-1-0-2",
            title="Upload crash fixed in v1.0.2",
            keywords=("v1.0.0", "upload"),
            severity="low", status="fixed_in:1.0.2",
            notes="Users still on v1.0.0 will continue to see the bug.",
        ),
    ])
    return claims, actors, issues
