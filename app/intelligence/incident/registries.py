"""NeuroBridge-seeded registries for the incident-response pipeline.

Reuses the generic registry dataclasses from
:mod:`app.intelligence.crisis.registries` (``DebunkedClaim``,
``KnownActor``, ``KnownIssue``) so a single mental model covers both
crisis-comms and incident-response scenarios.
"""
from __future__ import annotations

from typing import Tuple

from app.intelligence.crisis.registries import (
    DebunkedClaim,
    DebunkedClaimsRegistry,
    KnownActor,
    KnownActorRegistry,
    KnownIssue,
    KnownIssuesRegistry,
)


def build_default_registries() -> Tuple[
    DebunkedClaimsRegistry, KnownActorRegistry, KnownIssuesRegistry
]:
    """Seed the registries the NeuroBridge operator provides as memory."""
    claims = DebunkedClaimsRegistry([
        DebunkedClaim(
            claim_id="neurobridge-stores-ssn",
            canonical_text=(
                "NeuroBridge stores full Social Security numbers for "
                "every student."
            ),
            keywords=("neurobridge", "ssn", "social security"),
            debunked_at="yesterday",
            source_note=(
                "Investigated and publicly debunked yesterday. "
                "NeuroBridge does not collect or store full Social "
                "Security numbers."
            ),
            correction=(
                "NeuroBridge does not collect or store full Social "
                "Security numbers. This claim was debunked publicly "
                "the day before the incident."
            ),
        ),
    ])
    actors = KnownActorRegistry([
        KnownActor("k12privacywatch", "@K12PrivacyWatch",
                   "education privacy nonprofit", "high", "high",
                   "careful, evidence-based, widely cited by school districts"),
        KnownActor("mayasecops", "@MayaSecOps",
                   "security engineer", "high", "high",
                   "analyses breach rumours carefully; high technical credibility"),
        KnownActor("studentdatalawyer", "@StudentDataLawyer",
                   "student-privacy attorney", "high", "high",
                   "influential among parents and school administrators"),
        KnownActor("northviewpta", "@NorthviewPTA",
                   "parent-teacher association",
                   "high", "medium",
                   "real Northview High parent channel; directly affected stakeholder"),
        KnownActor("summitprepacademy", "@SummitPrepAcademy",
                   "private college-prep chain", "high", "medium",
                   "uses NeuroBridge across multiple locations"),
        KnownActor("eduadminforum", "@EduAdminForum",
                   "school IT admin community", "high", "medium",
                   "school admin audience; useful for roster-template context"),
        KnownActor("techclassroomdaily", "@TechClassroomDaily",
                   "education-tech blog", "medium", "medium",
                   "fast but sometimes publishes before full verification"),
        KnownActor("appstoreleaks", "@AppStoreLeaks",
                   "screenshot aggregator", "medium", "medium",
                   "sometimes accurate, sometimes missing context"),
        KnownActor("breachalertnow", "@BreachAlertNow",
                   "breach-rumour amplifier", "low", "medium",
                   "frequently posts alleged breaches before verification"),
        KnownActor("exposeedtech", "@ExposeEdTech",
                   "anti-edtech account", "low", "medium",
                   "ideological; broad claims with weak sourcing"),
        KnownActor("darkwebindex", "@DarkWebIndex",
                   "alleged leak-market monitor", "low", "medium",
                   "has previously amplified false breach claims"),
    ])
    issues = KnownIssuesRegistry([
        KnownIssue(
            issue_id="duplicate-welcome-emails",
            title="Duplicate welcome emails",
            keywords=("duplicate", "welcome email"),
            severity="low", status="open",
            notes=(
                "Known operational bug; unrelated to security. Some users "
                "received duplicate welcome emails."
            ),
        ),
        KnownIssue(
            issue_id="delayed-password-reset",
            title="Delayed password-reset emails",
            keywords=("password reset", "password-reset", "reset email"),
            severity="low", status="open",
            notes=(
                "Email-queue latency; known issue. Do not conflate with "
                "account-takeover signal unless other indicators present."
            ),
        ),
        KnownIssue(
            issue_id="stale-android-notifications",
            title="Stale notification badges on Android",
            keywords=("stale notification", "notification badge"),
            severity="low", status="open",
            notes="Cosmetic Android bug; unrelated to data security.",
        ),
        KnownIssue(
            issue_id="duplicate-roster-upload",
            title="Duplicate CSV roster upload by school admin",
            keywords=("duplicate", "roster", "csv"),
            severity="low", status="open",
            notes=(
                "School administrator accidentally uploaded a duplicate "
                "CSV roster last week."
            ),
        ),
        KnownIssue(
            issue_id="parent-account-mislink",
            title="Parent accounts linked to inactive student accounts",
            keywords=("parent account", "inactive student"),
            severity="low", status="open",
            notes="Operational link bug; not a security issue.",
        ),
    ])
    return claims, actors, issues
