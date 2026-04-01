"""Shared Pydantic models for the Developer Change Intelligence package."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ChangeCategory(str, Enum):
    """Classification of a single changelog entry."""

    FEATURE = "feature"
    FIX = "fix"
    BREAKING = "breaking"
    DEPRECATION = "deprecation"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    REFACTOR = "refactor"
    DEPENDENCY = "dependency"
    MIGRATION = "migration"
    OTHER = "other"


class ImpactLevel(str, Enum):
    """Impact level of a breaking or notable change."""

    CRITICAL = "critical"    # Must upgrade immediately / breaks production
    HIGH = "high"            # Breaking change; requires code changes
    MEDIUM = "medium"        # Deprecation warning; plan migration
    LOW = "low"              # Minor behavioral shift; informational


class ChangeEntry(BaseModel, frozen=True):
    """A single normalized change log entry.

    Attributes:
        text:       Raw change description.
        category:   ``ChangeCategory`` classification.
        is_breaking: True if this entry is a breaking change.
        pr_number:  Pull request number if extractable.
        commit_sha: Commit SHA if present.
        author:     Author GitHub handle or name.
    """

    text: str = Field(..., min_length=1)
    category: ChangeCategory = ChangeCategory.OTHER
    is_breaking: bool = False
    pr_number: Optional[int] = None
    commit_sha: str = ""
    author: str = ""


class ReleaseNote(BaseModel):
    """Structured representation of a GitHub/changelog release.

    Attributes:
        version:      SemVer string (e.g. ``"2.1.0"``).
        repo:         ``owner/repo`` GitHub slug.
        published_at: UTC release date.
        url:          HTML URL for the release.
        title:        Release title (may differ from version).
        summary:      LLM-generated or extractive summary.
        features:     New feature entries.
        fixes:        Bug fix entries.
        breaking:     Breaking change entries.
        deprecations: Deprecation entries.
        security:     Security fix entries.
        migration_notes: Free-form migration guidance.
        raw_body:     Original release body Markdown.
    """

    version: str
    repo: str = ""
    published_at: Optional[datetime] = None
    url: str = ""
    title: str = ""
    summary: str = ""
    features: List[ChangeEntry] = Field(default_factory=list)
    fixes: List[ChangeEntry] = Field(default_factory=list)
    breaking: List[ChangeEntry] = Field(default_factory=list)
    deprecations: List[ChangeEntry] = Field(default_factory=list)
    security: List[ChangeEntry] = Field(default_factory=list)
    migration_notes: str = ""
    raw_body: str = ""

    @field_validator("version")
    @classmethod
    def _non_empty_version(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("'version' must be non-empty")
        return v.strip()

    def all_entries(self) -> List[ChangeEntry]:
        return self.features + self.fixes + self.breaking + self.deprecations + self.security

    @property
    def has_breaking_changes(self) -> bool:
        return bool(self.breaking)


class BreakingChange(BaseModel, frozen=True):
    """A single detected breaking change with impact assessment.

    Attributes:
        description:   Change description text.
        impact_level:  ``ImpactLevel`` classification.
        affected_apis: List of API/function names affected.
        migration_hint: Suggested migration action.
        source_entry:  The raw ``ChangeEntry`` that triggered this.
        confidence:    Detection confidence [0, 1].
    """

    description: str
    impact_level: ImpactLevel = ImpactLevel.HIGH
    affected_apis: List[str] = Field(default_factory=list)
    migration_hint: str = ""
    source_entry: Optional[ChangeEntry] = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class DependencyAlert(BaseModel, frozen=True):
    """Alert emitted when a tracked dependency releases a new version.

    Attributes:
        package_name:  Package name (e.g. ``"openai"``).
        old_version:   Previous pinned version.
        new_version:   Newly released version.
        repo:          GitHub repo of the package.
        impact_level:  Derived from version diff type.
        breaking_changes: Detected breaking changes in the new release.
        upgrade_urgency_score: Float [0, 1] combining impact + security signals.
    """

    package_name: str
    old_version: str = ""
    new_version: str
    repo: str = ""
    impact_level: ImpactLevel = ImpactLevel.MEDIUM
    breaking_changes: List[BreakingChange] = Field(default_factory=list)
    upgrade_urgency_score: float = Field(default=0.5, ge=0.0, le=1.0)


class RepoHealthScore(BaseModel):
    """Repository health assessment.

    Attributes:
        repo:              ``owner/repo`` slug.
        overall_score:     Composite health score [0, 1].
        stars:             Current star count.
        forks:             Current fork count.
        open_issues:       Open issue count.
        open_prs:          Open PR count.
        days_since_last_commit: Recency of last commit.
        contributor_count: Number of distinct contributors.
        has_ci:            Whether CI is configured.
        has_tests:         Whether tests exist.
        license:           License SPDX identifier.
        score_breakdown:   Per-dimension scores.
    """

    repo: str
    overall_score: float = Field(..., ge=0.0, le=1.0)
    stars: int = 0
    forks: int = 0
    open_issues: int = 0
    open_prs: int = 0
    days_since_last_commit: Optional[int] = None
    contributor_count: int = 0
    has_ci: bool = False
    has_tests: bool = False
    license: str = ""
    score_breakdown: Dict[str, float] = Field(default_factory=dict)

