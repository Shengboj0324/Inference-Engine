"""Developer Change Intelligence — Phase 2 structured release analysis.

Structures raw release notes, detects breaking changes, scores version
impact, and tracks watched dependency updates.

Public exports
--------------
Models: ChangeCategory, ImpactLevel, ChangeEntry, ReleaseNote,
        BreakingChange, DependencyAlert, RepoHealthScore
Components: ReleaseParser, ChangelogNormalizer, BreakingChangeDetector,
            VersionDiffAnalyzer, SemanticDiffSummarizer,
            RepoHealthScorer, DependencyAlertEngine
"""

from app.devintel.models import (
    BreakingChange,
    ChangeCategory,
    ChangeEntry,
    DependencyAlert,
    ImpactLevel,
    ReleaseNote,
    RepoHealthScore,
)
from app.devintel.release_parser import ReleaseParser
from app.devintel.changelog_normalizer import ChangelogNormalizer
from app.devintel.breaking_change_detector import BreakingChangeDetector
from app.devintel.version_diff_analyzer import VersionDiffAnalyzer
from app.devintel.semantic_diff_summarizer import SemanticDiffSummarizer
from app.devintel.repo_health import RepoHealthScorer
from app.devintel.dependency_alerts import DependencyAlertEngine

__all__ = [
    # Models
    "BreakingChange",
    "ChangeCategory",
    "ChangeEntry",
    "DependencyAlert",
    "ImpactLevel",
    "ReleaseNote",
    "RepoHealthScore",
    # Components
    "BreakingChangeDetector",
    "ChangelogNormalizer",
    "DependencyAlertEngine",
    "ReleaseParser",
    "RepoHealthScorer",
    "SemanticDiffSummarizer",
    "VersionDiffAnalyzer",
]

