"""Repository health scorer.

Computes a ``RepoHealthScore`` from GitHub repository metadata.  All
inputs are plain metadata dicts (as returned by the GitHub REST API or
``GitHubReleasesConnector`` / ``GitHubRepoEventsConnector``), so no
additional HTTP calls are needed.

Score dimensions (weights):
  - Recency            (0.30): Days since last commit
  - Community          (0.25): Stars + forks
  - Issue health       (0.20): Open issues / closed ratio proxy
  - CI / testing       (0.15): Presence of CI config and tests
  - License            (0.10): Open-source license presence
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.devintel.models import RepoHealthScore

logger = logging.getLogger(__name__)

_STAR_SCALE = 10_000     # stars for max community score
_OPEN_ISSUE_PENALTY = 500  # open issues for max penalty


class RepoHealthScorer:
    """Computes ``RepoHealthScore`` from GitHub repository metadata.

    Args:
        star_scale:         Star count for full community score.
        open_issue_penalty: Open issues at which issue score bottoms out.
    """

    def __init__(
        self,
        star_scale: int = _STAR_SCALE,
        open_issue_penalty: int = _OPEN_ISSUE_PENALTY,
    ) -> None:
        if star_scale <= 0:
            raise ValueError(f"'star_scale' must be positive, got {star_scale!r}")
        if open_issue_penalty <= 0:
            raise ValueError(f"'open_issue_penalty' must be positive, got {open_issue_penalty!r}")
        self._star_scale = star_scale
        self._open_issue_penalty = open_issue_penalty

    def score(self, metadata: Dict[str, Any]) -> RepoHealthScore:
        """Compute health score from a GitHub API repository metadata dict.

        Expected keys (all optional; missing → defaults):
          ``full_name``, ``stargazers_count``, ``forks_count``,
          ``open_issues_count``, ``pushed_at``, ``license``,
          ``has_ci``, ``has_tests``, ``subscribers_count``,
          ``open_prs``, ``contributor_count``.

        Args:
            metadata: GitHub API metadata dict.

        Returns:
            ``RepoHealthScore``.

        Raises:
            TypeError: If *metadata* is not a dict.
        """
        if not isinstance(metadata, dict):
            raise TypeError(f"'metadata' must be a dict, got {type(metadata)!r}")

        repo = str(metadata.get("full_name", "unknown/unknown"))
        stars = int(metadata.get("stargazers_count", 0))
        forks = int(metadata.get("forks_count", 0))
        open_issues = int(metadata.get("open_issues_count", 0))
        pushed_at_str = metadata.get("pushed_at", "")
        license_info = metadata.get("license") or {}
        has_ci = bool(metadata.get("has_ci", False))
        has_tests = bool(metadata.get("has_tests", False))
        contributor_count = int(metadata.get("contributor_count", 0))
        open_prs = int(metadata.get("open_prs", 0))
        license_id = license_info.get("spdx_id", "") if isinstance(license_info, dict) else ""

        # Compute days since last commit
        days_since = None
        if pushed_at_str:
            try:
                pushed_dt = datetime.fromisoformat(pushed_at_str.replace("Z", "+00:00"))
                days_since = (datetime.now(timezone.utc) - pushed_dt).days
            except ValueError:
                pass

        breakdown: Dict[str, float] = {}

        # 1. Recency (weight 0.30)
        if days_since is not None:
            # Exponential decay: 1.0 at 0 days, ~0.5 at 90 days, ~0.0 at 365+
            recency = math.exp(-days_since / 120.0)
        else:
            recency = 0.5
        breakdown["recency"] = round(recency, 3)

        # 2. Community (weight 0.25)
        community = min(math.log1p(stars + forks) / math.log1p(self._star_scale + 0), 1.0)
        breakdown["community"] = round(community, 3)

        # 3. Issue health (weight 0.20)
        issue_penalty = min(open_issues / self._open_issue_penalty, 1.0)
        issue_health = 1.0 - issue_penalty * 0.7  # never fully zero
        breakdown["issue_health"] = round(issue_health, 3)

        # 4. CI / testing (weight 0.15)
        ci_score = (0.6 if has_ci else 0.0) + (0.4 if has_tests else 0.0)
        breakdown["ci_testing"] = round(ci_score, 3)

        # 5. License (weight 0.10)
        license_score = 1.0 if license_id and license_id != "NOASSERTION" else 0.0
        breakdown["license"] = round(license_score, 3)

        overall = (
            0.30 * recency
            + 0.25 * community
            + 0.20 * issue_health
            + 0.15 * ci_score
            + 0.10 * license_score
        )
        overall = round(min(max(overall, 0.0), 1.0), 3)

        logger.debug(
            "RepoHealthScorer: %s score=%.3f (recency=%.2f community=%.2f issues=%.2f)",
            repo, overall, recency, community, issue_health,
        )
        return RepoHealthScore(
            repo=repo,
            overall_score=overall,
            stars=stars,
            forks=forks,
            open_issues=open_issues,
            open_prs=open_prs,
            days_since_last_commit=days_since,
            contributor_count=contributor_count,
            has_ci=has_ci,
            has_tests=has_tests,
            license=license_id,
            score_breakdown=breakdown,
        )

