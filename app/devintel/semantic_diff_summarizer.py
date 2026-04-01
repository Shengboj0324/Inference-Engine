"""Semantic diff summarizer.

Produces a human-readable LLM summary of changes between two software
versions, targeted at developers who need to understand:
- What changed
- Who is affected (user types / use-cases)
- Migration effort estimate (none / trivial / moderate / significant)
- Whether immediate action is needed

Falls back to a structured extractive summary when no LLM is available.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.devintel.models import ReleaseNote
from app.devintel.version_diff_analyzer import VersionDiff

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """\
You are a software engineering advisor. A developer needs to understand whether \
to upgrade from version {from_version} to {to_version} of {repo}.

Summarize the changes concisely. Use exactly these labeled sections:

WHAT CHANGED: (2-3 sentences covering headline changes)
WHO IS AFFECTED: (which user types / use-cases are impacted)
MIGRATION EFFORT: none | trivial | moderate | significant
ACTION REQUIRED: yes | no (should they upgrade urgently?)

RELEASE NOTES:
{release_notes}

BREAKING CHANGES:
{breaking_text}
"""


class SemanticDiffSummarizer:
    """Generates developer-focused upgrade summaries.

    Args:
        llm_router: LLM router; None → extractive fallback.
        max_tokens: Token budget for LLM summary.
        temperature: LLM temperature (lower = more factual).
    """

    def __init__(
        self,
        llm_router: Optional[Any] = None,
        max_tokens: int = 400,
        temperature: float = 0.2,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(f"'max_tokens' must be positive, got {max_tokens!r}")
        if not (0.0 <= temperature <= 2.0):
            raise ValueError(f"'temperature' must be in [0, 2], got {temperature!r}")
        self._router = llm_router
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def summarize(
        self,
        diff: VersionDiff,
        release_note: Optional[ReleaseNote] = None,
    ) -> str:
        """Generate a summary of changes between two versions.

        Args:
            diff:         ``VersionDiff`` from ``VersionDiffAnalyzer``.
            release_note: Structured release note for the new version.

        Returns:
            Multi-section summary string.

        Raises:
            TypeError: If *diff* is not a ``VersionDiff``.
        """
        if not isinstance(diff, VersionDiff):
            raise TypeError(f"'diff' must be VersionDiff, got {type(diff)!r}")

        if self._router is not None:
            try:
                return await self._llm_summarize(diff, release_note)
            except Exception as exc:
                logger.warning("SemanticDiffSummarizer: LLM failed (%s), using extractive", exc)

        return self._extractive_summary(diff, release_note)

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    async def _llm_summarize(self, diff: VersionDiff, note: Optional[ReleaseNote]) -> str:
        repo = (note.repo if note else "") or "this package"
        release_notes_text = self._format_note(note)
        breaking_text = "\n".join(
            f"- [{bc.impact_level.value}] {bc.description[:150]}"
            for bc in diff.breaking_changes[:5]
        ) or "None detected."
        prompt = _SUMMARY_PROMPT.format(
            from_version=diff.from_version,
            to_version=diff.to_version,
            repo=repo,
            release_notes=release_notes_text[:2000],
            breaking_text=breaking_text,
        )
        return await self._router.complete(prompt, max_tokens=self._max_tokens, temperature=self._temperature)

    # ------------------------------------------------------------------
    # Extractive fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _extractive_summary(diff: VersionDiff, note: Optional[ReleaseNote]) -> str:
        parts: list[str] = []
        parts.append(f"Version change: {diff.from_version} → {diff.to_version} ({diff.change_type.upper()})")
        parts.append(f"Upgrade urgency: {diff.upgrade_urgency_score:.0%}")

        if diff.breaking_changes:
            parts.append("\nBREAKING CHANGES:")
            for bc in diff.breaking_changes[:5]:
                parts.append(f"  [{bc.impact_level.value}] {bc.description[:120]}")
                if bc.migration_hint:
                    parts.append(f"    → Migration: {bc.migration_hint}")

        if note:
            if note.features:
                parts.append(f"\nNEW FEATURES ({len(note.features)}):")
                for f in note.features[:3]:
                    parts.append(f"  + {f.text[:100]}")
            if note.fixes:
                parts.append(f"\nFIXES ({len(note.fixes)}):")
                for f in note.fixes[:3]:
                    parts.append(f"  • {f.text[:100]}")
            if note.security:
                parts.append(f"\nSECURITY ({len(note.security)}):")
                for s in note.security[:3]:
                    parts.append(f"  ⚠ {s.text[:100]}")
            if note.migration_notes:
                parts.append(f"\nMIGRATION NOTES:\n{note.migration_notes[:500]}")

        return "\n".join(parts)

    @staticmethod
    def _format_note(note: Optional[ReleaseNote]) -> str:
        if not note:
            return "No release notes available."
        sections: list[str] = []
        if note.features:
            sections.append("Features:\n" + "\n".join(f"- {e.text[:120]}" for e in note.features[:5]))
        if note.fixes:
            sections.append("Fixes:\n" + "\n".join(f"- {e.text[:120]}" for e in note.fixes[:5]))
        if note.breaking:
            sections.append("Breaking:\n" + "\n".join(f"- {e.text[:120]}" for e in note.breaking[:5]))
        if note.security:
            sections.append("Security:\n" + "\n".join(f"- {e.text[:120]}" for e in note.security[:3]))
        return "\n\n".join(sections) or note.raw_body[:1000]

