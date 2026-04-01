"""Breaking change detector.

Scans ``ReleaseNote`` and ``ChangeEntry`` objects for breaking changes
using:
1. ``is_breaking`` flag set by ``ReleaseParser`` (highest confidence)
2. Keyword/regex heuristic scan of change text
3. Optional LLM confirmation pass for borderline cases

Each detected ``BreakingChange`` receives an ``ImpactLevel`` classification
driven by vocabulary signals and affected API surface area.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

from app.devintel.models import BreakingChange, ChangeEntry, ImpactLevel, ReleaseNote

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic patterns
# ---------------------------------------------------------------------------
_BREAKING_KEYWORDS = re.compile(
    r"\b(remov(ed?|ing)|delet(ed?|ing)|drop(ped)?|deprecat(ed?|ing)|"
    r"replac(ed?|ing)|renamed?|breaking|incompatible|no\s+longer\s+support|"
    r"must\s+(now|update|migrate)|required?\s+to\s+update|breaking\s+change|"
    r"backward[s]?\s+incompatible|api\s+change|signature\s+change|"
    r"return\s+type\s+changed|raise[sd]?\s+(?:now\s+)?(?:a\s+)?(\w+Error))\b",
    re.IGNORECASE,
)
_CRITICAL = re.compile(r"\b(security|auth|token|credential|data\s+loss|corrupt)\b", re.I)
_API_NAME = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]+(?:\(\))?)`")
_MIGRATION_HINT = re.compile(r"(?:use\s+|replace\s+with\s+|migrate\s+to\s+)(`[^`]+`|\w+)", re.I)

_LLM_PROMPT = """\
You are analyzing software release notes for breaking changes.

For each change below, determine:
1. Is it a breaking change? (yes/no)
2. Impact level: critical | high | medium | low
3. Affected APIs (list of function/class names, empty if none)
4. Migration hint (one sentence)

Return a JSON array: [{{"index": int, "is_breaking": bool, "impact_level": str, \
"affected_apis": [str], "migration_hint": str}}]

CHANGES:
{changes_text}
"""

_BORDERLINE = re.compile(
    r"\b(changed|modified|updated|adjusted|refactored|reworked)\b",
    re.IGNORECASE,
)


class BreakingChangeDetector:
    """Detects breaking changes in release notes with optional LLM confirmation.

    Args:
        llm_router:       Optional LLM for borderline cases.
        min_confidence:   Minimum confidence to emit a ``BreakingChange``.
        llm_on_borderline: Send heuristic-borderline entries to LLM (default True).
    """

    def __init__(
        self,
        llm_router: Optional[Any] = None,
        min_confidence: float = 0.5,
        llm_on_borderline: bool = True,
    ) -> None:
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(f"'min_confidence' must be in [0, 1], got {min_confidence!r}")
        self._router = llm_router
        self._min_conf = min_confidence
        self._llm_on_borderline = llm_on_borderline and (llm_router is not None)

    async def detect(self, note: ReleaseNote) -> List[BreakingChange]:
        """Detect breaking changes in a ``ReleaseNote``.

        Args:
            note: Structured release note.

        Returns:
            List of ``BreakingChange`` sorted by impact level (critical first).

        Raises:
            TypeError: If *note* is not a ``ReleaseNote``.
        """
        if not isinstance(note, ReleaseNote):
            raise TypeError(f"Expected ReleaseNote, got {type(note)!r}")

        confirmed: List[BreakingChange] = []
        borderline: List[tuple[int, ChangeEntry]] = []

        all_entries = note.all_entries()

        for i, entry in enumerate(all_entries):
            result, confidence, borderline_flag = self._heuristic_classify(entry)
            if result and confidence >= self._min_conf:
                confirmed.append(result)
            elif borderline_flag:
                borderline.append((i, entry))

        if self._llm_on_borderline and borderline:
            llm_results = await self._llm_confirm(borderline, all_entries)
            confirmed.extend(llm_results)

        # Sort: CRITICAL > HIGH > MEDIUM > LOW
        order = {ImpactLevel.CRITICAL: 0, ImpactLevel.HIGH: 1, ImpactLevel.MEDIUM: 2, ImpactLevel.LOW: 3}
        confirmed.sort(key=lambda bc: order.get(bc.impact_level, 99))
        logger.debug("BreakingChangeDetector: %d breaking changes detected in %s", len(confirmed), note.version)
        return confirmed

    def detect_from_entries(self, entries: List[ChangeEntry]) -> List[BreakingChange]:
        """Synchronous heuristic-only detection for a list of ``ChangeEntry``.

        Useful when LLM is not needed.
        """
        if not isinstance(entries, list):
            raise TypeError(f"Expected list, got {type(entries)!r}")
        results: List[BreakingChange] = []
        for entry in entries:
            bc, confidence, _ = self._heuristic_classify(entry)
            if bc and confidence >= self._min_conf:
                results.append(bc)
        return results

    # ------------------------------------------------------------------
    # Heuristic classification
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic_classify(entry: ChangeEntry) -> tuple[Optional[BreakingChange], float, bool]:
        """Return (BreakingChange|None, confidence, is_borderline)."""
        text = entry.text

        # Already flagged by parser
        if entry.is_breaking:
            impact = ImpactLevel.CRITICAL if _CRITICAL.search(text) else ImpactLevel.HIGH
            apis = _API_NAME.findall(text)
            hint_m = _MIGRATION_HINT.search(text)
            return (
                BreakingChange(
                    description=text,
                    impact_level=impact,
                    affected_apis=apis,
                    migration_hint=hint_m.group(0) if hint_m else "",
                    source_entry=entry,
                    confidence=0.95,
                ),
                0.95,
                False,
            )

        # Keyword heuristic
        if _BREAKING_KEYWORDS.search(text):
            impact = ImpactLevel.CRITICAL if _CRITICAL.search(text) else ImpactLevel.HIGH
            apis = _API_NAME.findall(text)
            hint_m = _MIGRATION_HINT.search(text)
            return (
                BreakingChange(
                    description=text,
                    impact_level=impact,
                    affected_apis=apis,
                    migration_hint=hint_m.group(0) if hint_m else "",
                    source_entry=entry,
                    confidence=0.75,
                ),
                0.75,
                False,
            )

        # Borderline (changed/modified language)
        if _BORDERLINE.search(text):
            return None, 0.0, True

        return None, 0.0, False

    # ------------------------------------------------------------------
    # LLM confirmation
    # ------------------------------------------------------------------

    async def _llm_confirm(
        self,
        borderline: List[tuple[int, ChangeEntry]],
        all_entries: List[ChangeEntry],
    ) -> List[BreakingChange]:
        changes_text = "\n".join(
            f"[{i}] {entry.text[:200]}"
            for i, entry in borderline
        )
        prompt = _LLM_PROMPT.format(changes_text=changes_text)
        raw = await self._router.complete(prompt, max_tokens=1000, temperature=0.1)
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip().rstrip("`")
        items: list[dict] = json.loads(raw)
        results: List[BreakingChange] = []
        for item in items:
            if not item.get("is_breaking", False):
                continue
            orig_idx = item["index"]
            if 0 <= orig_idx < len(borderline):
                _, entry = borderline[orig_idx]
            else:
                continue
            impact = ImpactLevel(item.get("impact_level", "medium"))
            results.append(BreakingChange(
                description=entry.text,
                impact_level=impact,
                affected_apis=list(item.get("affected_apis", [])),
                migration_hint=str(item.get("migration_hint", "")),
                source_entry=entry,
                confidence=0.80,
            ))
        return results

