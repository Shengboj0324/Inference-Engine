"""Deterministic evidence-strength scoring.

Drops the LLM from the loop on a question it was found to fail at in
the 66/100 baseline: rating "evidence strength" of a claim.  The
calculator multiplies three factors:

    score = corroboration_factor * credibility_factor * platform_factor

- ``corroboration_factor`` saturates around 6 sources via ``1 - exp(-k*n)``
- ``credibility_factor`` averages high/medium/low source weights
- ``platform_factor`` rewards diversity across platforms

The returned :class:`EvidenceScore` carries both the numeric strength
(0-1) and a human-readable label ("Low" / "Medium" / "High") for direct
rendering in the report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

from app.intelligence.crisis.registries import KnownActor, KnownActorRegistry
from app.intelligence.crisis.stream_parser import StreamItem


_CREDIBILITY_W = {"high": 1.0, "medium": 0.6, "low": 0.3}


@dataclass(frozen=True)
class EvidenceScore:
    label: str                       # "Low" | "Medium" | "High"
    value: float                     # 0..1
    corroborators: int
    platforms: Tuple[str, ...]
    rationale: str

    @property
    def display(self) -> str:
        return f"{self.label} ({self.value:.2f})"


class EvidenceStrengthCalculator:
    """Score evidence for a *claim* using the supporting StreamItems."""

    def __init__(self, actors: KnownActorRegistry,
                 *, corroboration_k: float = 0.45) -> None:
        self._actors = actors
        self._k = corroboration_k

    def score(self, claim: str, supporting: List[StreamItem],
              *, weight_by_repeat: bool = True,
              repeat_log_cap: float = 60.0) -> EvidenceScore:
        n = len(supporting)
        if n == 0:
            return EvidenceScore("Low", 0.05, 0, (),
                                 "No corroborating source in the stream.")
        # effective corroboration count: log-dampened repeat_count keeps
        # one bulk amplification row from completely saturating the
        # corroboration factor while still rewarding 300+ posts well
        # above a single isolated mention.
        if weight_by_repeat:
            eff_n = 0.0
            for it in supporting:
                r = max(1, it.repeat_count)
                eff_n += 1.0 + math.log1p(min(r - 1, repeat_log_cap))
        else:
            eff_n = float(n)
        corro = 1.0 - math.exp(-self._k * eff_n)

        # credibility factor (average source weight)
        weights: List[float] = []
        for it in supporting:
            actor = self._actors.get(it.handle) if it.handle else None
            if actor is not None:
                weights.append(_CREDIBILITY_W.get(actor.credibility_tier, 0.5))
            else:
                # platform default
                weights.append({
                    "blog": 0.7, "appstore": 0.7, "reddit": 0.55,
                    "discord": 0.55, "instagram": 0.5, "tiktok": 0.45,
                    "x": 0.45, "screenshot": 0.6, "system": 0.4,
                }.get(it.platform, 0.5))
        cred = sum(weights) / len(weights)

        # platform diversity factor (0.5..1)
        platforms = tuple(sorted({it.platform for it in supporting}))
        diversity = min(1.0, 0.5 + 0.1 * len(platforms))

        value = max(0.0, min(1.0, corro * cred * diversity))
        label = self._label(value)
        total_posts = sum(max(1, it.repeat_count) for it in supporting)
        rationale = (
            f"{n} corroborator row(s) (~{total_posts} effective posts) "
            f"across {len(platforms)} platform(s) "
            f"(corro={corro:.2f}, cred={cred:.2f}, diversity={diversity:.2f})"
        )
        return EvidenceScore(label, round(value, 3), n, platforms, rationale)

    @staticmethod
    def _label(value: float) -> str:
        if value >= 0.55:
            return "High"
        if value >= 0.30:
            return "Medium"
        return "Low"

    @staticmethod
    def from_value(value: float, *, rationale: str,
                   corroborators: int = 0,
                   platforms: Tuple[str, ...] = ()) -> "EvidenceScore":
        """Construct an :class:`EvidenceScore` from an externally-computed
        value. Used when detection confidence rather than source-credibility
        determines the strength (e.g. coordinated-amplification clusters).
        """
        v = max(0.0, min(1.0, value))
        return EvidenceScore(
            EvidenceStrengthCalculator._label(v), round(v, 3),
            corroborators, platforms, rationale,
        )
