"""Bayesian persona extensions (Tier 2.1 + 2.2).

- :class:`BetaBinomialTrait` (2.1) — for traits that are effectively binary
  (emoji on/off, step-by-step yes/no), model the preference as a recency-decayed
  **Beta-Binomial posterior**.  Confidence is derived from the posterior width
  (``1 - posterior_std / uniform_std``), so it reflects genuine statistical
  certainty and tightens monotonically with consistent evidence.

- :class:`PopulationPrior` (2.2) — an **empirical-Bayes** prior built by pooling
  trait estimates across many users.  New users seed their traits from this
  prior and shrink toward their own evidence as it accumulates, fixing the
  cold-start problem (a brand-new user is no longer assumed neutral at 0.5).

Pure standard library (math only); deterministic; JSON-serialisable.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

_BASE = 1.0                       # Beta(1,1) uniform base (immovable floor)
_UNIFORM_STD = 1.0 / math.sqrt(12.0)   # std of Beta(1,1) ≈ 0.288675


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


class BetaBinomialTrait:
    """Recency-decayed Beta-Binomial estimator for a binary-ish trait in [0, 1].

    ``alpha`` / ``beta`` accumulate decayed positive / negative evidence over a
    ``Beta(1, 1)`` base.  ``value`` is the posterior mean; ``confidence`` is
    ``1 - posterior_std / uniform_std`` (0 at the uniform prior, → 1 as the
    posterior tightens).
    """

    def __init__(self, alpha: float = _BASE, beta: float = _BASE,
                 last_updated: Optional[str] = None, update_count: int = 0) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.update_count = int(update_count)
        self.last_updated = last_updated
        self._recompute()

    @classmethod
    def from_prior(cls, mean: float, strength: float) -> "BetaBinomialTrait":
        mean = min(1.0, max(0.0, mean))
        strength = max(0.0, strength)
        return cls(alpha=_BASE + mean * strength, beta=_BASE + (1.0 - mean) * strength)

    def _recompute(self) -> None:
        a, b = self.alpha, self.beta
        s = a + b
        self.value = round(a / s, 6) if s > 0 else 0.5
        var = (a * b) / (s * s * (s + 1.0)) if s > 0 else _UNIFORM_STD ** 2
        std = math.sqrt(max(0.0, var))
        self.confidence = round(min(1.0, max(0.0, 1.0 - std / _UNIFORM_STD)), 6)

    def observe(self, x: float, strength: float, half_life_days: float,
                now: Optional[datetime] = None) -> None:
        x = min(1.0, max(0.0, float(x)))
        strength = max(1e-6, float(strength))
        now = now or _now()
        prev = _parse_iso(self.last_updated)
        decay = 0.5 ** (max(0.0, (now - prev).total_seconds() / 86400.0) / half_life_days) if prev else 1.0
        a_ev = (self.alpha - _BASE) * decay + strength * x
        b_ev = (self.beta - _BASE) * decay + strength * (1.0 - x)
        self.alpha = _BASE + a_ev
        self.beta = _BASE + b_ev
        self.update_count += 1
        self.last_updated = now.isoformat()
        self._recompute()

    def credible_interval(self, z: float = 1.96) -> Tuple[float, float]:
        """Normal-approximation credible interval for the posterior mean."""
        s = self.alpha + self.beta
        var = (self.alpha * self.beta) / (s * s * (s + 1.0)) if s > 0 else 0.0
        half = z * math.sqrt(max(0.0, var))
        return (max(0.0, self.value - half), min(1.0, self.value + half))

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "beta", "alpha": self.alpha, "beta": self.beta,
                "update_count": self.update_count, "last_updated": self.last_updated}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BetaBinomialTrait":
        return cls(alpha=float(d.get("alpha", _BASE)), beta=float(d.get("beta", _BASE)),
                   last_updated=d.get("last_updated"), update_count=int(d.get("update_count", 0)))


class PopulationPrior:
    """Empirical-Bayes population prior over persona traits.

    Pools confident per-user trait estimates into a per-trait population mean.
    New users seed each trait from this mean with a pseudo-count ``strength``,
    so estimates start informed and shrink toward the user's own evidence.

    Args:
        strength: Pseudo-count (k0) applied when seeding a new user's trait.
    """

    def __init__(self, strength: float = 2.0) -> None:
        self.strength = float(strength)
        self._sum: Dict[str, float] = {}
        self._count: Dict[str, float] = {}

    def observe_value(self, name: str, value: float, weight: float = 1.0) -> None:
        self._sum[name] = self._sum.get(name, 0.0) + value * weight
        self._count[name] = self._count.get(name, 0.0) + weight

    def build_from_personas(self, personas: List[Any], min_confidence: float = 0.0) -> "PopulationPrior":
        """Aggregate per-user trait estimates into an unbiased population mean.

        Each qualifying persona contributes its trait estimate with **uniform
        weight**.  Confidence-weighting is intentionally avoided here: the
        consistency-penalised confidence is systematically lower for genuinely
        mid-range users, so weighting by it would bias the pooled mean toward
        the extremes.  ``min_confidence`` (default 0) can still exclude users
        with essentially no evidence.
        """
        for p in personas:
            for name, value, conf in _iter_views(p):
                if conf >= min_confidence:
                    self.observe_value(name, value, weight=1.0)
        return self

    def mean_for(self, name: str) -> Optional[float]:
        c = self._count.get(name, 0.0)
        return (self._sum[name] / c) if c > 0 else None

    def prior_for(self, name: str) -> Optional[Tuple[float, float]]:
        """Return ``(mean, strength)`` for seeding, or ``None`` if unknown."""
        m = self.mean_for(name)
        return None if m is None else (m, self.strength)

    def to_dict(self) -> Dict[str, Any]:
        return {"version": "1.0", "strength": self.strength,
                "sum": self._sum, "count": self._count}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PopulationPrior":
        p = cls(strength=float(d.get("strength", 2.0)))
        p._sum = dict(d.get("sum", {}))
        p._count = dict(d.get("count", {}))
        return p


def _iter_views(persona: Any):
    """Yield (trait_name, value, confidence) from a persona's continuous + beta traits."""
    for name, t in getattr(persona, "traits", {}).items():
        yield name, t.value, t.confidence
    for name, t in getattr(persona, "beta_traits", {}).items():
        yield name, t.value, t.confidence
