"""Tier 2.3 — Thompson-sampling directive bandit.

When the persona is *uncertain* about a binary style choice (e.g. step-by-step
vs. conclusion-first), deterministically applying the point estimate can lock in
the wrong guess forever.  This module treats the choice as a **two-armed
Bernoulli bandit** per trait and uses **Thompson sampling** to balance
exploration and exploitation: it occasionally tries the other style, learns from
the next turn's implicit reward (did the user accept or push back), and converges
to the genuinely preferred directive — discovering preferences the user never
states explicitly.

Each arm's reward probability is tracked with a ``Beta(alpha, beta)`` posterior.
``select`` draws one sample per arm and picks the argmax; ``update`` folds the
observed reward into the chosen arm's posterior.  Pure standard library,
seedable for reproducibility, JSON-serialisable.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

#: Arm encoding for readability.
ARM_LOW = 0   # apply the low-pole directive
ARM_HIGH = 1  # apply the high-pole directive


class _TwoArmBeta:
    """Two Bernoulli arms, each with a Beta(alpha, beta) reward posterior."""

    def __init__(self) -> None:
        self.alpha = [1.0, 1.0]
        self.beta = [1.0, 1.0]

    def select(self, rng: random.Random) -> int:
        s0 = rng.betavariate(self.alpha[0], self.beta[0])
        s1 = rng.betavariate(self.alpha[1], self.beta[1])
        return ARM_HIGH if s1 >= s0 else ARM_LOW

    def update(self, arm: int, reward: float) -> None:
        reward = min(1.0, max(0.0, float(reward)))
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward

    def posterior_mean(self, arm: int) -> float:
        return self.alpha[arm] / (self.alpha[arm] + self.beta[arm])

    def to_dict(self) -> Dict[str, Any]:
        return {"alpha": list(self.alpha), "beta": list(self.beta)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_TwoArmBeta":
        a = cls()
        a.alpha = [float(x) for x in d.get("alpha", [1.0, 1.0])]
        a.beta = [float(x) for x in d.get("beta", [1.0, 1.0])]
        return a


class ThompsonDirectiveSelector:
    """Per-trait Thompson-sampling selector over {low-pole, high-pole} directives.

    Args:
        seed: Optional RNG seed for reproducible selection.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._bandits: Dict[str, _TwoArmBeta] = {}
        self._rng = random.Random(seed)

    def _bandit(self, trait: str) -> _TwoArmBeta:
        b = self._bandits.get(trait)
        if b is None:
            b = _TwoArmBeta()
            self._bandits[trait] = b
        return b

    def select(self, trait: str) -> int:
        """Return the chosen arm (``ARM_HIGH`` / ``ARM_LOW``) for ``trait``."""
        return self._bandit(trait).select(self._rng)

    def update(self, trait: str, arm: int, reward: float) -> None:
        """Fold an observed reward for ``(trait, arm)`` into its posterior."""
        self._bandit(trait).update(arm, reward)

    def best_arm(self, trait: str) -> int:
        """Return the current exploitation choice (higher posterior mean)."""
        b = self._bandit(trait)
        return ARM_HIGH if b.posterior_mean(ARM_HIGH) >= b.posterior_mean(ARM_LOW) else ARM_LOW

    def posterior_means(self, trait: str) -> Tuple[float, float]:
        b = self._bandit(trait)
        return (b.posterior_mean(ARM_LOW), b.posterior_mean(ARM_HIGH))

    def recommend(self, trait: str, min_pulls: float = 3.0, margin: float = 0.1) -> Optional[int]:
        """Exploit recommendation for ``trait``, or ``None`` if not yet confident.

        Returns the better arm only when enough reward evidence has accrued
        (``min_pulls`` beyond the uniform priors) *and* the two arms' posterior
        means are separated by at least ``margin`` — otherwise ``None`` so the
        caller declines to apply a directive it is unsure about.
        """
        b = self._bandits.get(trait)
        if b is None:
            return None
        pulls = (b.alpha[0] - 1.0) + (b.beta[0] - 1.0) + (b.alpha[1] - 1.0) + (b.beta[1] - 1.0)
        lo, hi = b.posterior_mean(ARM_LOW), b.posterior_mean(ARM_HIGH)
        if pulls < min_pulls or abs(hi - lo) < margin:
            return None
        return ARM_HIGH if hi >= lo else ARM_LOW

    def to_dict(self) -> Dict[str, Any]:
        return {"version": "1.0", "bandits": {t: b.to_dict() for t, b in self._bandits.items()}}

    @classmethod
    def from_dict(cls, d: Dict[str, Any], seed: Optional[int] = None) -> "ThompsonDirectiveSelector":
        s = cls(seed=seed)
        s._bandits = {t: _TwoArmBeta.from_dict(bd) for t, bd in d.get("bandits", {}).items()}
        return s
