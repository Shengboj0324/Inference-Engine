"""Tests for Tier 2.3 — ThompsonDirectiveSelector."""

from __future__ import annotations

import numpy as np

from app.personalization.persona_bandit import (
    ARM_HIGH,
    ARM_LOW,
    ThompsonDirectiveSelector,
)


class TestThompsonDirectiveSelector:
    def test_regret_far_below_random_and_converges(self):
        p_high, p_low, T = 0.8, 0.3, 500
        sel = ThompsonDirectiveSelector(seed=1)
        rng = np.random.default_rng(1)
        reg_ts = 0.0
        for _ in range(T):
            arm = sel.select("step")
            p = p_high if arm == ARM_HIGH else p_low
            reward = 1.0 if rng.random() < p else 0.0
            sel.update("step", arm, reward)
            reg_ts += max(p_high, p_low) - p
        reg_random = 0.0
        for _ in range(T):
            arm = ARM_HIGH if rng.random() < 0.5 else ARM_LOW
            p = p_high if arm == ARM_HIGH else p_low
            reg_random += max(p_high, p_low) - p
        assert reg_ts < reg_random * 0.4
        assert sel.best_arm("step") == ARM_HIGH

    def test_update_moves_posterior(self):
        sel = ThompsonDirectiveSelector(seed=0)
        before = sel.posterior_means("t")
        for _ in range(20):
            sel.update("t", ARM_HIGH, 1.0)
        after = sel.posterior_means("t")
        assert after[ARM_HIGH] > before[ARM_HIGH]

    def test_deterministic_with_seed(self):
        a = ThompsonDirectiveSelector(seed=7)
        b = ThompsonDirectiveSelector(seed=7)
        assert [a.select("x") for _ in range(10)] == [b.select("x") for _ in range(10)]

    def test_round_trip(self):
        sel = ThompsonDirectiveSelector(seed=2)
        for _ in range(5):
            sel.update("t", ARM_HIGH, 1.0)
        s2 = ThompsonDirectiveSelector.from_dict(sel.to_dict())
        assert s2.posterior_means("t") == sel.posterior_means("t")
