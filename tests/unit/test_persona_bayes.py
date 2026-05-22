"""Tests for Tier 2.1 (Beta-Binomial trait) and Tier 2.2 (population prior)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from app.personalization.persona_bayes import BetaBinomialTrait, PopulationPrior
from app.personalization.user_persona import UserPersonaProfile

NOW = datetime(2026, 5, 22, tzinfo=timezone.utc)


def _at(days):
    return NOW + timedelta(days=days)


class TestBetaBinomialTrait:
    def test_consistent_evidence_converges_and_tightens(self):
        t = BetaBinomialTrait()
        confs = []
        for i in range(10):
            t.observe(1.0, 1.0, 45.0, _at(i * 0.01))
            confs.append(t.confidence)
        assert t.value > 0.85
        assert all(confs[i + 1] >= confs[i] - 1e-9 for i in range(len(confs) - 1))

    def test_conflicting_evidence_is_less_confident(self):
        consistent = BetaBinomialTrait()
        conflict = BetaBinomialTrait()
        for i in range(10):
            consistent.observe(1.0, 1.0, 45.0, _at(i * 0.01))
            conflict.observe(1.0 if i % 2 == 0 else 0.0, 1.0, 45.0, _at(i * 0.01))
        assert conflict.confidence < consistent.confidence

    def test_credible_interval_shrinks(self):
        t = BetaBinomialTrait()
        t.observe(1.0, 1.0, 45.0, _at(0))
        lo1, hi1 = t.credible_interval()
        for i in range(20):
            t.observe(1.0, 1.0, 45.0, _at(i * 0.01))
        lo2, hi2 = t.credible_interval()
        assert (hi2 - lo2) < (hi1 - lo1)

    def test_recency_relaxes_posterior(self):
        t = BetaBinomialTrait()
        for i in range(15):
            t.observe(1.0, 1.0, 10.0, _at(i * 0.01))
        mass_before = t.alpha + t.beta
        t.observe(1.0, 1.0, 10.0, _at(80))  # ~8 half-lives later
        # the pre-gap evidence has decayed, so total mass is far below the
        # no-decay accumulation of ~17.
        assert t.alpha + t.beta < mass_before + 2.0

    def test_from_prior_seeds_mean(self):
        t = BetaBinomialTrait.from_prior(0.2, strength=10.0)
        assert abs(t.value - 0.2) < 0.05

    def test_round_trip(self):
        t = BetaBinomialTrait()
        for i in range(5):
            t.observe(1.0, 1.0, 45.0, _at(i))
        t2 = BetaBinomialTrait.from_dict(t.to_dict())
        assert abs(t2.value - t.value) < 1e-9 and abs(t2.confidence - t.confidence) < 1e-9


class TestPopulationPrior:
    def test_recovers_population_mean_uniformly(self):
        rng = np.random.default_rng(0)
        personas = []
        for u in range(200):
            tu = float(rng.beta(2, 5))
            p = UserPersonaProfile(f"u{u}")
            for _ in range(25):
                p.observe_trait("verbosity", 1.0 if rng.random() < tu else 0.0)
            personas.append(p)
        prior = PopulationPrior(strength=3.0).build_from_personas(personas)
        # E[Beta(2,5)] = 2/7 ≈ 0.286
        assert abs(prior.mean_for("verbosity") - 0.286) < 0.06
        assert prior.prior_for("verbosity")[1] == 3.0
        assert prior.prior_for("unknown_trait") is None

    def test_prior_seeded_cold_start_beats_flat(self):
        rng = np.random.default_rng(1)
        prior = PopulationPrior(strength=3.0)
        # population mean ~0.286
        for u in range(150):
            tu = float(rng.beta(2, 5))
            p = UserPersonaProfile(f"b{u}")
            for _ in range(25):
                p.observe_trait("verbosity", 1.0 if rng.random() < tu else 0.0)
            prior.build_from_personas([p])
        mse_no = mse_pr = 0.0
        M = 300
        for _ in range(M):
            tu = float(rng.beta(2, 5))
            pn = UserPersonaProfile("n")
            pp = UserPersonaProfile("p", prior=prior)
            for _ in range(2):
                x = 1.0 if rng.random() < tu else 0.0
                pn.observe_trait("verbosity", x)
                pp.observe_trait("verbosity", x)
            mse_no += (pn.trait("verbosity").value - tu) ** 2
            mse_pr += (pp.trait("verbosity").value - tu) ** 2
        assert mse_pr < mse_no

    def test_round_trip(self):
        prior = PopulationPrior(strength=2.0)
        prior.observe_value("verbosity", 0.3)
        prior.observe_value("verbosity", 0.5)
        p2 = PopulationPrior.from_dict(prior.to_dict())
        assert p2.mean_for("verbosity") == prior.mean_for("verbosity")


class TestPersonaIntegration:
    def test_binary_trait_routes_to_beta(self):
        p = UserPersonaProfile("u", binary_traits={"emoji_affinity"})
        for _ in range(6):
            p.observe_trait("emoji_affinity", 0.0)
        assert "emoji_affinity" in p.beta_traits
        assert p.trait("emoji_affinity").value < 0.3
        # round-trips through persona serialization
        p2 = UserPersonaProfile.from_dict(p.to_dict())
        assert p2.trait("emoji_affinity").value == p.trait("emoji_affinity").value

    def test_default_behaviour_unchanged(self):
        p = UserPersonaProfile("u")  # no binary, no prior
        for _ in range(5):
            p.observe_trait("verbosity", 0.0)
        assert not p.beta_traits
        assert "concise" in p.render_style_directive(min_confidence=0.3)
