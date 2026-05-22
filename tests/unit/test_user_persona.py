"""Tests for the dedicated per-user persona memory.

Covers the confidence-weighted, recency-decayed estimator
(:class:`UserPersonaProfile`) and its integration into
:class:`ContextMemoryStore` (durable, GDPR-complete per-user memory).

Pure-stdlib persona math + pydantic domain models; no torch / network.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from app.personalization.user_persona import Trait, UserPersonaProfile

NOW = datetime(2026, 5, 22, tzinfo=timezone.utc)


def _at(days: float) -> datetime:
    return NOW + timedelta(days=days)


class TestPersonaEstimator:
    def test_consistent_evidence_converges_with_high_confidence(self):
        p = UserPersonaProfile("u")
        for i in range(8):
            p.observe_trait("verbosity", 0.0, strength=1.0, now=_at(i * 0.01))
        t = p.trait("verbosity")
        assert t.value < 0.15
        assert t.confidence > 0.6

    def test_conflicting_evidence_lowers_confidence(self):
        consistent = UserPersonaProfile("c")
        conflict = UserPersonaProfile("k")
        for i in range(8):
            consistent.observe_trait("formality", 0.0, now=_at(i * 0.01))
            conflict.observe_trait("formality", 0.0 if i % 2 == 0 else 1.0, now=_at(i * 0.01))
        assert conflict.trait("formality").confidence < consistent.trait("formality").confidence

    def test_established_trait_is_more_stable_than_fresh(self):
        fresh = UserPersonaProfile("f")
        fresh.observe_trait("warmth", 1.0, now=_at(0))
        d_fresh = abs(fresh.trait("warmth").value - 0.5)

        est = UserPersonaProfile("e")
        for i in range(20):
            est.observe_trait("warmth", 1.0, now=_at(i * 0.01))
        before = est.trait("warmth").value
        est.observe_trait("warmth", 0.0, now=_at(0.3))
        d_est = abs(est.trait("warmth").value - before)
        assert d_est < d_fresh

    def test_long_gap_restores_plasticity(self):
        g = UserPersonaProfile("g", half_life_days=10.0)
        for i in range(15):
            g.observe_trait("directness", 1.0, now=_at(i * 0.01))
        n_before = g.trait("directness").n_eff
        g.observe_trait("directness", 1.0, now=_at(60))
        assert g.trait("directness").n_eff < n_before

    def test_value_and_confidence_bounded(self):
        p = UserPersonaProfile("b")
        for i in range(30):
            p.observe_trait("technical_depth", 1.0 if i % 3 else 0.0, now=_at(i * 0.01))
        t = p.trait("technical_depth")
        assert 0.0 <= t.value <= 1.0
        assert 0.0 <= t.confidence <= 1.0


class TestStyleDirective:
    def test_cold_start_directive_is_empty(self):
        assert UserPersonaProfile("cold").render_style_directive() == ""

    def test_directive_includes_confident_non_neutral_only(self):
        p = UserPersonaProfile("u")
        for i in range(6):
            p.observe_trait("verbosity", 0.0, now=_at(i * 0.01))   # confident, low
        p.observe_trait("warmth", 0.52, now=_at(0))                # ~neutral -> excluded
        text = p.render_style_directive(min_confidence=0.35)
        assert "concise" in text
        assert "warm" not in text.lower()
        # unseen trait never appears
        assert "anticipate next steps" not in text.lower()

    def test_directive_includes_hobbies_and_characteristics(self):
        p = UserPersonaProfile("u")
        for _ in range(5):
            p.observe_hobby("hiking", now=_at(0))
        p.observe_characteristic("role", "data scientist", strength=3.0, now=_at(0))
        text = p.render_style_directive(min_confidence=0.3)
        assert "hiking" in text
        assert "role=data scientist" in text


class TestCharacteristicsAndHobbies:
    def test_agreement_reinforces_value(self):
        p = UserPersonaProfile("u")
        p.observe_characteristic("tz", "US/Pacific", strength=2.0, now=_at(0))
        p.observe_characteristic("tz", "US/Pacific", strength=2.0, now=_at(1))
        assert p.characteristics["tz"]["value"] == "US/Pacific"
        assert p.characteristics["tz"]["confidence"] > 0.5

    def test_weak_conflict_does_not_flip_but_strong_does(self):
        p = UserPersonaProfile("u")
        p.observe_characteristic("role", "ds", strength=3.0, now=_at(0))
        p.observe_characteristic("role", "pm", strength=1.0, now=_at(1))
        assert p.characteristics["role"]["value"] == "ds"
        p.observe_characteristic("role", "pm", strength=10.0, now=_at(2))
        assert p.characteristics["role"]["value"] == "pm"

    def test_hobby_intensity_ranks_by_evidence(self):
        p = UserPersonaProfile("u")
        for _ in range(5):
            p.observe_hobby("hiking", now=_at(0))
        p.observe_hobby("jazz", now=_at(0))
        top = dict(p.top_hobbies(k=5))
        assert top["hiking"] > top["jazz"]


class TestSerialization:
    def test_round_trip_preserves_state(self):
        p = UserPersonaProfile("u", half_life_days=30.0)
        for i in range(4):
            p.observe_trait("verbosity", 0.1, now=_at(i * 0.01))
        p.observe_hobby("chess", now=_at(0))
        p.observe_characteristic("role", "analyst", strength=2.0, now=_at(0))
        d = p.to_dict()
        p2 = UserPersonaProfile.from_dict(d)
        assert p2.half_life_days == 30.0
        assert abs(p2.trait("verbosity").value - p.trait("verbosity").value) < 1e-9
        assert p2.characteristics["role"]["value"] == "analyst"
        assert p2.render_style_directive() == p.render_style_directive()

    def test_trait_dataclass_round_trip(self):
        t = Trait()
        t.observe(0.8, strength=1.0, half_life_days=30.0, now=_at(0))
        t2 = Trait.from_dict(t.to_dict())
        assert t2.value == t.value and t2.confidence == t.confidence


class TestContextMemoryIntegration:
    def test_observe_persist_reload_and_gdpr(self, tmp_path):
        from app.intelligence.context_memory import ContextMemoryStore
        store = ContextMemoryStore()
        uid = uuid4()
        store.observe_user_persona(
            uid,
            traits={"verbosity": 0.0, "formality": 0.0},
            hobbies=["hiking"],
            characteristics={"role": "data scientist"},
            acquisition={"reddit": 0.9},
        )
        for _ in range(4):
            store.observe_user_persona(uid, traits={"verbosity": 0.0, "formality": 0.0})

        persona = store.get_user_persona(uid)
        assert persona.trait("verbosity").value < 0.2
        directive = persona.render_style_directive(min_confidence=0.3)
        assert "concise" in directive

        p = tmp_path / "cm.json"
        store.persist(p)
        store2 = ContextMemoryStore()
        store2.load_from_disk(p)
        reloaded = store2.get_user_persona(uid)
        assert reloaded.render_style_directive(min_confidence=0.3) == directive
        assert reloaded.characteristics["role"]["value"] == "data scientist"

        exported = store.export_user_data(uid)
        assert exported["persona"]
        removed = store.clear_user_data(uid)
        assert removed["persona"] == 1
        assert store.get_user_persona(uid).render_style_directive() == ""
