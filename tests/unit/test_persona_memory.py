"""Tests for the desktop persona-memory answer-path bridge.

`extract_style_signals` is a pure function (no I/O).  The learn/render/persist
path is exercised against a temp user-data dir so it touches the real
``ContextMemoryStore`` without polluting the developer's machine.
"""

from __future__ import annotations

import importlib

import pytest

from app.local import persona_memory as pm


class TestExtractStyleSignals:
    def test_neutral_text_yields_nothing(self):
        assert pm.extract_style_signals("what's the weather in Paris today?") == {}

    def test_empty_text(self):
        assert pm.extract_style_signals("") == {}

    @pytest.mark.parametrize("text,trait,value", [
        ("please be concise", "verbosity", 0.0),
        ("can you give more detail and elaborate", "verbosity", 1.0),
        ("no emojis please", "emoji_affinity", 0.0),
        ("walk me through it step by step", "step_by_step_preference", 1.0),
        ("just the answer, skip the explanation", "step_by_step_preference", 0.0),
        ("be more technical", "technical_depth", 1.0),
        ("explain in simple terms, eli5", "technical_depth", 0.0),
        ("show me examples", "example_preference", 1.0),
        ("cite your sources", "evidence_depth", 1.0),
    ])
    def test_explicit_requests_detected(self, text, trait, value):
        assert pm.extract_style_signals(text).get(trait) == value

    def test_multiple_signals_in_one_turn(self):
        sig = pm.extract_style_signals("be concise and no emojis please")
        assert sig.get("verbosity") == 0.0
        assert sig.get("emoji_affinity") == 0.0


class TestLearnRenderPersist:
    @pytest.fixture(autouse=True)
    def _isolated_store(self, tmp_path, monkeypatch):
        # Point the user-data dir at a temp location and reset the singleton so
        # each test starts from a clean, isolated persona store.
        monkeypatch.setenv("SMR_DATA_DIR", str(tmp_path))
        monkeypatch.setattr(pm, "_store", None, raising=False)
        monkeypatch.setattr(pm, "_loaded", False, raising=False)
        self.tmp_path = tmp_path
        yield
        pm._store = None
        pm._loaded = False

    def test_cold_start_has_no_directive(self):
        assert pm.persona_system_prompt() is None

    def test_learn_then_render(self):
        for _ in range(4):
            pm.learn_from_user_turn("be concise please")
            pm.learn_from_user_turn("no emojis")
        directive = pm.persona_system_prompt(min_confidence=0.3)
        assert directive is not None
        assert "concise" in directive
        assert "emoji" in directive.lower()

    def test_persisted_and_reloaded(self):
        for _ in range(4):
            pm.learn_from_user_turn("be concise please")
        directive = pm.persona_system_prompt(min_confidence=0.3)
        assert (self.tmp_path / "persona_memory.json").exists()
        # Simulate a process restart: drop the singleton, reload from disk.
        pm._store = None
        pm._loaded = False
        assert pm.persona_system_prompt(min_confidence=0.3) == directive

    def test_neutral_turn_does_not_learn(self):
        assert pm.learn_from_user_turn("what is the capital of France?") == {}
        assert pm.persona_system_prompt() is None


class TestBanditRewardLoop:
    @pytest.fixture(autouse=True)
    def _isolated(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SMR_DATA_DIR", str(tmp_path))
        for attr in ("_store", "_bandit"):
            monkeypatch.setattr(pm, attr, None, raising=False)
        for attr in ("_loaded", "_bandit_loaded"):
            monkeypatch.setattr(pm, attr, False, raising=False)
        self.tmp_path = tmp_path
        yield
        pm._store = None; pm._loaded = False
        pm._bandit = None; pm._bandit_loaded = False

    def test_turn_signals_reward_bandit(self):
        from app.personalization.persona_bandit import ARM_LOW
        for _ in range(5):
            pm.learn_from_user_turn("please be concise")
        bandit = pm.get_directive_bandit()
        assert bandit.recommend("verbosity") == ARM_LOW
        assert (self.tmp_path / "directive_bandit.json").exists()

    def test_bandit_fallback_supplies_directive_at_high_threshold(self):
        for _ in range(5):
            pm.learn_from_user_turn("please be concise")
        # the estimator alone is silent at 0.95, but the bandit fills it in
        store = pm.get_persona_memory()
        persona = store.get_user_persona(pm.DESKTOP_USER_ID)
        assert persona.render_style_directive(min_confidence=0.95) == ""
        with_bandit = pm.persona_system_prompt(min_confidence=0.95)
        assert with_bandit is not None and "concise" in with_bandit

    def test_bandit_survives_restart(self):
        from app.personalization.persona_bandit import ARM_LOW
        for _ in range(5):
            pm.learn_from_user_turn("be concise")
        pm._bandit = None
        pm._bandit_loaded = False
        assert pm.get_directive_bandit().recommend("verbosity") == ARM_LOW
