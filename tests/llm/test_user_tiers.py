"""Unit + stress tests for the Lite/Medium/Turbo user-tier feature.

These tests run offline — every OpenRouter call is replaced by a stub so the
suite is deterministic and does not require ``OPENROUTER_API_KEY`` to be set.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import List, Optional
from unittest.mock import patch

import pytest

from app.core.config import settings
from app.llm.config import MODEL_REGISTRY, LLMServiceConfig, ensure_openrouter_model_registered
from app.llm.models import (
    FinishReason,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    MessageRole,
    PerformanceMetrics,
    TokenUsage,
)
from app.llm.router import LLMRouter, RoutingStrategy
from app.llm.user_tiers import (
    UserTier,
    get_active_tier,
    is_tier_mode_enabled,
    list_tiers,
    resolve_tier_model,
    set_active_tier,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_tier_state():
    """Snapshot/restore tier-related settings around every test."""
    saved_tier = settings.user_tier
    saved_key = settings.openrouter_api_key
    saved_override = get_active_tier()
    settings.user_tier = None
    settings.openrouter_api_key = "test-or-key"
    set_active_tier(None)
    try:
        yield
    finally:
        settings.user_tier = saved_tier
        settings.openrouter_api_key = saved_key
        set_active_tier(saved_override)


def _stub_response(model: str) -> LLMResponse:
    """Build a minimal valid ``LLMResponse`` for stubbed clients."""
    return LLMResponse(
        content=f"stub-from:{model}",
        model=model,
        provider=LLMProvider.OPENROUTER,
        usage=TokenUsage(
            prompt_tokens=5,
            completion_tokens=7,
            total_tokens=12,
            prompt_cost=0.0,
            completion_cost=0.0,
            total_cost=0.0,
        ),
        metrics=PerformanceMetrics(
            latency_ms=10,
            request_time=datetime.utcnow(),
            response_time=datetime.utcnow(),
        ),
        finish_reason=FinishReason.STOP,
    )


class _StubOpenRouterClient:
    """Drop-in stub for :class:`OpenRouterLLMClient`."""

    def __init__(self, model_name: str, service_config: Optional[LLMServiceConfig] = None):
        self.model_name = model_name
        self.service_config = service_config
        self.calls: List[List[LLMMessage]] = []

    async def generate(self, messages, temperature=0.7, max_tokens=None, **kwargs):
        self.calls.append(list(messages))
        return _stub_response(self.model_name)

    async def generate_simple(self, prompt, **kwargs):
        return _stub_response(self.model_name).content

    def get_stats(self) -> dict:
        return {"model": self.model_name, "total_requests": len(self.calls)}


# ---------------------------------------------------------------------------
# UserTier enum + helpers
# ---------------------------------------------------------------------------


class TestUserTierEnum:
    def test_parse_valid_values(self):
        assert UserTier.parse("lite") == UserTier.LITE
        assert UserTier.parse("MEDIUM") == UserTier.MEDIUM
        assert UserTier.parse(" Turbo ") == UserTier.TURBO

    def test_parse_none_and_empty(self):
        assert UserTier.parse(None) is None
        assert UserTier.parse("") is None
        assert UserTier.parse("   ") is None

    def test_parse_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown tier"):
            UserTier.parse("ultra")

    def test_parse_non_string_raises(self):
        with pytest.raises(ValueError, match="must be a string"):
            UserTier.parse(123)  # type: ignore[arg-type]


class TestTierHelpers:
    def test_list_tiers_contains_all_three(self):
        tiers = list_tiers()
        ids = {t["id"] for t in tiers}
        assert ids == {"lite", "medium", "turbo"}
        for t in tiers:
            assert t["model"] and isinstance(t["model"], str)

    def test_resolve_tier_model_registers_unknown(self):
        custom = "openai/gpt-4o-test-xyz"
        with patch.object(settings, "user_tier_medium_model", custom):
            assert custom not in MODEL_REGISTRY
            resolved = resolve_tier_model(UserTier.MEDIUM)
            assert resolved == custom
            assert custom in MODEL_REGISTRY
            assert MODEL_REGISTRY[custom].provider == LLMProvider.OPENROUTER

    def test_set_and_get_active_tier(self):
        assert get_active_tier() is None
        set_active_tier(UserTier.TURBO)
        assert get_active_tier() == UserTier.TURBO
        set_active_tier(None)
        assert get_active_tier() is None

    def test_is_tier_mode_enabled_requires_key_and_tier(self):
        assert not is_tier_mode_enabled()
        set_active_tier(UserTier.LITE)
        assert is_tier_mode_enabled()
        with patch.object(settings, "openrouter_api_key", None):
            assert not is_tier_mode_enabled()



# ---------------------------------------------------------------------------
# Router integration
# ---------------------------------------------------------------------------


class TestRouterTierSelection:
    def _router(self):
        return LLMRouter(service_config=LLMServiceConfig(
            primary_model="gpt-4o-mini",
            fallback_models=[],
            enable_caching=False,
        ))

    def test_select_model_with_explicit_tier(self):
        set_active_tier(None)
        decision = self._router().select_model(
            messages=[LLMMessage(role=MessageRole.USER, content="hi")],
            user_tier=UserTier.TURBO,
        )
        assert decision.provider == LLMProvider.OPENROUTER
        assert decision.model_name == settings.user_tier_turbo_model
        assert "turbo" in decision.reason.lower()

    def test_select_model_with_active_tier(self):
        set_active_tier(UserTier.LITE)
        decision = self._router().select_model(
            messages=[LLMMessage(role=MessageRole.USER, content="hi")],
        )
        assert decision.model_name == settings.user_tier_lite_model

    def test_select_model_legacy_path_when_tier_disabled(self):
        set_active_tier(None)
        decision = self._router().select_model(
            messages=[LLMMessage(role=MessageRole.USER, content="hi")],
        )
        assert decision.provider != LLMProvider.OPENROUTER

    def test_select_model_tier_mode_disabled_without_api_key(self):
        set_active_tier(UserTier.MEDIUM)
        with patch.object(settings, "openrouter_api_key", None):
            decision = self._router().select_model(
                messages=[LLMMessage(role=MessageRole.USER, content="hi")],
            )
        assert decision.provider != LLMProvider.OPENROUTER

    @pytest.mark.asyncio
    async def test_generate_routes_to_stub_openrouter(self):
        set_active_tier(UserTier.MEDIUM)
        router = self._router()
        stub = _StubOpenRouterClient(settings.user_tier_medium_model)
        with patch.object(router, "_get_client", return_value=stub):
            resp = await router.generate(
                messages=[LLMMessage(role=MessageRole.USER, content="hello")],
            )
        assert resp.model == settings.user_tier_medium_model
        assert stub.calls, "stub OpenRouter client was never invoked"


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


class TestTierAPIEndpoints:
    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.api.routes import llm as llm_routes
        app = FastAPI()
        app.include_router(llm_routes.router, prefix="/api/llm")
        return TestClient(app)

    def test_get_tiers(self, client):
        resp = client.get("/api/llm/tiers")
        assert resp.status_code == 200
        body = resp.json()
        assert {t["id"] for t in body["tiers"]} == {"lite", "medium", "turbo"}
        assert body["active_tier"] is None

    def test_post_tier_round_trip(self, client):
        resp = client.post("/api/llm/tier", json={"tier": "medium"})
        assert resp.status_code == 200
        assert resp.json()["active_tier"] == "medium"

        resp = client.get("/api/llm/tier")
        assert resp.json()["active_tier"] == "medium"

        resp = client.post("/api/llm/tier", json={"tier": None})
        assert resp.status_code == 200
        assert resp.json()["active_tier"] is None

    def test_post_tier_invalid_rejected(self, client):
        resp = client.post("/api/llm/tier", json={"tier": "ultra"})
        assert resp.status_code == 400

    def test_generate_with_tier_override(self, client):
        from app.api.routes import llm as llm_routes

        class _FakeRouter:
            async def generate(self, **kwargs):
                return _stub_response(settings.user_tier_lite_model)

        with patch.object(llm_routes, "get_router", return_value=_FakeRouter()):
            resp = client.post(
                "/api/llm/generate",
                json={"prompt": "hi", "tier": "lite"},
            )
        assert resp.status_code == 200
        assert resp.json()["model"] == settings.user_tier_lite_model


# ---------------------------------------------------------------------------
# Stress test — 200 concurrent tier-routed generations through the stub
# ---------------------------------------------------------------------------


class TestStress:
    @pytest.mark.asyncio
    async def test_concurrent_tier_routing_does_not_leak(self):
        set_active_tier(UserTier.MEDIUM)
        router = LLMRouter(service_config=LLMServiceConfig(
            primary_model="gpt-4o-mini",
            fallback_models=[],
            enable_caching=False,
        ))
        stub = _StubOpenRouterClient(settings.user_tier_medium_model)
        with patch.object(router, "_get_client", return_value=stub):
            results = await asyncio.gather(*(
                router.generate(
                    messages=[LLMMessage(role=MessageRole.USER, content=f"q-{i}")]
                )
                for i in range(200)
            ))
        assert len(results) == 200
        assert len(stub.calls) == 200
        assert all(r.model == settings.user_tier_medium_model for r in results)
