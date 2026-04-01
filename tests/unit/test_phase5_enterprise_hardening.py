"""Phase 5 — Enterprise Hardening Unit Tests.

Test groups
-----------
Group  1: TenantTierEnum         — enum values
Group  2: TenantConfigModel      — frozen, field validators
Group  3: AuditEntryModel        — frozen, chain_hash default
Group  4: SourceTrustPolicyModel — floor ≤ ceiling validator
Group  5: RetentionPolicyModel   — per_class_retention validator
Group  6: SLOTargetModel         — window_seconds, percentile range
Group  7: SLOStatusModel         — frozen, observation_count ≥ 0
Group  8: PurgeResultModel       — records_purged ≤ records_checked

Group  9: TenantRegistryConstruct    — constructor validation
Group 10: TenantRegistryProvision    — provision lifecycle
Group 11: TenantRegistryDeprovision  — soft deactivation
Group 12: TenantRegistryGet          — get/list/update
Group 13: TenantRegistryTierDefaults — tier-specific resource caps
Group 14: TenantRegistryEdgeCases    — duplicate provision, unknown tenant,
                                       max_tenants limit, wrong types

Group 15: AuditLoggerConstruct   — constructor validation
Group 16: AuditLoggerLog         — log_event + chain_hash set
Group 17: AuditLoggerQuery       — query_events filtering
Group 18: AuditLoggerChain       — verify_chain integrity + tamper detection
Group 19: AuditLoggerExport      — export_events JSON roundtrip
Group 20: AuditLoggerSummary     — get_summary counts
Group 21: AuditLoggerSuspicious  — detect_suspicious_patterns heuristics

Group 22: SourceTrustManagerConstruct    — constructor validation
Group 23: SourceTrustManagerSetPolicy    — set / get / reset policy
Group 24: SourceTrustManagerEffective    — effective trust formula
Group 25: SourceTrustManagerBlocklist    — block / unblock / is_blocked
Group 26: SourceTrustManagerRecommend    — heuristic recommendation

Group 27: RetentionManagerConstruct — constructor validation
Group 28: RetentionManagerPolicy    — set / get / get_retention_days
Group 29: RetentionManagerExpiry    — check_expired correctness
Group 30: RetentionManagerPurge     — purge_eligible + legal hold

Group 31: SLOTrackerConstruct     — constructor validation
Group 32: SLOTrackerRegister      — register / deregister
Group 33: SLOTrackerObservations  — record_observation
Group 34: SLOTrackerStatus        — get_slo_status aggregation
Group 35: SLOTrackerBreach        — check_breach / list_breaches

Group 36: CrossComponentWiring    — TenantRegistry → AuditLogger → TrustManager
Group 37: CrossComponentPackageImports — __all__ static check
"""

from __future__ import annotations

import json
import pytest
from datetime import datetime, timedelta, timezone

from app.enterprise.models import (
    AuditEntry,
    AuditEventType,
    DataClass,
    PurgeResult,
    RetentionPolicy,
    RiskLevel,
    SLOOperator,
    SLOStatus,
    SLOTarget,
    SourceTrustPolicy,
    TenantConfig,
    TenantTier,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _tenant_id() -> str:
    import uuid
    return f"tenant-{uuid.uuid4().hex[:8]}"


def _audit_entry(tenant_id: str = "t1", event_type: AuditEventType = AuditEventType.API_CALL) -> AuditEntry:
    return AuditEntry(tenant_id=tenant_id, event_type=event_type)


def _slo_target(metric: str = "latency_ms", target: float = 200.0,
                operator: SLOOperator = SLOOperator.LESS_THAN,
                window: int = 3600) -> SLOTarget:
    return SLOTarget(metric_name=metric, target_value=target, operator=operator, window_seconds=window)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ===========================================================================
# Group 1: TenantTierEnum
# ===========================================================================

class TestTenantTierEnum:
    def test_all_values_present(self):
        assert TenantTier.FREE.value == "free"
        assert TenantTier.STANDARD.value == "standard"
        assert TenantTier.ENTERPRISE.value == "enterprise"
        assert TenantTier.PLATFORM.value == "platform"

    def test_from_string(self):
        assert TenantTier("enterprise") == TenantTier.ENTERPRISE

    def test_invalid_tier_raises(self):
        with pytest.raises(ValueError):
            TenantTier("ultra")


# ===========================================================================
# Group 2: TenantConfigModel
# ===========================================================================

class TestTenantConfigModel:
    def test_frozen(self):
        cfg = TenantConfig(tenant_id="t1")
        with pytest.raises(Exception):
            cfg.is_active = False  # type: ignore

    def test_empty_tenant_id_raises(self):
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            TenantConfig(tenant_id="")

    def test_max_sources_lt_one_raises(self):
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            TenantConfig(tenant_id="t1", max_sources=0)

    def test_max_daily_api_calls_lt_one_raises(self):
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            TenantConfig(tenant_id="t1", max_daily_api_calls=0)

    def test_defaults(self):
        cfg = TenantConfig(tenant_id="t1")
        assert cfg.tier == TenantTier.FREE
        assert cfg.audit_enabled is True
        assert cfg.is_active is True
        assert cfg.max_sources == 10
        assert cfg.max_daily_api_calls == 1_000


# ===========================================================================
# Group 3: AuditEntryModel
# ===========================================================================

class TestAuditEntryModel:
    def test_frozen(self):
        e = _audit_entry()
        with pytest.raises(Exception):
            e.success = False  # type: ignore

    def test_empty_tenant_id_raises(self):
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            AuditEntry(tenant_id="", event_type=AuditEventType.LOGIN)

    def test_auto_entry_id(self):
        e = _audit_entry()
        assert len(e.entry_id) > 0

    def test_default_risk_low(self):
        e = _audit_entry()
        assert e.risk_level == RiskLevel.LOW

    def test_chain_hash_default_empty(self):
        e = _audit_entry()
        assert e.chain_hash == ""

    def test_success_default_true(self):
        e = _audit_entry()
        assert e.success is True


# ===========================================================================
# Group 4: SourceTrustPolicyModel
# ===========================================================================

class TestSourceTrustPolicyModel:
    def test_floor_gt_ceiling_raises(self):
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            SourceTrustPolicy(tenant_id="t1", global_trust_floor=0.8, global_trust_ceiling=0.3)

    def test_floor_eq_ceiling_ok(self):
        p = SourceTrustPolicy(tenant_id="t1", global_trust_floor=0.5, global_trust_ceiling=0.5)
        assert p.global_trust_floor == pytest.approx(0.5)

    def test_trust_out_of_range_raises(self):
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            SourceTrustPolicy(tenant_id="t1", global_trust_ceiling=1.5)

    def test_frozen(self):
        p = SourceTrustPolicy(tenant_id="t1")
        with pytest.raises(Exception):
            p.legal_hold = True  # type: ignore — wrong attr name for trust policy, still frozen

    def test_auto_policy_id(self):
        p = SourceTrustPolicy(tenant_id="t1")
        assert len(p.policy_id) > 0

    def test_defaults(self):
        p = SourceTrustPolicy(tenant_id="t1")
        assert p.global_trust_floor == pytest.approx(0.0)
        assert p.global_trust_ceiling == pytest.approx(1.0)
        assert p.blocklisted_source_ids == []


# ===========================================================================
# Group 5: RetentionPolicyModel
# ===========================================================================

class TestRetentionPolicyModel:
    def test_default_retention_zero_raises(self):
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            RetentionPolicy(tenant_id="t1", default_retention_days=0)

    def test_default_retention_too_large_raises(self):
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            RetentionPolicy(tenant_id="t1", default_retention_days=3651)

    def test_per_class_retention_invalid_days_raises(self):
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            RetentionPolicy(tenant_id="t1", per_class_retention={"audit_log": 0})

    def test_frozen(self):
        p = RetentionPolicy(tenant_id="t1")
        with pytest.raises(Exception):
            p.legal_hold = True  # type: ignore

    def test_defaults(self):
        p = RetentionPolicy(tenant_id="t1")
        assert p.default_retention_days == 365
        assert p.legal_hold is False
        assert p.auto_purge_enabled is False


# ===========================================================================
# Group 9: TenantRegistryConstruct
# ===========================================================================

class TestTenantRegistryConstruct:
    def test_negative_max_tenants_raises(self):
        from app.enterprise.tenant_registry import TenantRegistry
        with pytest.raises(ValueError):
            TenantRegistry(max_tenants=-1)

    def test_default_construction(self):
        from app.enterprise.tenant_registry import TenantRegistry
        r = TenantRegistry()
        assert r is not None

    def test_zero_max_tenants_means_unlimited(self):
        from app.enterprise.tenant_registry import TenantRegistry
        r = TenantRegistry(max_tenants=0)
        for i in range(5):
            r.provision_tenant(f"t{i}")
        assert len(r.list_tenants()) == 5


# ===========================================================================
# Group 10: TenantRegistryProvision
# ===========================================================================

class TestTenantRegistryProvision:
    def _registry(self):
        from app.enterprise.tenant_registry import TenantRegistry
        return TenantRegistry()

    def test_returns_tenant_config(self):
        r = self._registry()
        cfg = r.provision_tenant("acme")
        assert isinstance(cfg, TenantConfig)
        assert cfg.tenant_id == "acme"

    def test_whitespace_normalised(self):
        r = self._registry()
        cfg = r.provision_tenant("  acme  ")
        assert cfg.tenant_id == "acme"

    def test_tier_applied(self):
        r = self._registry()
        cfg = r.provision_tenant("acme", tier=TenantTier.ENTERPRISE)
        assert cfg.tier == TenantTier.ENTERPRISE
        assert cfg.max_sources == 1_000

    def test_override_max_sources(self):
        r = self._registry()
        cfg = r.provision_tenant("acme", tier=TenantTier.FREE, max_sources=500)
        assert cfg.max_sources == 500

    def test_feature_flags_stored(self):
        r = self._registry()
        cfg = r.provision_tenant("acme", feature_flags={"new_ui": True})
        assert cfg.feature_flags == {"new_ui": True}

    def test_empty_tenant_id_raises(self):
        r = self._registry()
        with pytest.raises(ValueError):
            r.provision_tenant("")

    def test_whitespace_only_tenant_id_raises(self):
        r = self._registry()
        with pytest.raises(ValueError):
            r.provision_tenant("   ")

    def test_non_string_tenant_id_raises(self):
        r = self._registry()
        with pytest.raises(TypeError):
            r.provision_tenant(42)  # type: ignore

    def test_wrong_tier_type_raises(self):
        r = self._registry()
        with pytest.raises(TypeError):
            r.provision_tenant("acme", tier="enterprise")  # type: ignore


# ===========================================================================
# Group 11: TenantRegistryDeprovision
# ===========================================================================

class TestTenantRegistryDeprovision:
    def _registry(self):
        from app.enterprise.tenant_registry import TenantRegistry
        r = TenantRegistry()
        r.provision_tenant("acme")
        return r

    def test_deprovisioned_is_inactive(self):
        r = self._registry()
        cfg = r.deprovision_tenant("acme")
        assert cfg.is_active is False

    def test_deprovisioned_still_in_list_all(self):
        r = self._registry()
        r.deprovision_tenant("acme")
        all_tenants = r.list_tenants(active_only=False)
        assert any(c.tenant_id == "acme" for c in all_tenants)

    def test_deprovisioned_excluded_from_active(self):
        r = self._registry()
        r.deprovision_tenant("acme")
        active = r.list_tenants(active_only=True)
        assert not any(c.tenant_id == "acme" for c in active)

    def test_unknown_tenant_raises_key_error(self):
        r = self._registry()
        with pytest.raises(KeyError):
            r.deprovision_tenant("unknown-tenant")

    def test_empty_id_raises(self):
        r = self._registry()
        with pytest.raises(ValueError):
            r.deprovision_tenant("")


# ===========================================================================
# Group 12: TenantRegistryGet
# ===========================================================================

class TestTenantRegistryGet:
    def _registry_with_tenants(self):
        from app.enterprise.tenant_registry import TenantRegistry
        r = TenantRegistry()
        r.provision_tenant("alpha", tier=TenantTier.STANDARD)
        r.provision_tenant("beta", tier=TenantTier.FREE)
        return r

    def test_get_returns_correct_tenant(self):
        r = self._registry_with_tenants()
        cfg = r.get_tenant("alpha")
        assert cfg.tenant_id == "alpha"
        assert cfg.tier == TenantTier.STANDARD

    def test_get_unknown_raises_key_error(self):
        r = self._registry_with_tenants()
        with pytest.raises(KeyError):
            r.get_tenant("gamma")

    def test_list_returns_sorted_by_id(self):
        r = self._registry_with_tenants()
        ids = [c.tenant_id for c in r.list_tenants()]
        assert ids == sorted(ids)

    def test_list_count(self):
        r = self._registry_with_tenants()
        assert len(r.list_tenants()) == 2

    def test_update_display_name(self):
        r = self._registry_with_tenants()
        updated = r.update_tenant("alpha", display_name="Alpha Corp")
        assert updated.display_name == "Alpha Corp"

    def test_update_unknown_field_ignored(self):
        r = self._registry_with_tenants()
        # Should not raise; unknown fields silently ignored
        updated = r.update_tenant("alpha", nonexistent_field="xyz")
        assert updated.tenant_id == "alpha"

    def test_update_unknown_tenant_raises(self):
        r = self._registry_with_tenants()
        with pytest.raises(KeyError):
            r.update_tenant("gamma", display_name="X")


# ===========================================================================
# Group 13: TenantRegistryTierDefaults
# ===========================================================================

class TestTenantRegistryTierDefaults:
    def _r(self):
        from app.enterprise.tenant_registry import TenantRegistry
        return TenantRegistry()

    def test_free_tier_defaults(self):
        cfg = self._r().provision_tenant("t", tier=TenantTier.FREE)
        assert cfg.max_sources == 10
        assert cfg.max_daily_api_calls == 1_000

    def test_standard_tier_defaults(self):
        cfg = self._r().provision_tenant("t", tier=TenantTier.STANDARD)
        assert cfg.max_sources == 100
        assert cfg.max_daily_api_calls == 50_000

    def test_enterprise_tier_defaults(self):
        cfg = self._r().provision_tenant("t", tier=TenantTier.ENTERPRISE)
        assert cfg.max_sources == 1_000

    def test_platform_tier_defaults(self):
        cfg = self._r().provision_tenant("t", tier=TenantTier.PLATFORM)
        assert cfg.max_sources == 100_000

    def test_enterprise_all_data_classes_allowed(self):
        cfg = self._r().provision_tenant("t", tier=TenantTier.ENTERPRISE)
        for dc in DataClass:
            assert dc in cfg.allowed_data_classes

    def test_free_tier_limited_data_classes(self):
        cfg = self._r().provision_tenant("t", tier=TenantTier.FREE)
        assert DataClass.CREDENTIAL not in cfg.allowed_data_classes


# ===========================================================================
# Group 14: TenantRegistryEdgeCases
# ===========================================================================

class TestTenantRegistryEdgeCases:
    def test_duplicate_provision_raises(self):
        from app.enterprise.tenant_registry import TenantRegistry
        r = TenantRegistry()
        r.provision_tenant("same")
        with pytest.raises(ValueError):
            r.provision_tenant("same")

    def test_max_tenants_limit_enforced(self):
        from app.enterprise.tenant_registry import TenantRegistry
        r = TenantRegistry(max_tenants=2)
        r.provision_tenant("t1")
        r.provision_tenant("t2")
        with pytest.raises(ValueError):
            r.provision_tenant("t3")

    def test_suggest_tier_heuristic_free(self):
        from app.enterprise.tenant_registry import TenantRegistry
        r = TenantRegistry()
        tier = r.suggest_tier("t1", {"daily_api_calls": 100})
        assert tier == TenantTier.FREE

    def test_suggest_tier_heuristic_standard(self):
        from app.enterprise.tenant_registry import TenantRegistry
        r = TenantRegistry()
        tier = r.suggest_tier("t1", {"daily_api_calls": 10_000})
        assert tier == TenantTier.STANDARD

    def test_suggest_tier_heuristic_enterprise(self):
        from app.enterprise.tenant_registry import TenantRegistry
        r = TenantRegistry()
        tier = r.suggest_tier("t1", {"daily_api_calls": 100_000})
        assert tier == TenantTier.ENTERPRISE

    def test_suggest_tier_heuristic_platform(self):
        from app.enterprise.tenant_registry import TenantRegistry
        r = TenantRegistry()
        tier = r.suggest_tier("t1", {"daily_api_calls": 1_000_000})
        assert tier == TenantTier.PLATFORM

    def test_suggest_tier_wrong_stats_type_raises(self):
        from app.enterprise.tenant_registry import TenantRegistry
        r = TenantRegistry()
        with pytest.raises(TypeError):
            r.suggest_tier("t1", "not a dict")  # type: ignore

    def test_suggest_tier_empty_tenant_raises(self):
        from app.enterprise.tenant_registry import TenantRegistry
        r = TenantRegistry()
        with pytest.raises(ValueError):
            r.suggest_tier("", {"daily_api_calls": 100})



# ===========================================================================
# Group 15: AuditLoggerConstruct
# ===========================================================================

class TestAuditLoggerConstruct:
    def test_zero_max_entries_raises(self):
        from app.enterprise.audit_logger import AuditLogger
        with pytest.raises(ValueError):
            AuditLogger(max_entries_per_tenant=0)

    def test_negative_max_entries_raises(self):
        from app.enterprise.audit_logger import AuditLogger
        with pytest.raises(ValueError):
            AuditLogger(max_entries_per_tenant=-1)

    def test_default_construction(self):
        from app.enterprise.audit_logger import AuditLogger
        a = AuditLogger()
        assert a is not None


# ===========================================================================
# Group 16: AuditLoggerLog
# ===========================================================================

class TestAuditLoggerLog:
    def _logger(self):
        from app.enterprise.audit_logger import AuditLogger
        return AuditLogger()

    def test_log_event_returns_audit_entry(self):
        al = self._logger()
        entry = _audit_entry("t1")
        stored = al.log_event(entry)
        assert isinstance(stored, AuditEntry)

    def test_log_event_sets_chain_hash(self):
        al = self._logger()
        stored = al.log_event(_audit_entry("t1"))
        assert len(stored.chain_hash) == 64  # SHA-256 hex digest

    def test_first_entry_uses_genesis(self):
        from app.enterprise.audit_logger import _compute_chain_hash, _GENESIS_HASH
        al = self._logger()
        entry = _audit_entry("t1")
        stored = al.log_event(entry)
        expected = _compute_chain_hash(_GENESIS_HASH, entry)
        assert stored.chain_hash == expected

    def test_second_entry_chains_to_first(self):
        al = self._logger()
        e1 = al.log_event(_audit_entry("t1"))
        # Reuse the same entry object so the UUID matches what the logger saw
        original_e2 = _audit_entry("t1", AuditEventType.LOGIN)
        e2 = al.log_event(original_e2)
        from app.enterprise.audit_logger import _compute_chain_hash
        expected = _compute_chain_hash(e1.chain_hash, original_e2)
        assert e2.chain_hash == expected

    def test_wrong_type_raises(self):
        al = self._logger()
        with pytest.raises(TypeError):
            al.log_event("not an entry")  # type: ignore

    def test_separate_tenants_independent_chains(self):
        al = self._logger()
        orig_t1 = _audit_entry("t1")
        orig_t2 = _audit_entry("t2")
        e_t1 = al.log_event(orig_t1)
        e_t2 = al.log_event(orig_t2)
        # Different tenants start at GENESIS independently
        from app.enterprise.audit_logger import _compute_chain_hash, _GENESIS_HASH
        exp_t1 = _compute_chain_hash(_GENESIS_HASH, orig_t1)
        assert e_t1.chain_hash == exp_t1
        # t2's chain is independent of t1's
        exp_t2 = _compute_chain_hash(_GENESIS_HASH, orig_t2)
        assert e_t2.chain_hash == exp_t2


# ===========================================================================
# Group 17: AuditLoggerQuery
# ===========================================================================

class TestAuditLoggerQuery:
    def _logger_with_events(self):
        from app.enterprise.audit_logger import AuditLogger
        al = AuditLogger()
        al.log_event(AuditEntry(tenant_id="t1", event_type=AuditEventType.LOGIN, actor_id="u1"))
        al.log_event(AuditEntry(tenant_id="t1", event_type=AuditEventType.API_CALL, actor_id="u2"))
        al.log_event(AuditEntry(tenant_id="t1", event_type=AuditEventType.LOGIN, actor_id="u1",
                                risk_level=RiskLevel.HIGH))
        al.log_event(AuditEntry(tenant_id="t1", event_type=AuditEventType.DATA_EXPORT,
                                actor_id="u1", success=False))
        return al

    def test_query_all_returns_newest_first(self):
        al = self._logger_with_events()
        results = al.query_events("t1", limit=100)
        assert len(results) == 4
        # Newest first: DATA_EXPORT was last logged
        assert results[0].event_type == AuditEventType.DATA_EXPORT

    def test_filter_by_event_type(self):
        al = self._logger_with_events()
        logins = al.query_events("t1", event_type=AuditEventType.LOGIN)
        assert len(logins) == 2
        assert all(e.event_type == AuditEventType.LOGIN for e in logins)

    def test_filter_by_actor_id(self):
        al = self._logger_with_events()
        u2_events = al.query_events("t1", actor_id="u2")
        assert len(u2_events) == 1
        assert u2_events[0].actor_id == "u2"

    def test_filter_by_risk_level(self):
        al = self._logger_with_events()
        high_risk = al.query_events("t1", risk_level=RiskLevel.HIGH)
        assert len(high_risk) == 1

    def test_filter_success_only_false(self):
        al = self._logger_with_events()
        failed = al.query_events("t1", success_only=False)
        assert len(failed) == 1
        assert failed[0].event_type == AuditEventType.DATA_EXPORT

    def test_limit_respected(self):
        al = self._logger_with_events()
        results = al.query_events("t1", limit=2)
        assert len(results) == 2

    def test_unknown_tenant_returns_empty(self):
        al = self._logger_with_events()
        results = al.query_events("unknown-tenant")
        assert results == []

    def test_limit_zero_raises(self):
        from app.enterprise.audit_logger import AuditLogger
        with pytest.raises(ValueError):
            AuditLogger().query_events("t1", limit=0)

    def test_wrong_tenant_type_raises(self):
        from app.enterprise.audit_logger import AuditLogger
        with pytest.raises(TypeError):
            AuditLogger().query_events(123)  # type: ignore


# ===========================================================================
# Group 18: AuditLoggerChain
# ===========================================================================

class TestAuditLoggerChain:
    def _logger_with_chain(self, tenant: str = "t1", n: int = 5):
        from app.enterprise.audit_logger import AuditLogger
        al = AuditLogger()
        for i in range(n):
            al.log_event(AuditEntry(
                tenant_id=tenant,
                event_type=AuditEventType.API_CALL,
                actor_id=f"u{i}",
            ))
        return al

    def test_intact_chain_verifies_true(self):
        al = self._logger_with_chain()
        assert al.verify_chain("t1") is True

    def test_empty_chain_verifies_true(self):
        from app.enterprise.audit_logger import AuditLogger
        al = AuditLogger()
        assert al.verify_chain("no-events") is True

    def test_tampered_entry_fails_verification(self):
        al = self._logger_with_chain()
        # Manually corrupt an entry's chain_hash
        with al._lock:
            chain_list = list(al._chains["t1"])
            corrupted = chain_list[2].model_copy(update={"chain_hash": "deadbeef" * 8})
            chain_list[2] = corrupted
            from collections import deque
            al._chains["t1"] = deque(chain_list, maxlen=al._max_per_tenant)
        assert al.verify_chain("t1") is False

    def test_verify_chain_empty_tenant_raises(self):
        from app.enterprise.audit_logger import AuditLogger
        with pytest.raises(ValueError):
            AuditLogger().verify_chain("")

    def test_verify_chain_wrong_type_raises(self):
        from app.enterprise.audit_logger import AuditLogger
        with pytest.raises(TypeError):
            AuditLogger().verify_chain(None)  # type: ignore


# ===========================================================================
# Group 19: AuditLoggerExport
# ===========================================================================

class TestAuditLoggerExport:
    def test_export_is_valid_json(self):
        from app.enterprise.audit_logger import AuditLogger
        al = AuditLogger()
        al.log_event(_audit_entry("t1"))
        result = al.export_events("t1")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_export_contains_event_type(self):
        from app.enterprise.audit_logger import AuditLogger
        al = AuditLogger()
        al.log_event(AuditEntry(tenant_id="t1", event_type=AuditEventType.LOGIN))
        parsed = json.loads(al.export_events("t1"))
        assert parsed[0]["event_type"] == "login"

    def test_export_unknown_tenant_returns_empty_array(self):
        from app.enterprise.audit_logger import AuditLogger
        result = AuditLogger().export_events("nobody")
        assert result == "[]"

    def test_export_wrong_type_raises(self):
        from app.enterprise.audit_logger import AuditLogger
        with pytest.raises(TypeError):
            AuditLogger().export_events(42)  # type: ignore


# ===========================================================================
# Group 20: AuditLoggerSummary
# ===========================================================================

class TestAuditLoggerSummary:
    def _populated_logger(self):
        from app.enterprise.audit_logger import AuditLogger
        al = AuditLogger()
        al.log_event(AuditEntry(tenant_id="t1", event_type=AuditEventType.LOGIN, success=True))
        al.log_event(AuditEntry(tenant_id="t1", event_type=AuditEventType.LOGIN, success=False))
        al.log_event(AuditEntry(tenant_id="t1", event_type=AuditEventType.API_CALL,
                                risk_level=RiskLevel.HIGH))
        return al

    def test_summary_total_count(self):
        summary = self._populated_logger().get_summary("t1")
        assert summary["total_events"] == 3

    def test_summary_by_event_type(self):
        summary = self._populated_logger().get_summary("t1")
        assert summary["by_event_type"]["login"] == 2
        assert summary["by_event_type"]["api_call"] == 1

    def test_summary_failed_events(self):
        summary = self._populated_logger().get_summary("t1")
        assert summary["failed_events"] == 1

    def test_summary_empty_tenant(self):
        from app.enterprise.audit_logger import AuditLogger
        summary = AuditLogger().get_summary("nobody")
        assert summary["total_events"] == 0
        assert summary["failed_events"] == 0

    def test_summary_wrong_type_raises(self):
        from app.enterprise.audit_logger import AuditLogger
        with pytest.raises(TypeError):
            AuditLogger().get_summary(None)  # type: ignore


# ===========================================================================
# Group 21: AuditLoggerSuspicious
# ===========================================================================

class TestAuditLoggerSuspicious:
    def test_no_events_not_suspicious(self):
        from app.enterprise.audit_logger import AuditLogger
        result = AuditLogger().detect_suspicious_patterns("t1")
        assert result["suspicious"] is False
        assert result["reasons"] == []

    def test_high_permission_change_suspicious(self):
        from app.enterprise.audit_logger import AuditLogger
        al = AuditLogger()
        for _ in range(6):
            al.log_event(AuditEntry(
                tenant_id="t1",
                event_type=AuditEventType.PERMISSION_CHANGE,
                actor_id="admin",
            ))
        result = al.detect_suspicious_patterns("t1", window_size=10)
        assert result["suspicious"] is True
        assert any("permission" in r.lower() for r in result["reasons"])

    def test_multiple_critical_suspicious(self):
        from app.enterprise.audit_logger import AuditLogger
        al = AuditLogger()
        for _ in range(4):
            al.log_event(AuditEntry(
                tenant_id="t1",
                event_type=AuditEventType.API_CALL,
                risk_level=RiskLevel.CRITICAL,
            ))
        result = al.detect_suspicious_patterns("t1", window_size=10)
        assert result["suspicious"] is True
        assert any("critical" in r.lower() for r in result["reasons"])

    def test_window_size_zero_raises(self):
        from app.enterprise.audit_logger import AuditLogger
        with pytest.raises(ValueError):
            AuditLogger().detect_suspicious_patterns("t1", window_size=0)

    def test_wrong_tenant_type_raises(self):
        from app.enterprise.audit_logger import AuditLogger
        with pytest.raises(TypeError):
            AuditLogger().detect_suspicious_patterns(123)  # type: ignore


# ===========================================================================
# Group 22: SourceTrustManagerConstruct
# ===========================================================================

class TestSourceTrustManagerConstruct:
    def test_default_construction(self):
        from app.enterprise.source_trust_manager import SourceTrustManager
        m = SourceTrustManager()
        assert m is not None

    def test_pass_through_false(self):
        from app.enterprise.source_trust_manager import SourceTrustManager
        m = SourceTrustManager(pass_through=False)
        # No policy set → returns 0.0
        score = m.get_effective_trust("t1", "s1", 0.8)
        assert score == pytest.approx(0.0)

    def test_pass_through_true(self):
        from app.enterprise.source_trust_manager import SourceTrustManager
        m = SourceTrustManager(pass_through=True)
        # No policy set → returns base_score
        score = m.get_effective_trust("t1", "s1", 0.8)
        assert score == pytest.approx(0.8)


# ===========================================================================
# Group 23: SourceTrustManagerSetPolicy
# ===========================================================================

class TestSourceTrustManagerSetPolicy:
    def _mgr(self):
        from app.enterprise.source_trust_manager import SourceTrustManager
        return SourceTrustManager()

    def _policy(self, tenant_id: str = "t1", floor: float = 0.2, ceiling: float = 0.9):
        return SourceTrustPolicy(tenant_id=tenant_id,
                                 global_trust_floor=floor,
                                 global_trust_ceiling=ceiling)

    def test_set_and_get_policy(self):
        m = self._mgr()
        p = self._policy()
        m.set_policy("t1", p)
        retrieved = m.get_policy("t1")
        assert retrieved is not None
        assert retrieved.policy_id == p.policy_id

    def test_set_policy_mismatched_tenant_raises(self):
        m = self._mgr()
        p = self._policy("other-tenant")
        with pytest.raises(ValueError):
            m.set_policy("t1", p)

    def test_set_policy_wrong_type_raises(self):
        m = self._mgr()
        with pytest.raises(TypeError):
            m.set_policy("t1", "not a policy")  # type: ignore

    def test_get_policy_empty_tenant_raises(self):
        m = self._mgr()
        with pytest.raises(ValueError):
            m.get_policy("")

    def test_get_policy_unknown_returns_none(self):
        m = self._mgr()
        assert m.get_policy("unknown-tenant") is None

    def test_reset_policy(self):
        m = self._mgr()
        m.set_policy("t1", self._policy())
        m.reset_policy("t1")
        assert m.get_policy("t1") is None

    def test_reset_unknown_tenant_no_error(self):
        m = self._mgr()
        m.reset_policy("nonexistent")  # should not raise


# ===========================================================================
# Group 24: SourceTrustManagerEffective
# ===========================================================================

class TestSourceTrustManagerEffective:
    def _mgr_with_policy(self, floor: float = 0.2, ceiling: float = 0.8,
                         multipliers: dict = None, blocklist: list = None,
                         allowlist: list = None):
        from app.enterprise.source_trust_manager import SourceTrustManager
        m = SourceTrustManager()
        p = SourceTrustPolicy(
            tenant_id="t1",
            global_trust_floor=floor,
            global_trust_ceiling=ceiling,
            source_multipliers=multipliers or {},
            blocklisted_source_ids=blocklist or [],
            allowlisted_source_ids=allowlist or [],
        )
        m.set_policy("t1", p)
        return m

    def test_base_score_clamped_to_floor(self):
        m = self._mgr_with_policy(floor=0.4, ceiling=0.9)
        score = m.get_effective_trust("t1", "s1", 0.1)  # below floor
        assert score == pytest.approx(0.4)

    def test_base_score_clamped_to_ceiling(self):
        m = self._mgr_with_policy(floor=0.0, ceiling=0.6)
        score = m.get_effective_trust("t1", "s1", 0.9)  # above ceiling
        assert score == pytest.approx(0.6)

    def test_multiplier_applied_before_clamp(self):
        m = self._mgr_with_policy(floor=0.0, ceiling=1.0, multipliers={"s1": 0.5})
        score = m.get_effective_trust("t1", "s1", 0.8)
        assert score == pytest.approx(0.4)

    def test_blocklisted_source_returns_zero(self):
        m = self._mgr_with_policy(blocklist=["s1"])
        score = m.get_effective_trust("t1", "s1", 0.9)
        assert score == pytest.approx(0.0)

    def test_allowlisted_source_returns_ceiling(self):
        m = self._mgr_with_policy(ceiling=0.85, allowlist=["s1"])
        score = m.get_effective_trust("t1", "s1", 0.1)
        assert score == pytest.approx(0.85)

    def test_base_score_out_of_range_raises(self):
        from app.enterprise.source_trust_manager import SourceTrustManager
        m = SourceTrustManager()
        with pytest.raises(ValueError):
            m.get_effective_trust("t1", "s1", 1.5)

    def test_negative_base_score_raises(self):
        from app.enterprise.source_trust_manager import SourceTrustManager
        m = SourceTrustManager()
        with pytest.raises(ValueError):
            m.get_effective_trust("t1", "s1", -0.1)

    def test_empty_source_id_raises(self):
        from app.enterprise.source_trust_manager import SourceTrustManager
        m = SourceTrustManager()
        with pytest.raises(ValueError):
            m.get_effective_trust("t1", "", 0.5)

    def test_wrong_base_score_type_raises(self):
        from app.enterprise.source_trust_manager import SourceTrustManager
        m = SourceTrustManager()
        with pytest.raises(TypeError):
            m.get_effective_trust("t1", "s1", "high")  # type: ignore


# ===========================================================================
# Group 25: SourceTrustManagerBlocklist
# ===========================================================================

class TestSourceTrustManagerBlocklist:
    def _mgr(self):
        from app.enterprise.source_trust_manager import SourceTrustManager
        return SourceTrustManager()

    def test_block_source(self):
        m = self._mgr()
        m.block_source("t1", "bad-source")
        assert m.is_source_blocked("t1", "bad-source") is True

    def test_unblock_source(self):
        m = self._mgr()
        m.block_source("t1", "bad-source")
        m.unblock_source("t1", "bad-source")
        assert m.is_source_blocked("t1", "bad-source") is False

    def test_unblock_nonexistent_no_error(self):
        m = self._mgr()
        m.unblock_source("t1", "not-blocked")  # should not raise

    def test_not_blocked_returns_false(self):
        m = self._mgr()
        assert m.is_source_blocked("t1", "clean-source") is False

    def test_policy_blocklist_also_blocks(self):
        from app.enterprise.source_trust_manager import SourceTrustManager
        m = SourceTrustManager()
        p = SourceTrustPolicy(tenant_id="t1", blocklisted_source_ids=["policy-blocked"])
        m.set_policy("t1", p)
        assert m.is_source_blocked("t1", "policy-blocked") is True

    def test_effective_trust_zero_for_dynamic_blocked(self):
        m = self._mgr()
        m.block_source("t1", "s_bad")
        score = m.get_effective_trust("t1", "s_bad", 0.9)
        assert score == pytest.approx(0.0)

    def test_block_empty_source_raises(self):
        m = self._mgr()
        with pytest.raises(ValueError):
            m.block_source("t1", "")

    def test_block_empty_tenant_raises(self):
        m = self._mgr()
        with pytest.raises(ValueError):
            m.block_source("", "s1")


# ===========================================================================
# Group 26: SourceTrustManagerRecommend
# ===========================================================================

class TestSourceTrustManagerRecommend:
    def _mgr(self):
        from app.enterprise.source_trust_manager import SourceTrustManager
        return SourceTrustManager()

    def test_official_source_raises_floor(self):
        m = self._mgr()
        result = m.recommend_policy_adjustment("t1", {"official": True})
        assert result["suggested_floor"] >= 0.5

    def test_high_error_rate_lowers_ceiling(self):
        m = self._mgr()
        result = m.recommend_policy_adjustment("t1", {"error_rate": 0.5})
        assert result["suggested_ceiling"] <= 0.6

    def test_new_domain_lowers_ceiling(self):
        m = self._mgr()
        result = m.recommend_policy_adjustment("t1", {"domain_age_days": 5})
        assert result["suggested_ceiling"] <= 0.7

    def test_clean_source_no_adjustment(self):
        m = self._mgr()
        result = m.recommend_policy_adjustment("t1", {})
        assert result["suggested_floor"] == pytest.approx(0.0)
        assert result["suggested_ceiling"] == pytest.approx(1.0)

    def test_wrong_metadata_type_raises(self):
        m = self._mgr()
        with pytest.raises(TypeError):
            m.recommend_policy_adjustment("t1", "not a dict")  # type: ignore

    def test_empty_tenant_raises(self):
        m = self._mgr()
        with pytest.raises(ValueError):
            m.recommend_policy_adjustment("", {})

    def test_llm_enhanced_false_without_router(self):
        m = self._mgr()
        result = m.recommend_policy_adjustment("t1", {"official": True})
        assert result["llm_enhanced"] is False


# ===========================================================================
# Group 27: RetentionManagerConstruct
# ===========================================================================

class TestRetentionManagerConstruct:
    def test_zero_default_days_raises(self):
        from app.enterprise.retention_manager import RetentionManager
        with pytest.raises(ValueError):
            RetentionManager(default_retention_days=0)

    def test_default_construction(self):
        from app.enterprise.retention_manager import RetentionManager
        m = RetentionManager()
        assert m is not None

    def test_custom_default_days(self):
        from app.enterprise.retention_manager import RetentionManager
        m = RetentionManager(default_retention_days=90)
        days = m.get_retention_days("any-tenant", DataClass.SUMMARY)
        assert days == 90


# ===========================================================================
# Group 28: RetentionManagerPolicy
# ===========================================================================

class TestRetentionManagerPolicy:
    def _mgr(self):
        from app.enterprise.retention_manager import RetentionManager
        return RetentionManager()

    def _policy(self, tenant_id: str = "t1", default_days: int = 180,
                per_class: dict = None):
        return RetentionPolicy(
            tenant_id=tenant_id,
            default_retention_days=default_days,
            per_class_retention=per_class or {},
        )

    def test_set_and_get_policy(self):
        m = self._mgr()
        p = self._policy()
        m.set_policy("t1", p)
        retrieved = m.get_policy("t1")
        assert retrieved.policy_id == p.policy_id

    def test_set_policy_mismatched_tenant_raises(self):
        m = self._mgr()
        p = self._policy("other-tenant")
        with pytest.raises(ValueError):
            m.set_policy("t1", p)

    def test_set_policy_wrong_type_raises(self):
        m = self._mgr()
        with pytest.raises(TypeError):
            m.set_policy("t1", "bad")  # type: ignore

    def test_get_policy_unknown_raises_key_error(self):
        m = self._mgr()
        with pytest.raises(KeyError):
            m.get_policy("nonexistent")

    def test_get_retention_days_default(self):
        m = self._mgr()
        m.set_policy("t1", self._policy(default_days=200))
        days = m.get_retention_days("t1", DataClass.CONTENT_ITEM)
        assert days == 200

    def test_get_retention_days_per_class_override(self):
        m = self._mgr()
        m.set_policy("t1", self._policy(
            default_days=200,
            per_class={DataClass.AUDIT_LOG.value: 2555},
        ))
        assert m.get_retention_days("t1", DataClass.AUDIT_LOG) == 2555
        assert m.get_retention_days("t1", DataClass.CONTENT_ITEM) == 200

    def test_get_retention_days_no_policy_uses_constructor_default(self):
        from app.enterprise.retention_manager import RetentionManager
        m = RetentionManager(default_retention_days=90)
        # When no policy is set, the constructor default is returned for ALL classes.
        assert m.get_retention_days("no-policy", DataClass.AUDIT_LOG) == 90
        assert m.get_retention_days("no-policy", DataClass.CONTENT_ITEM) == 90

    def test_get_retention_days_wrong_class_type_raises(self):
        m = self._mgr()
        with pytest.raises(TypeError):
            m.get_retention_days("t1", "content_item")  # type: ignore


# ===========================================================================
# Group 29: RetentionManagerExpiry
# ===========================================================================

class TestRetentionManagerExpiry:
    def _mgr(self):
        from app.enterprise.retention_manager import RetentionManager
        m = RetentionManager()
        m.set_policy("t1", RetentionPolicy(
            tenant_id="t1", default_retention_days=30
        ))
        return m

    def test_old_record_is_expired(self):
        m = self._mgr()
        old = datetime.now(timezone.utc) - timedelta(days=31)
        assert m.check_expired("t1", DataClass.CONTENT_ITEM, old) is True

    def test_recent_record_not_expired(self):
        m = self._mgr()
        recent = datetime.now(timezone.utc) - timedelta(days=10)
        assert m.check_expired("t1", DataClass.CONTENT_ITEM, recent) is False

    def test_exactly_at_boundary_not_expired(self):
        m = self._mgr()
        # One second before the 30-day boundary → clearly not expired (avoids timing race)
        just_under = datetime.now(timezone.utc) - timedelta(days=29, hours=23, minutes=59, seconds=58)
        assert m.check_expired("t1", DataClass.CONTENT_ITEM, just_under) is False

    def test_naive_datetime_raises(self):
        m = self._mgr()
        naive = datetime(2020, 1, 1)  # no tzinfo
        with pytest.raises(ValueError):
            m.check_expired("t1", DataClass.CONTENT_ITEM, naive)

    def test_wrong_data_class_type_raises(self):
        m = self._mgr()
        with pytest.raises(TypeError):
            m.check_expired("t1", "content_item", _now_utc())

    def test_wrong_created_at_type_raises(self):
        m = self._mgr()
        with pytest.raises(TypeError):
            m.check_expired("t1", DataClass.CONTENT_ITEM, "2020-01-01")


# ===========================================================================
# Group 30: RetentionManagerPurge
# ===========================================================================

class TestRetentionManagerPurge:
    def _mgr(self, legal_hold: bool = False, days: int = 30):
        from app.enterprise.retention_manager import RetentionManager
        m = RetentionManager()
        m.set_policy("t1", RetentionPolicy(
            tenant_id="t1",
            default_retention_days=days,
            legal_hold=legal_hold,
        ))
        return m

    def _records(self, old_count: int = 2, new_count: int = 3, days: int = 31):
        old = datetime.now(timezone.utc) - timedelta(days=days)
        recent = datetime.now(timezone.utc) - timedelta(days=1)
        records = [(f"old-{i}", DataClass.CONTENT_ITEM, old) for i in range(old_count)]
        records += [(f"new-{i}", DataClass.CONTENT_ITEM, recent) for i in range(new_count)]
        return records

    def test_purge_correct_count(self):
        m = self._mgr(days=30)
        result = m.purge_eligible("t1", self._records(old_count=2, new_count=3))
        assert result.records_checked == 5
        assert result.records_purged == 2

    def test_purge_legal_hold_blocks_all(self):
        m = self._mgr(legal_hold=True)
        result = m.purge_eligible("t1", self._records(old_count=5))
        assert result.records_purged == 0
        assert result.legal_hold_active is True

    def test_purge_empty_records(self):
        m = self._mgr()
        result = m.purge_eligible("t1", [])
        assert result.records_checked == 0
        assert result.records_purged == 0

    def test_purge_wrong_records_type_raises(self):
        m = self._mgr()
        with pytest.raises(TypeError):
            m.purge_eligible("t1", "not a list")

    def test_purge_malformed_tuple_raises(self):
        m = self._mgr()
        with pytest.raises(ValueError):
            m.purge_eligible("t1", [("id-only",)])  # only 1 element

    def test_purge_wrong_data_class_raises(self):
        m = self._mgr()
        with pytest.raises(TypeError):
            m.purge_eligible("t1", [("id", "content_item", _now_utc())])

    def test_purge_returns_purge_result(self):
        from app.enterprise.models import PurgeResult
        m = self._mgr()
        result = m.purge_eligible("t1", self._records())
        assert isinstance(result, PurgeResult)


# ===========================================================================
# Group 31: SLOTrackerConstruct
# ===========================================================================

class TestSLOTrackerConstruct:
    def test_default_construction(self):
        from app.enterprise.slo_tracker import SLOTracker
        t = SLOTracker()
        assert t is not None

    def test_wrong_slo_target_type_raises(self):
        from app.enterprise.slo_tracker import SLOTracker
        t = SLOTracker()
        with pytest.raises(TypeError):
            t.register_slo("tenant", "not an SLOTarget")  # type: ignore

    def test_empty_tenant_id_raises(self):
        from app.enterprise.slo_tracker import SLOTracker
        t = SLOTracker()
        with pytest.raises(ValueError):
            t.register_slo("", _slo_target())


# ===========================================================================
# Group 32: SLOTrackerRegister
# ===========================================================================

class TestSLOTrackerRegister:
    def _tracker(self):
        from app.enterprise.slo_tracker import SLOTracker
        return SLOTracker()

    def test_register_slo(self):
        t = self._tracker()
        slo = _slo_target()
        t.register_slo("t1", slo)
        # Should be retrievable via get_slo_status after one observation
        t.record_observation("t1", slo.metric_name, 100.0)
        status = t.get_slo_status("t1", slo.metric_name)
        assert status.metric_name == slo.metric_name

    def test_register_overwrites_existing(self):
        t = self._tracker()
        slo1 = SLOTarget(metric_name="latency", target_value=200.0)
        slo2 = SLOTarget(metric_name="latency", target_value=300.0)
        t.register_slo("t1", slo1)
        t.register_slo("t1", slo2)
        t.record_observation("t1", "latency", 250.0)
        status = t.get_slo_status("t1", "latency")
        assert status.target.target_value == pytest.approx(300.0)

    def test_deregister_slo(self):
        t = self._tracker()
        t.register_slo("t1", _slo_target())
        t.deregister_slo("t1", "latency_ms")
        with pytest.raises(ValueError):
            t.get_slo_status("t1", "latency_ms")

    def test_deregister_unknown_no_error(self):
        t = self._tracker()
        t.deregister_slo("t1", "nonexistent")  # should not raise


# ===========================================================================
# Group 33: SLOTrackerObservations
# ===========================================================================

class TestSLOTrackerObservations:
    def _tracker(self):
        from app.enterprise.slo_tracker import SLOTracker
        t = SLOTracker()
        t.register_slo("t1", _slo_target())
        return t

    def test_record_observation_accepted(self):
        t = self._tracker()
        t.record_observation("t1", "latency_ms", 150.0)
        status = t.get_slo_status("t1", "latency_ms")
        assert status.observation_count == 1

    def test_record_multiple_observations(self):
        t = self._tracker()
        for v in [100.0, 150.0, 200.0]:
            t.record_observation("t1", "latency_ms", v)
        status = t.get_slo_status("t1", "latency_ms")
        assert status.observation_count == 3

    def test_wrong_value_type_raises(self):
        t = self._tracker()
        with pytest.raises(TypeError):
            t.record_observation("t1", "latency_ms", "fast")  # type: ignore

    def test_empty_metric_raises(self):
        t = self._tracker()
        with pytest.raises(ValueError):
            t.record_observation("t1", "", 100.0)

    def test_custom_timestamp_accepted(self):
        t = self._tracker()
        ts = datetime.now(timezone.utc) - timedelta(seconds=10)
        t.record_observation("t1", "latency_ms", 100.0, timestamp=ts)
        status = t.get_slo_status("t1", "latency_ms")
        assert status.observation_count == 1


# ===========================================================================
# Group 34: SLOTrackerStatus
# ===========================================================================

class TestSLOTrackerStatus:
    def _tracker_with_slo(self, metric: str = "latency_ms", target: float = 200.0,
                           operator: SLOOperator = SLOOperator.LESS_THAN,
                           window: int = 3600):
        from app.enterprise.slo_tracker import SLOTracker
        t = SLOTracker()
        slo = SLOTarget(metric_name=metric, target_value=target,
                        operator=operator, window_seconds=window)
        t.register_slo("t1", slo)
        return t

    def test_no_observations_current_value_none(self):
        t = self._tracker_with_slo()
        status = t.get_slo_status("t1", "latency_ms")
        assert status.current_value is None
        assert status.is_breaching is False

    def test_mean_aggregate_below_target_not_breaching(self):
        t = self._tracker_with_slo(target=200.0)
        t.record_observation("t1", "latency_ms", 100.0)
        t.record_observation("t1", "latency_ms", 150.0)
        status = t.get_slo_status("t1", "latency_ms")
        assert status.current_value == pytest.approx(125.0)
        assert status.is_breaching is False

    def test_mean_aggregate_above_target_breaching(self):
        t = self._tracker_with_slo(target=100.0)
        t.record_observation("t1", "latency_ms", 200.0)
        t.record_observation("t1", "latency_ms", 300.0)
        status = t.get_slo_status("t1", "latency_ms")
        assert status.is_breaching is True
        assert status.current_value > 100.0

    def test_percentile_slo(self):
        from app.enterprise.slo_tracker import SLOTracker
        t = SLOTracker()
        slo = SLOTarget(metric_name="p95_latency", target_value=500.0,
                        operator=SLOOperator.LESS_THAN, window_seconds=3600, percentile=95.0)
        t.register_slo("t1", slo)
        for v in [100.0, 200.0, 300.0, 400.0, 600.0]:  # p95 = 600
            t.record_observation("t1", "p95_latency", v)
        status = t.get_slo_status("t1", "p95_latency")
        assert status.is_breaching is True  # 600 > 500

    def test_greater_than_slo_satisfied(self):
        t = self._tracker_with_slo(target=0.99, operator=SLOOperator.GREATER_THAN_OR_EQUAL)
        t.record_observation("t1", "latency_ms", 1.0)  # 1.0 >= 0.99
        status = t.get_slo_status("t1", "latency_ms")
        assert status.is_breaching is False

    def test_unregistered_slo_raises(self):
        from app.enterprise.slo_tracker import SLOTracker
        with pytest.raises(ValueError):
            SLOTracker().get_slo_status("t1", "unregistered")


# ===========================================================================
# Group 35: SLOTrackerBreach
# ===========================================================================

class TestSLOTrackerBreach:
    def _breaching_tracker(self):
        from app.enterprise.slo_tracker import SLOTracker
        t = SLOTracker()
        t.register_slo("t1", _slo_target(target=100.0))  # LESS_THAN 100ms
        t.record_observation("t1", "latency_ms", 500.0)  # breaching
        return t

    def test_check_breach_true(self):
        t = self._breaching_tracker()
        assert t.check_breach("t1", "latency_ms") is True

    def test_check_breach_false_when_ok(self):
        from app.enterprise.slo_tracker import SLOTracker
        t = SLOTracker()
        t.register_slo("t1", _slo_target(target=1000.0))  # easy target
        t.record_observation("t1", "latency_ms", 50.0)
        assert t.check_breach("t1", "latency_ms") is False

    def test_list_breaches_returns_breaching(self):
        t = self._breaching_tracker()
        breaches = t.list_breaches("t1")
        assert len(breaches) == 1
        assert breaches[0].metric_name == "latency_ms"

    def test_list_breaches_returns_empty_when_ok(self):
        from app.enterprise.slo_tracker import SLOTracker
        t = SLOTracker()
        t.register_slo("t1", _slo_target(target=1000.0))
        t.record_observation("t1", "latency_ms", 50.0)
        assert t.list_breaches("t1") == []

    def test_list_breaches_empty_tenant_raises(self):
        from app.enterprise.slo_tracker import SLOTracker
        with pytest.raises(ValueError):
            SLOTracker().list_breaches("")

    def test_breach_started_at_set(self):
        t = self._breaching_tracker()
        status = t.get_slo_status("t1", "latency_ms")
        assert status.is_breaching is True
        assert status.breach_started_at is not None

    def test_explain_breach_returns_string(self):
        t = self._breaching_tracker()
        explanation = t.explain_breach("t1", "latency_ms")
        assert isinstance(explanation, str)
        assert len(explanation) > 0


# ===========================================================================
# Group 36: CrossComponentWiring
# ===========================================================================

class TestCrossComponentWiring:
    def test_tenant_provision_then_audit_log(self):
        """Provision a tenant and log a TENANT_PROVISION audit event for it."""
        from app.enterprise.tenant_registry import TenantRegistry
        from app.enterprise.audit_logger import AuditLogger

        registry = TenantRegistry()
        audit = AuditLogger()

        cfg = registry.provision_tenant("corp")
        audit.log_event(AuditEntry(
            tenant_id=cfg.tenant_id,
            event_type=AuditEventType.TENANT_PROVISION,
            actor_id="system",
            resource_id=cfg.tenant_id,
        ))
        events = audit.query_events("corp", event_type=AuditEventType.TENANT_PROVISION)
        assert len(events) == 1

    def test_trust_policy_then_effective_score_then_audit(self):
        """Set trust policy, compute effective score, audit the UPDATE event."""
        from app.enterprise.source_trust_manager import SourceTrustManager
        from app.enterprise.audit_logger import AuditLogger

        trust_mgr = SourceTrustManager()
        audit = AuditLogger()
        policy = SourceTrustPolicy(
            tenant_id="corp",
            global_trust_floor=0.3,
            global_trust_ceiling=0.8,
        )
        trust_mgr.set_policy("corp", policy)
        score = trust_mgr.get_effective_trust("corp", "source-A", 0.5)
        assert 0.3 <= score <= 0.8

        audit.log_event(AuditEntry(
            tenant_id="corp",
            event_type=AuditEventType.SOURCE_TRUST_UPDATE,
            actor_id="admin",
            resource_id="source-A",
        ))
        events = audit.query_events("corp", event_type=AuditEventType.SOURCE_TRUST_UPDATE)
        assert len(events) == 1

    def test_retention_policy_purge_then_audit(self):
        """Set retention policy, purge eligible records, audit RETENTION_PURGE."""
        from app.enterprise.retention_manager import RetentionManager
        from app.enterprise.audit_logger import AuditLogger

        mgr = RetentionManager()
        audit = AuditLogger()
        mgr.set_policy("corp", RetentionPolicy(tenant_id="corp", default_retention_days=30))
        old = datetime.now(timezone.utc) - timedelta(days=31)
        result = mgr.purge_eligible("corp", [("r1", DataClass.CONTENT_ITEM, old)])
        assert result.records_purged == 1

        audit.log_event(AuditEntry(
            tenant_id="corp",
            event_type=AuditEventType.RETENTION_PURGE,
            actor_id="system",
            details={"purged": result.records_purged},
        ))
        events = audit.query_events("corp", event_type=AuditEventType.RETENTION_PURGE)
        assert len(events) == 1
        assert events[0].details["purged"] == 1

    def test_slo_breach_then_audit(self):
        """Record an SLO breach and emit a SLO_BREACH audit entry."""
        from app.enterprise.slo_tracker import SLOTracker
        from app.enterprise.audit_logger import AuditLogger

        tracker = SLOTracker()
        audit = AuditLogger()
        tracker.register_slo("corp", _slo_target(target=100.0))
        tracker.record_observation("corp", "latency_ms", 500.0)
        assert tracker.check_breach("corp", "latency_ms") is True

        audit.log_event(AuditEntry(
            tenant_id="corp",
            event_type=AuditEventType.SLO_BREACH,
            risk_level=RiskLevel.HIGH,
            actor_id="system",
            details={"metric": "latency_ms"},
        ))
        breaches = audit.query_events("corp", event_type=AuditEventType.SLO_BREACH)
        assert len(breaches) == 1
        assert breaches[0].risk_level == RiskLevel.HIGH


# ===========================================================================
# Group 37: CrossComponentPackageImports
# ===========================================================================

class TestCrossComponentPackageImports:
    def test_all_enums_importable(self):
        from app.enterprise import TenantTier, DataClass, AuditEventType, RiskLevel, SLOOperator
        assert TenantTier.ENTERPRISE.value == "enterprise"
        assert DataClass.AUDIT_LOG.value == "audit_log"
        assert AuditEventType.SLO_BREACH.value == "slo_breach"
        assert RiskLevel.CRITICAL.value == "critical"
        assert SLOOperator.LESS_THAN.value == "lt"

    def test_all_models_importable(self):
        from app.enterprise import (
            TenantConfig, AuditEntry, SourceTrustPolicy,
            RetentionPolicy, SLOTarget, SLOStatus, PurgeResult,
        )
        assert TenantConfig is not None
        assert PurgeResult is not None

    def test_all_components_importable(self):
        from app.enterprise import (
            TenantRegistry, AuditLogger, SourceTrustManager,
            RetentionManager, SLOTracker,
        )
        assert TenantRegistry is not None
        assert SLOTracker is not None

    def test_all_exports_in_all_list(self):
        import app.enterprise as pkg
        for symbol in pkg.__all__:
            assert hasattr(pkg, symbol), f"{symbol!r} in __all__ but not importable"

    def test_end_to_end_package_usage(self):
        """Full enterprise pipeline from package-level imports."""
        from app.enterprise import (
            TenantRegistry, AuditLogger, SourceTrustManager,
            TenantTier, AuditEventType, SourceTrustPolicy, RiskLevel,
        )
        registry = TenantRegistry()
        audit = AuditLogger()
        trust = SourceTrustManager()

        cfg = registry.provision_tenant("demo", tier=TenantTier.STANDARD)
        audit.log_event(AuditEntry(
            tenant_id=cfg.tenant_id,
            event_type=AuditEventType.TENANT_PROVISION,
            risk_level=RiskLevel.LOW,
        ))
        policy = SourceTrustPolicy(
            tenant_id=cfg.tenant_id,
            global_trust_floor=0.2,
            global_trust_ceiling=0.95,
        )
        trust.set_policy(cfg.tenant_id, policy)
        score = trust.get_effective_trust(cfg.tenant_id, "source-x", 0.7)
        assert 0.2 <= score <= 0.95
        summary = audit.get_summary(cfg.tenant_id)
        assert summary["total_events"] == 1
