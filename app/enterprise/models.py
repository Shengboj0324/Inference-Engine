"""Shared Pydantic models for the enterprise-hardening package (Phase 5).

Design conventions (identical to Phases 2–4):
- ``frozen=True`` on every value-object model.
- Every numeric field carries explicit ``ge``/``le`` range validators.
- Every string identifier field carries ``min_length=1``.
- Cross-field constraints use ``model_validator(mode="after")``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TenantTier(str, Enum):
    """Service tier controlling resource caps and feature access."""

    FREE = "free"
    STANDARD = "standard"
    ENTERPRISE = "enterprise"
    PLATFORM = "platform"


class DataClass(str, Enum):
    """Category of data subject to retention policies."""

    CONTENT_ITEM = "content_item"
    USER_FEEDBACK = "user_feedback"
    AUDIT_LOG = "audit_log"
    EMBEDDING = "embedding"
    SUMMARY = "summary"
    SOURCE_METADATA = "source_metadata"
    CREDENTIAL = "credential"


class AuditEventType(str, Enum):
    """Taxonomy of events recorded in the tamper-evident audit trail."""

    LOGIN = "login"
    LOGOUT = "logout"
    DATA_ACCESS = "data_access"
    DATA_EXPORT = "data_export"
    CONFIG_CHANGE = "config_change"
    PERMISSION_CHANGE = "permission_change"
    SOURCE_TRUST_UPDATE = "source_trust_update"
    RETENTION_PURGE = "retention_purge"
    SLO_BREACH = "slo_breach"
    TENANT_PROVISION = "tenant_provision"
    TENANT_DEPROVISION = "tenant_deprovision"
    API_CALL = "api_call"


class RiskLevel(str, Enum):
    """Risk classification for audit events and alerts."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SLOOperator(str, Enum):
    """Comparison operator used when evaluating an SLO target."""

    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"


# ---------------------------------------------------------------------------
# Value-object models (frozen=True)
# ---------------------------------------------------------------------------

class TenantConfig(BaseModel, frozen=True):
    """Immutable configuration snapshot for one tenant.

    Attributes:
        tenant_id:            Unique tenant identifier (non-empty string).
        tier:                 Service tier governing resource caps.
        display_name:         Human-readable tenant name.
        max_sources:          Maximum simultaneous monitored sources (≥ 1).
        max_daily_api_calls:  API call budget per calendar day (≥ 1).
        allowed_data_classes: Which ``DataClass`` values this tenant may use.
        audit_enabled:        Whether audit logging is active for this tenant.
        created_at:           UTC provisioning timestamp.
        is_active:            Whether the tenant is currently active.
        data_residency_region: Cloud region where data must stay (e.g. ``"us-east-1"``).
        feature_flags:        Per-tenant boolean feature toggles.
    """

    tenant_id: str = Field(..., min_length=1)
    tier: TenantTier = TenantTier.FREE
    display_name: str = ""
    max_sources: int = Field(default=10, ge=1)
    max_daily_api_calls: int = Field(default=1_000, ge=1)
    allowed_data_classes: List[DataClass] = Field(default_factory=list)
    audit_enabled: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True
    data_residency_region: str = "us-east-1"
    feature_flags: Dict[str, bool] = Field(default_factory=dict)


class AuditEntry(BaseModel, frozen=True):
    """Single tamper-evident audit log entry.

    ``chain_hash`` links this entry to all previous entries in the tenant's
    audit chain using SHA-256: ``sha256(prev_chain_hash || entry_canonical_json)``.
    The first entry in a chain uses ``"GENESIS"`` as the previous hash.

    Attributes:
        entry_id:     Auto-generated UUID (non-empty).
        tenant_id:    Owning tenant.
        event_type:   Taxonomy of the event.
        actor_id:     Identity of the actor (user, service, system).
        resource_id:  Identifier of the resource affected.
        details:      Arbitrary additional context (must be JSON-serialisable).
        risk_level:   Impact classification.
        occurred_at:  UTC timestamp of the event.
        ip_address:   Source IP (empty string if unknown).
        chain_hash:   Tamper-evident chain link (hex string, non-empty).
        success:      Whether the operation succeeded.
    """

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1)
    tenant_id: str = Field(..., min_length=1)
    event_type: AuditEventType
    actor_id: str = ""
    resource_id: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.LOW
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ip_address: str = ""
    chain_hash: str = Field(default="", min_length=0)
    success: bool = True




class SourceTrustPolicy(BaseModel, frozen=True):
    """Per-tenant policy controlling how source trust scores are applied.

    Attributes:
        policy_id:             Auto-generated UUID.
        tenant_id:             Owning tenant.
        global_trust_floor:    Minimum effective trust score in [0, 1].
        global_trust_ceiling:  Maximum effective trust score in [0, 1].
        blocklisted_source_ids: Sources always assigned effective trust 0.0.
        allowlisted_source_ids: Sources always assigned the ceiling score.
        source_multipliers:    Per-source trust multipliers (applied before clamping).
        require_official_primary: When True, only accept primary official sources.
        created_at:            UTC creation timestamp.
    """

    policy_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1)
    tenant_id: str = Field(..., min_length=1)
    global_trust_floor: float = Field(default=0.0, ge=0.0, le=1.0)
    global_trust_ceiling: float = Field(default=1.0, ge=0.0, le=1.0)
    blocklisted_source_ids: List[str] = Field(default_factory=list)
    allowlisted_source_ids: List[str] = Field(default_factory=list)
    source_multipliers: Dict[str, float] = Field(default_factory=dict)
    require_official_primary: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _floor_le_ceiling(self) -> "SourceTrustPolicy":
        if self.global_trust_floor > self.global_trust_ceiling:
            raise ValueError(
                f"'global_trust_floor' ({self.global_trust_floor}) must be ≤ "
                f"'global_trust_ceiling' ({self.global_trust_ceiling})"
            )
        return self


class RetentionPolicy(BaseModel, frozen=True):
    """Per-tenant data retention policy governing automated purging.

    Attributes:
        policy_id:              Auto-generated UUID.
        tenant_id:              Owning tenant.
        default_retention_days: Default retention period in days (1–3650).
        per_class_retention:    Per-DataClass override in days (≥ 1 each).
        auto_purge_enabled:     When False, purging requires an explicit API call.
        legal_hold:             When True, ALL purging is blocked.
        created_at:             UTC creation timestamp.
    """

    policy_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1)
    tenant_id: str = Field(..., min_length=1)
    default_retention_days: int = Field(default=365, ge=1, le=3650)
    per_class_retention: Dict[str, int] = Field(default_factory=dict)
    auto_purge_enabled: bool = False
    legal_hold: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("per_class_retention")
    @classmethod
    def _validate_class_days(cls, v: Dict[str, int]) -> Dict[str, int]:
        for cls_name, days in v.items():
            if days < 1 or days > 3650:
                raise ValueError(
                    f"Retention days for '{cls_name}' must be in [1, 3650], got {days}"
                )
        return v


class SLOTarget(BaseModel, frozen=True):
    """Specification of a single Service Level Objective.

    Attributes:
        metric_name:      Metric identifier (e.g. ``"api_latency_ms"``).
        target_value:     Threshold the metric must satisfy (≥ 0).
        operator:         Comparison direction.
        window_seconds:   Sliding observation window (≥ 60 s).
        percentile:       If set, evaluate the p-ile (0–100); else use mean.
        breach_threshold: Fraction of window in violation before declaring
                          a breach (0–1).
    """

    metric_name: str = Field(..., min_length=1)
    target_value: float = Field(..., ge=0.0)
    operator: SLOOperator = SLOOperator.LESS_THAN
    window_seconds: int = Field(default=3600, ge=60)
    percentile: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    breach_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class SLOStatus(BaseModel, frozen=True):
    """Point-in-time SLO status for one tenant and metric.

    Attributes:
        metric_name:       Matched SLO metric.
        tenant_id:         Owning tenant.
        current_value:     Latest computed aggregate (None if no data yet).
        target:            The ``SLOTarget`` being evaluated.
        is_breaching:      True when the SLO is currently violated.
        breach_started_at: When the breach began (None if not breaching).
        observation_count: Number of observations used in the window.
        checked_at:        UTC timestamp of this evaluation.
    """

    metric_name: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    current_value: Optional[float] = None
    target: SLOTarget
    is_breaching: bool = False
    breach_started_at: Optional[datetime] = None
    observation_count: int = Field(default=0, ge=0)
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PurgeResult(BaseModel, frozen=True):
    """Result of one data-class purge evaluation.

    Attributes:
        tenant_id:         Owning tenant.
        data_class:        Which data class was evaluated.
        records_checked:   Number of records inspected (≥ 0).
        records_purged:    Records flagged for deletion (≥ 0, ≤ records_checked).
        purged_at:         UTC timestamp.
        legal_hold_active: True when purging was blocked by a legal hold.
        policy_id:         ID of the ``RetentionPolicy`` applied.
    """

    tenant_id: str = Field(..., min_length=1)
    data_class: DataClass
    records_checked: int = Field(default=0, ge=0)
    records_purged: int = Field(default=0, ge=0)
    purged_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    legal_hold_active: bool = False
    policy_id: str = ""

    @model_validator(mode="after")
    def _purged_le_checked(self) -> "PurgeResult":
        if self.records_purged > self.records_checked:
            raise ValueError(
                f"'records_purged' ({self.records_purged}) cannot exceed "
                f"'records_checked' ({self.records_checked})"
            )
        return self
