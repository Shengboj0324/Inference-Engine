"""Source Coverage Graph — Phase 6.

Tracks which sources cover which entities and computes coverage
completeness, freshness, gap severity, and derivative over-reliance.

Key concepts
------------
* **Entity**       — A tracked real-world object (company, product, repo, …).
* **SourceFamily** — The type of source providing coverage (news, research, …).
* **Freshness**    — How recently a source produced content for an entity.
* **Completeness** — Fraction of required source families that are covered.
* **Derivative**   — A source that re-publishes rather than breaks original news.

Public API
----------
    EntityCategory       — enum of tracked entity types
    SourceFreshnessEntry — per-(entity, source) freshness record
    EntityCoverageScore  — frozen snapshot of one entity's coverage quality
    SourceCoverageGraph  — main component (thread-safe)
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EntityCategory(str, Enum):
    """High-level category of a tracked entity."""

    COMPANY = "company"
    PRODUCT = "product"
    PERSON = "person"
    REPO = "repo"
    PODCAST = "podcast"
    PAPER = "paper"
    OFFICIAL_BLOG = "official_blog"
    DOCS_SITE = "docs_site"
    CHANGELOG = "changelog"
    NEWS = "news"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class SourceFreshnessEntry(BaseModel):
    """Freshness record for one (entity, source_id) pair.

    Attributes:
        entity_name:       Canonical entity name.
        source_id:         Source identifier.
        last_fetched_at:   UTC timestamp of the last successful fetch.
        is_derivative:     True when the source re-publishes rather than breaks news.
        item_count:        Total items seen from this source for this entity.
    """

    model_config = {"frozen": True}

    entity_name: str = Field(..., min_length=1)
    source_id: str = Field(..., min_length=1)
    last_fetched_at: datetime
    is_derivative: bool = False
    item_count: int = Field(default=0, ge=0)

    @field_validator("last_fetched_at")
    @classmethod
    def _must_be_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("'last_fetched_at' must be timezone-aware (UTC)")
        return v


class EntityCoverageScore(BaseModel):
    """Frozen snapshot of one entity's coverage quality.

    Attributes:
        entity_name:          Canonical entity name.
        category:             EntityCategory of this entity.
        total_sources:        Number of distinct sources attached.
        stale_sources:        Sources whose freshness exceeds the staleness threshold.
        derivative_sources:   Sources flagged as derivative (re-publishers).
        completeness:         Fraction [0, 1] of required families covered.
        derivative_ratio:     Fraction [0, 1] of sources that are derivative.
        gap_count:            Number of required source families not yet covered.
        computed_at:          UTC timestamp when this score was computed.
    """

    model_config = {"frozen": True}

    entity_name: str = Field(..., min_length=1)
    category: EntityCategory
    total_sources: int = Field(ge=0)
    stale_sources: int = Field(ge=0)
    derivative_sources: int = Field(ge=0)
    completeness: float = Field(ge=0.0, le=1.0)
    derivative_ratio: float = Field(ge=0.0, le=1.0)
    gap_count: int = Field(ge=0)
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _counts_le_total(self) -> "EntityCoverageScore":
        if self.stale_sources > self.total_sources:
            raise ValueError(
                f"'stale_sources' ({self.stale_sources}) cannot exceed "
                f"'total_sources' ({self.total_sources})"
            )
        if self.derivative_sources > self.total_sources:
            raise ValueError(
                f"'derivative_sources' ({self.derivative_sources}) cannot exceed "
                f"'total_sources' ({self.total_sources})"
            )
        return self


# Required source families per entity category — used for completeness scoring.
_REQUIRED_FAMILIES_BY_CATEGORY: Dict[EntityCategory, FrozenSet[str]] = {
    EntityCategory.COMPANY:       frozenset({"news", "official_blog", "social"}),
    EntityCategory.PRODUCT:       frozenset({"developer_release", "docs_site", "changelog"}),
    EntityCategory.REPO:          frozenset({"developer_release", "changelog"}),
    EntityCategory.PAPER:         frozenset({"research"}),
    EntityCategory.PERSON:        frozenset({"social", "media_audio"}),
    EntityCategory.PODCAST:       frozenset({"media_audio"}),
    EntityCategory.OFFICIAL_BLOG: frozenset({"official_blog"}),
    EntityCategory.DOCS_SITE:     frozenset({"docs_site"}),
    EntityCategory.CHANGELOG:     frozenset({"changelog"}),
    EntityCategory.NEWS:          frozenset({"news"}),
}

_DEFAULT_STALENESS_HOURS = 48.0


# ---------------------------------------------------------------------------
# SourceCoverageGraph
# ---------------------------------------------------------------------------

class SourceCoverageGraph:
    """Thread-safe graph tracking source coverage across entities.

    Args:
        staleness_threshold_hours: Hours after which a source is considered stale.
        derivative_overreliance_ratio: Fraction of derivative sources above which
            an entity is flagged for over-reliance.

    Raises:
        ValueError: If numeric arguments are out of valid range.
    """

    def __init__(
        self,
        staleness_threshold_hours: float = _DEFAULT_STALENESS_HOURS,
        derivative_overreliance_ratio: float = 0.6,
    ) -> None:
        if not isinstance(staleness_threshold_hours, (int, float)):
            raise TypeError(
                f"'staleness_threshold_hours' must be numeric, got {type(staleness_threshold_hours)!r}"
            )
        if staleness_threshold_hours <= 0:
            raise ValueError(
                f"'staleness_threshold_hours' must be > 0, got {staleness_threshold_hours!r}"
            )
        if not isinstance(derivative_overreliance_ratio, (int, float)):
            raise TypeError(
                f"'derivative_overreliance_ratio' must be numeric, got {type(derivative_overreliance_ratio)!r}"
            )
        if not (0.0 < derivative_overreliance_ratio <= 1.0):
            raise ValueError(
                f"'derivative_overreliance_ratio' must be in (0, 1], got {derivative_overreliance_ratio!r}"
            )
        self._staleness_hours = float(staleness_threshold_hours)
        self._overreliance_ratio = float(derivative_overreliance_ratio)
        # entity_name → EntityCategory
        self._entities: Dict[str, EntityCategory] = {}
        # (entity_name, source_id) → SourceFreshnessEntry
        self._freshness: Dict[tuple, SourceFreshnessEntry] = {}
        # entity_name → set of source_ids attached
        self._entity_sources: Dict[str, Set[str]] = {}
        # source_id → family name (for completeness computation)
        self._source_families: Dict[str, str] = {}
        # source_id → is_derivative flag set at attach time
        self._source_is_derivative: Dict[str, bool] = {}
        self._lock = threading.RLock()
        logger.info("SourceCoverageGraph created (staleness=%.1fh, overreliance=%.0f%%)",
                    self._staleness_hours, self._overreliance_ratio * 100)

    # ------------------------------------------------------------------
    # Entity management
    # ------------------------------------------------------------------

    def add_entity(self, entity_name: str, category: EntityCategory) -> None:
        """Register an entity in the graph.

        Args:
            entity_name: Canonical entity name (non-empty string).
            category:    EntityCategory for coverage requirement lookup.

        Raises:
            TypeError:  If arguments have wrong types.
            ValueError: If entity_name is empty.
        """
        if not isinstance(entity_name, str):
            raise TypeError(f"'entity_name' must be str, got {type(entity_name)!r}")
        if not entity_name.strip():
            raise ValueError("'entity_name' must be a non-empty string")
        if not isinstance(category, EntityCategory):
            raise TypeError(f"'category' must be EntityCategory, got {type(category)!r}")
        key = entity_name.strip()
        with self._lock:
            if key not in self._entities:
                self._entities[key] = category
                self._entity_sources[key] = set()
                logger.debug("add_entity: %r (%s)", key, category.value)

    def remove_entity(self, entity_name: str) -> None:
        """Remove an entity and all its source attachments.

        Args:
            entity_name: Entity to remove (silently ignored if unknown).

        Raises:
            TypeError:  If entity_name is not a string.
        """
        if not isinstance(entity_name, str):
            raise TypeError(f"'entity_name' must be str, got {type(entity_name)!r}")
        key = entity_name.strip()
        with self._lock:
            if key in self._entities:
                del self._entities[key]
                # Remove all freshness entries for this entity
                stale_keys = [k for k in self._freshness if k[0] == key]
                for sk in stale_keys:
                    del self._freshness[sk]
                self._entity_sources.pop(key, None)
                logger.debug("remove_entity: %r", key)

    def list_entities(self) -> List[str]:
        """Return a sorted list of all tracked entity names."""
        with self._lock:
            return sorted(self._entities.keys())

    # ------------------------------------------------------------------
    # Source attachment
    # ------------------------------------------------------------------

    def attach_source(
        self,
        entity_name: str,
        source_id: str,
        family: str,
        is_derivative: bool = False,
    ) -> None:
        """Attach a source to an entity.

        Idempotent: calling twice for the same (entity, source) pair is safe.

        Args:
            entity_name:   Entity to attach the source to.
            source_id:     Unique source identifier.
            family:        Source family name (e.g. ``"news"``, ``"research"``).
            is_derivative: True when the source re-publishes original news.

        Raises:
            TypeError:  Wrong argument types.
            ValueError: entity_name not registered or empty source_id.
        """
        if not isinstance(entity_name, str):
            raise TypeError(f"'entity_name' must be str, got {type(entity_name)!r}")
        if not isinstance(source_id, str):
            raise TypeError(f"'source_id' must be str, got {type(source_id)!r}")
        if not isinstance(family, str):
            raise TypeError(f"'family' must be str, got {type(family)!r}")
        if not isinstance(is_derivative, bool):
            raise TypeError(f"'is_derivative' must be bool, got {type(is_derivative)!r}")
        if not source_id.strip():
            raise ValueError("'source_id' must be a non-empty string")
        if not family.strip():
            raise ValueError("'family' must be a non-empty string")
        entity_key = entity_name.strip()
        sid = source_id.strip()
        with self._lock:
            if entity_key not in self._entities:
                raise ValueError(f"Entity {entity_name!r} is not registered; call add_entity() first")
            self._entity_sources[entity_key].add(sid)
            self._source_families[sid] = family.strip()
            self._source_is_derivative[sid] = is_derivative
            logger.debug("attach_source: entity=%r source=%r family=%r derivative=%s",
                         entity_key, sid, family, is_derivative)

    def detach_source(self, entity_name: str, source_id: str) -> None:
        """Detach a source from an entity (silently ignored if absent).

        Raises:
            TypeError: Wrong argument types.
        """
        if not isinstance(entity_name, str):
            raise TypeError(f"'entity_name' must be str, got {type(entity_name)!r}")
        if not isinstance(source_id, str):
            raise TypeError(f"'source_id' must be str, got {type(source_id)!r}")
        entity_key = entity_name.strip()
        sid = source_id.strip()
        with self._lock:
            if entity_key in self._entity_sources:
                self._entity_sources[entity_key].discard(sid)
            fkey = (entity_key, sid)
            self._freshness.pop(fkey, None)
            # Only remove derivative flag if no other entity still uses this source
            all_other = any(
                sid in srcs
                for ent, srcs in self._entity_sources.items()
                if ent != entity_key
            )
            if not all_other:
                self._source_is_derivative.pop(sid, None)
            logger.debug("detach_source: entity=%r source=%r", entity_key, sid)



    # ------------------------------------------------------------------
    # Freshness tracking
    # ------------------------------------------------------------------

    def record_fetch(
        self,
        entity_name: str,
        source_id: str,
        fetched_at: Optional[datetime] = None,
        is_derivative: bool = False,
    ) -> None:
        """Record a successful fetch for a (entity, source) pair.

        Args:
            entity_name: Canonical entity name (must be registered).
            source_id:   Source identifier (must be attached).
            fetched_at:  UTC timestamp; defaults to now.
            is_derivative: True if this source re-publishes content.

        Raises:
            TypeError:  Wrong argument types.
            ValueError: entity_name not registered, source_id not attached,
                        or naive datetime provided.
        """
        if not isinstance(entity_name, str):
            raise TypeError(f"'entity_name' must be str, got {type(entity_name)!r}")
        if not isinstance(source_id, str):
            raise TypeError(f"'source_id' must be str, got {type(source_id)!r}")
        entity_key = entity_name.strip()
        sid = source_id.strip()
        if fetched_at is None:
            fetched_at = datetime.now(timezone.utc)
        if not isinstance(fetched_at, datetime):
            raise TypeError(f"'fetched_at' must be datetime, got {type(fetched_at)!r}")
        if fetched_at.tzinfo is None:
            raise ValueError("'fetched_at' must be timezone-aware (UTC)")
        with self._lock:
            if entity_key not in self._entities:
                raise ValueError(f"Entity {entity_name!r} is not registered")
            if sid not in self._entity_sources.get(entity_key, set()):
                raise ValueError(
                    f"Source {source_id!r} is not attached to entity {entity_name!r}; "
                    "call attach_source() first"
                )
            fkey = (entity_key, sid)
            existing = self._freshness.get(fkey)
            item_count = (existing.item_count if existing else 0) + 1
            self._freshness[fkey] = SourceFreshnessEntry(
                entity_name=entity_key,
                source_id=sid,
                last_fetched_at=fetched_at,
                is_derivative=is_derivative,
                item_count=item_count,
            )
            logger.debug("record_fetch: entity=%r source=%r ts=%s", entity_key, sid, fetched_at.isoformat())

    def get_freshness(self, entity_name: str, source_id: str) -> Optional[SourceFreshnessEntry]:
        """Return the freshness entry for a (entity, source) pair, or None."""
        if not isinstance(entity_name, str):
            raise TypeError(f"'entity_name' must be str, got {type(entity_name)!r}")
        if not isinstance(source_id, str):
            raise TypeError(f"'source_id' must be str, got {type(source_id)!r}")
        with self._lock:
            return self._freshness.get((entity_name.strip(), source_id.strip()))

    def stale_sources(
        self, entity_name: str, reference_time: Optional[datetime] = None
    ) -> List[str]:
        """Return source IDs for a given entity that are past the staleness threshold.

        Args:
            entity_name:    Entity to inspect.
            reference_time: UTC now-reference; defaults to ``datetime.now(UTC)``.

        Raises:
            TypeError:  Wrong argument type.
            ValueError: entity_name not registered.
        """
        if not isinstance(entity_name, str):
            raise TypeError(f"'entity_name' must be str, got {type(entity_name)!r}")
        entity_key = entity_name.strip()
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)
        if not isinstance(reference_time, datetime):
            raise TypeError(f"'reference_time' must be datetime, got {type(reference_time)!r}")
        if reference_time.tzinfo is None:
            raise ValueError("'reference_time' must be timezone-aware")
        threshold = timedelta(hours=self._staleness_hours)
        stale: List[str] = []
        with self._lock:
            if entity_key not in self._entities:
                raise ValueError(f"Entity {entity_name!r} is not registered")
            for sid in self._entity_sources.get(entity_key, set()):
                entry = self._freshness.get((entity_key, sid))
                if entry is None or (reference_time - entry.last_fetched_at) > threshold:
                    stale.append(sid)
        return sorted(stale)

    # ------------------------------------------------------------------
    # Coverage scoring
    # ------------------------------------------------------------------

    def coverage_score(self, entity_name: str) -> EntityCoverageScore:
        """Compute a full coverage score for one entity.

        Args:
            entity_name: Must be registered.

        Raises:
            TypeError:  Wrong argument type.
            ValueError: entity_name not registered.
        """
        if not isinstance(entity_name, str):
            raise TypeError(f"'entity_name' must be str, got {type(entity_name)!r}")
        entity_key = entity_name.strip()
        now = datetime.now(timezone.utc)
        threshold = timedelta(hours=self._staleness_hours)
        with self._lock:
            if entity_key not in self._entities:
                raise ValueError(f"Entity {entity_name!r} is not registered")
            category = self._entities[entity_key]
            source_ids = set(self._entity_sources.get(entity_key, set()))
            total = len(source_ids)
            # Stale count
            n_stale = 0
            for sid in source_ids:
                entry = self._freshness.get((entity_key, sid))
                if entry is None or (now - entry.last_fetched_at) > threshold:
                    n_stale += 1
            # Derivative count: freshness flag takes precedence; fall back to attach-time flag
            n_deriv = sum(
                1 for sid in source_ids
                if (
                    self._freshness.get((entity_key, sid)) is not None
                    and self._freshness[(entity_key, sid)].is_derivative
                ) or (
                    self._freshness.get((entity_key, sid)) is None
                    and self._source_is_derivative.get(sid, False)
                )
            )
            # Completeness: fraction of required families covered
            required = _REQUIRED_FAMILIES_BY_CATEGORY.get(category, frozenset())
            covered_families: Set[str] = set()
            for sid in source_ids:
                fam = self._source_families.get(sid)
                if fam and fam in required:
                    covered_families.add(fam)
            completeness = len(covered_families) / len(required) if required else 1.0
            gap_count = len(required - covered_families)
            deriv_ratio = (n_deriv / total) if total > 0 else 0.0
        return EntityCoverageScore(
            entity_name=entity_key,
            category=category,
            total_sources=total,
            stale_sources=n_stale,
            derivative_sources=n_deriv,
            completeness=completeness,
            derivative_ratio=deriv_ratio,
            gap_count=gap_count,
            computed_at=now,
        )

    def identify_gaps(self, min_completeness: float = 1.0) -> List[EntityCoverageScore]:
        """Return coverage scores for all entities below *min_completeness*, sorted worst-first.

        Args:
            min_completeness: Threshold [0, 1]; entities strictly below this are returned.

        Raises:
            ValueError: min_completeness out of [0, 1].
        """
        if not isinstance(min_completeness, (int, float)):
            raise TypeError(f"'min_completeness' must be numeric, got {type(min_completeness)!r}")
        if not (0.0 <= min_completeness <= 1.0):
            raise ValueError(f"'min_completeness' must be in [0, 1], got {min_completeness!r}")
        with self._lock:
            names = list(self._entities.keys())
        gaps = []
        for name in names:
            score = self.coverage_score(name)
            if score.completeness < min_completeness:
                gaps.append(score)
        # Worst coverage first, then by gap_count descending
        gaps.sort(key=lambda s: (s.completeness, -s.gap_count))
        logger.info("identify_gaps: found %d entity/entities below completeness=%.2f", len(gaps), min_completeness)
        return gaps

    def derivative_overreliance(self) -> List[EntityCoverageScore]:
        """Return coverage scores for entities whose derivative ratio exceeds the threshold."""
        with self._lock:
            names = list(self._entities.keys())
        result = []
        for name in names:
            score = self.coverage_score(name)
            if score.total_sources > 0 and score.derivative_ratio >= self._overreliance_ratio:
                result.append(score)
        result.sort(key=lambda s: -s.derivative_ratio)
        logger.debug(
            "derivative_overreliance: %d/%d entities exceed %.0f%% derivative ratio",
            len(result), len(names), self._overreliance_ratio * 100,
        )
        return result

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the graph to a plain dict (JSON-safe)."""
        with self._lock:
            return {
                "staleness_threshold_hours": self._staleness_hours,
                "derivative_overreliance_ratio": self._overreliance_ratio,
                "entities": {k: v.value for k, v in self._entities.items()},
                "source_families": dict(self._source_families),
                "source_is_derivative": dict(self._source_is_derivative),
                "entity_sources": {k: sorted(v) for k, v in self._entity_sources.items()},
                "freshness": {
                    f"{k[0]}::{k[1]}": {
                        "last_fetched_at": v.last_fetched_at.isoformat(),
                        "is_derivative": v.is_derivative,
                        "item_count": v.item_count,
                    }
                    for k, v in self._freshness.items()
                },
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceCoverageGraph":
        """Reconstruct a SourceCoverageGraph from a dict produced by ``to_dict()``.

        Raises:
            TypeError:  data is not a dict.
            KeyError:   Required key is missing.
        """
        if not isinstance(data, dict):
            raise TypeError(f"'data' must be a dict, got {type(data)!r}")
        graph = cls(
            staleness_threshold_hours=data["staleness_threshold_hours"],
            derivative_overreliance_ratio=data["derivative_overreliance_ratio"],
        )
        with graph._lock:
            for name, cat_val in data.get("entities", {}).items():
                graph._entities[name] = EntityCategory(cat_val)
                graph._entity_sources[name] = set(data.get("entity_sources", {}).get(name, []))
            graph._source_families.update(data.get("source_families", {}))
            graph._source_is_derivative.update(data.get("source_is_derivative", {}))
            for composite_key, fdata in data.get("freshness", {}).items():
                entity_name, source_id = composite_key.split("::", 1)
                graph._freshness[(entity_name, source_id)] = SourceFreshnessEntry(
                    entity_name=entity_name,
                    source_id=source_id,
                    last_fetched_at=datetime.fromisoformat(fdata["last_fetched_at"]),
                    is_derivative=fdata.get("is_derivative", False),
                    item_count=fdata.get("item_count", 0),
                )
        logger.info("from_dict: loaded %d entities", len(graph._entities))
        return graph
