"""Coverage planner.

Analyses the ``SourceRegistryStore`` and produces ``CoverageGap`` objects
describing categories of sources that are missing or under-represented
for a given set of strategic entities and topics.

Gaps are assessed along three axes:
1. **Family gap**      — A ``SourceFamily`` has zero registered sources.
2. **Entity gap**      — A known entity has no sources in a required family.
3. **Capability gap**  — No registered source has a required ``SourceCapability``.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Dict, List, Optional, Set

from app.source_intelligence.source_registry import (
    SourceCapability,
    SourceFamily,
    SourceRegistryStore,
    SourceSpec,
)

logger = logging.getLogger(__name__)


class GapSeverity(Enum):
    """Importance of a coverage gap."""

    CRITICAL = "critical"    # Required for basic operation; block without it
    HIGH = "high"            # Important; should be resolved within a sprint
    MEDIUM = "medium"        # Nice to have; tracked but not urgent
    LOW = "low"              # Informational only


@dataclass
class CoverageGap:
    """A single identified gap in source coverage.

    Attributes:
        gap_id:        Unique identifier for this gap instance.
        gap_type:      One of ``"family"``, ``"entity"``, ``"capability"``.
        severity:      ``GapSeverity`` for prioritisation.
        description:   Human-readable explanation.
        affected_entity: Entity name if this is an entity-level gap.
        missing_family:  The ``SourceFamily`` that is absent (for family gaps).
        missing_capability: The ``SourceCapability`` that is absent.
        suggested_sources: Source IDs from the discovery catalogue that would
                            resolve this gap.
    """

    gap_id: str
    gap_type: str
    severity: GapSeverity
    description: str
    affected_entity: Optional[str] = None
    missing_family: Optional[SourceFamily] = None
    missing_capability: Optional[SourceCapability] = None
    suggested_sources: List[str] = field(default_factory=list)


class CoveragePlanner:
    """Analyses source coverage and returns prioritised ``CoverageGap`` objects.

    Thread-safe: analysis methods take a snapshot of the registry at call time
    and then process the snapshot without holding the lock.

    Args:
        registry: The ``SourceRegistryStore`` to analyse.
        required_families: Set of ``SourceFamily`` values that must be present.
        required_capabilities: Set of ``SourceCapability`` values that must
                                be present in at least one registered source.
    """

    def __init__(
        self,
        registry: SourceRegistryStore,
        required_families: Optional[Set[SourceFamily]] = None,
        required_capabilities: Optional[Set[SourceCapability]] = None,
    ) -> None:
        if not isinstance(registry, SourceRegistryStore):
            raise TypeError(f"'registry' must be SourceRegistryStore, got {type(registry)!r}")
        self._registry = registry
        self._required_families: Set[SourceFamily] = required_families or {
            SourceFamily.DEVELOPER_RELEASE,
            SourceFamily.RESEARCH,
            SourceFamily.MEDIA_AUDIO,
        }
        self._required_capabilities: Set[SourceCapability] = required_capabilities or {
            SourceCapability.SUPPORTS_SINCE,
            SourceCapability.PROVIDES_FULL_TEXT,
        }
        self._lock = threading.Lock()

    def analyse(self, entities: Optional[List[str]] = None) -> List[CoverageGap]:
        """Run a full coverage analysis and return gaps sorted by severity.

        Args:
            entities: Optional list of entity names to check for entity-level
                      gaps.  E.g. ``["OpenAI", "Anthropic"]``.

        Returns:
            List of ``CoverageGap`` objects sorted by severity (CRITICAL first).
        """
        t0 = time.perf_counter()
        specs = self._registry.all_specs()
        gaps: List[CoverageGap] = []
        gaps.extend(self._check_family_gaps(specs))
        gaps.extend(self._check_capability_gaps(specs))
        if entities:
            for entity in entities:
                if not isinstance(entity, str) or not entity.strip():
                    continue
                gaps.extend(self._check_entity_gaps(specs, entity))

        severity_order = {
            GapSeverity.CRITICAL: 0,
            GapSeverity.HIGH: 1,
            GapSeverity.MEDIUM: 2,
            GapSeverity.LOW: 3,
        }
        gaps.sort(key=lambda g: severity_order.get(g.severity, 4))

        logger.info(
            "CoveragePlanner.analyse: specs=%d gaps=%d latency_ms=%.1f",
            len(specs), len(gaps), (time.perf_counter() - t0) * 1000,
        )
        return gaps

    def _check_family_gaps(self, specs: List[SourceSpec]) -> List[CoverageGap]:
        present_families: Set[SourceFamily] = {s.family for s in specs}
        gaps: List[CoverageGap] = []
        for family in self._required_families:
            if family not in present_families:
                severity = GapSeverity.CRITICAL if family == SourceFamily.DEVELOPER_RELEASE else GapSeverity.HIGH
                gaps.append(CoverageGap(
                    gap_id=f"family_gap_{family.value}",
                    gap_type="family",
                    severity=severity,
                    description=f"No sources registered for family '{family.value}'",
                    missing_family=family,
                ))
        return gaps

    def _check_capability_gaps(self, specs: List[SourceSpec]) -> List[CoverageGap]:
        present_caps: Set[SourceCapability] = set()
        for spec in specs:
            present_caps.update(spec.capabilities)
        gaps: List[CoverageGap] = []
        for cap in self._required_capabilities:
            if cap not in present_caps:
                gaps.append(CoverageGap(
                    gap_id=f"cap_gap_{cap.name}",
                    gap_type="capability",
                    severity=GapSeverity.MEDIUM,
                    description=f"No registered source has capability '{cap.name}'",
                    missing_capability=cap,
                ))
        return gaps

    def _check_entity_gaps(self, specs: List[SourceSpec], entity: str) -> List[CoverageGap]:
        lower = entity.lower()
        entity_specs = [s for s in specs if lower in s.source_id.lower() or lower in s.display_name.lower()]
        entity_families: Set[SourceFamily] = {s.family for s in entity_specs}
        gaps: List[CoverageGap] = []
        for family in self._required_families:
            if family not in entity_families:
                gaps.append(CoverageGap(
                    gap_id=f"entity_gap_{lower}_{family.value}",
                    gap_type="entity",
                    severity=GapSeverity.HIGH,
                    description=f"Entity '{entity}' has no {family.value} sources",
                    affected_entity=entity,
                    missing_family=family,
                ))
        return gaps

    def summarise(self, gaps: Optional[List[CoverageGap]] = None) -> Dict[str, int]:
        """Return a count of gaps by severity level.

        Args:
            gaps: Pre-computed gap list; if None, runs ``analyse()`` first.
        """
        all_gaps = gaps if gaps is not None else self.analyse()
        return {sev.value: sum(1 for g in all_gaps if g.severity == sev) for sev in GapSeverity}

