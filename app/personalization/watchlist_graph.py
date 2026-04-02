"""Watchlist graph — user-followed entities, topics, and repositories.

``WatchlistGraph`` answers the question *"Which things is this user tracking,
and how well is the system actually covering them?"*

It holds a directed bipartite graph:

    WatchlistNode  ──coverage_edge──▶  SourceEntry

A ``WatchlistNode`` represents something the user is actively watching
(a technology, company, open-source repo, person, or topic).  A
``SourceEntry`` represents a specific source that has delivered content about
that node.  Coverage is scored per node and aggregated into a
``WatchlistCoverageReport`` which downstream systems (digest builder, gap
planner) consume.

Key concepts
------------
- **Coverage score** (0–1): fraction of the expected source families that
  have reported on this node within the staleness window.
- **Coverage gap**: a node whose coverage score is below a threshold — or
  that has not been updated in more than ``stale_after_hours`` hours.
- **Source family**: coarse category of the source (``"research"``,
  ``"developer_release"``, ``"media_audio"``, ``"social"``, ``"news"``).

Typical usage::

    graph = WatchlistGraph(user_id="u42")
    graph.watch("vLLM", node_type="repo",
                expected_families=["developer_release", "research"])
    graph.record_coverage("vLLM", source_family="developer_release",
                          source_id="github/vllm-project/vllm",
                          trust_score=0.95)

    report = graph.coverage_report()
    for gap in report.gaps:
        print(gap)   # WatchlistCoverageGap(node="vLLM", missing=["research"])
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    ENTITY    = "entity"
    TOPIC     = "topic"
    REPO      = "repo"
    PERSON    = "person"
    COMPANY   = "company"
    KEYWORD   = "keyword"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class WatchlistNode:
    """A single thing the user is tracking.

    Attributes
    ----------
    node_id:           Unique identifier (e.g. ``"vLLM"``, ``"attention:AI"``).
    node_type:         Semantic category.
    expected_families: Source families that *should* cover this node.  When
                       one is absent in the coverage window the gap is reported.
    priority:          User-supplied importance weight (0–1; higher = more urgent).
    added_at:          UTC timestamp when the node was added to the watchlist.
    """
    node_id:           str
    node_type:         NodeType          = NodeType.KEYWORD
    expected_families: List[str]         = field(default_factory=list)
    priority:          float             = 0.5
    added_at:          datetime          = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class SourceEntry:
    """One source that has delivered content for a ``WatchlistNode``.

    Attributes
    ----------
    source_id:     Connector-level source identifier.
    source_family: Coarse family string (``"developer_release"``, etc.).
    trust_score:   Connector trust score (0–1).
    last_seen_at:  UTC timestamp of the most recent content delivery.
    item_count:    Total items delivered for this node from this source.
    """
    source_id:     str
    source_family: str
    trust_score:   float    = 0.7
    last_seen_at:  datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    item_count:    int = 1


@dataclass
class WatchlistCoverageGap:
    """A detected coverage gap for one ``WatchlistNode``.

    Attributes
    ----------
    node_id:           The node with insufficient coverage.
    node_type:         Type of the node.
    missing_families:  Expected source families with zero recent coverage.
    stale_families:    Families with coverage but older than the stale window.
    coverage_score:    Current coverage score (0–1).
    priority:          Priority of the node (from ``WatchlistNode.priority``).
    recommendation:    Human-readable suggestion for closing the gap.
    """
    node_id:           str
    node_type:         NodeType
    missing_families:  List[str]
    stale_families:    List[str]
    coverage_score:    float
    priority:          float
    recommendation:    str = ""


@dataclass
class NodeCoverage:
    """Aggregated coverage statistics for one ``WatchlistNode``."""
    node_id:        str
    node_type:      NodeType
    priority:       float
    coverage_score: float                    # 0–1
    covered_families:  FrozenSet[str]        = field(default_factory=frozenset)
    missing_families:  FrozenSet[str]        = field(default_factory=frozenset)
    stale_families:    FrozenSet[str]        = field(default_factory=frozenset)
    sources:           List[SourceEntry]     = field(default_factory=list)
    latest_update:     Optional[datetime]    = None


@dataclass
class WatchlistCoverageReport:
    """Full coverage assessment across all watched nodes.

    Attributes
    ----------
    user_id:           Owner of this watchlist.
    generated_at:      UTC timestamp.
    node_coverages:    Per-node coverage statistics.
    gaps:              Nodes with insufficient or stale coverage.
    overall_score:     Weighted mean coverage score across all nodes.
    nodes_at_risk:     Nodes with coverage_score < gap_threshold.
    """
    user_id:         str
    generated_at:    datetime
    node_coverages:  List[NodeCoverage]
    gaps:            List[WatchlistCoverageGap]
    overall_score:   float
    nodes_at_risk:   int


# ---------------------------------------------------------------------------
# WatchlistGraph
# ---------------------------------------------------------------------------

class WatchlistGraph:
    """Tracks user watchlist nodes and their source coverage.

    Args:
        user_id:          Identifier for the user who owns this watchlist.
        gap_threshold:    Coverage score below which a node is flagged as a gap.
        stale_after_hours: Hours after which a source entry is considered stale.
    """

    def __init__(
        self,
        user_id:          str,
        gap_threshold:    float = 0.60,
        stale_after_hours: float = 72.0,
    ) -> None:
        if not user_id.strip():
            raise ValueError("'user_id' must be non-empty")
        if not (0.0 < gap_threshold <= 1.0):
            raise ValueError(f"'gap_threshold' must be in (0, 1]; got {gap_threshold!r}")
        if stale_after_hours <= 0:
            raise ValueError(f"'stale_after_hours' must be positive; got {stale_after_hours!r}")

        self.user_id           = user_id
        self._gap_threshold    = gap_threshold
        self._stale_after      = timedelta(hours=stale_after_hours)
        self._lock             = threading.Lock()
        # node_id → WatchlistNode
        self._nodes: Dict[str, WatchlistNode]               = {}
        # node_id → {source_id → SourceEntry}
        self._coverage: Dict[str, Dict[str, SourceEntry]]   = {}

    # ------------------------------------------------------------------
    # Watchlist management
    # ------------------------------------------------------------------

    def watch(
        self,
        node_id:           str,
        *,
        node_type:         str          = "keyword",
        expected_families: List[str]    = (),
        priority:          float        = 0.5,
    ) -> None:
        """Add *node_id* to the watchlist (idempotent).

        Args:
            node_id:           Unique identifier for the watched node.
            node_type:         One of the ``NodeType`` values (as a string).
            expected_families: Source families that should cover this node.
            priority:          Importance weight (0–1).

        Raises:
            ValueError: If ``priority`` is outside [0, 1].
        """
        if not node_id.strip():
            raise ValueError("'node_id' must be non-empty")
        if not 0.0 <= priority <= 1.0:
            raise ValueError(f"'priority' must be in [0, 1]; got {priority!r}")
        nt = NodeType(node_type) if node_type not in NodeType._value2member_map_ else NodeType(node_type)
        node = WatchlistNode(
            node_id=node_id.strip(),
            node_type=nt,
            expected_families=list(expected_families),
            priority=priority,
        )
        with self._lock:
            if node_id not in self._nodes:
                self._nodes[node_id]    = node
                self._coverage[node_id] = {}
            else:
                # Update mutable fields in place
                existing = self._nodes[node_id]
                self._nodes[node_id] = WatchlistNode(
                    node_id=existing.node_id,
                    node_type=nt,
                    expected_families=list(expected_families) or existing.expected_families,
                    priority=priority,
                    added_at=existing.added_at,
                )

    def unwatch(self, node_id: str) -> bool:
        """Remove *node_id* from the watchlist.  Returns ``True`` if found."""
        with self._lock:
            found = node_id in self._nodes
            self._nodes.pop(node_id, None)
            self._coverage.pop(node_id, None)
        return found

    def is_watched(self, node_id: str) -> bool:
        with self._lock:
            return node_id in self._nodes

    def watched_nodes(self) -> List[WatchlistNode]:
        """Return all watched nodes sorted by priority (descending)."""
        with self._lock:
            return sorted(self._nodes.values(), key=lambda n: -n.priority)

    def count(self) -> int:
        with self._lock:
            return len(self._nodes)

    # ------------------------------------------------------------------
    # Coverage recording
    # ------------------------------------------------------------------

    def record_coverage(
        self,
        node_id:       str,
        *,
        source_id:     str,
        source_family: str,
        trust_score:   float = 0.7,
        observed_at:   Optional[datetime] = None,
    ) -> None:
        """Record that *source_id* delivered content relevant to *node_id*.

        Silently ignored if *node_id* is not in the watchlist (coverage for
        unwatched nodes is not tracked to keep memory bounded).

        Raises:
            ValueError: If ``trust_score`` is outside [0, 1].
        """
        if not 0.0 <= trust_score <= 1.0:
            raise ValueError(f"'trust_score' must be in [0, 1]; got {trust_score!r}")
        ts = observed_at or datetime.now(timezone.utc)
        with self._lock:
            if node_id not in self._nodes:
                return
            entries = self._coverage[node_id]
            if source_id in entries:
                old = entries[source_id]
                entries[source_id] = SourceEntry(
                    source_id=source_id,
                    source_family=source_family,
                    trust_score=trust_score,
                    last_seen_at=max(old.last_seen_at, ts),
                    item_count=old.item_count + 1,
                )
            else:
                entries[source_id] = SourceEntry(
                    source_id=source_id,
                    source_family=source_family,
                    trust_score=trust_score,
                    last_seen_at=ts,
                )

    def record_coverage_from_result(self, result: Any) -> None:
        """Convenience: record coverage for all entities in a pipeline result.

        Matches each entity in ``result.entities`` against watched nodes.
        All matched nodes get a coverage entry for ``result.source_family``.

        Args:
            result: An ``IntelligencePipelineResult`` (imported lazily to avoid
                    circular imports).
        """
        if not result.is_actionable():
            return
        source_id     = str(result.content_item_id)
        source_family = result.source_family
        trust_score   = result.confidence
        with self._lock:
            watched_ids = set(self._nodes.keys())
        for entity in result.entities:
            if entity in watched_ids:
                self.record_coverage(
                    entity,
                    source_id=source_id,
                    source_family=source_family,
                    trust_score=trust_score,
                    observed_at=result.produced_at,
                )

    # ------------------------------------------------------------------
    # Coverage assessment
    # ------------------------------------------------------------------

    def node_coverage(self, node_id: str) -> Optional[NodeCoverage]:
        """Return coverage statistics for a single node.  ``None`` if not watched."""
        with self._lock:
            if node_id not in self._nodes:
                return None
            node    = self._nodes[node_id]
            entries = list(self._coverage[node_id].values())
        return self._compute_node_coverage(node, entries)

    def coverage_report(self) -> WatchlistCoverageReport:
        """Generate a full coverage report across all watched nodes.

        Returns:
            ``WatchlistCoverageReport`` with per-node statistics and gap list.
        """
        with self._lock:
            nodes_snapshot    = dict(self._nodes)
            coverage_snapshot = {nid: list(e.values()) for nid, e in self._coverage.items()}

        node_coverages: List[NodeCoverage] = []
        gaps: List[WatchlistCoverageGap]   = []

        for node_id, node in nodes_snapshot.items():
            entries  = coverage_snapshot.get(node_id, [])
            nc       = self._compute_node_coverage(node, entries)
            node_coverages.append(nc)
            if nc.coverage_score < self._gap_threshold or nc.missing_families or nc.stale_families:
                gaps.append(WatchlistCoverageGap(
                    node_id=node_id,
                    node_type=node.node_type,
                    missing_families=sorted(nc.missing_families),
                    stale_families=sorted(nc.stale_families),
                    coverage_score=nc.coverage_score,
                    priority=node.priority,
                    recommendation=self._recommend(node, nc),
                ))

        gaps.sort(key=lambda g: (-g.priority, g.coverage_score))
        if node_coverages:
            weights = [nc.coverage_score * nodes_snapshot[nc.node_id].priority
                       for nc in node_coverages]
            total_priority = sum(nodes_snapshot[nc.node_id].priority for nc in node_coverages)
            overall = sum(weights) / total_priority if total_priority > 0 else 0.0
        else:
            overall = 1.0

        return WatchlistCoverageReport(
            user_id=self.user_id,
            generated_at=datetime.now(timezone.utc),
            node_coverages=node_coverages,
            gaps=gaps,
            overall_score=min(1.0, max(0.0, overall)),
            nodes_at_risk=sum(1 for nc in node_coverages if nc.coverage_score < self._gap_threshold),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_node_coverage(
        self, node: WatchlistNode, entries: List[SourceEntry]
    ) -> NodeCoverage:
        now = datetime.now(timezone.utc)
        expected: FrozenSet[str] = frozenset(node.expected_families)
        # Partition entries into fresh vs. stale
        fresh_entries = [e for e in entries if (now - e.last_seen_at) <= self._stale_after]
        stale_entries = [e for e in entries if (now - e.last_seen_at) > self._stale_after]
        covered_fresh:  FrozenSet[str] = frozenset(e.source_family for e in fresh_entries)
        covered_stale:  FrozenSet[str] = frozenset(e.source_family for e in stale_entries)
        missing  = expected - covered_fresh - covered_stale
        stale    = (expected & covered_stale) - covered_fresh
        # Score = fraction of expected families with fresh coverage
        if expected:
            score = len(covered_fresh & expected) / len(expected)
        elif entries:
            # No expectations set — presence of any fresh coverage = 1.0
            score = 1.0 if fresh_entries else 0.3
        else:
            score = 0.0

        latest = max((e.last_seen_at for e in entries), default=None)
        return NodeCoverage(
            node_id=node.node_id,
            node_type=node.node_type,
            priority=node.priority,
            coverage_score=round(score, 4),
            covered_families=covered_fresh,
            missing_families=missing,
            stale_families=stale,
            sources=entries,
            latest_update=latest,
        )

    @staticmethod
    def _recommend(node: WatchlistNode, nc: NodeCoverage) -> str:
        parts: List[str] = []
        if nc.missing_families:
            fam = ", ".join(sorted(nc.missing_families))
            parts.append(f"Add {fam} connector(s) for '{node.node_id}'.")
        if nc.stale_families:
            fam = ", ".join(sorted(nc.stale_families))
            parts.append(f"Refresh {fam} source(s) — data is stale.")
        if not nc.sources:
            parts.append(f"No sources have reported on '{node.node_id}' yet.")
        return " ".join(parts) or "Coverage is below threshold; investigate source health."


# Avoid import error when TYPE_CHECKING uses Any
from typing import Any  # noqa: E402

