"""Event-First Ingestion Pipeline — Phase 6.

Orchestrates the full pipeline:
  raw item → extract entities/claims
           → map to canonical events (via EventClusterer)
           → merge supporting evidence
           → score event importance
           → generate user-facing updates

Public API
----------
    RawItem          — input dataclass for a single content item
    ProcessedEvent   — frozen Pydantic model representing a merged, scored event
    PipelineStats    — aggregate statistics about pipeline throughput
    EventFirstPipeline — main orchestrator (thread-safe)
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------

@dataclass
class RawItem:
    """A single raw content item entering the pipeline.

    Attributes:
        item_id:      Unique identifier (auto-generated if not provided).
        title:        Item title or headline.
        body:         Full text body (may be empty).
        source_id:    Originating source identifier.
        published_at: UTC publish timestamp (defaults to now).
        entities:     Pre-extracted entity strings (may be empty; pipeline adds more).
        trust_score:  Source trust score in [0, 1].
        platform:     Platform/source family name.
        metadata:     Arbitrary extra data.
    """

    title: str
    source_id: str
    body: str = ""
    published_at: Optional[datetime] = None
    entities: List[str] = field(default_factory=list)
    trust_score: float = 0.5
    platform: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    item_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("'title' must be a non-empty string")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("'source_id' must be a non-empty string")
        if not isinstance(self.trust_score, (int, float)):
            raise TypeError(f"'trust_score' must be numeric, got {type(self.trust_score)!r}")
        if not (0.0 <= self.trust_score <= 1.0):
            raise ValueError(f"'trust_score' must be in [0, 1], got {self.trust_score!r}")
        if self.published_at is None:
            self.published_at = datetime.now(timezone.utc)
        if isinstance(self.published_at, datetime) and self.published_at.tzinfo is None:
            raise ValueError("'published_at' must be timezone-aware")


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class ProcessedEvent(BaseModel):
    """A merged, scored event produced by the pipeline.

    Attributes:
        event_id:          Unique identifier for this event.
        canonical_title:   Representative title selected from supporting items.
        entities:          Deduplicated list of all entity mentions.
        claims:            Extracted factual claims from all supporting items.
        source_ids:        All source IDs that contributed evidence.
        importance_score:  Heuristic importance in [0, 1].
        evidence_count:    Number of distinct items merged into this event.
        trust_weighted_score: Trust-weighted aggregate.
        first_seen_at:     Earliest published_at across supporting items.
        last_seen_at:      Latest published_at across supporting items.
        pipeline_version:  Pipeline version tag for traceability.
    """

    model_config = {"frozen": True}

    event_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    canonical_title: str = Field(..., min_length=1)
    entities: List[str] = Field(default_factory=list)
    claims: List[str] = Field(default_factory=list)
    source_ids: List[str] = Field(default_factory=list)
    importance_score: float = Field(ge=0.0, le=1.0)
    evidence_count: int = Field(ge=1)
    trust_weighted_score: float = Field(ge=0.0, le=1.0)
    first_seen_at: datetime
    last_seen_at: datetime
    pipeline_version: str = Field(default="1.0", min_length=1)

    @field_validator("first_seen_at", "last_seen_at")
    @classmethod
    def _must_be_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamps must be timezone-aware")
        return v

    @model_validator(mode="after")
    def _first_le_last(self) -> "ProcessedEvent":
        if self.first_seen_at > self.last_seen_at:
            raise ValueError("'first_seen_at' must be ≤ 'last_seen_at'")
        return self



class PipelineStats(BaseModel):
    """Aggregate statistics for a completed pipeline run."""

    model_config = {"frozen": True}

    items_processed: int = Field(ge=0)
    events_produced: int = Field(ge=0)
    items_merged: int = Field(ge=0)
    run_duration_ms: float = Field(ge=0.0)
    errors: int = Field(ge=0)

    @model_validator(mode="after")
    def _merged_le_processed(self) -> "PipelineStats":
        if self.items_merged > self.items_processed:
            raise ValueError(
                f"'items_merged' ({self.items_merged}) cannot exceed "
                f"'items_processed' ({self.items_processed})"
            )
        return self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s]")
_STOP = frozenset(
    "a an the and or but in on at to for of with is are was were be been "
    "being have has had do does did will would could should may might "
    "this that these those it its by from as into".split()
)


def _extract_entities(text: str) -> List[str]:
    """Heuristic: capitalised n-grams that are not stop words."""
    words = _PUNCT.sub(" ", text).split()
    entities: List[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w and w[0].isupper() and w.lower() not in _STOP and len(w) > 1:
            entity_parts = [w]
            j = i + 1
            while j < len(words) and words[j] and words[j][0].isupper() and len(words[j]) > 1:
                entity_parts.append(words[j])
                j += 1
            entities.append(" ".join(entity_parts))
            i = j
        else:
            i += 1
    return list(dict.fromkeys(entities))


def _extract_claims(text: str) -> List[str]:
    """Heuristic: sentences containing claim-trigger verbs."""
    claim_triggers = {
        "announce", "release", "launch", "publish", "report", "find",
        "introduce", "reveal", "confirm", "state", "claim", "show",
        "demonstrate", "prove", "support", "reject", "deprecate",
    }
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims: List[str] = []
    for s in sentences:
        tokens = set(_PUNCT.sub(" ", s).lower().split())
        if tokens & claim_triggers:
            claims.append(s.strip())
    return claims[:10]


def _title_similarity(a: str, b: str) -> float:
    """Jaccard similarity on lowercased word tokens."""
    ta = set(_PUNCT.sub(" ", a.lower()).split())
    tb = set(_PUNCT.sub(" ", b.lower()).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _importance_score(
    evidence_count: int,
    trust_weighted: float,
    entity_count: int,
    claim_count: int,
) -> float:
    """Composite importance heuristic → [0, 1]."""
    import math
    ev_factor = math.log2(min(evidence_count, 10) + 1) / math.log2(11)
    ent_factor = min(entity_count, 10) / 10.0
    clm_factor = min(claim_count, 10) / 10.0
    raw = 0.35 * ev_factor + 0.35 * trust_weighted + 0.20 * ent_factor + 0.10 * clm_factor
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# EventFirstPipeline
# ---------------------------------------------------------------------------

class EventFirstPipeline:
    """Orchestrates event-first ingestion of raw content items (thread-safe).

    Stages per item:
    1. Validate + enrich (entity extraction, claim extraction).
    2. Find best-matching existing event via title + entity similarity.
    3. If similarity ≥ merge_threshold → merge (evidence stacking).
    4. Otherwise → create new event.
    5. Importance re-scored after each merge.

    Args:
        merge_threshold:  Combined similarity threshold to merge items [0, 1].
        entity_weight:    Entity-overlap weight (entity_weight + title_weight == 1.0).
        title_weight:     Title-similarity weight.
        pipeline_version: Version tag attached to every event.
    """

    def __init__(
        self,
        merge_threshold: float = 0.3,
        entity_weight: float = 0.5,
        title_weight: float = 0.5,
        pipeline_version: str = "1.0",
    ) -> None:
        if not isinstance(merge_threshold, (int, float)):
            raise TypeError(f"'merge_threshold' must be numeric, got {type(merge_threshold)!r}")
        if not (0.0 <= merge_threshold <= 1.0):
            raise ValueError(f"'merge_threshold' must be in [0, 1], got {merge_threshold!r}")
        if not isinstance(entity_weight, (int, float)):
            raise TypeError(f"'entity_weight' must be numeric, got {type(entity_weight)!r}")
        if not isinstance(title_weight, (int, float)):
            raise TypeError(f"'title_weight' must be numeric, got {type(title_weight)!r}")
        if not (0.0 <= entity_weight <= 1.0):
            raise ValueError(f"'entity_weight' must be in [0, 1], got {entity_weight!r}")
        if not (0.0 <= title_weight <= 1.0):
            raise ValueError(f"'title_weight' must be in [0, 1], got {title_weight!r}")
        if abs(entity_weight + title_weight - 1.0) > 1e-6:
            raise ValueError(
                f"'entity_weight' + 'title_weight' must equal 1.0, "
                f"got sum={entity_weight + title_weight}"
            )
        if not isinstance(pipeline_version, str) or not pipeline_version.strip():
            raise ValueError("'pipeline_version' must be a non-empty string")
        self._merge_threshold = float(merge_threshold)
        self._entity_weight = float(entity_weight)
        self._title_weight = float(title_weight)
        self._pipeline_version = pipeline_version.strip()
        self._events: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._errors = 0
        logger.info("EventFirstPipeline ready (merge_threshold=%.2f, v%s)",
                    self._merge_threshold, self._pipeline_version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_item(self, item: RawItem) -> str:
        """Ingest one raw item; return the event_id it was assigned to.

        Raises:
            TypeError: item is not a RawItem.
        """
        if not isinstance(item, RawItem):
            raise TypeError(f"'item' must be RawItem, got {type(item)!r}")
        try:
            event_id = self._ingest(item)
            logger.debug("process_item: item=%r → event=%r", item.item_id, event_id)
            return event_id
        except Exception as exc:
            logger.warning("process_item error for %r: %s", item.item_id, exc)
            with self._lock:
                self._errors += 1
            raise

    def process_batch(self, items: List[RawItem]) -> List[str]:
        """Process a list of RawItems; return event_ids in the same order.

        Raises:
            TypeError:  items is not a list.
            ValueError: items is empty.
        """
        if not isinstance(items, list):
            raise TypeError(f"'items' must be list, got {type(items)!r}")
        if not items:
            raise ValueError("'items' must be non-empty")
        return [self.process_item(i) for i in items]

    def get_events(self) -> List[ProcessedEvent]:
        """All ProcessedEvents ordered by importance descending."""
        with self._lock:
            # Snapshot each staging dict to prevent race with concurrent _ingest calls
            staging_snapshots = [dict(s) for s in self._events.values()]
        return sorted(
            [self._build_event(s) for s in staging_snapshots],
            key=lambda e: -e.importance_score,
        )

    def get_event(self, event_id: str) -> Optional[ProcessedEvent]:
        """Retrieve one event by ID, or None.

        Raises:
            TypeError: event_id is not str.
            ValueError: event_id is empty.
        """
        if not isinstance(event_id, str):
            raise TypeError(f"'event_id' must be str, got {type(event_id)!r}")
        if not event_id.strip():
            raise ValueError("'event_id' must be non-empty")
        with self._lock:
            # Snapshot the staging dict to prevent race with concurrent _ingest calls
            raw = self._events.get(event_id.strip())
            staging = dict(raw) if raw is not None else None
        return self._build_event(staging) if staging is not None else None

    def generate_updates(self, top_n: int = 10) -> List[ProcessedEvent]:
        """Top-N most important events for user-facing delivery.

        Raises:
            TypeError:  top_n not int.
            ValueError: top_n < 1.
        """
        if not isinstance(top_n, int):
            raise TypeError(f"'top_n' must be int, got {type(top_n)!r}")
        if top_n < 1:
            raise ValueError(f"'top_n' must be ≥ 1, got {top_n!r}")
        return self.get_events()[:top_n]

    def get_stats(self, run_duration_ms: float = 0.0) -> PipelineStats:
        """Return PipelineStats for the current state.

        Raises:
            TypeError:  run_duration_ms not numeric.
            ValueError: run_duration_ms < 0.
        """
        if not isinstance(run_duration_ms, (int, float)):
            raise TypeError(f"'run_duration_ms' must be numeric, got {type(run_duration_ms)!r}")
        if run_duration_ms < 0:
            raise ValueError(f"'run_duration_ms' must be ≥ 0, got {run_duration_ms!r}")
        with self._lock:
            events_produced = len(self._events)
            items_processed = sum(s["evidence_count"] for s in self._events.values())
            items_merged = max(0, items_processed - events_produced)
            errors = self._errors
        return PipelineStats(
            items_processed=items_processed,
            events_produced=events_produced,
            items_merged=items_merged,
            run_duration_ms=float(run_duration_ms),
            errors=errors,
        )

    def reset(self) -> None:
        """Clear all events and error counters."""
        with self._lock:
            self._events.clear()
            self._errors = 0
        logger.info("EventFirstPipeline reset")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ingest(self, item: RawItem) -> str:
        """Core ingestion: enrich → find match → merge or create."""
        all_entities = list(dict.fromkeys(
            [e.strip() for e in item.entities if e.strip()]
            + _extract_entities(item.title)
            + _extract_entities(item.body[:500])
        ))
        all_claims = _extract_claims(item.body)
        published_at = item.published_at or datetime.now(timezone.utc)

        with self._lock:
            best_id: Optional[str] = None
            best_score = 0.0
            for eid, s in self._events.items():
                ex_ents = set(e.lower() for e in s["entities"])
                nw_ents = set(e.lower() for e in all_entities)
                union = ex_ents | nw_ents
                ent_overlap = len(ex_ents & nw_ents) / len(union) if union else 0.0
                title_sim = _title_similarity(s["canonical_title"], item.title)
                score = self._entity_weight * ent_overlap + self._title_weight * title_sim
                if score > best_score:
                    best_score = score
                    best_id = eid

            if best_score >= self._merge_threshold and best_id is not None:
                s = self._events[best_id]
                n = s["evidence_count"]
                new_tw = (s["trust_weighted_score"] * n + item.trust_score) / (n + 1)
                s.update({
                    "entities": list(dict.fromkeys(s["entities"] + all_entities)),
                    "claims": list(dict.fromkeys(s["claims"] + all_claims)),
                    "source_ids": list(dict.fromkeys(s["source_ids"] + [item.source_id])),
                    "evidence_count": n + 1,
                    "trust_weighted_score": new_tw,
                    "first_seen_at": min(s["first_seen_at"], published_at),
                    "last_seen_at": max(s["last_seen_at"], published_at),
                })
                logger.debug(
                    "_ingest: MERGE item=%r into event=%r (score=%.3f, evidence=%d)",
                    item.item_id, best_id, best_score, n + 1,
                )
                return best_id
            else:
                new_id = str(uuid4())
                self._events[new_id] = {
                    "event_id": new_id,
                    "canonical_title": item.title.strip(),
                    "entities": all_entities,
                    "claims": all_claims,
                    "source_ids": [item.source_id],
                    "evidence_count": 1,
                    "trust_weighted_score": float(item.trust_score),
                    "first_seen_at": published_at,
                    "last_seen_at": published_at,
                }
                logger.debug(
                    "_ingest: CREATE event=%r from item=%r (best_score=%.3f)",
                    new_id, item.item_id, best_score,
                )
                return new_id

    def _build_event(self, staging: Dict[str, Any]) -> ProcessedEvent:
        importance = _importance_score(
            evidence_count=staging["evidence_count"],
            trust_weighted=staging["trust_weighted_score"],
            entity_count=len(staging["entities"]),
            claim_count=len(staging["claims"]),
        )
        return ProcessedEvent(
            event_id=staging["event_id"],
            canonical_title=staging["canonical_title"],
            entities=list(staging["entities"]),
            claims=list(staging["claims"]),
            source_ids=list(staging["source_ids"]),
            importance_score=importance,
            evidence_count=staging["evidence_count"],
            trust_weighted_score=staging["trust_weighted_score"],
            first_seen_at=staging["first_seen_at"],
            last_seen_at=staging["last_seen_at"],
            pipeline_version=self._pipeline_version,
        )
