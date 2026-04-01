"""Digest Delivery Modes — Phase 6.

Implements four user-facing delivery modes described in the roadmap:
  1. Morning Brief  — top-N items grouped by entity/theme with confidence.
  2. Watchlist      — per-entity digesting with staleness detection.
  3. Deep Dive      — consolidated timeline, claims, and business implications.
  4. Personalized Stream — feedback-weighted item ordering.

Each mode handler operates on plain Python dicts so it has zero external
dependencies beyond the stdlib.  An optional LLM router path can annotate
the output with richer explanations; graceful None-fallback always applies.

Public API
----------
    DeliveryMode            — enum of the four modes
    BriefItem               — one item in a morning brief
    WatchlistEntry          — one entity entry in a watchlist digest
    DeepDiveResult          — result of a deep dive on one event/entity
    PersonalizedStreamItem  — one item in a personalized stream
    MorningBrief            — full morning brief output
    WatchlistDigest         — full watchlist output
    PersonalizedStream      — full personalized stream output
    DigestModeRouter        — selects and runs the appropriate mode handler
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------

class DeliveryMode(str, Enum):
    """The four user-facing delivery modes."""

    MORNING_BRIEF = "morning_brief"
    WATCHLIST = "watchlist"
    DEEP_DIVE = "deep_dive"
    PERSONALIZED_STREAM = "personalized_stream"


# ---------------------------------------------------------------------------
# Item models (frozen value objects)
# ---------------------------------------------------------------------------

class BriefItem(BaseModel):
    """One item in a morning brief.

    Attributes:
        item_id:        Content item identifier.
        title:          Display title.
        entity_group:   Entity or theme this item belongs to.
        importance:     Importance score [0, 1].
        confidence:     System confidence in importance [0, 1].
        why_it_matters: One-line explanation of relevance.
        sources:        List of contributing source IDs.
        published_at:   UTC publication timestamp.
    """

    model_config = {"frozen": True}

    item_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    entity_group: str = Field(default="general")
    importance: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    why_it_matters: str = ""
    sources: List[str] = Field(default_factory=list)
    published_at: Optional[datetime] = None

    @field_validator("published_at")
    @classmethod
    def _aware(cls, v: Optional[datetime]) -> Optional[datetime]:
        if v is not None and v.tzinfo is None:
            raise ValueError("'published_at' must be timezone-aware")
        return v


class WatchlistEntry(BaseModel):
    """One entity entry in a watchlist digest.

    Attributes:
        entity_name:   Canonical entity being watched.
        update_count:  Number of updates in this period.
        top_items:     The most important item IDs for this entity.
        last_activity: UTC timestamp of the most recent update.
        is_stale:      True if no updates within the staleness window.
        alert_level:   'low' | 'medium' | 'high' — urgency of updates.
    """

    model_config = {"frozen": True}

    entity_name: str = Field(..., min_length=1)
    update_count: int = Field(ge=0)
    top_items: List[str] = Field(default_factory=list)
    last_activity: Optional[datetime] = None
    is_stale: bool = False
    alert_level: str = Field(default="low")

    @field_validator("alert_level")
    @classmethod
    def _valid_level(cls, v: str) -> str:
        if v not in {"low", "medium", "high"}:
            raise ValueError(f"'alert_level' must be 'low', 'medium', or 'high', got {v!r}")
        return v


class DeepDiveResult(BaseModel):
    """Result of a deep-dive analysis on one event or entity.

    Attributes:
        subject:             Entity name or event description.
        timeline:            Ordered list of (timestamp, description) tuples as dicts.
        factual_claims:      Extracted factual claims from all sources.
        business_implications: Practical implications for the reader.
        source_bundle:       All source IDs contributing evidence.
        confidence:          Overall confidence [0, 1].
        generated_at:        UTC timestamp of generation.
    """

    model_config = {"frozen": True}

    subject: str = Field(..., min_length=1)
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    factual_claims: List[str] = Field(default_factory=list)
    business_implications: List[str] = Field(default_factory=list)
    source_bundle: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))




class PersonalizedStreamItem(BaseModel):
    """One item in a personalized stream."""

    model_config = {"frozen": True}

    item_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    base_score: float = Field(ge=0.0, le=1.0)
    feedback_boost: float = Field(default=0.0, ge=0.0, le=1.0)
    penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)
    entity_ids: List[str] = Field(default_factory=list)
    topic_ids: List[str] = Field(default_factory=list)


class MorningBrief(BaseModel):
    """Full morning brief output."""

    model_config = {"frozen": True}

    date: str = Field(..., min_length=1)
    items: List[BriefItem] = Field(default_factory=list)
    entity_groups: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WatchlistDigest(BaseModel):
    """Full watchlist digest output."""

    model_config = {"frozen": True}

    entries: List[WatchlistEntry] = Field(default_factory=list)
    high_alert_entities: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PersonalizedStream(BaseModel):
    """Full personalized stream output."""

    model_config = {"frozen": True}

    user_id: str = Field(..., min_length=1)
    items: List[PersonalizedStreamItem] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))



# Type alias
_CandidateDict = Dict[str, Any]


class DigestModeRouter:
    """Routes digest generation to the appropriate mode handler (heuristic + optional LLM).

    Args:
        llm_router: Optional callable(prompt: str) → str used for LLM-enhanced output.
                    Falls back to heuristics when None or when call raises.
    """

    def __init__(self, llm_router: Optional[Callable[[str], str]] = None) -> None:
        self._llm: Optional[Callable[[str], str]] = llm_router
        logger.info("DigestModeRouter ready (LLM=%s)", "yes" if llm_router else "no")

    # ------------------------------------------------------------------
    # Mode 1: Morning Brief
    # ------------------------------------------------------------------

    def render_morning_brief(
        self,
        candidates: List[_CandidateDict],
        top_n: int = 10,
        date_str: Optional[str] = None,
    ) -> MorningBrief:
        """Produce a morning brief of top-N items grouped by entity.

        Raises:
            TypeError:  candidates not list, top_n not int.
            ValueError: empty candidates, top_n < 1.
        """
        if not isinstance(candidates, list):
            raise TypeError(f"'candidates' must be list, got {type(candidates)!r}")
        if not candidates:
            raise ValueError("'candidates' must be non-empty")
        if not isinstance(top_n, int):
            raise TypeError(f"'top_n' must be int, got {type(top_n)!r}")
        if top_n < 1:
            raise ValueError(f"'top_n' must be ≥ 1, got {top_n!r}")
        self._validate_candidates(candidates)
        if date_str is None:
            date_str = datetime.now(timezone.utc).date().isoformat()
        sorted_cands = sorted(
            candidates, key=lambda c: float(c.get("importance", 0.0)), reverse=True
        )[:top_n]
        brief_items: List[BriefItem] = []
        entity_groups: set = set()
        for c in sorted_cands:
            entity_ids = c.get("entity_ids", []) or []
            group = entity_ids[0] if entity_ids else "general"
            entity_groups.add(group)
            brief_items.append(BriefItem(
                item_id=str(c["item_id"]),
                title=str(c["title"]),
                entity_group=group,
                importance=float(c.get("importance", 0.0)),
                confidence=self._confidence(c),
                why_it_matters=self._why_it_matters(c),
                sources=list(c.get("sources", c.get("source_ids", []))),
                published_at=c.get("published_at"),
            ))
        logger.info("morning_brief: %d/%d items", len(brief_items), len(candidates))
        return MorningBrief(date=date_str, items=brief_items, entity_groups=sorted(entity_groups))

    # ------------------------------------------------------------------
    # Mode 2: Watchlist
    # ------------------------------------------------------------------

    def render_watchlist(
        self,
        candidates: List[_CandidateDict],
        watched_entities: List[str],
        staleness_hours: float = 48.0,
    ) -> WatchlistDigest:
        """Per-entity watchlist digest.

        Raises:
            TypeError:  Wrong argument types.
            ValueError: watched_entities empty, staleness_hours ≤ 0.
        """
        if not isinstance(candidates, list):
            raise TypeError(f"'candidates' must be list, got {type(candidates)!r}")
        if not isinstance(watched_entities, list):
            raise TypeError(f"'watched_entities' must be list, got {type(watched_entities)!r}")
        if not watched_entities:
            raise ValueError("'watched_entities' must be non-empty")
        if not isinstance(staleness_hours, (int, float)):
            raise TypeError(f"'staleness_hours' must be numeric, got {type(staleness_hours)!r}")
        if staleness_hours <= 0:
            raise ValueError(f"'staleness_hours' must be > 0, got {staleness_hours!r}")
        if candidates:
            self._validate_candidates(candidates)
        from datetime import timedelta
        staleness_delta = timedelta(hours=staleness_hours)
        now = datetime.now(timezone.utc)
        entity_items: Dict[str, List[_CandidateDict]] = {e: [] for e in watched_entities}
        for c in candidates:
            for eid in (c.get("entity_ids") or []):
                if eid in entity_items:
                    entity_items[eid].append(c)
        entries: List[WatchlistEntry] = []
        high_alert: List[str] = []
        for entity in watched_entities:
            items_for = sorted(
                entity_items[entity],
                key=lambda c: float(c.get("importance", 0.0)), reverse=True,
            )
            last_activity: Optional[datetime] = None
            for c in items_for:
                pa = c.get("published_at")
                if isinstance(pa, datetime):
                    if last_activity is None or pa > last_activity:
                        last_activity = pa
            is_stale = last_activity is None or (now - last_activity) > staleness_delta
            top_imp = float(items_for[0].get("importance", 0.0)) if items_for else 0.0
            alert = "high" if top_imp >= 0.7 else ("medium" if top_imp >= 0.4 else "low")
            if alert == "high":
                high_alert.append(entity)
            entries.append(WatchlistEntry(
                entity_name=entity,
                update_count=len(items_for),
                top_items=[str(c["item_id"]) for c in items_for[:3]],
                last_activity=last_activity,
                is_stale=is_stale,
                alert_level=alert,
            ))
        logger.info("watchlist: %d entities, %d high-alert", len(entries), len(high_alert))
        return WatchlistDigest(entries=entries, high_alert_entities=high_alert)

    # ------------------------------------------------------------------
    # Mode 3: Deep Dive
    # ------------------------------------------------------------------

    def render_deep_dive(
        self,
        candidates: List[_CandidateDict],
        subject: str,
    ) -> DeepDiveResult:
        """Consolidated deep-dive on one subject.

        Raises:
            TypeError:  Wrong argument types.
            ValueError: Empty candidates or empty subject.
        """
        if not isinstance(candidates, list):
            raise TypeError(f"'candidates' must be list, got {type(candidates)!r}")
        if not candidates:
            raise ValueError("'candidates' must be non-empty for a deep dive")
        if not isinstance(subject, str):
            raise TypeError(f"'subject' must be str, got {type(subject)!r}")
        if not subject.strip():
            raise ValueError("'subject' must be a non-empty string")
        self._validate_candidates(candidates)
        sorted_by_time = sorted(
            candidates,
            key=lambda c: c.get("published_at") or datetime.min.replace(tzinfo=timezone.utc),
        )
        timeline: List[Dict[str, Any]] = [
            {
                "timestamp": (c["published_at"].isoformat() if isinstance(c.get("published_at"), datetime) else None),
                "description": str(c.get("title", "")),
                "source": (c.get("sources") or c.get("source_ids") or ["unknown"])[0],
            }
            for c in sorted_by_time
        ]
        all_claims: List[str] = []
        for c in candidates:
            all_claims.extend(c.get("claims", []))
        claims = list(dict.fromkeys(all_claims))[:20]
        implications = self._business_implications(candidates, subject)
        all_sources: List[str] = []
        for c in candidates:
            all_sources.extend(c.get("sources", c.get("source_ids", [])))
        ts = [float(c.get("trust_score", 0.5)) for c in candidates]
        confidence = sum(ts) / len(ts)
        logger.info("deep_dive: subject=%r, %d items", subject, len(candidates))
        return DeepDiveResult(
            subject=subject.strip(), timeline=timeline, factual_claims=claims,
            business_implications=implications,
            source_bundle=list(dict.fromkeys(all_sources)),
            confidence=min(1.0, max(0.0, confidence)),
        )


    # ------------------------------------------------------------------
    # Mode 4: Personalized Stream
    # ------------------------------------------------------------------

    def render_personalized_stream(
        self,
        candidates: List[_CandidateDict],
        user_id: str,
        feedback_history: Optional[Dict[str, float]] = None,
        max_items: Optional[int] = None,
    ) -> PersonalizedStream:
        """Feedback-weighted personalized stream with optional depth control.

        Args:
            candidates:       List of candidate dicts.
            user_id:          User identifier (non-empty string).
            feedback_history: Mapping of item_id → signed float delta from prior
                              interactions.  Positive = liked/saved; negative =
                              dismissed/skipped.
            max_items:        Maximum items to include in the stream (depth control).
                              ``None`` means no limit.  Must be ≥ 1 when provided.

        Raises:
            TypeError:  Wrong argument types.
            ValueError: user_id empty, or max_items < 1.
        """
        if not isinstance(candidates, list):
            raise TypeError(f"'candidates' must be list, got {type(candidates)!r}")
        if not isinstance(user_id, str):
            raise TypeError(f"'user_id' must be str, got {type(user_id)!r}")
        if not user_id.strip():
            raise ValueError("'user_id' must be non-empty")
        if feedback_history is not None and not isinstance(feedback_history, dict):
            raise TypeError(f"'feedback_history' must be dict or None, got {type(feedback_history)!r}")
        if max_items is not None:
            if not isinstance(max_items, int):
                raise TypeError(f"'max_items' must be int or None, got {type(max_items)!r}")
            if max_items < 1:
                raise ValueError(f"'max_items' must be ≥ 1, got {max_items!r}")
        if candidates:
            self._validate_candidates(candidates)
        fb = feedback_history or {}
        stream_items: List[PersonalizedStreamItem] = []
        for c in candidates:
            iid = str(c["item_id"])
            base = float(c.get("importance", 0.5))
            delta = fb.get(iid, 0.0)
            boost = max(0.0, min(1.0, delta)) if delta > 0 else 0.0
            penalty = max(0.0, min(1.0, -delta)) if delta < 0 else 0.0
            final = max(0.0, min(1.0, base + boost - penalty))
            stream_items.append(PersonalizedStreamItem(
                item_id=iid,
                title=str(c["title"]),
                base_score=base,
                feedback_boost=boost,
                penalty=penalty,
                final_score=final,
                entity_ids=list(c.get("entity_ids", [])),
                topic_ids=list(c.get("topic_ids", [])),
            ))
        stream_items.sort(key=lambda i: -i.final_score)
        if max_items is not None:
            stream_items = stream_items[:max_items]
        logger.info(
            "personalized_stream: user=%r, %d items (max_items=%s)",
            user_id, len(stream_items), max_items,
        )
        return PersonalizedStream(user_id=user_id.strip(), items=stream_items)

    # ------------------------------------------------------------------
    # Unified dispatcher
    # ------------------------------------------------------------------

    def render(self, mode: DeliveryMode, candidates: List[_CandidateDict], **kwargs: Any) -> Any:
        """Dispatch to the appropriate mode handler.

        Raises:
            TypeError:  mode not DeliveryMode.
        """
        if not isinstance(mode, DeliveryMode):
            raise TypeError(f"'mode' must be DeliveryMode, got {type(mode)!r}")
        logger.debug(
            "render: dispatching mode=%r with %d candidates",
            mode.value, len(candidates) if isinstance(candidates, list) else -1,
        )
        dispatch = {
            DeliveryMode.MORNING_BRIEF: self.render_morning_brief,
            DeliveryMode.WATCHLIST: self.render_watchlist,
            DeliveryMode.DEEP_DIVE: self.render_deep_dive,
            DeliveryMode.PERSONALIZED_STREAM: self.render_personalized_stream,
        }
        return dispatch[mode](candidates, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_candidates(self, candidates: List[_CandidateDict]) -> None:
        for i, c in enumerate(candidates):
            if not isinstance(c, dict):
                raise TypeError(f"candidates[{i}] must be dict, got {type(c)!r}")
            if "item_id" not in c:
                raise ValueError(f"candidates[{i}] missing required key 'item_id'")
            if "title" not in c:
                raise ValueError(f"candidates[{i}] missing required key 'title'")
            imp = c.get("importance", 0.0)
            if not isinstance(imp, (int, float)):
                raise TypeError(f"candidates[{i}]['importance'] must be numeric, got {type(imp)!r}")
            if not (0.0 <= float(imp) <= 1.0):
                raise ValueError(f"candidates[{i}]['importance'] out of [0,1]: {imp!r}")

    def _why_it_matters(self, c: _CandidateDict) -> str:
        importance = float(c.get("importance", 0.0))
        tier = "high-importance" if importance >= 0.8 else ("notable" if importance >= 0.5 else "informational")
        heuristic = f"[{tier}] {str(c.get('title', ''))[:120]}"
        if self._llm is not None:
            try:
                prompt = f"One sentence: why does this matter to an AI professional? '{c.get('title', '')}'"
                response = self._llm(prompt)
                if isinstance(response, str) and response.strip():
                    return response.strip()
            except Exception as exc:
                logger.warning("LLM fallback _why_it_matters: %s", exc)
        return heuristic

    def _confidence(self, c: _CandidateDict) -> float:
        import math
        trust = float(c.get("trust_score", 0.5))
        evidence = int(c.get("evidence_count", 1))
        evidence_factor = math.log2(evidence + 1) / math.log2(11)
        return min(1.0, 0.6 * trust + 0.4 * evidence_factor)

    def _business_implications(self, candidates: List[_CandidateDict], subject: str) -> List[str]:
        avg_imp = sum(float(c.get("importance", 0.5)) for c in candidates) / len(candidates)
        urgency = "High urgency" if avg_imp >= 0.8 else ("Moderate urgency" if avg_imp >= 0.5 else "Low urgency")
        heuristic = [
            f"{urgency}: {subject} has {len(candidates)} related update(s).",
            f"Monitor {subject} for downstream impacts on tooling and dependencies.",
        ]
        if self._llm is not None:
            try:
                prompt = f"Two concise business implications (< 15 words each) for an AI team re: '{subject}'."
                response = self._llm(prompt)
                if isinstance(response, str) and response.strip():
                    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
                    if lines:
                        return lines[:2]
            except Exception as exc:
                logger.warning("LLM fallback _business_implications: %s", exc)
        return heuristic
