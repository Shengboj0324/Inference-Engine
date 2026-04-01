"""Shared Pydantic models for the entity resolution package."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator


class EntityType(str, Enum):
    """Taxonomy of entity types tracked by the system."""

    MODEL = "model"                  # AI model (GPT-4, Claude, LLaMA)
    ORGANIZATION = "organization"   # Company / lab (OpenAI, Anthropic)
    PERSON = "person"               # Researcher / founder
    PAPER = "paper"                 # Academic paper
    DATASET = "dataset"             # Training or evaluation dataset
    BENCHMARK = "benchmark"         # Evaluation benchmark
    PRODUCT = "product"             # Software product / API
    CONCEPT = "concept"             # Technical concept (RLHF, RAG, LoRA)
    EVENT = "event"                 # Real-world event (NeurIPS 2024)
    OTHER = "other"


class CanonicalEntity(BaseModel):
    """The authoritative record for a named entity.

    Attributes:
        entity_id:    Unique stable identifier (slug, e.g. ``"openai/gpt-4o"``).
        entity_type:  ``EntityType`` classification.
        canonical_name: Primary display name (e.g. ``"GPT-4o"``).
        aliases:      All known name variants (lowercased).
        description:  Short entity description.
        properties:   Arbitrary typed properties (e.g. ``{"params": "1T"}``).
        source_urls:  URLs where entity was first observed.
        first_seen_at: First observation datetime.
        updated_at:   Last update datetime.
    """

    entity_id: str
    entity_type: EntityType = EntityType.OTHER
    canonical_name: str
    aliases: List[str] = Field(default_factory=list)
    description: str = ""
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_urls: List[str] = Field(default_factory=list)
    first_seen_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @field_validator("entity_id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("'entity_id' must be a non-empty string")
        return v.strip().lower()

    @field_validator("canonical_name")
    @classmethod
    def _non_empty_name(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("'canonical_name' must be a non-empty string")
        return v.strip()

    def all_names(self) -> Set[str]:
        """Return all known names including canonical (lowercase)."""
        return {self.canonical_name.lower()} | {a.lower() for a in self.aliases}


class EventBundle(BaseModel):
    """A cluster of ``ContentItem``-like dicts about the same real-world event.

    Attributes:
        bundle_id:      Unique cluster identifier.
        canonical_title: Most representative title.
        event_time:     Best-estimate event datetime.
        entity_ids:     IDs of entities involved.
        source_items:   The raw content item dicts (source_id, title, url, etc.).
        trust_scores:   Per-item trust score (used for deduplication).
        primary_item_id: The highest-trust item (representative).
        duplicate_ids:  Items superseded by *primary_item_id*.
    """

    bundle_id: str
    canonical_title: str
    event_time: Optional[datetime] = None
    entity_ids: List[str] = Field(default_factory=list)
    source_items: List[Dict[str, Any]] = Field(default_factory=list)
    trust_scores: Dict[str, float] = Field(default_factory=dict)
    primary_item_id: str = ""
    duplicate_ids: List[str] = Field(default_factory=list)

    @field_validator("bundle_id")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("'bundle_id' must be non-empty")
        return v

    def size(self) -> int:
        return len(self.source_items)


class DedupeResult(BaseModel, frozen=True):
    """Result of deduplication within a single ``EventBundle``.

    Attributes:
        bundle_id:       The bundle that was deduplicated.
        kept_item_id:    Source ID of the item that was kept.
        removed_ids:     Source IDs of items that were removed.
        similarity_scores: Pairwise similarity scores used.
        strategy:        Deduplication strategy used (``"trust"`` or ``"similarity"``).
    """

    bundle_id: str
    kept_item_id: str
    removed_ids: List[str] = Field(default_factory=list)
    similarity_scores: Dict[str, float] = Field(default_factory=dict)
    strategy: str = "trust"

    @field_validator("strategy")
    @classmethod
    def _valid_strategy(cls, v: str) -> str:
        if v not in {"trust", "similarity", "combined"}:
            raise ValueError(f"'strategy' must be one of trust/similarity/combined, got {v!r}")
        return v

