"""Source capability registry.

Maintains a thread-safe catalogue of every monitored source with its
capability contract.  Each entry (``SourceSpec``) records:

- Which content *families* the source belongs to (``SourceFamily``)
- What ingestion *capabilities* it supports (``SourceCapability`` flags)
- Authentication requirements, content types, and rate-limit metadata

The ``SourceRegistryStore`` exposes CRUD operations and lookup by family /
capability.  It is the authoritative record for the Source Intelligence Layer.

Typical usage::

    registry = SourceRegistryStore()
    registry.register(SourceSpec(
        source_id="openai/openai-python",
        platform=SourcePlatform.GITHUB_RELEASES,
        family=SourceFamily.DEVELOPER_RELEASE,
        capabilities={SourceCapability.SUPPORTS_SINCE, SourceCapability.VERSIONED},
        display_name="OpenAI Python SDK Releases",
        homepage="https://github.com/openai/openai-python",
    ))
    specs = registry.list_by_family(SourceFamily.RESEARCH)
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set

from app.core.models import SourcePlatform

logger = logging.getLogger(__name__)


class SourceFamily(str, Enum):
    """High-level grouping of sources for the Source Intelligence Layer."""

    SOCIAL = "social"
    NEWS = "news"
    DEVELOPER_RELEASE = "developer_release"
    RESEARCH = "research"
    MEDIA_AUDIO = "media_audio"
    UNKNOWN = "unknown"


class SourceCapability(Enum):
    """Bitmask-compatible capability flags for sources."""

    SUPPORTS_SINCE = auto()     # connector honours `since` param for incremental fetch
    VERSIONED = auto()          # content carries explicit version/tag identifiers
    HAS_STRUCTURED_DATA = auto()  # response is machine-parseable JSON/XML (vs. scraped)
    SUPPORTS_SEARCH = auto()    # supports keyword/query search (not just chronological)
    PROVIDES_FULL_TEXT = auto() # full body available without follow-up HTTP call
    PROVIDES_PDF = auto()       # source may provide PDF binaries (Phase 2: extraction)
    PROVIDES_AUDIO = auto()     # source may provide audio files (Phase 2: Whisper)
    PROVIDES_TRANSCRIPT = auto()# transcript text already available
    REQUIRES_AUTH = auto()      # auth is mandatory (not just recommended)
    AUTH_OPTIONAL = auto()      # auth improves rate limits but is not required
    RATE_SENSITIVE = auto()     # very low rate limit; requires back-off strategy


@dataclass
class SourceSpec:
    """Capability contract for a single monitored source.

    Attributes:
        source_id:     Unique identifier (e.g. ``"openai/openai-python"``,
                       ``"cs.AI"`` for arXiv category, ``"lex_fridman"`` for podcast).
        platform:      The ``SourcePlatform`` enum value.
        family:        ``SourceFamily`` group this source belongs to.
        capabilities:  Set of ``SourceCapability`` flags.
        display_name:  Human-readable name for UI/logging.
        homepage:      Canonical URL for the source (may be empty).
        rate_limit_rph: Requests per hour (0 = unknown).
        auth_token_env: Environment variable name holding the credential.
        metadata:      Arbitrary extra metadata for downstream consumers.
    """

    source_id: str
    platform: SourcePlatform
    family: SourceFamily
    capabilities: FrozenSet[SourceCapability] = field(default_factory=frozenset)
    display_name: str = ""
    homepage: str = ""
    rate_limit_rph: int = 0
    auth_token_env: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id or not self.source_id.strip():
            raise ValueError("'source_id' must be a non-empty string")
        if not isinstance(self.platform, SourcePlatform):
            raise TypeError(f"'platform' must be SourcePlatform, got {type(self.platform)!r}")
        if not isinstance(self.family, SourceFamily):
            raise TypeError(f"'family' must be SourceFamily, got {type(self.family)!r}")
        # Normalise capabilities to a frozenset for hashability
        if not isinstance(self.capabilities, frozenset):
            self.capabilities = frozenset(self.capabilities)

    def has(self, cap: SourceCapability) -> bool:
        """Return True if this source has *cap*."""
        return cap in self.capabilities


class SourceRegistryStore:
    """Thread-safe store of ``SourceSpec`` objects.

    Supports:
    - ``register(spec)``               — add or replace a source spec
    - ``deregister(source_id)``        — remove a source spec
    - ``get(source_id)``               — lookup by ID
    - ``list_by_family(family)``       — filter by ``SourceFamily``
    - ``list_by_capability(cap)``      — filter by ``SourceCapability``
    - ``list_by_platform(platform)``   — filter by ``SourcePlatform``
    - ``all_specs()``                  — snapshot of all registered specs
    """

    def __init__(self) -> None:
        self._store: Dict[str, SourceSpec] = {}
        self._lock = threading.Lock()

    def register(self, spec: SourceSpec) -> None:
        """Register or update a ``SourceSpec``.

        Args:
            spec: The source specification to store.

        Raises:
            TypeError: If *spec* is not a ``SourceSpec``.
        """
        if not isinstance(spec, SourceSpec):
            raise TypeError(f"Expected SourceSpec, got {type(spec)!r}")
        t0 = time.perf_counter()
        with self._lock:
            existed = spec.source_id in self._store
            self._store[spec.source_id] = spec
        logger.debug(
            "SourceRegistryStore.register: source_id=%r platform=%s action=%s latency_ms=%.2f",
            spec.source_id, spec.platform.value, "updated" if existed else "added",
            (time.perf_counter() - t0) * 1000,
        )

    def deregister(self, source_id: str) -> bool:
        """Remove a source spec.

        Returns:
            True if it existed and was removed; False if not found.
        """
        t0 = time.perf_counter()
        with self._lock:
            existed = source_id in self._store
            if existed:
                del self._store[source_id]
        logger.debug(
            "SourceRegistryStore.deregister: source_id=%r found=%s latency_ms=%.2f",
            source_id, existed, (time.perf_counter() - t0) * 1000,
        )
        return existed

    def get(self, source_id: str) -> Optional[SourceSpec]:
        """Return the ``SourceSpec`` for *source_id*, or ``None``."""
        if not source_id:
            raise ValueError("'source_id' must be a non-empty string")
        with self._lock:
            return self._store.get(source_id)

    def list_by_family(self, family: SourceFamily) -> List[SourceSpec]:
        """Return all specs belonging to *family*."""
        if not isinstance(family, SourceFamily):
            raise TypeError(f"'family' must be SourceFamily, got {type(family)!r}")
        with self._lock:
            return [s for s in self._store.values() if s.family == family]

    def list_by_capability(self, cap: SourceCapability) -> List[SourceSpec]:
        """Return all specs that have *cap*."""
        if not isinstance(cap, SourceCapability):
            raise TypeError(f"'cap' must be SourceCapability, got {type(cap)!r}")
        with self._lock:
            return [s for s in self._store.values() if s.has(cap)]

    def list_by_platform(self, platform: SourcePlatform) -> List[SourceSpec]:
        """Return all specs for *platform*."""
        if not isinstance(platform, SourcePlatform):
            raise TypeError(f"'platform' must be SourcePlatform, got {type(platform)!r}")
        with self._lock:
            return [s for s in self._store.values() if s.platform == platform]

    def all_specs(self) -> List[SourceSpec]:
        """Return a snapshot of all registered specs (ordered by source_id)."""
        with self._lock:
            return sorted(self._store.values(), key=lambda s: s.source_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

