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
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

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
    priority: float = 0.5  # Acquisition scheduling priority in [0, 1]; 1 = highest

    def __post_init__(self) -> None:
        if not self.source_id or not self.source_id.strip():
            raise ValueError("'source_id' must be a non-empty string")
        if not isinstance(self.platform, SourcePlatform):
            raise TypeError(f"'platform' must be SourcePlatform, got {type(self.platform)!r}")
        if not isinstance(self.family, SourceFamily):
            raise TypeError(f"'family' must be SourceFamily, got {type(self.family)!r}")
        if not (0.0 <= self.priority <= 1.0):
            raise ValueError(f"'priority' must be in [0, 1], got {self.priority!r}")
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

    def next_batch(
        self,
        n: int,
        *,
        min_priority: float = 0.0,
        family: Optional[SourceFamily] = None,
    ) -> List[SourceSpec]:
        """Return up to *n* specs ordered by priority (descending).

        Args:
            n:            Maximum number of specs to return.
            min_priority: Only return specs with ``priority >= min_priority``.
            family:       If given, restrict to specs from this family.

        Returns:
            List of ``SourceSpec`` sorted by priority descending, length ≤ n.

        Raises:
            ValueError: If *n* is not a positive integer or *min_priority* is
                        outside [0, 1].
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"'n' must be a positive int, got {n!r}")
        if not (0.0 <= min_priority <= 1.0):
            raise ValueError(f"'min_priority' must be in [0, 1], got {min_priority!r}")
        with self._lock:
            candidates = list(self._store.values())
        if family is not None:
            candidates = [s for s in candidates if s.family == family]
        candidates = [s for s in candidates if s.priority >= min_priority]
        candidates.sort(key=lambda s: (-s.priority, s.source_id))
        return candidates[:n]

    def update_priority(self, source_id: str, priority: float) -> bool:
        """Update the acquisition priority of an existing source spec.

        Args:
            source_id: Identifier of the spec to update.
            priority:  New priority in [0, 1].

        Returns:
            ``True`` if the spec was found and updated; ``False`` if not found.

        Raises:
            ValueError: If *priority* is outside [0, 1].
        """
        if not (0.0 <= priority <= 1.0):
            raise ValueError(f"'priority' must be in [0, 1], got {priority!r}")
        with self._lock:
            spec = self._store.get(source_id)
            if spec is None:
                return False
            import dataclasses
            self._store[source_id] = dataclasses.replace(spec, priority=priority)
        logger.debug(
            "SourceRegistryStore.update_priority: source_id=%r priority=%.3f",
            source_id, priority,
        )
        return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# ---------------------------------------------------------------------------
# AcquisitionScheduler
# ---------------------------------------------------------------------------

class AcquisitionScheduler:
    """Priority-weighted acquisition scheduler with exponential back-off.

    Wraps a ``SourceRegistryStore`` and adds:

    1. **Priority-weighted ordering** — ``next_batch()`` returns sources in
       descending priority order, respecting any active back-off windows.
    2. **Configurable retry/back-off** — consecutive failures trigger an
       exponentially increasing cooldown period.  After *max_retries* failures
       the source is suspended until ``record_success()`` is called.
    3. **Trust gate** — When a ``SourceTrustScorer`` is injected, sources whose
       composite trust score falls below *min_authority_threshold* are silently
       excluded from ``next_batch()``.
    4. **HealthMonitor integration** — ``record_failure()`` calls
       ``health_monitor.record_connector_failure(source_id)``; ``record_success()``
       calls ``health_monitor.record_connector_success(source_id)`` so the
       ``PipelineHealthMonitor`` circuit-breaker state stays accurate.

    All mutable state is protected by ``threading.Lock``.

    Args:
        registry:               The ``SourceRegistryStore`` to schedule from.
        trust_scorer:           Optional ``SourceTrustScorer``; when ``None``
                                the trust gate is disabled.
        min_authority_threshold: Minimum composite trust score for a source to
                                 be eligible (default ``0.0`` = no gate).
        base_backoff_s:         Initial back-off window in seconds (default 60).
        max_backoff_s:          Cap on back-off window (default 3 600 = 1 h).
        max_retries:            Consecutive failures before source is suspended
                                indefinitely (default 5).
        health_monitor:         Optional monitor; receives failure/success calls.
    """

    def __init__(
        self,
        registry: "SourceRegistryStore",
        trust_scorer: Optional[Any] = None,
        min_authority_threshold: float = 0.0,
        base_backoff_s: float = 60.0,
        max_backoff_s: float = 3600.0,
        max_retries: int = 5,
        health_monitor: Optional[Any] = None,
    ) -> None:
        if not isinstance(registry, SourceRegistryStore):
            raise TypeError(
                f"'registry' must be SourceRegistryStore, got {type(registry)!r}"
            )
        if not (0.0 <= min_authority_threshold <= 1.0):
            raise ValueError(
                f"'min_authority_threshold' must be in [0, 1], got {min_authority_threshold!r}"
            )
        if base_backoff_s <= 0:
            raise ValueError(f"'base_backoff_s' must be > 0, got {base_backoff_s!r}")
        if max_backoff_s < base_backoff_s:
            raise ValueError(
                f"'max_backoff_s' must be >= base_backoff_s ({base_backoff_s}), "
                f"got {max_backoff_s!r}"
            )
        if not isinstance(max_retries, int) or max_retries < 1:
            raise ValueError(f"'max_retries' must be a positive int, got {max_retries!r}")

        self._registry  = registry
        self._scorer    = trust_scorer
        self._min_trust = min_authority_threshold
        self._base_s    = base_backoff_s
        self._max_s     = max_backoff_s
        self._max_retries = max_retries
        self._monitor   = health_monitor
        self._lock      = threading.Lock()
        # failure_counts[source_id] = consecutive failure count
        self._failure_counts: Dict[str, int] = {}
        # backoff_until[source_id] = monotonic timestamp when back-off expires
        self._backoff_until: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next_batch(self, n: int, *, family: Optional[SourceFamily] = None) -> List[SourceSpec]:
        """Return up to *n* eligible specs sorted by priority (descending).

        Eligibility requires:
        - Source is not in an active back-off window.
        - Source has not exceeded ``max_retries`` (suspended).
        - Source composite trust ≥ ``min_authority_threshold`` (when scorer set).

        Args:
            n:      Maximum number of specs to return (must be ≥ 1).
            family: Optional family filter applied before eligibility check.

        Returns:
            List of ``SourceSpec`` objects, length ≤ n.

        Raises:
            ValueError: If *n* < 1.
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"'n' must be a positive int, got {n!r}")
        candidates = self._registry.next_batch(
            len(self._registry._store) + 1,  # fetch all then filter
            family=family,
        )
        eligible = [s for s in candidates if self.is_eligible(s.source_id)]
        return eligible[:n]

    def record_failure(self, source_id: str) -> None:
        """Record one acquisition failure for *source_id*.

        Applies exponential back-off: ``backoff = base * 2^(failures - 1)``
        capped at ``max_backoff_s``.  After ``max_retries`` failures the source
        is suspended (back-off = ``max_backoff_s``) until ``record_success`` is
        called.

        Surfaces the failure to the injected ``health_monitor`` via
        ``record_connector_failure(source_id)``.

        Args:
            source_id: Identifier of the source that failed.

        Raises:
            ValueError: If *source_id* is empty.
        """
        if not source_id or not source_id.strip():
            raise ValueError("'source_id' must be a non-empty string")
        now = time.monotonic()
        with self._lock:
            count = self._failure_counts.get(source_id, 0) + 1
            self._failure_counts[source_id] = count
            exponent = min(count - 1, self._max_retries - 1)
            backoff = min(self._base_s * (2 ** exponent), self._max_s)
            self._backoff_until[source_id] = now + backoff

        logger.warning(
            "AcquisitionScheduler: source=%r failure #%d backoff=%.0fs",
            source_id, count, backoff,
        )
        if self._monitor is not None:
            try:
                self._monitor.record_connector_failure(source_id)
            except Exception as exc:
                logger.warning(
                    "AcquisitionScheduler: health_monitor.record_connector_failure failed: %s", exc
                )

    def record_success(self, source_id: str) -> None:
        """Record a successful acquisition; resets back-off and failure count.

        Surfaces the success to the injected ``health_monitor`` via
        ``record_connector_success(source_id)``.

        Args:
            source_id: Identifier of the source that succeeded.

        Raises:
            ValueError: If *source_id* is empty.
        """
        if not source_id or not source_id.strip():
            raise ValueError("'source_id' must be a non-empty string")
        with self._lock:
            self._failure_counts.pop(source_id, None)
            self._backoff_until.pop(source_id, None)

        logger.debug("AcquisitionScheduler: source=%r success — back-off cleared", source_id)
        if self._monitor is not None:
            try:
                self._monitor.record_connector_success(source_id)
            except Exception as exc:
                logger.warning(
                    "AcquisitionScheduler: health_monitor.record_connector_success failed: %s", exc
                )

    def is_eligible(self, source_id: str) -> bool:
        """Return ``True`` when *source_id* is eligible for acquisition.

        A source is eligible when:
        1. It is registered in the registry.
        2. Its back-off window has expired (or it has never failed).
        3. Its failure count is below ``max_retries``.
        4. Its composite trust score ≥ ``min_authority_threshold`` (when a
           scorer is configured).

        Args:
            source_id: Identifier to check.

        Returns:
            ``bool``
        """
        if not source_id:
            return False
        # Registry check
        if self._registry.get(source_id) is None:
            return False
        now = time.monotonic()
        with self._lock:
            count  = self._failure_counts.get(source_id, 0)
            until  = self._backoff_until.get(source_id, 0.0)
        # Suspended?
        if count >= self._max_retries:
            return False
        # In active back-off?
        if now < until:
            return False
        # Trust gate
        if self._scorer is not None:
            try:
                ts = self._scorer.score(source_id)
                if ts.composite < self._min_trust:
                    return False
            except Exception as exc:
                logger.warning(
                    "AcquisitionScheduler.is_eligible: scorer failed for %r: %s",
                    source_id, exc,
                )
        return True

    def backoff_remaining_s(self, source_id: str) -> float:
        """Return remaining back-off seconds for *source_id* (0.0 if eligible).

        Args:
            source_id: Source identifier.

        Returns:
            Seconds until the source is eligible again (0.0 = already eligible).
        """
        now = time.monotonic()
        with self._lock:
            until = self._backoff_until.get(source_id, 0.0)
        return max(0.0, until - now)

    def failure_count(self, source_id: str) -> int:
        """Return the current consecutive failure count for *source_id*.

        Args:
            source_id: Source identifier.

        Returns:
            Non-negative integer.
        """
        with self._lock:
            return self._failure_counts.get(source_id, 0)

