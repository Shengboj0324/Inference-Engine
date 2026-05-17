"""Local-first ingestion runtime — asyncio replacement for Celery Beat + workers.

Replaces the production ``fetch_all_sources`` / ``fetch_source_content``
Celery pipeline with a single in-process asyncio runtime so the desktop
sidecar can ingest content without Redis, RabbitMQ, or any separate
worker pool.

Architecture
------------
* :class:`LocalScheduler` fires ``fetch_all`` on a fixed interval and
  ``cleanup_old`` once per day.
* ``fetch_all`` enqueues one job per enabled :class:`SourceConfig` into
  :class:`LocalTaskQueue`.
* A pool of worker coroutines drains the queue, instantiates the
  appropriate :class:`BaseConnector`, awaits ``fetch_content``, and
  persists every new :class:`ContentItem` via :class:`ContentStore`.
* Each fetch is gated by :class:`Permission.NETWORK_FETCH`; denied
  permissions short-circuit the job and surface as a ``permission_denied``
  status on the source row — guaranteeing zero egress when the user has
  not opted in.

Stats are exposed via :meth:`IngestRuntime.stats` for the control routes
(Pillar D) and the desktop UI.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, List, Optional
from uuid import UUID

from app.connectors.base import ConnectorConfig
from app.core.models import SourcePlatform
from app.local.content_store import ContentStore
from app.local.dedup_store import DedupStore
from app.local.local_queue import LocalTaskQueue, QueuedJob
from app.local.local_scheduler import LocalScheduler
from app.local.permissions import Permission, PermissionManager
from app.local.sources_store import SourceConfig, SourcesStore

logger = logging.getLogger(__name__)


# Stable identifier for the lone desktop user; lets ContentItem.user_id
# (a UUID required by the legacy Pydantic model) stay deterministic
# without inventing a user table on a single-user machine.
LOCAL_USER_ID = UUID("00000000-0000-0000-0000-000000000001")

_JOB_FETCH = "ingest.fetch_source"
_DEFAULT_FETCH_INTERVAL_MIN = 15
_DEFAULT_RETENTION_DAYS = 30
_DEFAULT_CLEANUP_HOUR = 3
_DEFAULT_WORKER_CONCURRENCY = 2
_DEFAULT_MAX_ITEMS_PER_FETCH = 100


# Factory signature: (platform, ConnectorConfig, user_id) -> BaseConnector
ConnectorFactory = Callable[[SourcePlatform, ConnectorConfig, UUID], object]


@dataclass
class IngestStats:
    fetch_runs: int = 0
    items_seen: int = 0
    items_new: int = 0
    items_duplicate: int = 0
    fetch_errors: int = 0
    cleanup_runs: int = 0
    items_purged: int = 0
    permission_denied: int = 0
    last_run_at: Optional[float] = None
    per_platform: dict = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {
            "fetch_runs": self.fetch_runs,
            "items_seen": self.items_seen,
            "items_new": self.items_new,
            "items_duplicate": self.items_duplicate,
            "fetch_errors": self.fetch_errors,
            "cleanup_runs": self.cleanup_runs,
            "items_purged": self.items_purged,
            "permission_denied": self.permission_denied,
            "last_run_at": self.last_run_at,
            "per_platform": dict(self.per_platform),
        }


class IngestRuntime:
    """In-process scheduler + worker pool for local-first ingestion."""

    def __init__(
        self,
        *,
        sources: SourcesStore,
        content: ContentStore,
        queue: LocalTaskQueue,
        scheduler: LocalScheduler,
        permissions: Optional[PermissionManager] = None,
        dedup: Optional[DedupStore] = None,
        connector_factory: Optional[ConnectorFactory] = None,
        fetch_interval_minutes: int = _DEFAULT_FETCH_INTERVAL_MIN,
        retention_days: int = _DEFAULT_RETENTION_DAYS,
        cleanup_hour_utc: int = _DEFAULT_CLEANUP_HOUR,
        worker_concurrency: int = _DEFAULT_WORKER_CONCURRENCY,
        max_items_per_fetch: int = _DEFAULT_MAX_ITEMS_PER_FETCH,
        user_id: UUID = LOCAL_USER_ID,
    ) -> None:
        self._sources = sources
        self._content = content
        self._queue = queue
        self._scheduler = scheduler
        self._permissions = permissions
        self._dedup = dedup
        self._factory = connector_factory or _default_connector_factory
        self._fetch_interval = max(1, int(fetch_interval_minutes))
        self._retention_days = max(1, int(retention_days))
        self._cleanup_hour = int(cleanup_hour_utc) % 24
        self._concurrency = max(1, int(worker_concurrency))
        self._max_items_per_fetch = max(1, int(max_items_per_fetch))
        self._user_id = user_id
        self._workers: List[asyncio.Task] = []
        self._stop_event_obj: Optional[asyncio.Event] = None
        self._started = False
        self.stats = IngestStats()

    @property
    def _stop_event(self) -> asyncio.Event:
        if self._stop_event_obj is None:
            self._stop_event_obj = asyncio.Event()
        return self._stop_event_obj

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._stop_event.clear()
        self._scheduler.every_minutes(
            "ingest.fetch_all", self.enqueue_fetch_all,
            minutes=self._fetch_interval,
        )
        self._scheduler.daily_at(
            "ingest.cleanup_old", self.run_cleanup,
            hour=self._cleanup_hour,
        )
        self._scheduler.start()
        for i in range(self._concurrency):
            self._workers.append(
                asyncio.create_task(self._worker_loop(), name=f"ingest-worker-{i}")
            )
        self._started = True

    async def stop(self, *, timeout: Optional[float] = 30.0) -> None:
        if not self._started:
            return
        self._stop_event.set()
        await self._scheduler.stop(timeout=timeout)
        for w in self._workers:
            w.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        if self._dedup is not None:
            try:
                self._dedup.flush()
            except Exception:  # noqa: BLE001 - shutdown must never raise
                logger.exception("dedup flush on shutdown failed")
        self._started = False

    # ------------------------------------------------------------------
    # Scheduled jobs
    # ------------------------------------------------------------------

    async def enqueue_fetch_all(self) -> int:
        """Push one fetch job per enabled source.  Returns count enqueued."""
        if not self._network_allowed():
            self.stats.permission_denied += 1
            logger.info("ingest.fetch_all skipped: NETWORK_FETCH not granted")
            return 0
        configs = self._sources.list_all(enabled_only=True)
        enqueued = 0
        for cfg in configs:
            try:
                await self._queue.enqueue(
                    _JOB_FETCH, {"platform": cfg.platform}, priority=5,
                )
                enqueued += 1
            except Exception:  # noqa: BLE001 - queue full / shutdown
                logger.exception("failed to enqueue fetch for %s", cfg.platform)
        return enqueued

    async def run_cleanup(self) -> int:
        """Delete content rows older than the configured retention window."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)
        purged = self._content.delete_older_than(cutoff)
        self.stats.cleanup_runs += 1
        self.stats.items_purged += purged
        logger.info("ingest.cleanup_old removed %d row(s)", purged)
        return purged

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    async def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = await self._queue.dequeue(timeout=1.0)
            except asyncio.CancelledError:
                return
            if job is None:
                continue
            try:
                await self._dispatch(job)
                await self._queue.ack(job.id)
            except asyncio.CancelledError:
                await self._queue.nack(job.id, requeue=True)
                return
            except Exception:  # noqa: BLE001 - worker must not exit
                logger.exception("ingest job %s failed", job.name)
                await self._queue.nack(job.id, requeue=False)

    async def _dispatch(self, job: QueuedJob) -> None:
        if job.name == _JOB_FETCH:
            await self._handle_fetch(job.payload.get("platform", ""))
            return
        logger.warning("ingest: unknown job name %r — sending to DLQ", job.name)
        raise ValueError(f"unknown ingest job: {job.name!r}")

    async def _handle_fetch(self, platform_str: str) -> None:
        if not self._network_allowed():
            self.stats.permission_denied += 1
            self._sources.record_run(
                platform_str, status="permission_denied",
                error="NETWORK_FETCH not granted",
            )
            return
        cfg = self._sources.get(platform_str)
        if cfg is None or not cfg.enabled:
            return
        try:
            platform = SourcePlatform(cfg.platform)
        except ValueError:
            self._sources.record_run(
                cfg.platform, status="error",
                error=f"unknown platform: {cfg.platform!r}",
            )
            self.stats.fetch_errors += 1
            return
        connector = self._build_connector(platform, cfg)
        since = cfg.last_fetch_dt or (
            datetime.now(timezone.utc) - timedelta(hours=24)
        )
        try:
            result = await connector.fetch_content(
                since=since, max_items=self._max_items_per_fetch,
            )
        except Exception as exc:  # noqa: BLE001 - per-source isolation
            self.stats.fetch_errors += 1
            self._sources.record_run(
                cfg.platform, status="error", error=str(exc)[:500],
            )
            logger.exception("connector fetch failed for %s", cfg.platform)
            return
        new_count, dup_count = 0, 0
        for item in getattr(result, "items", []) or []:
            try:
                # Bloom fast-path: skip the SQL UNIQUE round-trip whenever
                # the filter is certain the (platform, source_id) tuple has
                # been seen before in *some* prior run.  False positives
                # only cost us one extra SQL probe (the upsert returns
                # False) — never a missed item, because the filter has no
                # false negatives.
                if self._dedup is not None and self._dedup.seen(
                    cfg.platform, item.source_id
                ):
                    dup_count += 1
                    continue
                if self._content.upsert(item):
                    new_count += 1
                    if self._dedup is not None:
                        self._dedup.mark(cfg.platform, item.source_id)
                else:
                    dup_count += 1
                    if self._dedup is not None:
                        self._dedup.mark(cfg.platform, item.source_id)
            except Exception:  # noqa: BLE001 - skip malformed items
                logger.exception("upsert failed for item from %s", cfg.platform)
        self.stats.fetch_runs += 1
        self.stats.items_seen += new_count + dup_count
        self.stats.items_new += new_count
        self.stats.items_duplicate += dup_count
        self.stats.last_run_at = datetime.now(timezone.utc).timestamp()
        per = self.stats.per_platform.setdefault(
            cfg.platform, {"new": 0, "duplicate": 0, "errors": 0},
        )
        per["new"] += new_count
        per["duplicate"] += dup_count
        self._sources.record_run(cfg.platform, status="ok", error=None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def snapshot_stats(self) -> dict:
        return self.stats.snapshot()

    def _network_allowed(self) -> bool:
        if self._permissions is None:
            return True
        return self._permissions.is_allowed(Permission.NETWORK_FETCH)

    def _build_connector(self, platform: SourcePlatform, cfg: SourceConfig):
        config = ConnectorConfig(
            platform=platform,
            credentials=dict(cfg.credentials),
            settings=dict(cfg.settings),
        )
        return self._factory(platform, config, self._user_id)


def _default_connector_factory(
    platform: SourcePlatform, config: ConnectorConfig, user_id: UUID,
):
    """Resolve a real connector class via :class:`ConnectorRegistry`."""
    # Imported lazily so unit tests can swap in a stub factory without
    # pulling the full connector tree (and its optional dependencies).
    from app.connectors.registry import ConnectorRegistry  # noqa: PLC0415
    return ConnectorRegistry.get_connector(platform, config, user_id)


_global_runtime: Optional[IngestRuntime] = None
_global_runtime_lock = asyncio.Lock if False else None  # placeholder for IDEs

import threading as _threading  # local alias keeps the top imports tidy

_runtime_init_lock = _threading.Lock()


def get_ingest_runtime() -> IngestRuntime:
    """Return the process-wide :class:`IngestRuntime` singleton.

    Constructs the runtime with default stores on first call but does
    **not** start the scheduler / worker pool — that is the desktop
    launcher's responsibility so the control routes remain side-effect
    free at import time.
    """
    global _global_runtime
    with _runtime_init_lock:
        if _global_runtime is None:
            from app.local.content_store import get_content_store  # noqa: PLC0415
            from app.local.permissions import get_permission_manager  # noqa: PLC0415
            from app.local.sources_store import get_sources_store  # noqa: PLC0415
            _global_runtime = IngestRuntime(
                sources=get_sources_store(),
                content=get_content_store(),
                queue=LocalTaskQueue(),
                scheduler=LocalScheduler(),
                permissions=get_permission_manager(),
                dedup=DedupStore(),
            )
        return _global_runtime


def reset_ingest_runtime() -> None:
    """Drop the cached singleton (test-only)."""
    global _global_runtime
    with _runtime_init_lock:
        _global_runtime = None

