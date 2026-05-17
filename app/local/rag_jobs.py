"""In-process background reindex job manager for the local RAG layer.

Replaces the synchronous ``POST /rag/reindex`` for large corpora.  Each
job runs as a long-lived ``asyncio.Task`` that drains stale embeddings
batch-by-batch, updates a shared :class:`ReindexJob` snapshot the UI
can poll, and exits cleanly on cancellation.

State is process-local — losing the sidecar mid-job means the operator
re-fires the start endpoint; stale embeddings remain selectable so no
work is lost beyond the in-flight batch.  This is the right trade-off
for a single-user desktop tool: avoids a job table on disk while still
giving the UI a progress signal.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)

#: Job status state machine.  Linear and small on purpose; the UI maps
#: each value to an icon and never branches further.
_STATUS_QUEUED = "queued"
_STATUS_RUNNING = "running"
_STATUS_DONE = "done"
_STATUS_CANCELLED = "cancelled"
_STATUS_ERROR = "error"


@dataclass
class ReindexJob:
    """Snapshot of a single background reindex run."""

    id: str
    status: str = _STATUS_QUEUED
    scanned: int = 0
    embedded: int = 0
    skipped: int = 0
    errors: int = 0
    total: int = 0
    batch_size: int = 16
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    detail: Optional[str] = None
    _task: Optional[asyncio.Task] = field(default=None, repr=False)
    _cancel: bool = field(default=False, repr=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "status": self.status,
            "scanned": self.scanned, "embedded": self.embedded,
            "skipped": self.skipped, "errors": self.errors, "total": self.total,
            "batch_size": self.batch_size,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "detail": self.detail,
        }


# Callable signature implemented by ``app.api.routes.rag``: takes a
# ``ReindexJob`` it mutates in-place and a ``should_cancel`` predicate
# the worker polls between batches.  Injected from the route so the
# job manager has zero knowledge of ContentStore / embedder internals.
WorkerCallable = Callable[
    ["ReindexJob", Callable[[], bool]], Awaitable[None]
]


class ReindexJobManager:
    """Thread- and asyncio-safe registry of in-flight reindex jobs."""

    def __init__(self, *, max_history: int = 20) -> None:
        self._jobs: Dict[str, ReindexJob] = {}
        self._lock = threading.RLock()
        self._max_history = max_history

    def start(
        self, worker: WorkerCallable, *, batch_size: int = 16, total: int = 0,
    ) -> ReindexJob:
        """Schedule ``worker`` and return the freshly-created job snapshot."""
        job = ReindexJob(
            id=uuid.uuid4().hex, batch_size=batch_size, total=total,
            started_at=time.time(),
        )
        with self._lock:
            # Refuse to start a second job while one is running; the UI
            # is expected to wait or cancel the prior job first.
            for existing in self._jobs.values():
                if existing.status in (_STATUS_QUEUED, _STATUS_RUNNING):
                    raise RuntimeError(
                        f"reindex job {existing.id} is already {existing.status}; "
                        "cancel it before starting a new one"
                    )
            self._jobs[job.id] = job
            self._trim_locked()
        job.status = _STATUS_RUNNING
        job._task = asyncio.create_task(self._run(job, worker), name=f"rag-reindex-{job.id}")
        return job

    async def _run(self, job: ReindexJob, worker: WorkerCallable) -> None:
        try:
            await worker(job, lambda: job._cancel)
        except asyncio.CancelledError:
            job.status = _STATUS_CANCELLED
            job.detail = "cancelled by operator"
            raise
        except Exception as exc:  # noqa: BLE001 - long-running surface
            logger.exception("reindex job %s failed", job.id)
            job.status = _STATUS_ERROR
            job.detail = str(exc)[:500]
        else:
            if job._cancel:
                job.status = _STATUS_CANCELLED
                job.detail = job.detail or "cancelled by operator"
            else:
                job.status = _STATUS_DONE
        finally:
            job.finished_at = time.time()

    def cancel(self, job_id: str) -> bool:
        """Mark ``job_id`` for cancellation.  Returns True if it was running."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.status not in (_STATUS_QUEUED, _STATUS_RUNNING):
                return False
            job._cancel = True
        # Cancel the asyncio task so the worker's batch await wakes up.
        if job._task is not None and not job._task.done():
            job._task.cancel()
        return True

    def get(self, job_id: str) -> Optional[ReindexJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> List[ReindexJob]:
        with self._lock:
            return sorted(
                self._jobs.values(),
                key=lambda j: j.started_at or 0.0, reverse=True,
            )

    def _trim_locked(self) -> None:
        """Drop finished jobs beyond ``max_history`` (LRU by start time)."""
        if len(self._jobs) <= self._max_history:
            return
        finished = [j for j in self._jobs.values()
                    if j.status in (_STATUS_DONE, _STATUS_CANCELLED, _STATUS_ERROR)]
        finished.sort(key=lambda j: j.finished_at or 0.0)
        while len(self._jobs) > self._max_history and finished:
            self._jobs.pop(finished.pop(0).id, None)


_global_manager: Optional[ReindexJobManager] = None
_global_manager_lock = threading.Lock()


def get_reindex_jobs() -> ReindexJobManager:
    global _global_manager
    with _global_manager_lock:
        if _global_manager is None:
            _global_manager = ReindexJobManager()
        return _global_manager


def reset_reindex_jobs() -> None:
    global _global_manager
    with _global_manager_lock:
        _global_manager = None
