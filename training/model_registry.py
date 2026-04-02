"""Training model artifact registry.

``ModelArtifactRegistry`` provides versioned tracking of all model training
artifacts (checkpoints + calibration runs) and controls which artifact is
the active production deployment.

Design goals
------------
- **Single source of truth** for "what is in production?".  The CI
  ``eval-gate`` writes passing checkpoints to ``training/checkpoints/``;
  the registry wraps those files and adds deployment-state metadata.
- **Promote / rollback** — production promotion is explicit; rolling back
  simply re-promotes the previous production artifact.
- **Persistence** — the registry state is stored as a plain JSON file
  (``training/model_registry.json``) so it survives process restarts.
- **Gate enforcement** — ``get_production_candidate()`` only returns
  artifacts that pass both ECE ≤ 0.10 and macro_F1 ≥ 0.70.

Typical usage::

    registry = ModelArtifactRegistry()

    # Register a newly calibrated checkpoint
    artifact_id = registry.register_from_checkpoint(
        path=Path("training/checkpoints/z_calibrated_epoch_005_isotonic_ece_0.0103_*.json")
    )

    # Promote to production
    registry.promote(artifact_id)

    # Look up what is in production
    prod = registry.get_production()
    print(prod.ece, prod.macro_f1, prod.calibration_method)

    # Roll back if something goes wrong
    registry.rollback()
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

_REGISTRY_PATH   = Path(__file__).parent / "model_registry.json"
_AUDIT_LOG_PATH  = Path(__file__).parent / "model_audit_log.jsonl"

ECE_THRESHOLD      = 0.10
MACRO_F1_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class ArtifactRecord(BaseModel):
    """Metadata for one model training / calibration artifact.

    Attributes
    ----------
    artifact_id:
        Auto-generated UUID identifying this record.
    model_name:
        Human-readable model name (e.g. ``"signal_classifier"``).
    epoch:
        Training epoch number.
    ece:
        Expected Calibration Error on the validation split.
    macro_f1:
        Macro-averaged F1 on the validation split.
    calibration_method:
        ``"temperature_scaling"``, ``"isotonic"``, or ``None`` for uncalibrated.
    checkpoint_path:
        Relative or absolute path to the checkpoint JSON file.
    registered_at:
        UTC timestamp of registration.
    promoted_at:
        UTC timestamp of most recent production promotion, or ``None``.
    is_production:
        ``True`` when this artifact is the current production deployment.
    notes:
        Free-form annotation (e.g. reason for rollback).
    """

    artifact_id:        str      = Field(default_factory=lambda: str(uuid4()))
    model_name:         str      = "signal_classifier"
    epoch:              int
    ece:                float
    macro_f1:           float
    calibration_method: Optional[str] = None
    checkpoint_path:    str
    registered_at:      datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    promoted_at:        Optional[datetime] = None
    is_production:      bool     = False
    notes:              str      = ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def passes_gate(
        self,
        ece_threshold:   float = ECE_THRESHOLD,
        f1_threshold:    float = MACRO_F1_THRESHOLD,
    ) -> bool:
        """Return True when this artifact satisfies the deployment thresholds."""
        return self.ece <= ece_threshold and self.macro_f1 >= f1_threshold

    def gate_failures(self) -> List[str]:
        """Return human-readable list of threshold violations (empty = all pass)."""
        failures = []
        if self.ece > ECE_THRESHOLD:
            failures.append(
                f"ECE={self.ece:.6f} exceeds threshold of {ECE_THRESHOLD}"
            )
        if self.macro_f1 < MACRO_F1_THRESHOLD:
            failures.append(
                f"macro_F1={self.macro_f1:.6f} below threshold of {MACRO_F1_THRESHOLD}"
            )
        return failures


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ModelArtifactRegistry:
    """Versioned artifact registry with promote / rollback support.

    Args:
        registry_path:  Path to the JSON file used for persistence.  Defaults
            to ``training/model_registry.json``.
        audit_log_path: Path to the append-only JSONL audit log.  Defaults to
            ``training/model_audit_log.jsonl``.  Pass an alternative path (e.g.
            a ``tmp_path`` in tests) to avoid writing to the real audit log.
    """

    def __init__(
        self,
        registry_path:  Path = _REGISTRY_PATH,
        audit_log_path: Path = _AUDIT_LOG_PATH,
    ) -> None:
        self._path       = registry_path
        self._audit_path = audit_log_path
        self._lock  = threading.Lock()
        self._store: Dict[str, ArtifactRecord] = {}
        self._load()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, record: ArtifactRecord) -> str:
        """Add *record* to the registry without promoting it.

        Returns:
            The ``artifact_id``.

        Raises:
            ValueError: If an artifact with the same ``artifact_id`` already
                exists.
        """
        with self._lock:
            if record.artifact_id in self._store:
                raise ValueError(
                    f"Artifact {record.artifact_id!r} is already registered; "
                    "create a new record instead of re-registering."
                )
            self._store[record.artifact_id] = record
            self._save()
        return record.artifact_id

    def register_from_checkpoint(
        self,
        path: Path,
        model_name: str = "signal_classifier",
        notes:      str = "",
    ) -> str:
        """Parse a calibration checkpoint JSON and register it.

        Supports both ``ece`` and ``val_ece`` field aliases.

        Returns:
            The ``artifact_id`` of the registered record.

        Raises:
            FileNotFoundError: If *path* does not exist.
            KeyError:          If required fields are missing from the JSON.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))

        ece      = data.get("ece") or data.get("val_ece")
        macro_f1 = data.get("macro_f1")
        epoch    = data.get("epoch")

        if ece is None or epoch is None:
            raise KeyError(
                f"Checkpoint {path.name!r} must contain 'ece' (or 'val_ece') "
                f"and 'epoch' fields; found keys: {list(data.keys())}"
            )

        record = ArtifactRecord(
            model_name=model_name,
            epoch=int(epoch),
            ece=float(ece),
            macro_f1=float(macro_f1) if macro_f1 is not None else 0.0,
            calibration_method=data.get("calibration_method"),
            checkpoint_path=str(path),
            notes=notes,
        )
        return self.register(record)

    # ------------------------------------------------------------------
    # Production promotion / rollback
    # ------------------------------------------------------------------

    def promote(self, artifact_id: str, notes: str = "") -> ArtifactRecord:
        """Set *artifact_id* as the production deployment.

        Every call appends one JSON entry to the immutable audit log at
        ``audit_log_path``.

        Raises:
            KeyError:   If *artifact_id* is not registered.
            ValueError: If the artifact does not pass the deployment gate.
        """
        with self._lock:
            record = self._store.get(artifact_id)
            if record is None:
                raise KeyError(f"Artifact {artifact_id!r} not found in registry")
            failures = record.gate_failures()
            if failures:
                raise ValueError(
                    f"Cannot promote artifact {artifact_id!r}: {'; '.join(failures)}"
                )
            # Demote all others
            for other in self._store.values():
                if other.artifact_id != artifact_id and other.is_production:
                    self._store[other.artifact_id] = other.model_copy(
                        update={"is_production": False}
                    )
            # Promote
            self._store[artifact_id] = record.model_copy(
                update={
                    "is_production": True,
                    "promoted_at": datetime.now(timezone.utc),
                }
            )
            self._save()
            self._append_audit_log(
                event="promote",
                artifact_id=artifact_id,
                notes=notes,
            )
        return self._store[artifact_id]

    def check_and_rollback(
        self,
        monitor: object,
        notes: str = "",
    ) -> Optional["ArtifactRecord"]:
        """Trigger a rollback when the ``PipelineHealthMonitor`` reports RED.

        This is the single integration point between model deployment and
        operational health monitoring.  It queries :meth:`monitor.health_report`
        and, if the report's ``overall_status`` is ``SLOStatus.RED``, calls
        :meth:`rollback` automatically.

        No action is taken for GREEN or YELLOW status — the current production
        artifact is preserved.

        Args:
            monitor: A :class:`~app.intelligence.health_monitor.PipelineHealthMonitor`
                     instance, or any object that has a ``health_report()`` method
                     returning an object with an ``overall_status`` attribute
                     (duck-typing is intentional to avoid a hard circular import).
            notes:   Optional annotation forwarded to :meth:`rollback` and the
                     audit log when a rollback is triggered.

        Returns:
            The newly promoted :class:`ArtifactRecord` if a rollback was performed,
            or ``None`` when:
            - Status is not RED (no rollback needed).
            - A rollback was attempted but no fallback artifact exists.

        Raises:
            TypeError:  If *monitor* does not have a ``health_report()`` callable.
            RuntimeError: If ``monitor.health_report()`` itself raises.

        Examples::

            monitor = PipelineHealthMonitor()
            monitor.record_ece(0.25)   # ECE way above threshold → RED
            rolled_back = registry.check_and_rollback(monitor, notes="auto-rollback")
            if rolled_back:
                print("Rolled back to:", rolled_back.artifact_id)
        """
        if not callable(getattr(monitor, "health_report", None)):
            raise TypeError(
                "'monitor' must have a callable 'health_report' method; "
                f"got {type(monitor)!r}"
            )

        import logging as _logging
        _log = _logging.getLogger(__name__)

        try:
            report = monitor.health_report()
        except Exception as exc:
            raise RuntimeError(
                f"ModelArtifactRegistry.check_and_rollback: "
                f"monitor.health_report() failed: {exc}"
            ) from exc

        # Lazy import to avoid circular dependency at module import time.
        try:
            from app.intelligence.health_monitor import SLOStatus
            is_red = report.overall_status == SLOStatus.RED
        except ImportError:
            # Duck-type fallback: compare the string value
            is_red = str(getattr(report.overall_status, "value", report.overall_status)) == "red"

        if not is_red:
            _log.debug(
                "ModelArtifactRegistry.check_and_rollback: status=%s — no action",
                report.overall_status,
            )
            return None

        _log.warning(
            "ModelArtifactRegistry.check_and_rollback: RED health status detected; "
            "triggering automatic rollback. violations=%d",
            len(report.violations),
        )
        rb_notes = notes or (
            f"Automatic rollback triggered by RED health status "
            f"({len(report.violations)} violation(s))"
        )
        return self.rollback(notes=rb_notes)

    def rollback(self, notes: str = "") -> Optional[ArtifactRecord]:
        """Demote the current production artifact and re-promote the previous one.

        "Previous" is defined as the most recently promoted passing artifact
        that is not currently in production.

        Returns:
            The newly promoted artifact, or ``None`` if no fallback exists.
        """
        with self._lock:
            # Find current production
            current = self._get_production_unlocked()
            # Collect all passing artifacts sorted by promoted_at (most recent first)
            candidates = sorted(
                [a for a in self._store.values() if a.passes_gate()],
                key=lambda a: (a.promoted_at or datetime.min.replace(tzinfo=timezone.utc)),
                reverse=True,
            )
            # Pick the first one that is NOT the current production
            fallback = next(
                (c for c in candidates if current is None or c.artifact_id != current.artifact_id),
                None,
            )
            if fallback is None:
                return None
            # Demote current
            if current:
                self._store[current.artifact_id] = current.model_copy(
                    update={"is_production": False, "notes": notes or "Rolled back"}
                )
            # Promote fallback
            self._store[fallback.artifact_id] = fallback.model_copy(
                update={
                    "is_production": True,
                    "promoted_at": datetime.now(timezone.utc),
                    "notes": f"Rollback restore. {fallback.notes}".strip(),
                }
            )
            self._save()
            # Capture IDs for audit log (do this inside the lock so fallback_id is stable)
            fallback_id = fallback.artifact_id
            current_id  = current.artifact_id if current else None
        self._append_audit_log(
            event="rollback",
            artifact_id=fallback_id,
            notes=notes,
            extra={"previous_production_id": current_id},
        )
        return self._store[fallback_id]

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def _append_audit_log(
        self,
        event:       str,
        artifact_id: str,
        notes:       str = "",
        extra:       Optional[Dict[str, object]] = None,
    ) -> None:
        """Append a single JSON line to the immutable audit log.

        The file is opened in *append* mode each call so that concurrent
        processes never overwrite each other's entries.  The directory is
        created automatically if it does not exist.

        Each log line is a JSON object with at minimum:
        ``event``, ``artifact_id``, ``timestamp``, ``notes``.
        """
        entry: Dict[str, object] = {
            "event":        event,
            "artifact_id":  artifact_id,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "notes":        notes,
        }
        if extra:
            entry.update(extra)
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._audit_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def read_audit_log(self) -> List[Dict[str, object]]:
        """Read all entries from the audit log.

        Returns:
            List of dicts, one per logged event, in chronological order.
            Empty list if the log does not yet exist.
        """
        if not self._audit_path.exists():
            return []
        entries = []
        with open(self._audit_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # skip corrupt lines
        return entries

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_production(self) -> Optional[ArtifactRecord]:
        """Return the current production artifact or ``None``."""
        with self._lock:
            return self._get_production_unlocked()

    def get_production_candidate(self) -> Optional[ArtifactRecord]:
        """Return the best passing artifact (lowest ECE) regardless of production status."""
        with self._lock:
            passing = [a for a in self._store.values() if a.passes_gate()]
            return min(passing, key=lambda a: a.ece, default=None)

    def get(self, artifact_id: str) -> Optional[ArtifactRecord]:
        """Retrieve a record by ``artifact_id``."""
        with self._lock:
            return self._store.get(artifact_id)

    def list_all(self) -> List[ArtifactRecord]:
        """Return all records sorted by epoch then registered_at."""
        with self._lock:
            return sorted(
                self._store.values(),
                key=lambda a: (a.epoch, a.registered_at),
            )

    def list_by_epoch(self, epoch: int) -> List[ArtifactRecord]:
        """Return all records for a given training epoch."""
        with self._lock:
            return [a for a in self._store.values() if a.epoch == epoch]

    def count(self) -> int:
        """Total number of registered artifacts."""
        with self._lock:
            return len(self._store)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Serialise registry to JSON (must be called under self._lock)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            aid: rec.model_dump(mode="json")
            for aid, rec in self._store.items()
        }
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self._path)

    def _load(self) -> None:
        """Deserialise registry from JSON if it exists."""
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            with self._lock:
                for aid, rec_dict in raw.items():
                    self._store[aid] = ArtifactRecord.model_validate(rec_dict)
        except Exception:  # noqa: BLE001 — corrupt registry is recoverable
            self._store = {}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_production_unlocked(self) -> Optional[ArtifactRecord]:
        return next((a for a in self._store.values() if a.is_production), None)

