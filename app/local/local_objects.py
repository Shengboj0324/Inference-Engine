"""Filesystem-backed object store mirroring the boto3 / minio surface.

The desktop sidecar persists media blobs (images, audio, video) directly
to a per-bucket directory under :func:`get_user_data_dir` / ``objects``.
The exposed methods match the subset of boto3/minio that
``app/media/media_downloader.py`` and ``app/core/health.py`` use:

* :meth:`put_object` (bytes or file path)
* :meth:`get_object` (returns bytes)
* :meth:`exists`
* :meth:`delete`
* :meth:`list_objects` (prefix scan, returns ``ObjectMetadata`` rows)
* :meth:`get_presigned_url` (returns a ``file://`` URI — the desktop UI
  reads blobs directly from disk so no actual signing is needed)

Write atomicity
---------------
``put_object`` writes to a sibling ``<key>.tmp.<pid>.<uuid>`` file and
``os.replace`` it into place so half-written files are never observable.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Iterator, List, Optional, Union

from app.local.user_data_dir import get_user_data_dir

_BytesOrPath = Union[bytes, str, os.PathLike]


@dataclass
class ObjectMetadata:
    """Lightweight metadata row returned by :meth:`list_objects`."""

    key: str
    size: int
    etag: str
    last_modified: datetime


class LocalObjectStore:
    """Per-bucket filesystem store rooted at ``<data_dir>/objects/<bucket>``."""

    def __init__(self, bucket: str, *, root: Optional[Path] = None) -> None:
        if not bucket or "/" in bucket or "\\" in bucket:
            raise ValueError(f"invalid bucket name: {bucket!r}")
        self.bucket = bucket
        base = root or (get_user_data_dir(ensure=True) / "objects")
        self._root: Path = base / bucket
        self._root.mkdir(parents=True, exist_ok=True)

    # ---- path helpers ---------------------------------------------------
    def _resolve(self, key: str) -> Path:
        if not key or key.startswith("/") or ".." in Path(key).parts:
            raise ValueError(f"invalid object key: {key!r}")
        path = (self._root / key).resolve()
        # Defence-in-depth: refuse paths that escape the bucket root.
        if self._root.resolve() not in path.parents and path != self._root.resolve():
            raise ValueError(f"key escapes bucket root: {key!r}")
        return path

    @staticmethod
    def _etag(data: bytes) -> str:
        return hashlib.md5(data).hexdigest()  # noqa: S324 - parity with S3 ETag

    # ---- public API -----------------------------------------------------
    def put_object(self, key: str, data: _BytesOrPath) -> ObjectMetadata:
        """Write ``data`` to ``key``.  Atomic replace; parent dirs auto-created."""
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
        if isinstance(data, (bytes, bytearray)):
            payload = bytes(data)
            tmp.write_bytes(payload)
        else:
            src = Path(os.fspath(data))
            shutil.copyfile(src, tmp)
            payload = tmp.read_bytes()
        os.replace(tmp, target)
        stat = target.stat()
        return ObjectMetadata(
            key=key,
            size=stat.st_size,
            etag=self._etag(payload),
            last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        )

    def get_object(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.is_file():
            raise FileNotFoundError(f"object not found: {self.bucket}/{key}")
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        try:
            return self._resolve(key).is_file()
        except ValueError:
            return False

    def delete(self, key: str) -> bool:
        path = self._resolve(key)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def list_objects(self, prefix: str = "") -> Iterator[ObjectMetadata]:
        """Yield metadata for every object whose key starts with ``prefix``.

        Always walks from the bucket root and filters on the rendered key so
        partial-segment prefixes (e.g. ``"e."`` matching ``"e.txt"``) work
        identically to the boto3/minio behaviour.
        """
        if prefix.startswith("/") or ".." in Path(prefix).parts:
            raise ValueError(f"invalid prefix: {prefix!r}")
        if not self._root.exists():
            return
        for path in self._root.rglob("*"):
            if not path.is_file():
                continue
            if ".tmp." in path.name:
                continue
            rel = path.relative_to(self._root).as_posix()
            if not rel.startswith(prefix):
                continue
            stat = path.stat()
            data = path.read_bytes()
            yield ObjectMetadata(
                key=rel,
                size=stat.st_size,
                etag=self._etag(data),
                last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            )

    def get_presigned_url(self, key: str, *, expires_in: int = 3600) -> str:
        """Return a ``file://`` URI.  ``expires_in`` is accepted for parity."""
        del expires_in  # filesystem URIs have no expiry
        return self._resolve(key).as_uri()
