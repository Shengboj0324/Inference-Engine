"""Local-first backend implementations for the desktop sidecar.

This subpackage provides drop-in replacements for the external services
the server topology depends on:

* ``user_data_dir``  — cross-platform persistent storage root.
* ``local_pubsub``   — in-process replacement for ``redis.asyncio`` pub/sub.
* ``local_cache``    — TTL key/value cache with a CacheManager-compatible API.
* ``local_queue``    — bounded asyncio task queue (priority + ack/nack).
* ``local_objects``  — filesystem object store with the boto3/minio surface.
* ``local_scheduler``— native asyncio cron-style scheduler.
* ``sqlite_store``   — aiosqlite engine factory with optional sqlite-vec.

Selection between local and server backends is governed by
``app.core.config.settings.deployment_mode``.
"""

from app.local.user_data_dir import (
    get_user_data_dir,
    get_user_cache_dir,
    get_user_log_dir,
)

__all__ = [
    "get_user_data_dir",
    "get_user_cache_dir",
    "get_user_log_dir",
]
