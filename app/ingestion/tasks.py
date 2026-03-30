"""Celery tasks for content ingestion and processing."""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from uuid import UUID

from celery import Task
from sqlalchemy import select

from app.connectors.base import ConnectorConfig
from app.core.config import settings
from app.core.db import SessionLocal
from app.core.db_models import ContentItemDB, PlatformConfigDB, User
from app.core.models import ContentItem
from app.core.monitoring import MetricsCollector
from app.ingestion.celery_app import celery_app
from app.llm.openai_client import OpenAISyncEmbeddingClient

logger = logging.getLogger(__name__)


class DatabaseTask(Task):
    """Base task with database session management."""

    _db = None

    @property
    def db(self):
        """Get database session."""
        if self._db is None:
            self._db = SessionLocal()
        return self._db

    def after_return(self, *args, **kwargs):
        """Clean up database session after task completion."""
        if self._db is not None:
            self._db.close()
            self._db = None


@celery_app.task(base=DatabaseTask, bind=True)
def fetch_all_sources(self):
    """Fetch content from all configured sources for all users.

    This task runs periodically to pull new content from each user's
    configured platforms.
    """
    db = self.db

    # Get all active users
    users = db.execute(select(User).where(User.is_active.is_(True))).scalars().all()

    for user in users:
        # Get user's platform configs
        configs = (
            db.execute(
                select(PlatformConfigDB)
                .where(PlatformConfigDB.user_id == user.id)
                .where(PlatformConfigDB.enabled.is_(True))
            )
            .scalars()
            .all()
        )

        for config in configs:
            # Trigger fetch for each source
            fetch_source_content.delay(user.id, config.id)


@celery_app.task(base=DatabaseTask, bind=True)
def fetch_source_content(self, user_id: UUID, config_id: UUID):
    """Fetch content from a specific source.

    Args:
        user_id: User ID
        config_id: Platform configuration ID
    """
    db = self.db

    # Get platform config
    config = db.get(PlatformConfigDB, config_id)
    if not config:
        return {"error": "Config not found"}

    # Decrypt credentials using credential vault
    credentials = {}
    if config.encrypted_credentials:
        try:
            # Decrypt credentials using CredentialEncryption
            from app.core.security import CredentialEncryption

            encryption = CredentialEncryption()
            credentials = encryption.decrypt(config.encrypted_credentials)
        except Exception as e:
            logger.error(f"Failed to decrypt credentials: {e}")
            # Security: Do not fall back to empty credentials in production
            raise ValueError(f"Credential decryption failed: {e}") from e

    # Create connector using registry
    from app.connectors.registry import ConnectorRegistry

    try:
        connector = ConnectorRegistry.get_connector(
            platform=config.platform,
            config=ConnectorConfig(
                platform=config.platform,
                credentials=credentials,
                settings=config.settings or {},
            ),
            user_id=user_id,
        )
    except Exception as e:
        logger.error(f"Failed to create connector for {config.platform}: {e}")
        return {"error": f"Failed to create connector: {str(e)}"}

    # Fetch content
    try:
        # Get last fetch time from database
        # Default to 24 hours ago if no previous fetch
        since = config.last_fetch_time or (datetime.utcnow() - timedelta(hours=24))

        # Measure end-to-end ingestion latency for the Prometheus histogram.
        # The clock starts just before the connector call and stops after the
        # last ContentItem has been queued for processing.
        _fetch_start = time.monotonic()

        # Run async connector in sync context using asyncio.run()
        result = asyncio.run(
            connector.fetch_content(
                since=since, max_items=settings.max_items_per_fetch
            )
        )

        _fetch_elapsed = time.monotonic() - _fetch_start
        if result.items:
            # Record per-item latency (total fetch time ÷ items avoids
            # artificially large values for large batches).
            per_item_latency = _fetch_elapsed / len(result.items)
            MetricsCollector.record_ingestion_latency(
                platform=str(config.platform),
                latency_seconds=per_item_latency,
            )
        else:
            # Still record a zero-items fetch so the histogram baseline exists.
            MetricsCollector.record_ingestion_latency(
                platform=str(config.platform),
                latency_seconds=_fetch_elapsed,
            )

        # Check for duplicates and upsert instead of blind insert
        new_items = 0
        for item in result.items:
            # Check if item already exists
            existing = db.execute(
                select(ContentItemDB).where(
                    ContentItemDB.user_id == user_id,
                    ContentItemDB.source_platform == config.platform,
                    ContentItemDB.source_id == item.source_id,
                )
            ).scalar_one_or_none()

            if not existing:
                process_content_item.delay(item.dict())
                new_items += 1
            else:
                logger.debug(f"Skipping duplicate item: {item.source_id}")

        # Update last fetch time
        config.last_fetch_time = datetime.utcnow()
        db.commit()

        return {
            "status": "success",
            "items_fetched": len(result.items),
            "new_items": new_items,
            "duplicates_skipped": len(result.items) - new_items,
            "errors": result.errors,
        }

    except Exception as e:
        logger.error(f"Error fetching content from {config.platform}: {e}")
        return {"status": "error", "error": str(e)}


@celery_app.task(base=DatabaseTask, bind=True)
def process_content_item(self, item_dict: dict):
    """Process a single content item.

    This includes:
    - Generating embeddings
    - Storing in database
    - Topic extraction (future enhancement)
    - Language detection (future enhancement)

    Args:
        item_dict: ContentItem as dictionary
    """
    db = self.db

    # Reconstruct ContentItem
    item = ContentItem(**item_dict)

    # Generate embedding if text available (using synchronous client for Celery)
    embedding = None
    if item.raw_text or item.title:
        text = f"{item.title}\n\n{item.raw_text or ''}"
        try:
            embedding_client = OpenAISyncEmbeddingClient()
            response = embedding_client.embed_text(text)
            embedding = response.embedding
        except Exception as e:
            logger.error(f"Error generating embedding for item {item.id}: {e}")

    # Create database record
    db_item = ContentItemDB(
        id=item.id,
        user_id=item.user_id,
        source_platform=item.source_platform,
        source_id=item.source_id,
        source_url=item.source_url,
        author=item.author,
        channel=item.channel,
        title=item.title,
        raw_text=item.raw_text,
        media_type=item.media_type,
        media_urls=item.media_urls,
        published_at=item.published_at,
        fetched_at=item.fetched_at,
        topics=item.topics,
        lang=item.lang,
        embedding=embedding,
        metadata_=item.metadata,
    )

    _write_start = time.monotonic()
    db.add(db_item)
    db.commit()
    _write_elapsed = time.monotonic() - _write_start

    # Record DB-write latency under the same platform label so operators can
    # distinguish connector-fetch latency from storage-write latency.
    MetricsCollector.record_ingestion_latency(
        platform=str(item.source_platform),
        latency_seconds=_write_elapsed,
    )

    return {"status": "success", "item_id": str(item.id)}


@celery_app.task(base=DatabaseTask, bind=True)
def cleanup_old_content(self):
    """Clean up old content items based on retention policy."""
    db = self.db

    cutoff_date = datetime.utcnow() - timedelta(days=settings.max_content_age_days)

    # Delete old content items
    result = db.execute(
        select(ContentItemDB).where(ContentItemDB.fetched_at < cutoff_date)
    )
    old_items = result.scalars().all()

    for item in old_items:
        db.delete(item)

    db.commit()

    return {"status": "success", "items_deleted": len(old_items)}


# ---------------------------------------------------------------------------
# Req 4 — Contextual Thread Expansion
# ---------------------------------------------------------------------------

#: Minimum classification confidence required to trigger thread expansion.
_THREAD_EXPANSION_CONFIDENCE_THRESHOLD: float = 0.8
#: Minimum impact score required to trigger thread expansion.
_THREAD_EXPANSION_IMPACT_THRESHOLD: float = 0.7
#: Maximum number of parent/child nodes to ingest per thread expansion.
_THREAD_EXPANSION_MAX_NODES: int = 50


@celery_app.task(base=DatabaseTask, bind=True, queue="high_priority")
def expand_conversation_thread(
    self,
    signal_id: str,
    source_id: str,
    platform: str,
    user_id: str,
    max_nodes: int = _THREAD_EXPANSION_MAX_NODES,
) -> dict:
    """High-priority task: ingest the full conversation thread for a signal.

    Triggered automatically when a signal is classified with
    ``confidence_score > 0.8`` AND ``impact_score > 0.7``.  Ingests up to
    *max_nodes* parent and child posts from the originating thread, stores
    them as ``ContentItemDB`` rows, and updates
    ``ActionableSignalDB.context`` to record that thread expansion has run.

    All new items are linked to the signal by appending their IDs to the
    ``context`` JSON field so the ``DeepResearchAgent`` can surface them via
    the ``VectorSearchTool``.

    Args:
        signal_id: UUID string of the ``ActionableSignalDB`` that triggered
            expansion.
        source_id: Platform-native thread/post identifier (e.g. Reddit
            ``t1_abc123`` or Twitter conversation ID).
        platform: Lower-case platform name (e.g. ``"reddit"``).
        user_id: UUID string of the signal owner — used to scope DB writes.
        max_nodes: Maximum parent/child nodes to ingest (default 50).

    Returns:
        Dict with ``status``, ``signal_id``, ``nodes_ingested``, and
        ``platform``.
    """
    db = self.db
    logger.info(
        "expand_conversation_thread: signal=%s platform=%s source_id=%s max_nodes=%d",
        signal_id, platform, source_id, max_nodes,
    )
    start_time = time.time()
    nodes_ingested = 0

    try:
        from app.core.db_models import ActionableSignalDB
        from app.connectors.registry import ConnectorRegistry

        # Resolve the signal row
        sig_row = db.execute(
            select(ActionableSignalDB).where(
                ActionableSignalDB.id == UUID(signal_id)
            )
        ).scalar_one_or_none()
        if sig_row is None:
            logger.warning("expand_conversation_thread: signal %s not found", signal_id)
            return {"status": "not_found", "signal_id": signal_id, "nodes_ingested": 0}

        # Try to fetch the full thread via the platform connector
        try:
            registry = ConnectorRegistry()
            connector = registry.get_connector(platform)
            thread_items: list = connector.fetch_thread(
                source_id=source_id, max_nodes=max_nodes
            ) if hasattr(connector, "fetch_thread") else []
        except Exception as conn_exc:
            logger.warning(
                "expand_conversation_thread: connector fetch failed for %s/%s: %s",
                platform, source_id, conn_exc,
            )
            thread_items = []

        # Persist each thread node as a ContentItemDB
        expanded_ids: list[str] = []
        user_uuid = UUID(user_id)
        for raw_item in thread_items[:max_nodes]:
            try:
                existing = db.execute(
                    select(ContentItemDB).where(
                        ContentItemDB.source_id == raw_item.get("source_id", ""),
                    )
                ).scalar_one_or_none()
                if existing:
                    expanded_ids.append(str(existing.id))
                    continue

                db_item = ContentItemDB(
                    user_id=user_uuid,
                    source_platform=platform,
                    source_id=raw_item.get("source_id", f"{source_id}_node_{nodes_ingested}"),
                    source_url=raw_item.get("url", ""),
                    title=raw_item.get("title", "")[:255],
                    raw_text=raw_item.get("text", "")[:10000],
                    author=raw_item.get("author"),
                    published_at=raw_item.get("published_at", datetime.utcnow()),
                    fetched_at=datetime.utcnow(),
                )
                db.add(db_item)
                db.flush()
                expanded_ids.append(str(db_item.id))
                nodes_ingested += 1
            except Exception as item_exc:
                logger.debug("expand_conversation_thread: item write failed: %s", item_exc)

        # Update ActionableSignalDB.context to record expansion
        import json as _json
        existing_ctx: dict = {}
        try:
            existing_ctx = _json.loads(sig_row.context or "{}") if isinstance(
                sig_row.context, str
            ) else (sig_row.context or {})
        except Exception:
            existing_ctx = {}
        existing_ctx["thread_expansion"] = {
            "source_id": source_id,
            "platform": platform,
            "nodes_ingested": nodes_ingested,
            "expanded_content_ids": expanded_ids,
            "expanded_at": datetime.utcnow().isoformat(),
        }
        sig_row.context = _json.dumps(existing_ctx)
        db.commit()

        elapsed = time.time() - start_time
        logger.info(
            "expand_conversation_thread: done — signal=%s nodes=%d elapsed=%.2fs",
            signal_id, nodes_ingested, elapsed,
        )
        MetricsCollector.record_ingestion(
            platform=platform,
            latency_seconds=elapsed,
        )
        return {
            "status": "success",
            "signal_id": signal_id,
            "nodes_ingested": nodes_ingested,
            "platform": platform,
        }

    except Exception as exc:
        logger.error(
            "expand_conversation_thread: unhandled error for signal %s: %s",
            signal_id, exc, exc_info=True,
        )
        return {"status": "error", "signal_id": signal_id, "error": str(exc), "nodes_ingested": 0}


# Connector creation is now handled by ConnectorRegistry
# See app/connectors/registry.py for all 13 supported platforms
