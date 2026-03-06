"""Celery tasks for content ingestion and processing."""

from datetime import datetime, timedelta
from typing import List
from uuid import UUID

from celery import Task
from sqlalchemy import select

from app.connectors.base import BaseConnector, ConnectorConfig
from app.connectors.reddit import RedditConnector
from app.connectors.rss import RSSConnector
from app.connectors.youtube import YouTubeConnector
from app.core.config import settings
from app.core.db import SessionLocal
from app.core.db_models import ContentItemDB, PlatformConfigDB, User
from app.core.models import ContentItem, SourcePlatform
from app.ingestion.celery_app import celery_app
from app.llm.openai_client import OpenAISyncEmbeddingClient
import logging

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
    users = db.execute(select(User).where(User.is_active == True)).scalars().all()

    for user in users:
        # Get user's platform configs
        configs = (
            db.execute(
                select(PlatformConfigDB)
                .where(PlatformConfigDB.user_id == user.id)
                .where(PlatformConfigDB.enabled == True)
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
            raise ValueError(f"Credential decryption failed: {e}")

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

        result = connector.fetch_content(
            since=since, max_items=settings.max_items_per_fetch
        )

        # Process each item
        for item in result.items:
            process_content_item.delay(item.dict())

        return {
            "status": "success",
            "items_fetched": len(result.items),
            "errors": result.errors,
        }

    except Exception as e:
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
        metadata=item.metadata,
    )

    db.add(db_item)
    db.commit()

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


# Connector creation is now handled by ConnectorRegistry
# See app/connectors/registry.py for all 13 supported platforms

