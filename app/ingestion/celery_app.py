"""Celery application for background task processing."""

from celery import Celery
from celery.schedules import crontab

from app.core.config import settings

# Create Celery app
celery_app = Celery(
    "social_media_radar",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Periodic task schedule
celery_app.conf.beat_schedule = {
    "fetch-all-sources": {
        "task": "app.ingestion.tasks.fetch_all_sources",
        "schedule": crontab(minute=f"*/{settings.ingestion_interval_minutes}"),
    },
    "cleanup-old-content": {
        "task": "app.ingestion.tasks.cleanup_old_content",
        "schedule": crontab(hour=2, minute=0),  # Daily at 2 AM
    },
}

# Auto-discover tasks
celery_app.autodiscover_tasks(["app.ingestion"])

