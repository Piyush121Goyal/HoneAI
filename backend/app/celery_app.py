"""Celery application — embedding backfill + batch optimize jobs."""
from celery import Celery

from .config import settings

celery = Celery("hone", broker=settings.redis_url, backend=settings.redis_url)
celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

celery.autodiscover_tasks(["app"], related_name="tasks")
