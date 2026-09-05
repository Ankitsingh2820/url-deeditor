"""Celery application.

Run it with:
    celery -A app.worker.celery_app:celery worker -l info            # linux/docker
    celery -A app.worker.celery_app:celery worker -l info --pool=solo  # windows
"""

from __future__ import annotations

from celery import Celery

from app.config import settings

celery = Celery(
    "lightnote",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.worker.tasks"],
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Video work is long and CPU-bound: fetch one task at a time and never let a
    # job silently run forever.
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=settings.job_time_budget_s + 120,
    task_soft_time_limit=settings.job_time_budget_s,
    result_expires=3600,
)


@celery.on_after_configure.connect
def _warmup(sender, **_kwargs):  # pragma: no cover - worker boot hook
    """Create tables and preload models once per worker process."""
    from app.db.session import init_db
    from app.warmup import preload_in_background

    init_db()
    preload_in_background()
