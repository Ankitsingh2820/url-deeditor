"""Celery task wrappers.

Thin on purpose: all logic lives in app.pipeline.runner so it can also run in a
thread (queue_mode="thread") or straight from a test, with no Celery imported.
"""

from __future__ import annotations

import logging

from app.worker.celery_app import celery

log = logging.getLogger(__name__)


@celery.task(name="lightnote.deedit", bind=True, max_retries=0)
def deedit_task(self, job_id: str) -> str:
    from app.pipeline.runner import run_pipeline

    log.info("worker picked up job %s (task %s)", job_id, self.request.id)
    return run_pipeline(job_id)


@celery.task(name="lightnote.render", bind=True, max_retries=0)
def render_task(self, job_id: str, edits: dict) -> str:
    """Recompose an edited project back into a video (block I)."""
    raise NotImplementedError("Recompose lands in block I")
