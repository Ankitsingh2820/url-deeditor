"""Queue abstraction: Celery in production, threads for local dev.

Why both: the pipeline must never block the ASGI event loop, but requiring a
Redis broker to run `uvicorn` on a laptop (especially Windows, where Celery's
prefork pool is unavailable) is a bad developer experience. The API calls
`enqueue()` / `cancel()` and does not care which backend is live.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor

from app.config import settings

log = logging.getLogger(__name__)

_executor: ThreadPoolExecutor | None = None
_futures: dict[str, Future] = {}


def _pool() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=settings.thread_workers, thread_name_prefix="deedit"
        )
    return _executor


def enqueue(job_id: str) -> str | None:
    """Schedule the de-edit pipeline. Returns a task id when one exists."""
    if settings.queue_mode == "celery":
        from app.worker.tasks import deedit_task

        result = deedit_task.delay(job_id)
        log.info("enqueued job %s as celery task %s", job_id, result.id)
        return result.id

    from app.pipeline.runner import run_pipeline

    log.info("enqueued job %s on the local thread pool", job_id)
    _futures[job_id] = _pool().submit(run_pipeline, job_id)
    return None


def cancel(job_id: str, task_id: str | None) -> bool:
    """Best-effort cancellation.

    Cooperative either way: the runner checks the job's status between stages,
    so a job already inside a stage stops at the next boundary.
    """
    if settings.queue_mode == "celery" and task_id:
        from app.worker.celery_app import celery

        celery.control.revoke(task_id, terminate=False)
        return True

    future = _futures.get(job_id)
    if future is not None:
        return future.cancel()  # only succeeds if it has not started yet
    return False


def shutdown() -> None:
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None
