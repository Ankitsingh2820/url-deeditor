"""Job progress events.

Writers (the worker) append rows; readers (the SSE endpoint) tail them by
autoincrement id. Cursor-based, so a client that reconnects replays everything
it missed by sending Last-Event-ID.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass

from sqlmodel import select

from app.db.models import Job, JobEvent, JobStatus
from app.db.session import session_scope

# Keep SSE responsive without hammering SQLite.
POLL_INTERVAL_S = 0.4
IDLE_TIMEOUT_S = 900.0


@dataclass
class Event:
    id: int
    type: str
    stage: str | None
    progress: float
    message: str | None
    status: str

    def to_sse(self) -> dict:
        import json

        return {"id": str(self.id), "event": self.type, "data": json.dumps(asdict(self))}


def emit(job_id: str, *, type: str = "progress", stage: str | None = None,
         progress: float | None = None, message: str | None = None,
         status: JobStatus | None = None) -> None:
    """Append an event and mirror the latest state onto the job row.

    Called from worker threads/processes, so it opens its own session.
    """
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        if stage is not None:
            job.stage = stage
        if progress is not None:
            job.progress = max(0.0, min(1.0, progress))
        if status is not None:
            job.status = status
        if type == "error":
            job.error_detail = message
        from app.db.models import utcnow

        job.updated_at = utcnow()
        session.add(job)
        session.add(
            JobEvent(
                job_id=job_id,
                type=type,
                stage=stage or job.stage,
                progress=job.progress,
                message=message,
            )
        )


def read_since(job_id: str, after_id: int) -> tuple[list[Event], JobStatus]:
    with session_scope() as session:
        job = session.get(Job, job_id)
        status = job.status if job else JobStatus.FAILED
        rows = session.exec(
            select(JobEvent)
            .where(JobEvent.job_id == job_id, JobEvent.id > after_id)
            .order_by(JobEvent.id)
            .limit(200)
        ).all()
        events = [
            Event(
                id=r.id or 0,
                type=r.type,
                stage=r.stage,
                progress=r.progress,
                message=r.message,
                status=status.value,
            )
            for r in rows
        ]
        return events, status


async def stream(job_id: str, after_id: int = 0) -> AsyncIterator[dict]:
    """Async generator of SSE dicts; completes once the job reaches a terminal state."""
    import asyncio

    cursor = after_id
    waited = 0.0
    while True:
        events, status = await asyncio.to_thread(read_since, job_id, cursor)
        for event in events:
            cursor = event.id
            waited = 0.0
            yield event.to_sse()
        if status.is_terminal and not events:
            yield {"event": "done", "data": status.value}
            return
        await asyncio.sleep(POLL_INTERVAL_S)
        waited += POLL_INTERVAL_S
        if waited >= IDLE_TIMEOUT_S:
            yield {"event": "timeout", "data": status.value}
            return
