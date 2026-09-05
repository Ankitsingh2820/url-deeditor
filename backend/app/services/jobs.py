"""Job lifecycle: create (url/upload), read, list, cancel, delete.

The API layer stays thin -- routes validate and serialise, this module owns the
rules (limits, dedupe, idempotency, cancellation semantics).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from fastapi import UploadFile
from sqlmodel import Session, func, select

from app.config import settings
from app.db.models import Job, JobEvent, JobStatus, SourceKind, utcnow
from app.errors import Conflict, NotFound, TooLarge, UnsupportedMedia
from app.schemas.job import JobOptions
from app.storage import Workspace
from app.worker import dispatch

log = logging.getLogger(__name__)

CHUNK = 1024 * 1024


def get_job(session: Session, job_id: str) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise NotFound(f"No job {job_id}")
    return job


def list_jobs(session: Session, limit: int = 25, offset: int = 0) -> tuple[list[Job], int]:
    total = session.exec(select(func.count()).select_from(Job)).one()
    rows = session.exec(
        select(Job).order_by(Job.created_at.desc()).offset(offset).limit(limit)
    ).all()
    return list(rows), int(total)


def _find_cached(session: Session, sha256: str) -> Job | None:
    """Same bytes, already de-edited -> reuse the result instead of re-running."""
    return session.exec(
        select(Job)
        .where(Job.sha256 == sha256, Job.status == JobStatus.SUCCEEDED)
        .order_by(Job.created_at.desc())
    ).first()


def _find_idempotent(session: Session, key: str) -> Job | None:
    return session.exec(select(Job).where(Job.idempotency_key == key)).first()


def create_url_job(
    session: Session,
    url: str,
    options: JobOptions,
    idempotency_key: str | None = None,
) -> Job:
    if idempotency_key and (existing := _find_idempotent(session, idempotency_key)):
        return existing

    job = Job(
        source_kind=SourceKind.URL,
        source_ref=url,
        options=options.model_dump(),
        idempotency_key=idempotency_key,
    )
    return _persist_and_enqueue(session, job)


async def create_upload_job(
    session: Session,
    upload: UploadFile,
    options: JobOptions,
    idempotency_key: str | None = None,
) -> Job:
    if idempotency_key and (existing := _find_idempotent(session, idempotency_key)):
        return existing

    filename = upload.filename or "upload.mp4"
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.allowed_upload_suffixes:
        raise UnsupportedMedia(
            f"'{suffix or 'unknown'}' is not a supported container",
            hint=f"Allowed: {', '.join(settings.allowed_upload_suffixes)}",
        )

    job = Job(source_kind=SourceKind.UPLOAD, source_ref=filename, options=options.model_dump())
    workspace = Workspace.for_job(job.id, create=True)
    target = workspace.path(f"upload{suffix}")

    # Stream to disk, hashing as we go, and abort the moment the cap is passed --
    # never buffer an untrusted upload in memory.
    digest = hashlib.sha256()
    size = 0
    limit = settings.max_upload_mb * 1024 * 1024
    try:
        with target.open("wb") as fh:
            while chunk := await upload.read(CHUNK):
                size += len(chunk)
                if size > limit:
                    raise TooLarge(f"Upload exceeds {settings.max_upload_mb} MB")
                digest.update(chunk)
                fh.write(chunk)
    except Exception:
        workspace.destroy()
        raise
    finally:
        await upload.close()

    if size == 0:
        workspace.destroy()
        raise UnsupportedMedia("Uploaded file is empty")

    job.sha256 = digest.hexdigest()
    job.options = {**job.options, "upload_path": workspace.rel(target), "upload_bytes": size}

    if not options.force and (cached := _find_cached(session, job.sha256)):
        log.info("cache hit: %s reuses %s", job.id, cached.id)
        workspace.destroy()
        return cached

    return _persist_and_enqueue(session, job)


def _persist_and_enqueue(session: Session, job: Job) -> Job:
    session.add(job)
    session.commit()
    session.refresh(job)

    Workspace.for_job(job.id, create=True)
    task_id = dispatch.enqueue(job.id)
    if task_id:
        job.celery_task_id = task_id
        session.add(job)
        session.commit()
        session.refresh(job)
    return job


def cancel_job(session: Session, job_id: str) -> Job:
    job = get_job(session, job_id)
    if job.status.is_terminal:
        raise Conflict(f"Job is already {job.status.value}")

    dispatch.cancel(job.id, job.celery_task_id)
    job.status = JobStatus.CANCELLED  # the runner observes this between stages
    job.error_code = "E_CANCELLED"
    job.updated_at = utcnow()
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def delete_job(session: Session, job_id: str) -> None:
    job = get_job(session, job_id)
    if not job.status.is_terminal:
        dispatch.cancel(job.id, job.celery_task_id)

    for event in session.exec(select(JobEvent).where(JobEvent.job_id == job_id)).all():
        session.delete(event)
    session.delete(job)
    session.commit()
    Workspace.for_job(job_id).destroy()
