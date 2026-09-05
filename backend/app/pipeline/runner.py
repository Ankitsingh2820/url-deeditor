"""Pipeline driver: ordering, progress scaling, checkpoints, timing, failure policy."""

from __future__ import annotations

import logging
import time
import traceback

from app.db.models import Job, JobStatus, utcnow
from app.db.session import session_scope
from app.errors import AppError, StageFailed
from app.events import emit
from app.pipeline.base import Stage
from app.pipeline.context import JobCancelled, JobContext
from app.storage import Workspace, write_json

log = logging.getLogger(__name__)


def _spans(stages: list[Stage]) -> list[tuple[float, float]]:
    """Map each stage onto a slice of the overall 0..1 progress bar by weight."""
    total = sum(s.weight for s in stages) or 1.0
    spans, cursor = [], 0.0
    for stage in stages:
        share = stage.weight / total
        spans.append((cursor, cursor + share))
        cursor += share
    return spans


def _load_job(job_id: str) -> Job:
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise AppError(f"Job {job_id} disappeared")
        session.expunge(job)
        return job


def _cancel_requested(job_id: str) -> bool:
    with session_scope() as session:
        job = session.get(Job, job_id)
        return job is not None and job.status == JobStatus.CANCELLED


def _backfill_content_hash(job_id: str, digest: str) -> None:
    """Record the source hash only when the job has no content key yet.

    Uploads are already keyed by the bytes the client sent; overwriting that
    with the hash of the remuxed source would defeat the result cache. URL jobs
    have nothing to key on until ingest runs, so they get it here.
    """
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is not None and not job.sha256:
            job.sha256 = digest
            session.add(job)


def _persist(job_id: str, **fields) -> None:
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        for key, value in fields.items():
            setattr(job, key, value)
        job.updated_at = utcnow()
        session.add(job)


def run_pipeline(job_id: str, stages: list[Stage] | None = None) -> str:
    """Execute every stage for `job_id`. Returns the final status value.

    Safe to call from a Celery worker or a plain thread; it owns its own DB
    sessions and never assumes an event loop.
    """
    from app.pipeline.stages import build_pipeline

    stages = stages or build_pipeline()
    job = _load_job(job_id)
    ctx = JobContext.from_job(job)
    ctx.cancel_check = lambda: _cancel_requested(job_id)
    spans = _spans(stages)

    _persist(job_id, status=JobStatus.DOWNLOADING, started_at=utcnow(), progress=0.0)
    emit(job_id, type="log", message=f"pipeline started ({len(stages)} stages)")

    try:
        for stage, span in zip(stages, spans, strict=True):
            if _cancel_requested(job_id):
                raise JobCancelled()

            ctx._stage_name = stage.name
            ctx._stage_span = span
            _persist(job_id, status=stage.status, stage=stage.name)
            emit(job_id, stage=stage.name, progress=span[0], status=stage.status,
                 message=f"stage: {stage.name}")

            reason = stage.skip_if(ctx)
            if reason:
                ctx.degrade(f"{stage.name}_skipped:{reason}")
                continue

            if stage.resumable and ctx.workspace.has_checkpoint(stage.name):
                stage.restore(ctx, ctx.workspace.read_checkpoint(stage.name))
                ctx.log(f"{stage.name}: restored from checkpoint")
                continue

            started = time.perf_counter()
            try:
                result = stage.run(ctx)
            except JobCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - policy decision below
                log.exception("stage %s failed for job %s", stage.name, job_id)
                if not stage.required:
                    ctx.degrade(f"{stage.name}_failed:{type(exc).__name__}")
                    continue
                # Keep a typed error's own code (E_DOWNLOAD_BLOCKED, E_TOO_LONG,
                # ...) so the client can react; only wrap the unexpected ones.
                if isinstance(exc, AppError):
                    exc.extra.setdefault("stage", stage.name)
                    raise
                raise StageFailed(stage.name, str(exc)) from exc

            elapsed = (time.perf_counter() - started) * 1000
            ctx.timings_ms[stage.name] = round(elapsed, 1)
            if result is not None:
                ctx.set(stage.name, result)
                if stage.resumable:
                    ctx.workspace.write_checkpoint(stage.name, result)
            ctx.progress(1.0, f"{stage.name} done in {elapsed:.0f}ms")

        final: dict = {
            "status": JobStatus.SUCCEEDED,
            "stage": "done",
            "progress": 1.0,
            "finished_at": utcnow(),
            "degradations": ctx.degradations,
            "stage_timings_ms": ctx.timings_ms,
        }
        _persist(job_id, **final)
        if ctx.content_sha256:
            _backfill_content_hash(job_id, ctx.content_sha256)
        emit(job_id, type="done", stage="done", progress=1.0, status=JobStatus.SUCCEEDED,
             message="completed")
        return JobStatus.SUCCEEDED.value

    except JobCancelled:
        _persist(job_id, status=JobStatus.CANCELLED, finished_at=utcnow(),
                 error_code="E_CANCELLED", error_detail="Cancelled by user")
        emit(job_id, type="done", status=JobStatus.CANCELLED, message="cancelled")
        return JobStatus.CANCELLED.value

    except AppError as exc:
        _fail(job_id, ctx, exc.code, exc.detail)
        return JobStatus.FAILED.value

    except Exception as exc:  # noqa: BLE001 - last resort
        log.error("pipeline crashed: %s", traceback.format_exc())
        _fail(job_id, ctx, "E_INTERNAL", f"{type(exc).__name__}: {exc}")
        return JobStatus.FAILED.value


def _fail(job_id: str, ctx: JobContext, code: str, detail: str) -> None:
    _persist(
        job_id,
        status=JobStatus.FAILED,
        finished_at=utcnow(),
        error_code=code,
        error_detail=detail,
        degradations=ctx.degradations,
        stage_timings_ms=ctx.timings_ms,
    )
    emit(job_id, type="error", status=JobStatus.FAILED, message=f"{code}: {detail}")
    # Preserve partial results so a failed job is still inspectable.
    ws: Workspace = ctx.workspace
    write_json(ws.path("state", "_failure.json"),
               {"code": code, "detail": detail, "artifacts": list(ctx.artifacts)})
