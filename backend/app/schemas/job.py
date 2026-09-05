"""API request/response models.

Deliberately separate from the SQLModel table so the wire contract can evolve
independently of the schema, and so internals (celery_task_id, idempotency_key)
never leak.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

from app.db.models import Job, JobStatus, SourceKind

RemovalMethod = Literal["auto", "opencv", "lama", "mask_blur"]


class JobOptions(BaseModel):
    removal_method: RemovalMethod = "auto"
    ai_enabled: bool = True
    sample_fps: float = Field(default=3.0, ge=0.5, le=10.0)
    force: bool = Field(default=False, description="Bypass the content-hash result cache")


class CreateJobRequest(BaseModel):
    """JSON body for URL submissions. File uploads use multipart instead."""

    url: HttpUrl
    options: JobOptions = JobOptions()

    @field_validator("url")
    @classmethod
    def _http_only(cls, v: HttpUrl) -> HttpUrl:
        if v.scheme not in {"http", "https"}:
            raise ValueError("Only http(s) URLs are supported")
        return v


class JobRead(BaseModel):
    id: str
    status: JobStatus
    stage: str | None = None
    progress: float = 0.0
    source_kind: SourceKind
    source_ref: str
    error_code: str | None = None
    error_detail: str | None = None
    degradations: list[str] = []
    stage_timings_ms: dict[str, float] = {}
    options: dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None = None

    # Convenience links so the client never string-builds URLs.
    project_url: str | None = None
    events_url: str | None = None

    @classmethod
    def from_job(cls, job: Job, prefix: str = "/api/v1") -> JobRead:
        return cls(
            id=job.id,
            status=job.status,
            stage=job.stage,
            progress=job.progress,
            source_kind=job.source_kind,
            source_ref=job.source_ref,
            error_code=job.error_code,
            error_detail=job.error_detail,
            degradations=job.degradations or [],
            stage_timings_ms=job.stage_timings_ms or {},
            options=job.options or {},
            created_at=job.created_at,
            updated_at=job.updated_at,
            finished_at=job.finished_at,
            project_url=f"{prefix}/jobs/{job.id}/project",
            events_url=f"{prefix}/jobs/{job.id}/events",
        )


class JobList(BaseModel):
    items: list[JobRead]
    total: int
    limit: int
    offset: int
