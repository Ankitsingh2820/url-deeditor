"""Persistence models.

Two tables:
  job         - one row per de-edit request, the resource the API exposes.
  job_event   - append-only progress log; the SSE endpoint tails it by id.

Why an events table instead of Redis pub/sub: it works identically in
thread-mode and celery-mode, survives a page reload (the client can replay
from seq 0), and needs no extra infrastructure. Redis pub/sub is the drop-in
upgrade for multi-node deployments -- see app/events.py.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, Column, Index, Text
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class JobStatus(StrEnum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    ANALYZING = "analyzing"
    INPAINTING = "inpainting"
    EXPORTING = "exporting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}


class SourceKind(StrEnum):
    URL = "url"
    UPLOAD = "upload"


class Job(SQLModel, table=True):
    __tablename__ = "job"

    id: str = Field(default_factory=lambda: new_id("job"), primary_key=True)

    source_kind: SourceKind
    source_ref: str = Field(sa_column=Column(Text))  # the URL, or the original filename
    sha256: str | None = Field(default=None, index=True)  # content key -> result cache

    status: JobStatus = Field(default=JobStatus.QUEUED, index=True)
    stage: str | None = None
    progress: float = 0.0  # 0..1

    error_code: str | None = None
    error_detail: str | None = Field(default=None, sa_column=Column(Text))

    # Request-time knobs (removal_method, ai_enabled, ...) and honest reporting
    # of anything the pipeline had to downgrade.
    options: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    degradations: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    stage_timings_ms: dict[str, float] = Field(default_factory=dict, sa_column=Column(JSON))

    idempotency_key: str | None = Field(default=None, index=True)
    celery_task_id: str | None = None

    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def cancel_requested(self) -> bool:
        return self.status == JobStatus.CANCELLED


class JobEvent(SQLModel, table=True):
    __tablename__ = "job_event"
    __table_args__ = (Index("ix_job_event_job_id_id", "job_id", "id"),)

    id: int | None = Field(default=None, primary_key=True)
    job_id: str = Field(index=True)
    type: str = "progress"  # progress | log | error | done
    stage: str | None = None
    progress: float = 0.0
    message: str | None = Field(default=None, sa_column=Column(Text))
    created_at: datetime = Field(default_factory=utcnow)
