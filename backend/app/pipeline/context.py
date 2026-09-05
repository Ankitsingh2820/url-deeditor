"""Everything a stage is allowed to touch."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.db.models import Job, JobStatus
from app.events import emit
from app.schemas.job import JobOptions
from app.storage import Workspace


class JobCancelled(Exception):
    """Raised between stages when the API has flagged the job cancelled."""


@dataclass
class JobContext:
    """Carries state between stages.

    Stages read/write `artifacts` (plain JSON-able dicts, one key per stage) and
    never talk to the database directly -- that keeps them unit-testable with
    nothing but a temp directory.
    """

    job_id: str
    workspace: Workspace
    options: JobOptions
    source_kind: str
    source_ref: str

    artifacts: dict[str, Any] = field(default_factory=dict)
    degradations: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    #: request options as stored, including keys JobOptions does not model
    #: (e.g. upload_path, written by the upload handler)
    raw_options: dict[str, Any] = field(default_factory=dict)
    #: sha256 of source.mp4, set by the ingest stage; the runner persists it as
    #: the job's content key so identical videos hit the result cache.
    content_sha256: str | None = None

    #: set by the runner; lets a long stage notice a cancellation mid-flight
    #: instead of only at the next stage boundary
    cancel_check: Callable[[], bool] | None = None

    # Set by the runner before each stage so `progress()` can scale into the
    # stage's slice of the overall 0..1 bar.
    _stage_name: str = ""
    _stage_span: tuple[float, float] = (0.0, 1.0)
    _last_cancel_check: float = 0.0

    @classmethod
    def from_job(cls, job: Job) -> JobContext:
        return cls(
            job_id=job.id,
            workspace=Workspace.for_job(job.id, create=True),
            options=JobOptions(**(job.options or {})),
            source_kind=job.source_kind.value,
            source_ref=job.source_ref,
            raw_options=dict(job.options or {}),
            degradations=list(job.degradations or []),
            timings_ms=dict(job.stage_timings_ms or {}),
        )

    # ---- reporting -------------------------------------------------------
    def progress(self, fraction: float, message: str | None = None) -> None:
        """Report progress *within* the current stage (0..1)."""
        lo, hi = self._stage_span
        overall = lo + (hi - lo) * max(0.0, min(1.0, fraction))
        emit(self.job_id, stage=self._stage_name, progress=overall, message=message)

    def log(self, message: str) -> None:
        emit(self.job_id, type="log", stage=self._stage_name, message=message)

    def degrade(self, reason: str) -> None:
        """Record that the pipeline had to fall back. Surfaced in project.json."""
        if reason not in self.degradations:
            self.degradations.append(reason)
        self.log(f"degraded: {reason}")

    def raise_if_cancelled(self, min_interval: float = 1.0) -> None:
        """Cheap cancellation check for stages that loop over many frames.

        Throttled: a per-frame database round-trip would cost more than the work
        it is guarding.
        """
        if self.cancel_check is None:
            return
        now = time.monotonic()
        if now - self._last_cancel_check < min_interval:
            return
        self._last_cancel_check = now
        if self.cancel_check():
            raise JobCancelled()

    # ---- artifact helpers ------------------------------------------------
    def set(self, key: str, value: Any) -> Any:
        self.artifacts[key] = value
        return value

    def get(self, key: str, default: Any = None) -> Any:
        return self.artifacts.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.artifacts:
            raise KeyError(f"Stage artifact '{key}' is missing; pipeline order is wrong")
        return self.artifacts[key]


__all__ = ["JobContext", "JobCancelled", "JobStatus"]
