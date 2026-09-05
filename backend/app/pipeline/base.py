"""Stage contract.

A stage is a small, resumable, independently testable unit of work:

    class MyStage(Stage):
        name = "scenes"
        status = JobStatus.ANALYZING
        weight = 2.0
        def run(self, ctx): ...

The runner handles ordering, progress scaling, timing, checkpointing and error
wrapping, so a stage body stays pure pipeline logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.db.models import JobStatus
from app.pipeline.context import JobContext


class Stage(ABC):
    #: unique, stable identifier -- also the checkpoint filename
    name: str = "stage"
    #: job status to advertise while this stage runs
    status: JobStatus = JobStatus.ANALYZING
    #: relative cost, used to size the stage's slice of the progress bar
    weight: float = 1.0
    #: if False, a failure is logged as a degradation instead of failing the job
    required: bool = True
    #: skip work entirely when a checkpoint exists
    resumable: bool = True

    @abstractmethod
    def run(self, ctx: JobContext) -> Any:
        """Do the work. Return a JSON-able value to checkpoint, or None."""

    def restore(self, ctx: JobContext, checkpoint: Any) -> None:
        """Rehydrate ctx.artifacts from a checkpoint instead of re-running."""
        if checkpoint is not None:
            ctx.set(self.name, checkpoint)

    def skip_if(self, ctx: JobContext) -> str | None:
        """Return a reason string to skip this stage (e.g. AI disabled)."""
        return None

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Stage {self.name}>"
