"""Scaffold stages.

Block A ships the *shape* of the pipeline end to end so status, progress,
cancellation, checkpointing and the SSE stream can be exercised before any
heavy dependency is installed. Each placeholder is replaced in a later block
(the target block is noted on every one) -- the runner, API and UI do not
change when that happens.
"""

from __future__ import annotations

import time

from app.db.models import JobStatus
from app.pipeline.base import Stage
from app.pipeline.context import JobContext


class PlaceholderStage(Stage):
    """Simulates a stage: reports progress, respects timing, produces a stub artifact."""

    def __init__(
        self,
        name: str,
        *,
        status: JobStatus = JobStatus.ANALYZING,
        weight: float = 1.0,
        required: bool = True,
        block: str = "B",
        duration_s: float = 0.3,
        steps: int = 4,
    ) -> None:
        self.name = name
        self.status = status
        self.weight = weight
        self.required = required
        self.block = block
        self.duration_s = duration_s
        self.steps = steps
        self.resumable = False  # stubs should never be cached as if they were real

    def run(self, ctx: JobContext) -> dict:
        for i in range(self.steps):
            time.sleep(self.duration_s / self.steps)
            ctx.progress((i + 1) / self.steps, f"{self.name}: step {i + 1}/{self.steps}")
        ctx.degrade(f"{self.name}_not_implemented")
        return {"stub": True, "implemented_in_block": self.block}
