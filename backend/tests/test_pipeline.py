"""Unit tests for the pipeline framework -- no database, no media, no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.db.models import JobStatus
from app.pipeline.base import Stage
from app.pipeline.context import JobContext
from app.pipeline.runner import _spans
from app.schemas.job import JobOptions
from app.storage import Workspace


@pytest.fixture
def ctx(tmp_path: Path, monkeypatch) -> JobContext:
    # ctx.progress()/log() write events; silence them for pure unit tests.
    monkeypatch.setattr("app.pipeline.context.emit", lambda *a, **k: None)
    return JobContext(
        job_id="job_test",
        workspace=Workspace(job_id="job_test", root=tmp_path).ensure(),
        options=JobOptions(),
        source_kind="url",
        source_ref="https://example.com/v",
    )


class _Weighted(Stage):
    def __init__(self, name: str, weight: float):
        self.name, self.weight = name, weight

    def run(self, ctx):  # pragma: no cover - not executed here
        return None


def test_spans_partition_the_bar_by_weight():
    spans = _spans([_Weighted("a", 1.0), _Weighted("b", 3.0)])
    assert spans[0] == (0.0, 0.25)
    assert spans[1] == (0.25, 1.0)
    assert spans[-1][1] == pytest.approx(1.0)


def test_degradations_are_deduped(ctx: JobContext):
    ctx.degrade("asr_skipped")
    ctx.degrade("asr_skipped")
    assert ctx.degradations == ["asr_skipped"]


def test_require_missing_artifact_raises(ctx: JobContext):
    ctx.set("scenes", {"scenes": []})
    assert ctx.require("scenes") == {"scenes": []}
    with pytest.raises(KeyError):
        ctx.require("ocr")


def test_checkpoint_roundtrip(ctx: JobContext):
    assert not ctx.workspace.has_checkpoint("scenes")
    ctx.workspace.write_checkpoint("scenes", {"scenes": [{"id": "sc_001"}]})
    assert ctx.workspace.has_checkpoint("scenes")
    assert ctx.workspace.read_checkpoint("scenes")["scenes"][0]["id"] == "sc_001"


def test_workspace_rejects_path_escape(ctx: JobContext):
    from app.errors import NotFound

    with pytest.raises(NotFound):
        ctx.workspace.resolve_public("../secrets.txt")


def test_workspace_rel_is_posix(ctx: JobContext):
    target = ctx.workspace.path("clean", "sc_001.mp4")
    target.write_bytes(b"")
    assert ctx.workspace.rel(target) == "clean/sc_001.mp4"


def test_finalize_writes_contract(ctx: JobContext):
    from app.pipeline.stages.finalize import FinalizeStage

    ctx.set("probe", {"duration": 12.0, "fps": 30, "width": 1080, "height": 1920,
                      "has_audio": True})
    ctx.set("scenes", {"scenes": [{"id": "sc_001", "t_in": 0.0, "t_out": 3.0}]})

    result = FinalizeStage().run(ctx)

    assert result["counts"]["scenes"] == 1
    import json

    project = json.loads(ctx.workspace.project_json.read_text("utf-8"))
    assert project["media"]["duration"] == 12.0
    assert project["diagnostics"]["degradations"] == []


def test_job_status_terminal_flags():
    assert JobStatus.SUCCEEDED.is_terminal
    assert JobStatus.CANCELLED.is_terminal
    assert not JobStatus.ANALYZING.is_terminal
