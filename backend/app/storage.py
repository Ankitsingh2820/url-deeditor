"""Per-job workspace on disk.

One directory per job, one place that knows the layout. Every artifact path
stored in project.json is *relative to the workspace root*, so the same JSON is
valid whether it is served from local disk today or S3 later.

    storage/{job_id}/
      source.mp4  proxy.mp4  audio.wav
      scenes/    original per-scene clips
      clean/     de-edited full video + per-scene clips
      frames/    scene keyframes + sampled frames
      overlays/  RGBA cutouts of detected pop-ups
      masks/     mask timeline
      state/     per-stage checkpoints (resumability)
      project.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import settings
from app.errors import NotFound

SUBDIRS = ("scenes", "clean", "frames", "overlays", "masks", "state", "tmp")


@dataclass(frozen=True)
class Workspace:
    job_id: str
    root: Path

    @classmethod
    def for_job(cls, job_id: str, *, create: bool = False) -> Workspace:
        root = (settings.storage_dir / job_id).resolve()
        ws = cls(job_id=job_id, root=root)
        if create:
            ws.ensure()
        return ws

    def ensure(self) -> Workspace:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in SUBDIRS:
            (self.root / name).mkdir(exist_ok=True)
        return self

    # ---- well-known paths ------------------------------------------------
    @property
    def source(self) -> Path:
        return self.root / "source.mp4"

    @property
    def proxy(self) -> Path:
        return self.root / "proxy.mp4"

    @property
    def audio(self) -> Path:
        return self.root / "audio.wav"

    @property
    def project_json(self) -> Path:
        return self.root / "project.json"

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def rel(self, path: Path) -> str:
        """Workspace-relative POSIX path, as stored in project.json."""
        return path.resolve().relative_to(self.root).as_posix()

    # ---- stage checkpoints (resumability) --------------------------------
    def checkpoint(self, stage: str) -> Path:
        return self.root / "state" / f"{stage}.json"

    def has_checkpoint(self, stage: str) -> bool:
        return self.checkpoint(stage).exists()

    def read_checkpoint(self, stage: str) -> Any:
        return json.loads(self.checkpoint(stage).read_text("utf-8"))

    def write_checkpoint(self, stage: str, payload: Any) -> None:
        write_json(self.checkpoint(stage), payload)

    # ---- safe artifact serving -------------------------------------------
    def resolve_public(self, relative: str) -> Path:
        """Resolve a client-supplied relative path, refusing directory escapes."""
        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(self.root):
            raise NotFound("No such artifact")
        if not candidate.is_file():
            raise NotFound(f"No such artifact: {relative}")
        return candidate

    def destroy(self) -> None:
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)


def write_json(path: Path, payload: Any) -> None:
    """Atomic write so a crashed worker can never leave half a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
