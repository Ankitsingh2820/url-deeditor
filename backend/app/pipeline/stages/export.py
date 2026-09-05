"""S10 - cut the clean plate into per-scene clips.

The UI needs both halves of the comparison: the original scene clip (produced in
the scene stage) and the de-edited one. Cutting from `clean.mp4` rather than
re-running removal per scene guarantees the two are consistent.

If inpainting was skipped -- nothing detected to remove, or a failure earlier --
this stage still reports honestly: no clean video, no clean clips, and the
reason is already in `diagnostics.degradations`.
"""

from __future__ import annotations

from app.db.models import JobStatus
from app.pipeline.base import Stage
from app.pipeline.context import JobContext
from app.utils import ffmpeg


class ExportStage(Stage):
    name = "export"
    status = JobStatus.EXPORTING
    weight = 1.2

    def run(self, ctx: JobContext) -> dict:
        ws = ctx.workspace
        scenes = (ctx.get("scenes") or {}).get("scenes", [])
        inpaint = ctx.get("inpaint") or {}
        clean_rel = inpaint.get("clean_video")

        if not clean_rel:
            ctx.degrade("no_clean_video_to_export")
            return {"clean_video": None, "removal_method": None, "clean_clips": {}}

        clean = ws.path(clean_rel)
        clips: dict[str, str] = {}
        total = len(scenes) or 1

        for i, scene in enumerate(scenes):
            ctx.raise_if_cancelled()
            ctx.progress(i / total, f"clean clip {i + 1}/{total}")
            destination = ws.path("clean", f"{scene['id']}.mp4")
            try:
                ffmpeg.cut(clean, destination, scene["t_in"], scene["t_out"])
                clips[scene["id"]] = ws.rel(destination)
            except ffmpeg.FFmpegError as exc:
                # One bad clip must not lose the whole clean render.
                ctx.degrade(f"clean_clip_failed:{scene['id']}")
                ctx.log(f"{scene['id']}: clean clip failed ({exc.detail[:60]})")

        ctx.log(f"{len(clips)}/{len(scenes)} clean scene clips exported")
        return {
            "clean_video": clean_rel,
            "removal_method": inpaint.get("removal_method"),
            "clean_clips": clips,
            "frames_inpainted": inpaint.get("frames_inpainted", 0),
        }
