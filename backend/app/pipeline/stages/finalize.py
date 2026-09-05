"""Assemble project.json -- the editable deliverable.

This stage is real from block A onwards: it collects whatever the upstream
stages produced and serialises the contract documented in plan.md §5. As stages
graduate from placeholder to real, they populate more of it; the schema and the
consumers never change.
"""

from __future__ import annotations

from app.db.models import JobStatus
from app.pipeline.base import Stage
from app.pipeline.context import JobContext
from app.storage import write_json

SCHEMA_VERSION = "1.0"


class FinalizeStage(Stage):
    name = "finalize"
    status = JobStatus.EXPORTING
    weight = 0.5
    resumable = False

    def run(self, ctx: JobContext) -> dict:
        ws = ctx.workspace
        media = ctx.get("probe", {}) or {}
        scenes = [dict(scene) for scene in (ctx.get("scenes") or {}).get("scenes", [])]
        text_tracks = (ctx.get("text_tracks") or {}).get("tracks", [])
        overlays = (ctx.get("overlays") or {}).get("tracks", [])
        transcript = (ctx.get("asr") or {}).get("segments", [])
        exported = ctx.get("export") or {}
        masks = ctx.get("masks") or {}

        # Attach the de-edited counterpart to each scene so the UI can A/B them.
        clean_clips = exported.get("clean_clips") or {}
        for scene in scenes:
            scene["clean_clip"] = clean_clips.get(scene["id"])

        if masks.get("segments"):
            write_json(ws.path("masks", "timeline.json"), masks)

        project = {
            "schema_version": SCHEMA_VERSION,
            "job_id": ctx.job_id,
            "source": {
                "kind": ctx.source_kind,
                "ref": ctx.source_ref,
                "sha256": (ctx.get("ingest") or {}).get("sha256"),
            },
            "media": {
                "duration": media.get("duration"),
                "fps": media.get("fps"),
                "width": media.get("width"),
                "height": media.get("height"),
                "has_audio": media.get("has_audio", False),
                "proxy": media.get("proxy"),
            },
            "scenes": scenes,
            "text_tracks": text_tracks,
            "overlay_tracks": overlays,
            "transcript": transcript,
            "outputs": {
                "clean_video": exported.get("clean_video"),
                "removal_method": exported.get("removal_method"),
                "source_video": ws.rel(ws.source) if ws.source.exists() else None,
                "audio": media.get("audio_path"),
                "mask_timeline": "masks/timeline.json" if masks.get("segments") else None,
                "frames_inpainted": exported.get("frames_inpainted", 0),
                "masked_fraction": masks.get("masked_fraction", 0.0),
            },
            "diagnostics": {
                "stage_timings_ms": ctx.timings_ms,
                "degradations": ctx.degradations,
                "options": ctx.options.model_dump(),
            },
        }

        ctx.progress(0.8, "writing project.json")
        write_json(ws.project_json, project)
        return {"path": ws.rel(ws.project_json), "counts": {
            "scenes": len(scenes),
            "text_tracks": len(text_tracks),
            "overlay_tracks": len(overlays),
            "samples": (ctx.get("sample") or {}).get("count", 0),
        }}
