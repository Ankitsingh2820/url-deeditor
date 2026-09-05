"""S4 - sample frames for the detectors.

OCR and overlay detection do not need every frame. Sampling at ~3 fps turns a
30 s video into ~90 images: seconds of OCR instead of minutes, while still
resolving caption changes (short-form captions rarely change faster than ~3 Hz).

Sampling is uniform in one decode pass, then topped up for scenes that came out
under-sampled -- a 0.5 s cut would otherwise get one frame or none, and a scene
with no samples is invisible to every stage that follows.
"""

from __future__ import annotations

from bisect import bisect_right

from app.db.models import JobStatus
from app.pipeline.base import Stage
from app.pipeline.context import JobContext
from app.storage import write_json
from app.utils import ffmpeg

#: every scene gets at least this many frames, however short it is
MIN_SAMPLES_PER_SCENE = 2


def assign_scene(timestamp: float, starts: list[float], ids: list[str]) -> str | None:
    """Map a timestamp onto a scene id via binary search over scene starts."""
    if not starts:
        return None
    index = bisect_right(starts, timestamp) - 1
    return ids[index] if index >= 0 else ids[0]


class SampleStage(Stage):
    name = "sample"
    status = JobStatus.ANALYZING
    weight = 0.7

    def run(self, ctx: JobContext) -> dict:
        ws = ctx.workspace
        scenes = ctx.require("scenes")["scenes"]
        fps = ctx.options.sample_fps
        out_dir = ws.path("frames", "samples")

        ctx.progress(0.1, f"sampling at {fps} fps")
        files = ffmpeg.extract_frames(ws.proxy, out_dir, fps=fps)

        starts = [s["t_in"] for s in scenes]
        ids = [s["id"] for s in scenes]

        samples = [
            {
                "index": i,
                "t": round(ffmpeg.frame_timestamp(i, fps), 3),
                "path": ws.rel(path),
                "scene_id": assign_scene(ffmpeg.frame_timestamp(i, fps), starts, ids),
            }
            for i, path in enumerate(files, start=1)
        ]

        ctx.progress(0.7, "topping up short scenes")
        samples.extend(self._top_up(ctx, scenes, samples))
        samples.sort(key=lambda s: s["t"])

        manifest = {
            "fps": fps,
            "count": len(samples),
            "source": ws.rel(ws.proxy),
            "samples": samples,
        }
        write_json(out_dir / "index.json", manifest)
        ctx.log(f"{len(samples)} frames across {len(scenes)} scenes")
        return manifest

    def _top_up(self, ctx: JobContext, scenes: list[dict], samples: list[dict]) -> list[dict]:
        """Force-extract frames for scenes the uniform pass under-covered."""
        ws = ctx.workspace
        counts: dict[str, int] = {}
        for sample in samples:
            counts[sample["scene_id"]] = counts.get(sample["scene_id"], 0) + 1

        extra: list[dict] = []
        for scene in scenes:
            missing = MIN_SAMPLES_PER_SCENE - counts.get(scene["id"], 0)
            if missing <= 0:
                continue
            span = scene["t_out"] - scene["t_in"]
            for k in range(missing):
                # Spread the extra frames inside the scene, away from the cuts.
                t = scene["t_in"] + span * (k + 1) / (missing + 1)
                path = ws.path("frames", "samples", f"x_{scene['id']}_{k}.jpg")
                try:
                    ffmpeg.extract_frame(ws.proxy, t, path)
                except ffmpeg.FFmpegError:
                    ctx.degrade(f"sample_topup_failed:{scene['id']}")
                    continue
                extra.append({
                    "index": -1,
                    "t": round(t, 3),
                    "path": ws.rel(path),
                    "scene_id": scene["id"],
                })
        return extra
