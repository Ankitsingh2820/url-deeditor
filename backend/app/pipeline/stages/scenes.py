"""S3 - segment the video into scenes.

PySceneDetect's AdaptiveDetector compares each frame's HSV content delta against
a rolling window of its neighbours, rather than a fixed threshold. That matters
here: UGC ads are shot handheld, so a fixed threshold either fires on every
camera shake or misses cuts between two similarly-lit shots.

Post-processing is where the practical wins are -- raw detector output on
short-form content contains flash frames a human would never call a scene.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.db.models import JobStatus
from app.pipeline.base import Stage
from app.pipeline.context import JobContext
from app.utils import ffmpeg

#: shorter than this and it is a flash frame / transition artefact, not a scene
MIN_SCENE_S = 0.35
#: below this, uniform chunking is a better guess than "one scene"
FALLBACK_CHUNK_S = 3.0
#: a single detected scene on a video longer than this is suspicious
SINGLE_SCENE_SUSPICION_S = 6.0

Span = tuple[float, float]


@dataclass
class SceneSpan:
    index: int
    t_in: float
    t_out: float

    @property
    def id(self) -> str:
        return f"sc_{self.index + 1:03d}"

    @property
    def duration(self) -> float:
        return self.t_out - self.t_in

    @property
    def midpoint(self) -> float:
        return self.t_in + self.duration / 2


# --------------------------------------------------------------- pure helpers


def merge_short_spans(spans: list[Span], min_len: float = MIN_SCENE_S) -> list[Span]:
    """Fold sub-threshold spans into their neighbour.

    Merges forward into the previous scene where possible, otherwise into the
    next one, so a flash frame never survives as a scene of its own.
    """
    if not spans:
        return []
    merged: list[list[float]] = [list(spans[0])]
    for start, end in spans[1:]:
        previous = merged[-1]
        if (end - start) < min_len or (previous[1] - previous[0]) < min_len:
            previous[1] = end
        else:
            merged.append([start, end])
    # A trailing short span has no "next" to merge into; fold it backwards.
    if len(merged) > 1 and (merged[-1][1] - merged[-1][0]) < min_len:
        merged[-2][1] = merged[-1][1]
        merged.pop()
    return [(round(a, 3), round(b, 3)) for a, b in merged]


def uniform_spans(duration: float, chunk: float = FALLBACK_CHUNK_S) -> list[Span]:
    """Fallback segmentation when detection finds nothing."""
    if duration <= 0:
        return []
    spans: list[Span] = []
    start = 0.0
    while start < duration - 0.01:
        end = min(start + chunk, duration)
        spans.append((round(start, 3), round(end, 3)))
        start = end
    # Avoid leaving a sliver at the end.
    if len(spans) > 1 and (spans[-1][1] - spans[-1][0]) < chunk / 3:
        spans[-2] = (spans[-2][0], spans[-1][1])
        spans.pop()
    return spans


def detect_spans(video_path, *, min_scene_s: float = MIN_SCENE_S,
                 adaptive_threshold: float = 3.0) -> list[Span]:
    """Run PySceneDetect and return (t_in, t_out) pairs in seconds."""
    from scenedetect import AdaptiveDetector, SceneManager, open_video

    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(
        AdaptiveDetector(
            adaptive_threshold=adaptive_threshold,
            min_scene_len=max(1, int(min_scene_s * video.frame_rate)),
            window_width=2,
        )
    )
    manager.detect_scenes(video, show_progress=False)
    return [
        (round(start.seconds, 3), round(end.seconds, 3))
        for start, end in manager.get_scene_list()
    ]


# ------------------------------------------------------------------- stage


class SceneStage(Stage):
    name = "scenes"
    status = JobStatus.ANALYZING
    weight = 1.5

    def run(self, ctx: JobContext) -> dict:
        ws = ctx.workspace
        media = ctx.require("probe")
        duration = float(media["duration"])

        ctx.progress(0.05, "detecting cuts")
        try:
            raw = detect_spans(ws.proxy)
            method = "adaptive"
        except Exception as exc:  # noqa: BLE001 - never let detection fail the job
            ctx.degrade(f"scene_detect_failed:{type(exc).__name__}")
            raw, method = [], "fallback_uniform"

        spans = merge_short_spans(raw)
        if not spans or (len(spans) == 1 and duration > SINGLE_SCENE_SUSPICION_S):
            # Either the detector found nothing, or it claims one scene for a
            # video long enough that an edit is near-certain. Chunk instead, and
            # say so rather than pretending the result is a detection.
            ctx.degrade("scene_detection_inconclusive")
            spans = uniform_spans(duration)
            method = "fallback_uniform"
        if not spans:
            spans = [(0.0, duration)]
            method = "single"

        scenes = [SceneSpan(i, a, b) for i, (a, b) in enumerate(spans)]
        ctx.log(f"{len(scenes)} scenes via {method} (from {len(raw)} raw cuts)")

        exported = self._export(ctx, scenes)
        return {
            "method": method,
            "raw_cut_count": len(raw),
            "count": len(scenes),
            "scenes": exported,
        }

    def _export(self, ctx: JobContext, scenes: list[SceneSpan]) -> list[dict]:
        """Cut a clip and grab a keyframe per scene, reporting progress as we go."""
        ws = ctx.workspace
        out: list[dict] = []
        total = len(scenes) or 1

        for i, scene in enumerate(scenes):
            ctx.progress(0.15 + 0.85 * (i / total), f"exporting {scene.id} ({i + 1}/{total})")

            keyframe = ws.path("frames", f"{scene.id}_key.jpg")
            clip = ws.path("scenes", f"{scene.id}.mp4")
            entry: dict = {
                "id": scene.id,
                "index": scene.index,
                "t_in": round(scene.t_in, 3),
                "t_out": round(scene.t_out, 3),
                "duration": round(scene.duration, 3),
                "keyframe": None,
                "clip": None,
                "clean_clip": None,
                "ai": None,
            }

            # Keyframes come from the source so the VLM sees full detail.
            try:
                ffmpeg.extract_frame(ws.source, scene.midpoint, keyframe, width=640)
                entry["keyframe"] = ws.rel(keyframe)
            except ffmpeg.FFmpegError as exc:
                ctx.degrade(f"keyframe_failed:{scene.id}")
                ctx.log(f"{scene.id}: keyframe failed ({exc.detail[:60]})")

            try:
                ffmpeg.cut(ws.source, clip, scene.t_in, scene.t_out)
                entry["clip"] = ws.rel(clip)
            except ffmpeg.FFmpegError as exc:
                ctx.degrade(f"clip_failed:{scene.id}")
                ctx.log(f"{scene.id}: clip export failed ({exc.detail[:60]})")

            out.append(entry)
        return out
