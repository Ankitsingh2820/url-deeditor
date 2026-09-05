"""S8 - build the mask timeline.

Everything the pipeline wants to remove -- text tracks now, image/product
overlays from block F -- collapses into one time-ordered structure: a list of
segments, each holding the boxes active for that span.

Why segments and not per-frame masks: a 30 s clip is ~900 frames, and the set of
active boxes changes maybe a dozen times. Storing the *changes* keeps the
artifact small enough to live in project.json, makes the timeline directly
renderable in the UI, and lets the inpainter look up a frame's boxes with a
binary search instead of loading a mask image per frame.

Two adjustments make the masks actually usable:
  * spatial padding, because glyph antialiasing bleeds past the OCR box and an
    inpainter fed a tight mask leaves a visible halo
  * temporal padding, because text fades in and out between sampled frames
"""

from __future__ import annotations

from bisect import bisect_right

from app.db.models import JobStatus
from app.pipeline import geometry
from app.pipeline.base import Stage
from app.pipeline.context import JobContext

#: mask padding, as a fraction of frame *height* (applied equally in pixels on
#: both axes, so a portrait frame does not get a stretched margin)
PAD_FRACTION = 0.015
#: extra time either side of a track, to cover fades between sampled frames
PAD_SECONDS = 0.08


def pad_box(box: geometry.Box, aspect: float, fraction: float = PAD_FRACTION) -> geometry.Box:
    """Grow a box by the same number of *pixels* on every side.

    `aspect` is width/height. A uniform margin in normalized units would be
    wider than it is tall on a portrait frame, so the x margin is scaled by it.
    """
    margin_y = fraction
    margin_x = fraction / aspect if aspect > 0 else fraction
    x = geometry.clamp01(box[0] - margin_x)
    y = geometry.clamp01(box[1] - margin_y)
    x1 = geometry.clamp01(box[0] + box[2] + margin_x)
    y1 = geometry.clamp01(box[1] + box[3] + margin_y)
    return (x, y, x1 - x, y1 - y)


def build_segments(tracks: list[dict], *, aspect: float, duration: float,
                   pad_s: float = PAD_SECONDS) -> list[dict]:
    """Collapse overlapping track lifetimes into constant-box-set segments."""
    windows = []
    for track in tracks:
        t_in = max(0.0, float(track["t_in"]) - pad_s)
        t_out = min(duration, float(track["t_out"]) + pad_s) if duration > 0 else \
            float(track["t_out"]) + pad_s
        if t_out <= t_in:
            continue
        windows.append((t_in, t_out, track))

    if not windows:
        return []

    # Every start and end is a point where the active set can change.
    breakpoints = sorted({t for window in windows for t in window[:2]})
    segments: list[dict] = []
    for start, end in zip(breakpoints, breakpoints[1:], strict=False):
        if end - start < 1e-6:
            continue
        active = [w for w in windows if w[0] <= start and w[1] >= end]
        if not active:
            continue
        segments.append({
            "t_in": round(start, 3),
            "t_out": round(end, 3),
            "boxes": [
                {
                    **geometry.to_dict(pad_box(geometry.from_dict(track["box"]), aspect)),
                    "source_id": track["id"],
                    "kind": track.get("kind", "text"),
                }
                for *_span, track in active
            ],
        })

    # Adjacent segments with an identical box set are one segment.
    merged: list[dict] = []
    for segment in segments:
        previous = merged[-1] if merged else None
        same = previous is not None and (
            abs(previous["t_out"] - segment["t_in"]) < 1e-6
            and {b["source_id"] for b in previous["boxes"]}
            == {b["source_id"] for b in segment["boxes"]}
        )
        if same:
            previous["t_out"] = segment["t_out"]
        else:
            merged.append(segment)
    return merged


def boxes_at(segments: list[dict], starts: list[float], t: float) -> list[dict]:
    """Boxes active at time `t`; `starts` is the precomputed segment start list."""
    if not segments:
        return []
    index = bisect_right(starts, t) - 1
    if index < 0:
        return []
    segment = segments[index]
    return segment["boxes"] if segment["t_in"] <= t < segment["t_out"] else []


class MaskStage(Stage):
    name = "masks"
    status = JobStatus.INPAINTING
    weight = 0.4

    def run(self, ctx: JobContext) -> dict:
        media = ctx.require("probe")
        width, height = int(media["width"]), int(media["height"])
        duration = float(media["duration"])

        sources: list[dict] = []
        sources += (ctx.get("text_tracks") or {}).get("tracks", [])
        sources += (ctx.get("overlays") or {}).get("tracks", [])  # block F

        if not sources:
            ctx.degrade("nothing_to_mask")
            return {"segments": [], "count": 0, "coverage_s": 0.0, "masked_fraction": 0.0}

        ctx.progress(0.4, f"building mask timeline from {len(sources)} elements")
        segments = build_segments(sources, aspect=width / height, duration=duration)

        coverage = sum(s["t_out"] - s["t_in"] for s in segments)
        ctx.log(
            f"{len(segments)} mask segments covering {coverage:.2f}s "
            f"({coverage / duration:.0%} of the video)" if duration else ""
        )
        return {
            "segments": segments,
            "count": len(segments),
            "coverage_s": round(coverage, 3),
            "masked_fraction": round(coverage / duration, 4) if duration else 0.0,
            "pad_fraction": PAD_FRACTION,
            "pad_seconds": PAD_SECONDS,
        }
