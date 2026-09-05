"""S5b - group OCR detections into editable text tracks.

Output shape is the `text_tracks[]` contract from plan.md §5: one entry per
on-screen text element, with the timing, box and style needed to remove it from
the footage *and* to re-render it afterwards.
"""

from __future__ import annotations

import cv2

from app.db.models import JobStatus
from app.pipeline import geometry, tracking
from app.pipeline.base import Stage
from app.pipeline.context import JobContext
from app.pipeline.tracking import Detection
from app.utils.style import TextStyle, extract_style


class TextTrackStage(Stage):
    name = "text_tracks"
    status = JobStatus.ANALYZING
    weight = 0.6

    def skip_if(self, ctx: JobContext) -> str | None:
        if not (ctx.get("ocr") or {}).get("detection_count"):
            return "no_detections"
        return None

    def run(self, ctx: JobContext) -> dict:
        ocr = ctx.require("ocr")
        media = ctx.require("probe")
        sample_interval = 1.0 / max(0.1, ctx.require("sample")["fps"])

        detections = [
            Detection(
                t=frame["t"],
                box=geometry.from_dict(det["box"]),
                text=det["text"],
                conf=det["conf"],
                scene_id=frame.get("scene_id"),
                frame_path=frame.get("frame_path"),
            )
            for frame in ocr["frames"]
            for det in frame["detections"]
        ]

        ctx.progress(0.2, f"tracking {len(detections)} detections")
        tracks = tracking.track_detections(detections, sample_interval=sample_interval)
        kept = tracking.prune(tracks, sample_interval)
        ctx.log(f"{len(tracks)} raw tracks -> {len(kept)} after pruning")

        ctx.progress(0.5, "extracting style")
        entries = [
            self._build(ctx, index, track, sample_interval, float(media["duration"]))
            for index, track in enumerate(
                sorted(kept, key=lambda t: t.span(sample_interval)[0])
            )
        ]

        by_kind: dict[str, int] = {}
        for entry in entries:
            by_kind[entry["kind"]] = by_kind.get(entry["kind"], 0) + 1
        ctx.log(f"text tracks by kind: {by_kind or 'none'}")

        return {"tracks": entries, "count": len(entries), "by_kind": by_kind}

    def _build(self, ctx: JobContext, index: int, track: tracking.Track,
               sample_interval: float, media_duration: float) -> dict:
        t_in, t_out = track.span(sample_interval)
        box = track.box
        box_dict = geometry.to_dict(box)
        style = self._style(ctx, track, box)
        # Take the height straight from the serialised box so the two never
        # disagree by a rounding step.
        style.font_px_norm = box_dict["h"]
        style.align = geometry.horizontal_align(box)

        return {
            "id": f"tx_{index + 1:03d}",
            "kind": tracking.classify(box, t_out - t_in, media_duration, track.text),
            "text": track.text,
            "t_in": t_in,
            "t_out": t_out,
            "scene_id": track.scene_id,
            "box": box_dict,
            "style": style.to_dict(),
            "confidence": track.confidence,
            "source": "ocr",
            "samples": len(track.detections),
            "editable": True,
        }

    def _style(self, ctx: JobContext, track: tracking.Track, box: geometry.Box) -> TextStyle:
        """Measure style on the frame OCR was most confident about."""
        best = max(track.detections, key=lambda d: d.conf)
        if not best.frame_path:
            return TextStyle()

        image = cv2.imread(str(ctx.workspace.path(best.frame_path)))
        if image is None:
            ctx.degrade("style_frame_unreadable")
            return TextStyle()

        height, width = image.shape[:2]
        x, y, w, h = geometry.to_pixels(box, width, height)
        return extract_style(image[y:y + h, x:x + w])
