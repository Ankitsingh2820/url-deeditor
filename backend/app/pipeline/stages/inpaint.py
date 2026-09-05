"""S9 - remove the masked elements and render the clean plate.

Three tiers, chosen at runtime and always degradable:

  mask_blur  heavy blur composited through the mask. Instant, obviously not a
             removal, but it never fails -- the guaranteed floor.
  opencv     cv2.inpaint (Telea). Fast, no model, decent on flat backgrounds,
             smears on busy ones.
  lama       LaMa via simple-lama-inpainting. Big quality jump; used when the
             package is installed, skipped silently when it is not.

`auto` picks the best tier available. Whatever happens, the choice is recorded
in the output so project.json says how the clean video was actually produced
rather than implying a quality it does not have.

Only frames that carry a mask are inpainted. In short-form video that is
typically well under half of them, which is where most of the time is saved.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from app.db.models import JobStatus
from app.pipeline import geometry
from app.pipeline.base import Stage
from app.pipeline.context import JobContext
from app.pipeline.stages.masks import boxes_at
from app.utils import ffmpeg

log = logging.getLogger(__name__)

#: Telea radius, in pixels
INPAINT_RADIUS = 4
#: how often to report progress / check for cancellation, in frames
REPORT_EVERY = 15


def render_mask(height: int, width: int, boxes: list[dict]) -> np.ndarray:
    """Binary mask (255 inside the boxes) at source resolution."""
    mask = np.zeros((height, width), np.uint8)
    for box in boxes:
        x, y, w, h = geometry.to_pixels(geometry.from_dict(box), width, height)
        mask[y:y + h, x:x + w] = 255
    return mask


def _lama():
    """Return a LaMa inpainter, or None when the package is not installed."""
    try:
        from simple_lama_inpainting import SimpleLama
    except Exception:  # noqa: BLE001 - optional dependency, any import error is a skip
        return None
    try:
        return SimpleLama()
    except Exception as exc:  # noqa: BLE001 - weights download / torch issues
        log.warning("LaMa unavailable: %s", exc)
        return None


def resolve_method(requested: str) -> tuple[str, object | None]:
    """Pick the best tier that is actually available for `requested`."""
    if requested in ("auto", "lama"):
        model = _lama()
        if model is not None:
            return "lama", model
        if requested == "lama":
            return "opencv", None  # asked for LaMa, fall back and say so
        return "opencv", None
    return requested, None


def inpaint_frame(frame: np.ndarray, mask: np.ndarray, method: str, model=None) -> np.ndarray:
    if method == "mask_blur":
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=12, sigmaY=12)
        return np.where(mask[:, :, None] > 0, blurred, frame)
    if method == "lama" and model is not None:
        from PIL import Image

        result = model(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                       Image.fromarray(mask))
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    return cv2.inpaint(frame, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)


class InpaintStage(Stage):
    name = "inpaint"
    status = JobStatus.INPAINTING
    weight = 3.0

    def skip_if(self, ctx: JobContext) -> str | None:
        if not (ctx.get("masks") or {}).get("segments"):
            return "nothing_to_remove"
        return None

    def run(self, ctx: JobContext) -> dict:
        ws = ctx.workspace
        media = ctx.require("probe")
        segments = ctx.require("masks")["segments"]
        starts = [s["t_in"] for s in segments]

        requested = ctx.options.removal_method
        method, model = resolve_method(requested)
        if method != requested and requested != "auto":
            ctx.degrade(f"removal_downgraded:{requested}->{method}")
        ctx.log(f"removal method: {method} (requested {requested})")

        capture = cv2.VideoCapture(str(ws.source))
        if not capture.isOpened():
            raise ffmpeg.FFmpegError("could not open source for inpainting")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS) or float(media["fps"])
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        clean = ws.path("clean", "clean.mp4")

        processed = 0
        touched = 0
        try:
            sink = ffmpeg.FrameSink(
                clean, width=width, height=height, fps=fps,
                audio_from=ws.source if media.get("has_audio") else None,
            )
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    t = processed / fps if fps else 0.0
                    boxes = boxes_at(segments, starts, t)
                    if boxes:
                        mask = render_mask(height, width, boxes)
                        frame = inpaint_frame(frame, mask, method, model)
                        touched += 1
                    sink.write(np.ascontiguousarray(frame))
                    processed += 1

                    if processed % REPORT_EVERY == 0:
                        ctx.raise_if_cancelled()
                        fraction = processed / total if total else 0.5
                        ctx.progress(min(0.98, fraction),
                                     f"{method}: {processed}/{total or '?'} frames "
                                     f"({touched} masked)")
            finally:
                sink.close()
        finally:
            capture.release()

        if not processed:
            raise ffmpeg.FFmpegError("decoded no frames from the source")

        ctx.log(f"clean plate written: {processed} frames, {touched} inpainted")
        return {
            "clean_video": ws.rel(clean),
            "removal_method": method,
            "frames": processed,
            "frames_inpainted": touched,
            "width": width,
            "height": height,
            "fps": round(fps, 3),
        }
