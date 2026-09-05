"""S5a - detect on-screen text in the sampled frames.

RapidOCR runs PP-OCRv4 through onnxruntime: the same detection/recognition
quality as PaddleOCR, but with no PaddlePaddle dependency, which matters because
Paddle has no wheels for recent Pythons. It is CPU-only and fast enough that
sampling, not inference, is the cost driver.

Two things keep this cheap:
  * we only look at sampled frames (~3 fps), not every frame
  * consecutive frames that are visually identical reuse the previous result --
    short-form video holds a static frame far more often than it changes
"""

from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

from app.db.models import JobStatus
from app.pipeline import geometry
from app.pipeline.base import Stage
from app.pipeline.context import JobContext

log = logging.getLogger(__name__)

#: recogniser confidence floor
MIN_CONFIDENCE = 0.5
#: reject specks: fraction of frame area, and of frame height
MIN_BOX_AREA = 0.0004
MIN_BOX_HEIGHT = 0.012
#: mean per-pixel difference below which two frames count as identical
STATIC_FRAME_TOLERANCE = 1.5

_engine = None
_engine_lock = threading.Lock()


def get_engine():
    """Process-wide OCR engine. Loading the models costs ~0.6 s; do it once."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from rapidocr_onnxruntime import RapidOCR

                _engine = RapidOCR()
                log.info("RapidOCR engine loaded")
    return _engine


def frame_signature(image: np.ndarray) -> np.ndarray:
    """Tiny greyscale fingerprint used to spot unchanged frames."""
    return cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (32, 32),
                      interpolation=cv2.INTER_AREA).astype(np.int16)


def is_static(previous: np.ndarray | None, current: np.ndarray) -> bool:
    if previous is None:
        return False
    return float(np.abs(previous - current).mean()) < STATIC_FRAME_TOLERANCE


def normalise_result(raw, width: int, height: int) -> list[dict]:
    """RapidOCR gives [quad, text, confidence]; keep the usable ones."""
    detections: list[dict] = []
    for entry in raw or []:
        try:
            quad, text, confidence = entry[0], str(entry[1]), float(entry[2])
        except (IndexError, TypeError, ValueError):
            continue
        text = text.strip()
        if not text or confidence < MIN_CONFIDENCE:
            continue
        box = geometry.quad_to_box(quad, width, height)
        if geometry.area(box) < MIN_BOX_AREA or box[3] < MIN_BOX_HEIGHT:
            continue
        detections.append({
            "box": geometry.to_dict(box),
            "text": text,
            "conf": round(confidence, 4),
        })
    return detections


class OcrStage(Stage):
    name = "ocr"
    status = JobStatus.ANALYZING
    weight = 2.5

    def skip_if(self, ctx: JobContext) -> str | None:
        if not (ctx.get("sample") or {}).get("samples"):
            return "no_sampled_frames"
        return None

    def run(self, ctx: JobContext) -> dict:
        ws = ctx.workspace
        samples = ctx.require("sample")["samples"]
        proxy = ctx.require("probe")["proxy"]
        width, height = int(proxy["width"]), int(proxy["height"])

        engine = get_engine()
        ctx.progress(0.02, f"running OCR over {len(samples)} frames")

        frames: list[dict] = []
        previous_signature: np.ndarray | None = None
        previous_detections: list[dict] = []
        reused = 0
        unreadable = 0

        for i, sample in enumerate(samples):
            ctx.raise_if_cancelled()
            image = cv2.imread(str(ws.path(sample["path"])))
            if image is None:
                unreadable += 1
                continue

            signature = frame_signature(image)
            if is_static(previous_signature, signature):
                detections = [dict(d) for d in previous_detections]
                reused += 1
            else:
                raw, _elapsed = engine(image)
                detections = normalise_result(raw, width, height)
                previous_detections = detections
            previous_signature = signature

            frames.append({
                "t": sample["t"],
                "scene_id": sample["scene_id"],
                "frame_path": sample["path"],
                "detections": detections,
            })

            if i % 5 == 0 or i == len(samples) - 1:
                boxes = sum(len(f["detections"]) for f in frames)
                ctx.progress(
                    0.02 + 0.98 * ((i + 1) / len(samples)),
                    f"OCR {i + 1}/{len(samples)} · {boxes} boxes",
                )

        total = sum(len(f["detections"]) for f in frames)
        if unreadable:
            ctx.degrade(f"unreadable_frames:{unreadable}")
        if not total:
            ctx.degrade("no_text_detected")
        ctx.log(f"{total} detections across {len(frames)} frames ({reused} reused as static)")

        return {
            "engine": "rapidocr-ppocrv4",
            "frames": frames,
            "detection_count": total,
            "reused_frames": reused,
            "unreadable_frames": unreadable,
        }
