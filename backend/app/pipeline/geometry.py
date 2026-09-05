"""Normalized box maths.

Every box in the system is `(x, y, w, h)` in 0..1 of the *display* frame. That
one convention is why analysis can run on a 720p proxy and render at 1080p
without a single conversion bug, and why project.json survives a resolution
change.
"""

from __future__ import annotations

from statistics import median

Box = tuple[float, float, float, float]

Quad = list[list[float]] | list[tuple[float, float]]


def clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def quad_to_box(quad: Quad, width: int, height: int) -> Box:
    """Axis-aligned, normalized bounds of an OCR quadrilateral.

    Detectors return rotated quads; downstream we only ever need the enclosing
    rectangle (masks are dilated anyway, and text is near-horizontal in
    short-form video).
    """
    xs = [float(point[0]) for point in quad]
    ys = [float(point[1]) for point in quad]
    x0, x1 = min(xs) / width, max(xs) / width
    y0, y1 = min(ys) / height, max(ys) / height
    x0, x1 = clamp01(x0), clamp01(x1)
    y0, y1 = clamp01(y0), clamp01(y1)
    return (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))


def area(box: Box) -> float:
    return max(0.0, box[2]) * max(0.0, box[3])


def center(box: Box) -> tuple[float, float]:
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def iou(a: Box, b: Box) -> float:
    """Intersection over union; 0 when they do not overlap."""
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]

    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = ix * iy
    if intersection <= 0:
        return 0.0
    return intersection / (area(a) + area(b) - intersection)


def union(boxes: list[Box]) -> Box:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def median_box(boxes: list[Box]) -> Box:
    """Per-component median -- robust to the odd frame where OCR mis-bounds."""
    return (
        median([b[0] for b in boxes]),
        median([b[1] for b in boxes]),
        median([b[2] for b in boxes]),
        median([b[3] for b in boxes]),
    )


def dilate(box: Box, margin: float) -> Box:
    """Grow by `margin` (fraction of the frame) on every side, clamped."""
    x = clamp01(box[0] - margin)
    y = clamp01(box[1] - margin)
    x1 = clamp01(box[0] + box[2] + margin)
    y1 = clamp01(box[1] + box[3] + margin)
    return (x, y, x1 - x, y1 - y)


def to_pixels(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    """Normalized -> integer pixel rect, clipped to the frame, never empty."""
    x = int(round(box[0] * width))
    y = int(round(box[1] * height))
    w = int(round(box[2] * width))
    h = int(round(box[3] * height))
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    return x, y, max(1, min(w, width - x)), max(1, min(h, height - y))


def to_dict(box: Box) -> dict[str, float]:
    return {"x": round(box[0], 5), "y": round(box[1], 5),
            "w": round(box[2], 5), "h": round(box[3], 5)}


def from_dict(data: dict) -> Box:
    return (float(data["x"]), float(data["y"]), float(data["w"]), float(data["h"]))


def horizontal_align(box: Box) -> str:
    cx, _ = center(box)
    if cx < 0.35:
        return "left"
    if cx > 0.65:
        return "right"
    return "center"
