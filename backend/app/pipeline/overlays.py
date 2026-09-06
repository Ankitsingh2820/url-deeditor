"""Detect composited image layers -- product pop-ups, stickers, screenshots.

There is no off-the-shelf "is this an overlay" model, so the detector infers it
from how a pasted layer *behaves* differently from filmed content. Two candidate
sources, because the camera may or may not be moving:

  screen-space stability   A pasted layer is nailed to the screen. When the
                           camera moves, the world shifts while the overlay does
                           not, so its temporal variance collapses to ~0 while
                           the background's does not.

  transience               For a locked-off camera, where nothing moves and
                           stability tells you nothing. A pop-up changes its
                           pixels once or twice (it arrives, it leaves) and is
                           frozen in between, while a person moves in nearly
                           every frame.

The two are **mutually exclusive, selected by camera state**. Transience is
worthless while the camera pans: a pixel inside a large flat region does not
change at all as it slides past, and one near a boundary changes only as an edge
sweeps over it, so background counts land in the same 1-2 range a pop-up does.
Measured on a panning fixture, transition counts inside a pasted card (2.74) were
indistinguishable from the background (2.66). Stability has the mirror-image
flaw: with a locked-off camera everything is stable.

Two more things are needed to make either usable:

**Structure.** A flat wall is perfectly stable and never changes, so it looks
exactly like an overlay. Intersecting the candidate mask with dilated edges
restricts it to regions that actually contain something -- and, just as
importantly, stops flat background from merging with a real overlay into one
enormous blob that then fails the area filter.

**Windowing.** A pop-up that appears mid-shot is *not* stable across the whole
scene: its region holds background, then a frozen card, then background again.
Measuring over short sliding windows finds it in the windows where it is
present; `presence_span` then recovers its true in/out times.

Everything here takes arrays and returns numbers: no video files, no model
weights, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.pipeline import geometry
from app.pipeline.geometry import Box

# ---- tuning ---------------------------------------------------------------
#: a candidate must occupy at least this fraction of the frame (kills speckle)
MIN_AREA = 0.004
#: ...and at most this much (a near-full-frame "overlay" is a scene, not a layer)
MAX_AREA = 0.55
#: absolute temporal-std floor, in grey levels, for "this pixel never changes"
STABILITY_ABS = 8.0
#: ...or this fraction of the frame's own median std, whichever is larger
STABILITY_REL = 0.35
#: per-pixel difference counted as "this pixel changed" between two samples
CHANGE_DELTA = 28
#: a pop-up appears and leaves: at most this many changes over the scene
MAX_TRANSITIONS = 2
#: minimum edge density inside a candidate -- rejects flat walls and skies
MIN_EDGE_DENSITY = 0.02
#: median inter-frame shift, in pixels, above which the camera counts as moving
MOTION_THRESHOLD_PX = 1.2
#: fused score a candidate must reach
SCORE_THRESHOLD = 0.55
#: normalized correlation above which a patch counts as still present
PRESENCE_NCC = 0.55
#: overlap with a text track that makes a candidate "already accounted for"
TEXT_OVERLAP_IOU = 0.3
#: radius by which edges are grown to define "this region contains something"
STRUCTURE_DILATE = 9
#: closing kernel that fills a bordered card whose interior is flat
SOLIDIFY_KERNEL = 21
#: samples per analysis window, and the step between windows
WINDOW_SIZE = 4
WINDOW_STRIDE = 2
#: component area over bounding-box area, below which a blob is not an asset
MIN_SOLIDITY = 0.35
#: an overlay has duration: seen in one sampled frame only, it is noise
MIN_PRESENT_SAMPLES = 2
#: fraction of the smaller box inside a larger one that makes it a duplicate
CONTAINMENT_THRESHOLD = 0.6


@dataclass
class Candidate:
    box: Box
    scene_id: str
    t_in: float
    t_out: float
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return round(fuse(self.scores), 4)


# ---- camera motion --------------------------------------------------------


def estimate_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Global translation between two greyscale frames, via phase correlation.

    Phase correlation rather than feature matching: it is deterministic, needs
    no descriptors, and handles the low-texture frames where ORB returns too few
    keypoints to fit a transform. Pans and handheld drift are overwhelmingly
    translational, which is what this recovers; rotation/zoom would need an
    affine fit and is the natural upgrade.
    """
    if a.shape != b.shape or a.size == 0:
        return 0.0, 0.0, 0.0
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32), window)
    return float(dx), float(dy), float(response)


def camera_motion(frames: list[np.ndarray]) -> float:
    """Median inter-frame shift magnitude, in pixels."""
    if len(frames) < 2:
        return 0.0
    shifts = []
    for previous, current in zip(frames, frames[1:], strict=False):
        dx, dy, _ = estimate_shift(previous, current)
        shifts.append(float(np.hypot(dx, dy)))
    return float(np.median(shifts)) if shifts else 0.0


# ---- candidate masks ------------------------------------------------------


def stability_map(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel temporal standard deviation in screen space."""
    stack = np.stack(frames).astype(np.float32)
    return stack.std(axis=0)


def stability_mask(std_map: np.ndarray) -> np.ndarray:
    """Pixels that barely change while the rest of the frame does."""
    reference = float(np.median(std_map))
    threshold = max(STABILITY_ABS, STABILITY_REL * reference)
    return (std_map < threshold).astype(np.uint8) * 255


def transition_count(frames: list[np.ndarray], delta: int = CHANGE_DELTA) -> np.ndarray:
    """How many times each pixel changed materially across the samples.

    0 = static background, 1-2 = something appeared and/or left, many = motion.
    """
    if len(frames) < 2:
        return np.zeros(frames[0].shape, np.int16) if frames else np.zeros((1, 1), np.int16)
    counts = np.zeros(frames[0].shape, np.int16)
    for previous, current in zip(frames, frames[1:], strict=False):
        changed = cv2.absdiff(previous, current) > delta
        counts += changed.astype(np.int16)
    return counts


def transience_mask(counts: np.ndarray) -> np.ndarray:
    """Regions that changed once or twice -- the signature of a pop-up."""
    return ((counts >= 1) & (counts <= MAX_TRANSITIONS)).astype(np.uint8) * 255


def edge_map(frame: np.ndarray) -> np.ndarray:
    return cv2.Canny(cv2.GaussianBlur(frame, (3, 3), 0), 60, 160)


def structure_mask(frame: np.ndarray, dilate: int = STRUCTURE_DILATE) -> np.ndarray:
    """Regions that contain something: dilated edges.

    Intersecting a stability or transience mask with this is what separates a
    pasted asset from a flat wall, and keeps flat background from fusing with a
    real overlay into a single oversized component.
    """
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    return cv2.dilate(edge_map(frame), element, iterations=1)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes, so a bordered card becomes a solid block.

    A product card is mostly flat, so intersecting with edges leaves only its
    border -- a ring. A ring's bounding box is the card, but its *solidity* is
    near zero, so the shape filter throws the detection away. Flood-filling from
    the frame edge and inverting recovers everything the border encloses, which
    is what the object actually is.
    """
    height, width = mask.shape[:2]
    flood = mask.copy()
    # floodFill needs a mask 2px larger than the image on each axis.
    scratch = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(flood, scratch, (0, 0), 255)
    return mask | cv2.bitwise_not(flood)


def solidify(mask: np.ndarray, kernel: int = SOLIDIFY_KERNEL) -> np.ndarray:
    """Close gaps in an asset's outline, then fill what the outline encloses."""
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element, iterations=1)
    return fill_holes(closed)


def windows(count: int, size: int = WINDOW_SIZE, stride: int = WINDOW_STRIDE) -> list[range]:
    """Sliding index windows over a scene's samples.

    A pop-up occupying only part of a shot is stable *within* a window even
    though it is not across the whole scene.
    """
    if count <= size:
        return [range(count)] if count else []
    result = [range(i, i + size) for i in range(0, count - size + 1, stride)]
    # The stride can leave a tail uncovered (9 samples, size 4, stride 2 stops at
    # index 7). Anchor a final window to the end so an overlay that appears in
    # the last moments of a scene is still seen.
    if result[-1][-1] < count - 1:
        result.append(range(count - size, count))
    return result


def clean_mask(mask: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Close pinholes, drop speckle, so connected components are meaningful."""
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element, iterations=2)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, element, iterations=1)


def boxes_from_mask(mask: np.ndarray, *, min_area: float = MIN_AREA,
                    max_area: float = MAX_AREA) -> list[tuple[Box, float]]:
    """Connected components as (normalized box, solidity) pairs.

    Solidity -- the component's area over its bounding box's area -- is the
    robust way to ask "is this a pasted rectangle". Measuring edge strength
    along the bounding box's border instead sounds equivalent but is not: the
    mask a real overlay produces is a little looser than the asset itself, so
    the border band lands just outside the asset's edges and scores ~0 on a
    perfectly good detection.
    """
    height, width = mask.shape[:2]
    frame_area = float(height * width)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    found: list[tuple[Box, float]] = []
    for index in range(1, count):  # 0 is the background label
        x, y, w, h, area = stats[index]
        if not (min_area <= (w * h) / frame_area <= max_area):
            continue
        solidity = area / max(1.0, w * h)
        if solidity < MIN_SOLIDITY:
            continue  # a sparse, stringy blob is not a pasted rectangle
        found.append(((x / width, y / height, w / width, h / height), float(solidity)))
    return found


# ---- scoring --------------------------------------------------------------


def edge_density(edges: np.ndarray, box: Box) -> float:
    """Fraction of edge pixels inside the box -- the flat-wall rejector."""
    height, width = edges.shape[:2]
    x, y, w, h = geometry.to_pixels(box, width, height)
    patch = edges[y:y + h, x:x + w]
    return float(np.count_nonzero(patch)) / max(1.0, patch.size)


def stability_score(std_map: np.ndarray, box: Box) -> float:
    """How much stiller the box is than the frame as a whole."""
    height, width = std_map.shape[:2]
    x, y, w, h = geometry.to_pixels(box, width, height)
    inside = float(std_map[y:y + h, x:x + w].mean())
    overall = float(np.median(std_map))
    if overall <= 1e-6:
        return 0.0  # a completely still frame proves nothing
    return float(np.clip(1.0 - inside / overall, 0.0, 1.0))


def transience_score(counts: np.ndarray, box: Box) -> float:
    """Fraction of the box that changed only once or twice."""
    height, width = counts.shape[:2]
    x, y, w, h = geometry.to_pixels(box, width, height)
    patch = counts[y:y + h, x:x + w]
    if patch.size == 0:
        return 0.0
    transient = ((patch >= 1) & (patch <= MAX_TRANSITIONS)).sum()
    return float(transient) / patch.size


# Stability and transience are alternatives, never both live, so they share a
# single weight slot; detail and rectangularity are the spatial corroboration.
WEIGHTS = {"stability": 0.45, "transience": 0.45, "detail": 0.3, "solidity": 0.25}


def fuse(scores: dict[str, float]) -> float:
    """Weighted sum, normalised by the weights that are actually in play."""
    live = {k: w for k, w in WEIGHTS.items() if scores.get(k, 0.0) > 0.0}
    total = sum(live.values())
    if total <= 0:
        return 0.0
    return sum(live[k] * scores[k] for k in live) / total



# ---- presence over time ---------------------------------------------------


def patch_presence(frames: list[np.ndarray], reference_index: int, box: Box) -> list[float]:
    """Normalized correlation of each frame's ROI against a reference patch."""
    height, width = frames[0].shape[:2]
    x, y, w, h = geometry.to_pixels(box, width, height)
    reference = frames[reference_index][y:y + h, x:x + w]
    if reference.size == 0 or min(reference.shape[:2]) < 3:
        return [0.0] * len(frames)

    scores = []
    for frame in frames:
        patch = frame[y:y + h, x:x + w]
        if patch.shape != reference.shape:
            scores.append(0.0)
            continue
        result = cv2.matchTemplate(patch, reference, cv2.TM_CCOEFF_NORMED)
        scores.append(float(result[0][0]))
    return scores


def present_count(presence: list[float], threshold: float = PRESENCE_NCC) -> int:
    """How many sampled frames the patch is still recognisable in."""
    return sum(1 for score in presence if score >= threshold)


def presence_span(times: list[float], presence: list[float], *, reference_index: int,
                  threshold: float = PRESENCE_NCC) -> tuple[float, float]:
    """Longest run of "still present" containing the reference frame.

    Timing comes from the run, not the scene: a pop-up usually occupies only
    part of the shot it appears in, and masking the whole scene would blur
    footage that was never covered.
    """
    if not times:
        return 0.0, 0.0
    start = end = reference_index
    while start - 1 >= 0 and presence[start - 1] >= threshold:
        start -= 1
    while end + 1 < len(times) and presence[end + 1] >= threshold:
        end += 1
    return times[start], times[end]


def overlaps_text(box: Box, text_boxes: list[Box], threshold: float = TEXT_OVERLAP_IOU) -> bool:
    """A caption is already handled by the text pipeline; do not double-count."""
    return any(geometry.iou(box, other) >= threshold for other in text_boxes)


def containment(a: Box, b: Box) -> float:
    """Intersection over the *smaller* box's area.

    IoU alone cannot see that a small fragment sits inside a large detection --
    a sliver overlapping a card scores near zero against it while being entirely
    part of the same object. This ratio does.
    """
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = ix * iy
    smaller = min(geometry.area(a), geometry.area(b))
    return intersection / smaller if smaller > 0 else 0.0


def deduplicate(candidates: list[Candidate], iou_threshold: float = 0.4,
                containment_threshold: float = CONTAINMENT_THRESHOLD) -> list[Candidate]:
    """Greedy NMS over overlap *and* containment.

    The same asset is often found twice -- once cleanly, and once as a fragment
    picked up from a neighbouring scene or window. Suppressing on containment as
    well as IoU collapses those into the one detection that actually bounds it.
    """
    kept: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda c: geometry.area(c.box) * c.score,
                            reverse=True):
        duplicate = any(
            geometry.iou(candidate.box, other.box) >= iou_threshold
            or containment(candidate.box, other.box) >= containment_threshold
            for other in kept
        )
        if not duplicate:
            kept.append(candidate)
    return kept
