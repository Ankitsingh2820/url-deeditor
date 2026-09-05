"""Turn per-frame OCR noise into temporal text tracks.

This is the step that makes the result *editable*. Raw OCR gives a bag of boxes
per sampled frame; an editor needs objects with a start, an end, one canonical
string and one stable box. Two detections belong to the same track when they
overlap spatially **and** read the same -- either test alone is wrong:

  * box only  -> a caption that changes text in place becomes one long track
  * text only -> the same word in two corners merges into nonsense

Everything here is pure: dicts in, dicts out, no video, no models. That is what
makes the trickiest logic in the pipeline cheap to test.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from statistics import mean

from rapidfuzz import fuzz

from app.pipeline import geometry
from app.pipeline.geometry import Box

#: spatial overlap required to continue a track
IOU_THRESHOLD = 0.5
#: fuzzy string similarity (0-100) required to continue a track
TEXT_THRESHOLD = 80.0
#: how long a track may vanish before it counts as a new one. One *missing*
#: sample puts two intervals between neighbouring detections, so 2.0 is the
#: value that bridges a single dropped frame and nothing more.
MAX_GAP_SAMPLES = 2.0
#: tracks shorter than this are OCR flicker, not content
MIN_TRACK_S = 0.25
#: a lone detection needs to be confident to survive
MIN_SOLO_CONFIDENCE = 0.7


@dataclass
class Detection:
    t: float
    box: Box
    text: str
    conf: float
    scene_id: str | None = None
    frame_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Detection:
        return cls(
            t=float(data["t"]),
            box=geometry.from_dict(data["box"]),
            text=str(data["text"]),
            conf=float(data.get("conf", 0.0)),
            scene_id=data.get("scene_id"),
            frame_path=data.get("frame_path"),
        )


@dataclass
class Track:
    detections: list[Detection] = field(default_factory=list)

    @property
    def last(self) -> Detection:
        return self.detections[-1]

    @property
    def box(self) -> Box:
        return geometry.median_box([d.box for d in self.detections])

    @property
    def text(self) -> str:
        """Modal string: the reading OCR agreed on most often."""
        counts = Counter(d.text.strip() for d in self.detections if d.text.strip())
        if not counts:
            return ""
        best = counts.most_common(1)[0][1]
        # Tie-break on length -- truncated reads are the common OCR failure.
        return max((t for t, c in counts.items() if c == best), key=len)

    @property
    def confidence(self) -> float:
        return round(mean(d.conf for d in self.detections), 4)

    @property
    def scene_id(self) -> str | None:
        scenes = [d.scene_id for d in self.detections if d.scene_id]
        return Counter(scenes).most_common(1)[0][0] if scenes else None

    def span(self, sample_interval: float) -> tuple[float, float]:
        """On/off times.

        A detection at t means the text was present *around* t, so the true
        boundaries lie within half a sample either side. Extending by half an
        interval keeps a single-sample detection from having zero duration.
        """
        half = sample_interval / 2
        return (
            round(max(0.0, self.detections[0].t - half), 3),
            round(self.detections[-1].t + half, 3),
        )


def similarity(a: str, b: str) -> float:
    """Case/space-insensitive fuzzy ratio, 0-100."""
    return fuzz.ratio(a.strip().lower(), b.strip().lower())


def match_score(track: Track, detection: Detection) -> float | None:
    """Combined spatial+textual score, or None when the pair cannot match."""
    overlap = geometry.iou(track.last.box, detection.box)
    if overlap < IOU_THRESHOLD:
        return None
    ratio = similarity(track.last.text, detection.text)
    if ratio < TEXT_THRESHOLD:
        return None
    return overlap + ratio / 100.0


def track_detections(
    detections: list[Detection],
    *,
    sample_interval: float,
    max_gap_samples: float = MAX_GAP_SAMPLES,
) -> list[Track]:
    """Greedy nearest-match tracking over time-ordered detections.

    Greedy is the right complexity here: within one sampled frame text elements
    rarely overlap, so the assignment problem is trivial and a Hungarian solve
    would buy nothing.
    """
    max_gap = sample_interval * max_gap_samples
    ordered = sorted(detections, key=lambda d: (d.t, -d.conf))
    tracks: list[Track] = []
    open_tracks: list[Track] = []

    for detection in ordered:
        # Close tracks that have not been seen for too long.
        open_tracks = [t for t in open_tracks if detection.t - t.last.t <= max_gap]

        best, best_score = None, 0.0
        for track in open_tracks:
            if track.last.t == detection.t:
                continue  # one detection per track per frame
            score = match_score(track, detection)
            if score is not None and score > best_score:
                best, best_score = track, score

        if best is None:
            new = Track([detection])
            tracks.append(new)
            open_tracks.append(new)
        else:
            best.detections.append(detection)

    return tracks


def prune(tracks: list[Track], sample_interval: float) -> list[Track]:
    """Drop what is almost certainly OCR noise rather than on-screen text."""
    kept = []
    for track in tracks:
        t_in, t_out = track.span(sample_interval)
        if (t_out - t_in) < MIN_TRACK_S:
            continue
        if len(track.detections) == 1 and track.confidence < MIN_SOLO_CONFIDENCE:
            continue
        if not track.text:
            continue
        kept.append(track)
    return kept


# ------------------------------------------------------------- classification


def classify(box: Box, duration: float, media_duration: float, text: str) -> str:
    """Provisional kind for a text track.

    Position and lifetime only -- this is a heuristic, and a deliberately
    shallow one. The real separation between a burned-in *caption* (a subtitle
    of what is being said) and an *overlay_text* (a hook line or CTA that is
    never spoken) comes from aligning the text against the ASR transcript in the
    speech stage, which refines whatever we decide here.
    """
    cx, cy = geometry.center(box)
    centered = 0.2 < cx < 0.8
    lower_third = cy > 0.62
    persistent = media_duration > 0 and duration >= 0.75 * media_duration

    if persistent and (not centered or box[3] < 0.04):
        return "watermark"
    if lower_third and centered and duration <= 4.0 and len(text) <= 90:
        return "caption"
    return "overlay_text"
