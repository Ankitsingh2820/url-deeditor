"""Geometry, tracking, style and OCR normalisation.

The tracking logic is the subtlest code in the pipeline and it is pure, so it
gets tested with plain dicts -- no video, no models, milliseconds per case.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.pipeline import geometry, tracking
from app.pipeline.stages.ocr import (
    frame_signature,
    is_static,
    normalise_result,
)
from app.pipeline.tracking import Detection
from app.utils.style import extract_style, luminance, to_hex

# ------------------------------------------------------------------ geometry


def test_quad_to_box_normalises_and_bounds():
    quad = [[10, 20], [110, 22], [110, 60], [10, 58]]
    box = geometry.quad_to_box(quad, 200, 100)
    assert box == pytest.approx((0.05, 0.2, 0.5, 0.4), abs=1e-6)


def test_quad_to_box_clamps_out_of_frame_points():
    box = geometry.quad_to_box([[-50, -10], [250, -10], [250, 150], [-50, 150]], 200, 100)
    assert box == (0.0, 0.0, 1.0, 1.0)


def test_iou_matches_hand_computed_value():
    a = (0.0, 0.0, 0.5, 0.5)
    b = (0.25, 0.25, 0.5, 0.5)
    # intersection 0.0625, union 0.4375
    assert geometry.iou(a, b) == pytest.approx(0.0625 / 0.4375)


def test_iou_is_zero_when_disjoint():
    assert geometry.iou((0, 0, 0.1, 0.1), (0.5, 0.5, 0.1, 0.1)) == 0.0


def test_median_box_ignores_an_outlier_frame():
    boxes = [(0.1, 0.8, 0.5, 0.05), (0.1, 0.8, 0.5, 0.05), (0.9, 0.1, 0.9, 0.9)]
    assert geometry.median_box(boxes) == pytest.approx((0.1, 0.8, 0.5, 0.05))


def test_dilate_clamps_at_the_frame_edge():
    assert geometry.dilate((0.0, 0.0, 0.2, 0.2), 0.05) == pytest.approx((0.0, 0.0, 0.25, 0.25))


def test_to_pixels_never_returns_an_empty_rect():
    assert geometry.to_pixels((0.999, 0.999, 0.0005, 0.0005), 100, 100) == (99, 99, 1, 1)


@pytest.mark.parametrize(
    ("box", "expected"),
    [((0.0, 0, 0.2, 0.1), "left"), ((0.4, 0, 0.2, 0.1), "center"), ((0.8, 0, 0.2, 0.1), "right")],
)
def test_horizontal_align(box, expected):
    assert geometry.horizontal_align(box) == expected


# ------------------------------------------------------------------ tracking


def _det(t: float, box, text: str, conf: float = 0.9) -> Detection:
    return Detection(t=t, box=box, text=text, conf=conf, scene_id="sc_001", frame_path="f.jpg")


def test_same_text_in_place_becomes_one_track():
    box = (0.1, 0.8, 0.6, 0.06)
    dets = [_det(t, box, "this changed my skin") for t in (0.17, 0.5, 0.83)]
    tracks = tracking.track_detections(dets, sample_interval=1 / 3)
    assert len(tracks) == 1
    assert tracks[0].text == "this changed my skin"


def test_text_changing_in_place_splits_into_two_tracks():
    # Same position, different words: a caption cutting to the next line.
    box = (0.1, 0.8, 0.6, 0.06)
    dets = [_det(0.17, box, "before"), _det(0.5, box, "completely different words")]
    assert len(tracking.track_detections(dets, sample_interval=1 / 3)) == 2


def test_same_text_in_two_places_stays_separate():
    dets = [_det(0.17, (0.05, 0.1, 0.2, 0.05), "SALE"),
            _det(0.17, (0.75, 0.8, 0.2, 0.05), "SALE")]
    assert len(tracking.track_detections(dets, sample_interval=1 / 3)) == 2


def test_a_one_frame_gap_is_bridged():
    box = (0.1, 0.8, 0.6, 0.06)
    dets = [_det(0.17, box, "flicker"), _det(0.83, box, "flicker")]  # 0.5 missing
    assert len(tracking.track_detections(dets, sample_interval=1 / 3)) == 1


def test_a_long_gap_starts_a_new_track():
    box = (0.1, 0.8, 0.6, 0.06)
    dets = [_det(0.17, box, "again"), _det(5.0, box, "again")]
    assert len(tracking.track_detections(dets, sample_interval=1 / 3)) == 2


def test_span_extends_half_a_sample_either_side():
    track = tracking.Track([_det(1.0, (0.1, 0.8, 0.5, 0.05), "x")])
    assert track.span(1 / 3) == (pytest.approx(0.833, abs=1e-3), pytest.approx(1.167, abs=1e-3))


def test_modal_text_wins_over_a_bad_read():
    box = (0.1, 0.8, 0.6, 0.06)
    dets = [_det(0.17, box, "buy now"), _det(0.5, box, "buy n0w"), _det(0.83, box, "buy now")]
    assert tracking.track_detections(dets, sample_interval=1 / 3)[0].text == "buy now"


def test_prune_drops_a_lone_low_confidence_detection():
    weak = tracking.Track([_det(0.17, (0.1, 0.8, 0.5, 0.05), "?", conf=0.55)])
    strong = tracking.Track([_det(0.17, (0.1, 0.8, 0.5, 0.05), "SALE", conf=0.95)])
    assert tracking.prune([weak, strong], 1 / 3) == [strong]


def test_prune_drops_empty_text():
    empty = tracking.Track([_det(0.17, (0.1, 0.8, 0.5, 0.05), "   ", conf=0.99)])
    assert tracking.prune([empty], 1 / 3) == []


def test_similarity_is_case_and_space_insensitive():
    assert tracking.similarity(" Buy Now ", "buy now") == 100.0


@pytest.mark.parametrize(
    ("box", "duration", "expected"),
    [
        ((0.15, 0.80, 0.70, 0.06), 1.8, "caption"),       # lower third, centred, brief
        ((0.20, 0.10, 0.60, 0.09), 2.5, "overlay_text"),  # top hook line
        ((0.02, 0.02, 0.18, 0.025), 5.9, "watermark"),    # corner, whole video
    ],
)
def test_classify(box, duration, expected):
    assert tracking.classify(box, duration, 6.0, "some text") == expected


# --------------------------------------------------------------------- style


def _text_crop(bg=(20, 20, 20), fg=(255, 255, 255), stroke=None) -> np.ndarray:
    """Render a text crop with an optional outline.

    The outline is built from the glyph mask by morphology rather than by a
    second, thicker putText: OpenCV's text rasterisation differs between
    versions, and on OpenCV 5 the thin pass paints over the thick one, silently
    producing a crop with no outline at all.
    """
    glyph = np.zeros((60, 240), np.uint8)
    cv2.putText(glyph, "SAMPLE", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 2, cv2.LINE_AA)
    mask = glyph > 127

    crop = np.full((60, 240, 3), bg, np.uint8)
    if stroke:
        band = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~mask
        crop[band] = stroke
    crop[mask] = fg
    return crop


def test_extract_style_reads_white_on_dark():
    style = extract_style(_text_crop())
    assert style.color == "#FFFFFF"
    assert style.contrast > 100
    assert style.confidence > 0.7


def test_extract_style_reads_dark_on_white():
    style = extract_style(_text_crop(bg=(245, 245, 245), fg=(10, 10, 10)))
    assert luminance((10, 10, 10)) < luminance((245, 245, 245))
    assert style.color == "#0A0A0A"


def test_extract_style_detects_a_flat_caption_plate():
    assert extract_style(_text_crop()).has_bg_box is True


def test_extract_style_detects_an_outline():
    style = extract_style(_text_crop(bg=(120, 120, 120), fg=(255, 255, 255), stroke=(0, 0, 0)))
    assert style.stroke is not None


def test_extract_style_survives_a_degenerate_crop():
    assert extract_style(np.zeros((0, 0, 3), np.uint8)).confidence == 0.0
    assert extract_style(np.full((2, 2, 3), 128, np.uint8)).color == "#FFFFFF"


def test_to_hex_is_bgr_to_rgb():
    assert to_hex((0, 128, 255)) == "#FF8000"


# ----------------------------------------------------------- ocr normalisation


def test_normalise_result_filters_and_normalises():
    raw = [
        [[[10, 20], [110, 20], [110, 60], [10, 60]], "KEEP ME", "0.91"],
        [[[10, 20], [110, 20], [110, 60], [10, 60]], "low conf", "0.31"],
        [[[0, 0], [3, 0], [3, 3], [0, 3]], "tiny", "0.99"],
        [[[10, 20], [110, 20], [110, 60], [10, 60]], "   ", "0.99"],
    ]
    out = normalise_result(raw, 200, 400)
    assert [d["text"] for d in out] == ["KEEP ME"]
    assert out[0]["box"] == {"x": 0.05, "y": 0.05, "w": 0.5, "h": 0.1}


def test_normalise_result_ignores_malformed_entries():
    assert normalise_result([["not-a-quad"], None], 100, 100) == []
    assert normalise_result(None, 100, 100) == []


def test_static_frame_detection():
    a = np.full((80, 80, 3), 100, np.uint8)
    b = a.copy()
    c = a.copy()
    cv2.rectangle(c, (5, 5), (70, 70), (255, 255, 255), -1)
    assert is_static(frame_signature(a), frame_signature(b)) is True
    assert is_static(frame_signature(a), frame_signature(c)) is False
    assert is_static(None, frame_signature(a)) is False
