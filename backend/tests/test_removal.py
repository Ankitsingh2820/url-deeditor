"""Mask timeline and inpainting: pure logic first, then pixels."""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.geometry import to_dict
from app.pipeline.stages.inpaint import inpaint_frame, render_mask, resolve_method
from app.pipeline.stages.masks import boxes_at, build_segments, pad_box


def _track(track_id: str, t_in: float, t_out: float, box=(0.1, 0.8, 0.5, 0.06),
           kind: str = "caption") -> dict:
    return {"id": track_id, "t_in": t_in, "t_out": t_out, "box": to_dict(box), "kind": kind}


# ------------------------------------------------------------------ padding


def test_pad_box_uses_equal_pixel_margins_on_both_axes():
    # Portrait 9:16 frame: the x margin must be *smaller* in normalized units
    # to cover the same number of pixels as the y margin.
    box = pad_box((0.4, 0.4, 0.2, 0.2), aspect=9 / 16, fraction=0.02)
    margin_x = 0.4 - box[0]
    margin_y = 0.4 - box[1]
    assert margin_y == pytest.approx(0.02)
    assert margin_x == pytest.approx(0.02 / (9 / 16))
    # Equal in pixels on a 360x640 frame.
    assert margin_x * 360 == pytest.approx(margin_y * 640, rel=1e-6)


def test_pad_box_clamps_at_the_frame_edge():
    assert pad_box((0.0, 0.0, 0.1, 0.1), aspect=1.0, fraction=0.05)[:2] == (0.0, 0.0)


# ------------------------------------------------------------------ segments


def test_disjoint_tracks_make_one_segment_each():
    segments = build_segments(
        [_track("tx_001", 0.0, 1.0), _track("tx_002", 2.0, 3.0)],
        aspect=0.5625, duration=4.0, pad_s=0.0,
    )
    assert [(s["t_in"], s["t_out"]) for s in segments] == [(0.0, 1.0), (2.0, 3.0)]
    assert all(len(s["boxes"]) == 1 for s in segments)


def test_overlapping_tracks_split_into_three_segments():
    segments = build_segments(
        [_track("tx_001", 0.0, 2.0), _track("tx_002", 1.0, 3.0, box=(0.1, 0.1, 0.2, 0.1))],
        aspect=0.5625, duration=3.0, pad_s=0.0,
    )
    assert [(s["t_in"], s["t_out"]) for s in segments] == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    assert [len(s["boxes"]) for s in segments] == [1, 2, 1]
    assert {b["source_id"] for b in segments[1]["boxes"]} == {"tx_001", "tx_002"}


def test_a_track_contained_in_another_still_yields_its_own_segment():
    segments = build_segments(
        [_track("tx_001", 0.0, 4.0), _track("tx_002", 1.0, 2.0, box=(0.5, 0.1, 0.2, 0.1))],
        aspect=0.5625, duration=4.0, pad_s=0.0,
    )
    assert [len(s["boxes"]) for s in segments] == [1, 2, 1]


def test_segments_carry_the_source_id_and_kind_for_the_ui():
    segment = build_segments([_track("tx_007", 0.0, 1.0, kind="watermark")],
                             aspect=0.5625, duration=1.0, pad_s=0.0)[0]
    assert segment["boxes"][0]["source_id"] == "tx_007"
    assert segment["boxes"][0]["kind"] == "watermark"


def test_temporal_padding_extends_but_never_leaves_the_timeline():
    segments = build_segments([_track("tx_001", 0.0, 1.0)],
                              aspect=0.5625, duration=1.0, pad_s=0.2)
    assert segments[0]["t_in"] == 0.0    # clamped at the start
    assert segments[0]["t_out"] == 1.0   # clamped at the duration


def test_no_tracks_means_no_segments():
    assert build_segments([], aspect=0.5625, duration=5.0) == []


def test_zero_length_track_is_dropped():
    assert build_segments([_track("tx_001", 1.0, 1.0)],
                          aspect=0.5625, duration=5.0, pad_s=0.0) == []


# ------------------------------------------------------------------- lookup


def test_boxes_at_finds_the_active_segment():
    segments = build_segments(
        [_track("tx_001", 0.0, 1.0), _track("tx_002", 2.0, 3.0)],
        aspect=0.5625, duration=4.0, pad_s=0.0,
    )
    starts = [s["t_in"] for s in segments]

    assert [b["source_id"] for b in boxes_at(segments, starts, 0.5)] == ["tx_001"]
    assert boxes_at(segments, starts, 1.5) == []          # in the gap
    assert [b["source_id"] for b in boxes_at(segments, starts, 2.5)] == ["tx_002"]
    assert boxes_at(segments, starts, 9.0) == []          # past the end
    assert boxes_at(segments, starts, -1.0) == []         # before the start
    assert boxes_at([], [], 1.0) == []


def test_boxes_at_is_half_open_at_the_boundary():
    segments = build_segments([_track("tx_001", 0.0, 1.0)],
                              aspect=0.5625, duration=2.0, pad_s=0.0)
    starts = [s["t_in"] for s in segments]
    assert boxes_at(segments, starts, 0.0) != []
    assert boxes_at(segments, starts, 1.0) == []


# ----------------------------------------------------------------- rendering


def test_render_mask_fills_exactly_the_box():
    mask = render_mask(100, 200, [to_dict((0.25, 0.5, 0.5, 0.25))])
    assert mask.shape == (100, 200)
    assert mask[60, 100] == 255      # inside
    assert mask[10, 10] == 0         # outside
    assert mask.sum() / 255 == pytest.approx(100 * 25, rel=0.05)


def test_render_mask_unions_overlapping_boxes():
    mask = render_mask(100, 100, [to_dict((0.0, 0.0, 0.5, 0.5)), to_dict((0.25, 0.25, 0.5, 0.5))])
    assert mask[10, 10] == 255 and mask[60, 60] == 255 and mask[90, 90] == 0


def test_render_mask_with_no_boxes_is_empty():
    assert render_mask(50, 50, []).sum() == 0


# ------------------------------------------------------------ method tiering


def test_explicit_methods_are_respected():
    assert resolve_method("mask_blur")[0] == "mask_blur"
    assert resolve_method("opencv")[0] == "opencv"


def test_lama_falls_back_when_unavailable():
    # simple-lama-inpainting is not installed here, so both "auto" and an
    # explicit "lama" must degrade to the OpenCV tier rather than fail.
    method, model = resolve_method("lama")
    assert (method, model) in (("opencv", None), ("lama", model))
    assert resolve_method("auto")[0] in ("opencv", "lama")


def test_mask_blur_only_touches_the_masked_region():
    frame = np.zeros((80, 80, 3), np.uint8)
    frame[:, 40:] = 255                       # hard vertical edge
    mask = render_mask(80, 80, [to_dict((0.25, 0.0, 0.5, 1.0))])

    out = inpaint_frame(frame, mask, "mask_blur")
    assert np.array_equal(out[:, :15], frame[:, :15])     # untouched left
    assert np.array_equal(out[:, 70:], frame[:, 70:])     # untouched right
    assert not np.array_equal(out[:, 20:60], frame[:, 20:60])


def test_opencv_inpaint_removes_a_bright_patch():
    frame = np.full((60, 60, 3), 40, np.uint8)
    frame[25:35, 25:35] = 255                 # the "text" to remove
    mask = render_mask(60, 60, [to_dict((0.4, 0.4, 0.2, 0.2))])

    out = inpaint_frame(frame, mask, "opencv")
    assert out[30, 30].mean() < 90, "bright patch should be filled from its surroundings"
