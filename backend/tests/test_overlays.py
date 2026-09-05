"""Pop-up detection: the signals, the filters, and the fusion.

All of it operates on arrays, so the whole detector is testable with synthetic
scenes built in a few lines -- which matters, because this is the most heuristic
stage in the pipeline and the one most likely to regress silently.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.pipeline import overlays
from app.pipeline.geometry import to_dict
from app.pipeline.overlays import Candidate

W, H = 120, 200
CARD = (0.5, 0.1, 0.35, 0.25)


def _textured(seed: int = 0) -> np.ndarray:
    """A background with enough structure to have edges and phase correlation."""
    rng = np.random.default_rng(seed)
    canvas = np.full((H + 80, W + 80), 60, np.uint8)
    for _ in range(40):
        centre = (int(rng.integers(0, canvas.shape[1])), int(rng.integers(0, canvas.shape[0])))
        cv2.circle(canvas, centre, int(rng.integers(6, 22)), int(rng.integers(40, 230)), -1)
    return canvas


def _pan(frames: int, step: int = 6, card_range: tuple[int, int] | None = None) -> list[np.ndarray]:
    """A panning shot, optionally with a pasted card for part of it."""
    canvas = _textured()
    out = []
    for i in range(frames):
        frame = canvas[i * step:i * step + H, i * step:i * step + W].copy()
        if card_range and card_range[0] <= i < card_range[1]:
            _paste_card(frame)
        out.append(frame)
    return out


def _static(frames: int, card_range: tuple[int, int] | None = None) -> list[np.ndarray]:
    base = _textured(seed=3)[:H, :W]
    out = []
    for i in range(frames):
        frame = base.copy()
        if card_range and card_range[0] <= i < card_range[1]:
            _paste_card(frame)
        out.append(frame)
    return out


def _paste_card(frame: np.ndarray) -> None:
    x, y = int(CARD[0] * W), int(CARD[1] * H)
    w, h = int(CARD[2] * W), int(CARD[3] * H)
    cv2.rectangle(frame, (x, y), (x + w, y + h), 245, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), 20, 2)
    cv2.circle(frame, (x + w // 2, y + h // 2), min(w, h) // 4, 90, -1)


# ------------------------------------------------------------ camera motion


def test_estimate_shift_recovers_a_known_translation():
    base = _textured(seed=1)[:H, :W]
    shifted = _textured(seed=1)[8:8 + H, 5:5 + W]
    dx, dy, response = overlays.estimate_shift(base, shifted)
    assert np.hypot(dx, dy) == pytest.approx(np.hypot(5, 8), abs=2.0)
    assert response > 0.1


def test_estimate_shift_on_mismatched_input_is_zero():
    assert overlays.estimate_shift(np.zeros((4, 4), np.uint8), np.zeros((5, 5), np.uint8)) == (
        0.0, 0.0, 0.0,
    )


def test_camera_motion_separates_a_pan_from_a_locked_off_shot():
    assert overlays.camera_motion(_pan(6)) > overlays.MOTION_THRESHOLD_PX
    assert overlays.camera_motion(_static(6)) < overlays.MOTION_THRESHOLD_PX
    assert overlays.camera_motion([]) == 0.0


# ---------------------------------------------------------------- signals


def test_a_pasted_card_is_still_while_a_pan_is_not():
    frames = _pan(6, card_range=(0, 6))
    std = overlays.stability_map(frames)
    x, y = int(CARD[0] * W) + 6, int(CARD[1] * H) + 6
    assert std[y:y + 20, x:x + 20].mean() < 2.0        # the card never moves
    assert np.median(std) > 10.0                        # the world does


def test_transition_count_separates_background_popup_and_motion():
    # Static camera: background never changes, the card arrives once.
    frames = _static(8, card_range=(4, 8))
    counts = overlays.transition_count(frames)
    x, y = int(CARD[0] * W) + 6, int(CARD[1] * H) + 6

    assert counts[y:y + 20, x:x + 20].mean() == pytest.approx(1.0, abs=0.3)
    corner = counts[:20, :20]
    assert corner.mean() < 0.2                          # untouched background


def test_transience_is_unreliable_under_a_pan():
    """Documents why the two signals are exclusive rather than ORed.

    Panning over broad flat regions makes background pixels change once or
    twice too, so transience cannot distinguish them from a pop-up.
    """
    frames = _pan(8, card_range=(4, 8))
    counts = overlays.transition_count(frames)
    background = counts[:30, :30]
    assert ((background >= 1) & (background <= overlays.MAX_TRANSITIONS)).mean() > 0.2


def test_structure_mask_excludes_a_flat_region():
    frame = np.full((H, W), 128, np.uint8)
    cv2.rectangle(frame, (10, 10), (40, 40), 250, -1)
    mask = overlays.structure_mask(frame)
    assert mask[20, 20] > 0 or mask[10, 10] > 0        # near the square's edges
    assert mask[H - 5, W - 5] == 0                      # flat corner: nothing there


def test_windows_cover_the_samples():
    assert overlays.windows(3) == [range(3)]
    assert overlays.windows(0) == []
    result = overlays.windows(9, size=4, stride=2)
    assert list(result[0]) == [0, 1, 2, 3]
    assert result[-1][-1] == 8


# ---------------------------------------------------------- candidate boxes


def test_boxes_from_mask_reports_solidity_and_filters_area():
    mask = np.zeros((H, W), np.uint8)
    cv2.rectangle(mask, (20, 20), (70, 90), 255, -1)   # solid, big enough
    cv2.rectangle(mask, (2, 2), (4, 4), 255, -1)       # too small

    found = overlays.boxes_from_mask(mask)
    assert len(found) == 1
    box, solidity = found[0]
    assert solidity > 0.9
    assert box[0] == pytest.approx(20 / W, abs=0.02)


def test_boxes_from_mask_rejects_a_stringy_blob():
    mask = np.zeros((H, W), np.uint8)
    cv2.line(mask, (10, 10), (100, 180), 255, 2)       # spans a big bbox, fills none
    assert overlays.boxes_from_mask(mask) == []


def test_boxes_from_mask_rejects_a_near_full_frame_component():
    mask = np.full((H, W), 255, np.uint8)
    assert overlays.boxes_from_mask(mask) == []


# ---------------------------------------------------------------- scoring


def test_edge_density_rejects_flat_and_keeps_structured():
    frame = np.full((H, W), 120, np.uint8)
    _paste_card(frame)
    edges = overlays.edge_map(frame)
    assert overlays.edge_density(edges, CARD) > overlays.MIN_EDGE_DENSITY
    assert overlays.edge_density(edges, (0.02, 0.6, 0.3, 0.3)) < overlays.MIN_EDGE_DENSITY


def test_stability_score_is_high_for_the_card_and_low_elsewhere():
    std = overlays.stability_map(_pan(6, card_range=(0, 6)))
    assert overlays.stability_score(std, CARD) > 0.5
    assert overlays.stability_score(std, (0.02, 0.6, 0.3, 0.3)) < 0.5


def test_stability_score_of_a_perfectly_still_frame_proves_nothing():
    std = np.zeros((H, W), np.float32)
    assert overlays.stability_score(std, CARD) == 0.0


def test_fuse_normalises_over_the_live_signals_only():
    """Stability and transience are alternatives; whichever fires must be able
    to reach the threshold on its own rather than being halved by the other."""
    moving = overlays.fuse({"stability": 0.8, "transience": 0.0, "detail": 0.8, "solidity": 0.8})
    static = overlays.fuse({"stability": 0.0, "transience": 0.8, "detail": 0.8, "solidity": 0.8})
    assert moving == pytest.approx(0.8)
    assert static == pytest.approx(moving)


def test_fuse_of_nothing_is_zero():
    assert overlays.fuse({}) == 0.0


# ------------------------------------------------------------- persistence


def test_patch_presence_tracks_a_card_appearing_and_leaving():
    frames = _static(9, card_range=(3, 7))
    presence = overlays.patch_presence(frames, reference_index=5, box=CARD)

    assert overlays.present_count(presence) == 4
    assert all(p >= overlays.PRESENCE_NCC for p in presence[3:7])


def test_presence_span_returns_the_run_containing_the_reference():
    times = [i * 0.33 for i in range(9)]
    presence = [0.0, 0.0, 0.0, 0.9, 0.95, 0.99, 0.92, 0.0, 0.0]
    span = overlays.presence_span(times, presence, reference_index=5)
    assert span == (pytest.approx(times[3]), pytest.approx(times[6]))


def test_presence_span_of_a_lone_frame_is_a_point():
    times = [0.0, 0.33, 0.66]
    assert overlays.presence_span(times, [0.0, 0.9, 0.0], reference_index=1) == (0.33, 0.33)


def test_present_count_is_the_persistence_filter():
    assert overlays.present_count([0.9, 0.1, 0.1]) < overlays.MIN_PRESENT_SAMPLES
    assert overlays.present_count([0.9, 0.9, 0.1]) >= overlays.MIN_PRESENT_SAMPLES


# ------------------------------------------------------------ deduplication


def test_overlaps_text_prevents_double_counting_a_caption():
    caption = (0.1, 0.8, 0.6, 0.06)
    assert overlays.overlaps_text((0.1, 0.8, 0.6, 0.06), [caption]) is True
    assert overlays.overlaps_text((0.5, 0.1, 0.3, 0.2), [caption]) is False
    assert overlays.overlaps_text((0.5, 0.1, 0.3, 0.2), []) is False


def test_deduplicate_keeps_the_strongest_of_an_overlapping_cluster():
    weak = Candidate(box=(0.5, 0.1, 0.35, 0.25), scene_id="sc_001", t_in=1.0, t_out=2.0,
                     scores={"stability": 0.6, "detail": 0.6, "solidity": 0.6})
    strong = Candidate(box=(0.52, 0.11, 0.35, 0.25), scene_id="sc_001", t_in=1.0, t_out=2.0,
                       scores={"stability": 0.95, "detail": 0.95, "solidity": 0.95})
    apart = Candidate(box=(0.02, 0.7, 0.2, 0.2), scene_id="sc_001", t_in=1.0, t_out=2.0,
                      scores={"stability": 0.7, "detail": 0.7, "solidity": 0.7})

    kept = overlays.deduplicate([weak, strong, apart])
    assert strong in kept and weak not in kept and apart in kept


def test_masks_consume_overlay_tracks_the_same_as_text():
    """The mask stage is generic over element kinds; prove it, since the
    overlay pipeline gets its removal for free from that."""
    from app.pipeline.stages.masks import build_segments

    segments = build_segments(
        [{"id": "ov_001", "t_in": 1.0, "t_out": 2.0, "box": to_dict(CARD), "kind": "overlay"}],
        aspect=0.5625, duration=4.0, pad_s=0.0,
    )
    assert segments[0]["boxes"][0]["source_id"] == "ov_001"
    assert segments[0]["boxes"][0]["kind"] == "overlay"
