"""Media-layer tests: pure helpers first, then a real ffmpeg/PySceneDetect pass."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.pipeline.stages.sample import assign_scene
from app.pipeline.stages.scenes import detect_spans, merge_short_spans, uniform_spans
from app.utils import ffmpeg

# --------------------------------------------------------------- pure helpers


@pytest.mark.parametrize(
    ("value", "expected"),
    [("30000/1001", 29.97), ("30/1", 30.0), ("25", 25.0), ("0/0", 0.0), (None, 0.0), ("x", 0.0)],
)
def test_parse_rate(value, expected):
    assert ffmpeg._parse_rate(value) == pytest.approx(expected, abs=0.01)


def test_frame_timestamp_is_the_sample_centre():
    assert ffmpeg.frame_timestamp(1, 3.0) == pytest.approx(1 / 6)
    assert ffmpeg.frame_timestamp(4, 2.0) == pytest.approx(1.75)


def test_merge_short_spans_folds_flash_frames():
    spans = [(0.0, 1.0), (1.0, 1.1), (1.1, 3.0)]
    assert merge_short_spans(spans, 0.35) == [(0.0, 1.1), (1.1, 3.0)]


def test_merge_short_spans_folds_a_short_tail_backwards():
    assert merge_short_spans([(0.0, 2.0), (2.0, 2.1)], 0.35) == [(0.0, 2.1)]


def test_merge_short_spans_leaves_good_input_alone():
    spans = [(0.0, 1.0), (1.0, 2.0)]
    assert merge_short_spans(spans, 0.35) == spans


def test_merge_short_spans_handles_empty():
    assert merge_short_spans([], 0.35) == []


def test_uniform_spans_tile_the_timeline():
    spans = uniform_spans(10.0, 3.0)
    assert spans[0][0] == 0.0
    assert spans[-1][1] == 10.0
    for previous, current in zip(spans, spans[1:], strict=False):
        assert previous[1] == current[0]


def test_uniform_spans_absorbs_a_trailing_sliver():
    # 6.2s at 3s chunks would leave a 0.2s tail; it must be merged, not kept.
    assert uniform_spans(6.2, 3.0)[-1] == (3.0, 6.2)


def test_assign_scene_uses_the_containing_span():
    starts, ids = [0.0, 2.0, 4.0], ["sc_001", "sc_002", "sc_003"]
    assert assign_scene(0.0, starts, ids) == "sc_001"
    assert assign_scene(1.9, starts, ids) == "sc_001"
    assert assign_scene(2.0, starts, ids) == "sc_002"
    assert assign_scene(99.0, starts, ids) == "sc_003"
    assert assign_scene(1.0, [], []) is None


# ------------------------------------------------------------- real ffmpeg


def test_probe_media_reads_the_fixture(fixture_video: Path):
    info = ffmpeg.probe_media(fixture_video)
    assert (info.width, info.height) == (320, 568)
    assert info.duration == pytest.approx(6.0, abs=0.3)
    assert info.fps == pytest.approx(15.0, abs=0.1)
    assert info.has_audio is True
    assert info.codec == "h264"


def test_probe_media_rejects_a_non_video(tmp_path: Path):
    from app.errors import NoVideoStream

    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"\x00" * 512)
    with pytest.raises(NoVideoStream):
        ffmpeg.probe_media(junk)


def test_probe_media_rejects_an_empty_file(tmp_path: Path):
    from app.errors import NoVideoStream

    empty = tmp_path / "empty.mp4"
    empty.touch()
    with pytest.raises(NoVideoStream):
        ffmpeg.probe_media(empty)


def test_make_proxy_downscales_and_keeps_aspect(fixture_video: Path, tmp_path: Path):
    proxy = tmp_path / "proxy.mp4"
    ffmpeg.make_proxy(fixture_video, proxy, max_dim=240)
    info = ffmpeg.probe_media(proxy)
    assert max(info.width, info.height) <= 240
    assert info.aspect == pytest.approx(320 / 568, abs=0.02)
    assert info.has_audio is False  # analysis copy drops audio


def test_cut_is_frame_accurate(fixture_video: Path, tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    ffmpeg.cut(fixture_video, clip, 2.0, 3.5)
    assert ffmpeg.probe_media(clip).duration == pytest.approx(1.5, abs=0.2)


def test_extract_audio_produces_16k_mono(fixture_video: Path, tmp_path: Path):
    wav = tmp_path / "audio.wav"
    ffmpeg.extract_audio(fixture_video, wav)
    streams = ffmpeg.probe(wav)["streams"]
    assert streams[0]["sample_rate"] == "16000"
    assert streams[0]["channels"] == 1


def test_extract_frames_samples_at_the_requested_rate(fixture_video: Path, tmp_path: Path):
    files = ffmpeg.extract_frames(fixture_video, tmp_path / "frames", fps=2.0, width=120)
    assert 10 <= len(files) <= 14  # ~6 s at 2 fps
    assert all(f.stat().st_size > 0 for f in files)


def test_ffmpeg_error_carries_the_stderr_tail(tmp_path: Path):
    with pytest.raises(ffmpeg.FFmpegError) as excinfo:
        ffmpeg.ffmpeg(["-i", str(tmp_path / "nope.mp4"), str(tmp_path / "out.mp4")])
    assert excinfo.value.code == "E_FFMPEG"
    assert excinfo.value.detail


def test_detect_spans_finds_the_fixture_cuts(fixture_video: Path):
    spans = detect_spans(fixture_video)
    assert spans, "detector returned nothing"
    assert spans[0][0] == 0.0
    for previous, current in zip(spans, spans[1:], strict=False):
        assert previous[1] == current[0]
