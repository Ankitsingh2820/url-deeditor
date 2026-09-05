"""API + end-to-end pipeline tests, driven through real (tiny) video files."""

from __future__ import annotations

import pytest

from app.pipeline.tracking import similarity
from tests.conftest import wait_for_terminal


def _normalise(text: str) -> str:
    """OCR gets spacing wrong on synthetic fonts; compare on content, not layout."""
    return "".join(text.split()).upper()


def _run(client, payload: bytes) -> tuple[str, dict]:
    """Run (or reuse, via the content-hash cache) a job; return its id + project.json."""
    job = client.post("/api/v1/jobs", files={"file": ("clip.mp4", payload, "video/mp4")}).json()
    wait_for_terminal(client, job["id"])
    return job["id"], client.get(f"/api/v1/jobs/{job['id']}/project").json()


def _project_for(client, payload: bytes) -> dict:
    return _run(client, payload)[1]


def _find(tracks: list[dict], expected: str, threshold: float = 75.0) -> dict:
    match = max(tracks, key=lambda t: similarity(_normalise(t["text"]), expected), default=None)
    assert match and similarity(_normalise(match["text"]), expected) >= threshold, (
        f"no track resembling {expected!r} in {[t['text'] for t in tracks]}"
    )
    return match


def test_healthz(client):
    assert client.get("/healthz").json()["status"] == "ok"


def test_readyz_reports_checks(client):
    body = client.get("/readyz").json()
    assert set(body["checks"]) == {"database", "storage", "ffmpeg", "broker"}


def test_upload_runs_the_pipeline(client, fixture_bytes):
    r = client.post("/api/v1/jobs", files={"file": ("clip.mp4", fixture_bytes, "video/mp4")})
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["status"] == "queued"

    final = wait_for_terminal(client, job["id"])
    assert final["status"] == "succeeded", final
    assert final["progress"] == 1.0
    for stage in ("ingest", "probe", "scenes", "sample", "finalize"):
        assert stage in final["stage_timings_ms"], final["stage_timings_ms"]


def test_project_json_describes_the_video(client, fixture_bytes):
    job = client.post(
        "/api/v1/jobs", files={"file": ("clip.mp4", fixture_bytes, "video/mp4")}
    ).json()
    wait_for_terminal(client, job["id"])
    project = client.get(f"/api/v1/jobs/{job['id']}/project").json()

    assert project["schema_version"] == "1.0"
    media = project["media"]
    assert media["width"] == 320 and media["height"] == 568
    assert 5.5 < media["duration"] < 6.5
    assert media["has_audio"] is True

    # Three distinct 2 s segments: detection (or the uniform fallback) must find
    # more than one scene, and the spans must tile the timeline without gaps.
    scenes = project["scenes"]
    assert len(scenes) >= 2
    assert scenes[0]["t_in"] == 0.0
    for previous, current in zip(scenes, scenes[1:], strict=False):
        assert current["t_in"] == previous["t_out"]
    assert abs(scenes[-1]["t_out"] - media["duration"]) < 0.5


def test_scene_artifacts_are_served(client, fixture_bytes):
    job = client.post(
        "/api/v1/jobs", files={"file": ("clip.mp4", fixture_bytes, "video/mp4")}
    ).json()
    wait_for_terminal(client, job["id"])
    scene = client.get(f"/api/v1/jobs/{job['id']}/project").json()["scenes"][0]

    for key in ("clip", "keyframe"):
        assert scene[key], f"{key} was not exported"
        r = client.get(f"/api/v1/jobs/{job['id']}/files/{scene[key]}")
        assert r.status_code == 200
        assert len(r.content) > 0


def test_identical_upload_hits_the_result_cache(client, fixture_bytes):
    first = client.post(
        "/api/v1/jobs", files={"file": ("a.mp4", fixture_bytes, "video/mp4")}
    ).json()
    wait_for_terminal(client, first["id"])
    second = client.post(
        "/api/v1/jobs", files={"file": ("b.mp4", fixture_bytes, "video/mp4")}
    ).json()
    assert second["id"] == first["id"]


def test_unreachable_url_fails_with_a_typed_error(client):
    # Port 9 refuses immediately: exercises the failure path without the network.
    job = client.post(
        "/api/v1/jobs", json={"url": "http://127.0.0.1:9/video.mp4"}
    ).json()
    final = wait_for_terminal(client, job["id"], timeout=120)

    assert final["status"] == "failed"
    assert final["error_code"] == "E_DOWNLOAD_BLOCKED"
    assert final["stage"] == "ingest"


def test_corrupt_upload_fails_cleanly(client):
    job = client.post(
        "/api/v1/jobs", files={"file": ("broken.mp4", b"\x00\x01not-a-video", "video/mp4")}
    ).json()
    final = wait_for_terminal(client, job["id"], timeout=60)
    assert final["status"] == "failed"
    assert final["error_code"] == "E_NO_VIDEO_STREAM"


def test_rejects_unsupported_upload(client):
    r = client.post("/api/v1/jobs", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert r.status_code == 415
    assert r.json()["code"] == "E_UNSUPPORTED_MEDIA"


def test_rejects_bad_content_type(client):
    r = client.post("/api/v1/jobs", content=b"raw", headers={"Content-Type": "text/plain"})
    assert r.status_code == 400
    assert r.json()["code"] == "E_BAD_INPUT"


def test_unknown_job_is_problem_json(client):
    r = client.get("/api/v1/jobs/job_missing")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/problem+json")
    assert r.json()["code"] == "E_NOT_FOUND"


def test_artifact_path_traversal_is_blocked(client, fixture_bytes):
    job = client.post(
        "/api/v1/jobs", files={"file": ("clip.mp4", fixture_bytes, "video/mp4")}
    ).json()
    wait_for_terminal(client, job["id"])
    assert client.get(f"/api/v1/jobs/{job['id']}/files/../../lightnote.db").status_code == 404


def test_idempotency_key_returns_same_job(client):
    headers = {"Idempotency-Key": "abc-123"}
    a = client.post("/api/v1/jobs", json={"url": "http://127.0.0.1:9/v"}, headers=headers).json()
    b = client.post("/api/v1/jobs", json={"url": "http://127.0.0.1:9/v"}, headers=headers).json()
    assert a["id"] == b["id"]


def test_delete_removes_job_and_workspace(client, fixture_bytes):
    job = client.post(
        "/api/v1/jobs", files={"file": ("clip.mp4", fixture_bytes, "video/mp4")}
    ).json()
    wait_for_terminal(client, job["id"])
    assert client.delete(f"/api/v1/jobs/{job['id']}").status_code == 204
    assert client.get(f"/api/v1/jobs/{job['id']}").status_code == 404


# --------------------------------------------------------- text detection (C)


def test_ocr_finds_every_burned_in_text_element(client, caption_bytes):
    """Three captions plus a persistent watermark, all detected as tracks."""
    tracks = _project_for(client, caption_bytes)["text_tracks"]
    assert len(tracks) >= 4, [t["text"] for t in tracks]

    for expected in ("HELLOWORLD", "BUYNOW", "@BRANDCO"):
        _find(tracks, expected)


def test_text_track_timings_match_the_burned_in_spans(client, caption_bytes):
    tracks = _project_for(client, caption_bytes)["text_tracks"]

    first = _find(tracks, "HELLOWORLD")     # burned into scene 1: 0-2 s
    second = _find(tracks, "BUYNOW")        # scene 2: 2-4 s
    watermark = _find(tracks, "@BRANDCO")   # whole video

    assert first["t_in"] < 0.4 and 1.6 < first["t_out"] < 2.4
    assert 1.6 < second["t_in"] < 2.4 and 3.6 < second["t_out"] < 4.4
    assert watermark["t_in"] < 0.4 and watermark["t_out"] > 5.5
    # Each caption is confined to its own scene.
    assert first["scene_id"] != second["scene_id"]


def test_text_tracks_are_classified_by_role(client, caption_bytes):
    tracks = _project_for(client, caption_bytes)["text_tracks"]

    assert _find(tracks, "@BRANDCO")["kind"] == "watermark"    # corner, whole video
    assert _find(tracks, "HELLOWORLD")["kind"] == "caption"    # lower third, brief
    assert _find(tracks, "50%OFF")["kind"] == "overlay_text"   # upper third


def test_text_tracks_carry_style_and_geometry_for_rebuild(client, caption_bytes):
    track = _find(_project_for(client, caption_bytes)["text_tracks"], "HELLOWORLD")

    box = track["box"]
    assert all(0.0 <= box[k] <= 1.0 for k in "xywh")
    assert box["y"] > 0.6, "the caption sits in the lower third"

    style = track["style"]
    red, green, blue = (int(style["color"][i:i + 2], 16) for i in (1, 3, 5))
    assert min(red, green, blue) > 190, f"expected near-white text, got {style['color']}"
    assert style["contrast"] > 40
    assert style["font_px_norm"] == pytest.approx(box["h"])
    assert track["editable"] is True
    assert track["source"] == "ocr"


# ------------------------------------------------------- removal / export (D)


def test_clean_video_and_per_scene_clean_clips_are_produced(client, caption_bytes):
    job_id, project = _run(client, caption_bytes)

    outputs = project["outputs"]
    assert outputs["clean_video"] == "clean/clean.mp4"
    assert outputs["removal_method"] in {"opencv", "lama", "mask_blur"}
    assert outputs["frames_inpainted"] > 0
    assert outputs["mask_timeline"] == "masks/timeline.json"

    for scene in project["scenes"]:
        assert scene["clip"], f"{scene['id']} has no original clip"
        assert scene["clean_clip"], f"{scene['id']} has no de-edited clip"

    served = client.get(f"/api/v1/jobs/{job_id}/files/{outputs['clean_video']}")
    assert served.status_code == 200 and len(served.content) > 0


def test_clean_video_keeps_the_source_geometry_and_duration(client, caption_bytes):
    from app.storage import Workspace
    from app.utils import ffmpeg

    job_id, project = _run(client, caption_bytes)
    clean = ffmpeg.probe_media(Workspace.for_job(job_id).path(project["outputs"]["clean_video"]))
    media = project["media"]

    assert (clean.width, clean.height) == (media["width"], media["height"])
    assert clean.duration == pytest.approx(media["duration"], abs=0.3)


def test_mask_timeline_covers_every_detected_text_track(client, caption_bytes):
    import json

    from app.storage import Workspace

    job_id, project = _run(client, caption_bytes)
    timeline = json.loads(
        Workspace.for_job(job_id).path("masks", "timeline.json").read_text("utf-8")
    )

    masked_ids = {box["source_id"] for seg in timeline["segments"] for box in seg["boxes"]}
    assert masked_ids == {track["id"] for track in project["text_tracks"]}
    # Segments are ordered and non-overlapping.
    for previous, current in zip(timeline["segments"], timeline["segments"][1:], strict=False):
        assert previous["t_out"] <= current["t_in"]


def test_detected_text_is_actually_gone_from_the_clean_plate(client, caption_bytes):
    """The end-to-end proof: re-run OCR on the output and find nothing.

    Anything weaker (a file exists, some pixels changed) would pass even if the
    removal had missed the text entirely.
    """
    import cv2

    from app.pipeline.stages.ocr import get_engine, normalise_result
    from app.storage import Workspace

    job_id, project = _run(client, caption_bytes)
    workspace = Workspace.for_job(job_id)
    engine = get_engine()

    def read_text_at(path, seconds: float) -> list[str]:
        capture = cv2.VideoCapture(str(path))
        capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = capture.read()
        capture.release()
        assert ok, f"could not read {path.name} at {seconds}s"
        raw, _ = engine(frame)
        return [d["text"] for d in normalise_result(raw, frame.shape[1], frame.shape[0])]

    source = workspace.path(project["outputs"]["source_video"])
    clean = workspace.path(project["outputs"]["clean_video"])

    for seconds in (1.0, 3.0, 5.0):
        assert read_text_at(source, seconds), f"fixture should have text at {seconds}s"
        assert read_text_at(clean, seconds) == [], f"text survived removal at {seconds}s"


# ------------------------------------------------ image / product pop-ups (F)


def test_popup_is_detected_with_correct_geometry_and_timing(client, popup_bytes):
    """A panning shot with a pasted card for 1.0-3.0s at a known box."""
    from app.pipeline.geometry import from_dict, iou
    from tests.conftest import POPUP_BOX, POPUP_T_IN, POPUP_T_OUT

    project = _project_for(client, popup_bytes)
    tracks = project["overlay_tracks"]

    assert len(tracks) == 1, [t["box"] for t in tracks]
    track = tracks[0]

    assert iou(from_dict(track["box"]), POPUP_BOX) > 0.4, track["box"]
    # Timing is quantised to the 3 fps sampling grid, so allow one interval.
    assert abs(track["t_in"] - POPUP_T_IN) < 0.4
    assert abs(track["t_out"] - POPUP_T_OUT) < 0.4
    assert track["source"] == "cv"
    assert track["confidence"] >= 0.55


def test_popup_signals_are_reported_for_inspection(client, popup_bytes):
    track = _project_for(client, popup_bytes)["overlay_tracks"][0]
    signals = track["signals"]

    assert set(signals) == {"stability", "transience", "detail", "solidity"}
    # The fixture pans, so stability carries the signal and transience is unused.
    assert signals["stability"] > 0.4
    assert signals["transience"] == 0.0


def test_detected_popup_is_removed_from_the_clean_plate(client, popup_bytes):
    """Overlays flow into masking and inpainting with no extra wiring."""
    import json

    from app.storage import Workspace

    job_id, project = _run(client, popup_bytes)
    timeline = json.loads(
        Workspace.for_job(job_id).path("masks", "timeline.json").read_text("utf-8")
    )

    masked_ids = {box["source_id"] for seg in timeline["segments"] for box in seg["boxes"]}
    assert "ov_001" in masked_ids
    assert project["outputs"]["clean_video"]
    assert project["outputs"]["frames_inpainted"] > 0


def test_text_only_video_yields_no_false_positive_popups(client, caption_bytes):
    """The negative case: captions must not be re-reported as image overlays."""
    project = _project_for(client, caption_bytes)
    assert project["text_tracks"], "fixture should still detect its captions"
    assert project["overlay_tracks"] == []
