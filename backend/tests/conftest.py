"""Shared fixtures.

Environment is configured before any app module is imported, because
`app.config.settings` is a module-level singleton.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="lightnote-test-"))
os.environ.setdefault("STORAGE_DIR", str(_TMP / "storage"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{(_TMP / 'test.db').as_posix()}")
os.environ.setdefault("QUEUE_MODE", "thread")
os.environ.setdefault("PROXY_MAX_DIM", "720")  # no downscale for the tiny fixtures
os.environ.setdefault("AI_ENABLED", "false")

#: three visually distinct 2 s segments + a tone, so scene detection has real cuts
FIXTURE_SEGMENTS = ("testsrc", "smptebars", "rgbtestsrc")
FIXTURE_SIZE = "320x568"  # portrait, like the short-form videos this targets
FIXTURE_RATE = 15
FIXTURE_SEG_S = 2


def _build_fixture_video(path: Path) -> Path:
    inputs: list[str] = []
    for source in FIXTURE_SEGMENTS:
        inputs += ["-f", "lavfi", "-i",
                   f"{source}=size={FIXTURE_SIZE}:rate={FIXTURE_RATE}:duration={FIXTURE_SEG_S}"]
    inputs += ["-f", "lavfi", "-i",
               f"sine=frequency=440:duration={FIXTURE_SEG_S * len(FIXTURE_SEGMENTS)}"]

    concat = "".join(f"[{i}:v]" for i in range(len(FIXTURE_SEGMENTS)))
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
         "-filter_complex", f"{concat}concat=n={len(FIXTURE_SEGMENTS)}:v=1:a=0[v]",
         "-map", "[v]", "-map", f"{len(FIXTURE_SEGMENTS)}:a",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)],
        check=True, capture_output=True,
    )
    return path


@pytest.fixture(scope="session")
def fixture_video() -> Path:
    """A real, tiny mp4 with three detectable scenes and an audio track."""
    path = _TMP / "fixture.mp4"
    if not path.exists():
        _build_fixture_video(path)
    return path


#: what the caption fixture burns in: (text, scene index, position, scale, thickness)
CAPTIONS = [
    ("HELLO WORLD", 0, (24, 500), 0.75, 2),
    ("BUY NOW", 1, (60, 500), 0.75, 2),
    ("50% OFF", 2, (70, 110), 0.75, 2),
]
WATERMARK = ("@brandco", (10, 34), 0.45, 1)


def _build_caption_video(path: Path) -> Path:
    """A synthetic UGC-style clip: three scenes, burned-in captions, a watermark.

    Rendered frame by frame with OpenCV rather than ffmpeg's drawtext filter, so
    the test does not depend on a font being installed at a known path.
    """
    import cv2
    import numpy as np

    width, height = 320, 568
    frames_dir = path.parent / "caption_frames"
    frames_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    backgrounds = [(40, 60, 180), (30, 150, 60), (170, 70, 40)]  # BGR, distinct scenes

    index = 0
    for scene, colour in enumerate(backgrounds):
        for _ in range(FIXTURE_RATE * FIXTURE_SEG_S):
            frame = np.full((height, width, 3), colour, np.uint8)
            # Light noise keeps the detector from seeing a degenerate flat frame.
            frame = np.clip(
                frame.astype(np.int16) + rng.integers(-8, 9, frame.shape), 0, 255
            ).astype(np.uint8)
            cv2.putText(frame, WATERMARK[0], WATERMARK[1], cv2.FONT_HERSHEY_SIMPLEX,
                        WATERMARK[2], (255, 255, 255), WATERMARK[3], cv2.LINE_AA)
            for text, scene_index, origin, scale, thickness in CAPTIONS:
                if scene_index == scene:
                    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                                (255, 255, 255), thickness, cv2.LINE_AA)
            index += 1
            cv2.imwrite(str(frames_dir / f"{index:05d}.png"), frame)

    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-framerate", str(FIXTURE_RATE), "-i", str(frames_dir / "%05d.png"),
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p",
         str(path)],
        check=True, capture_output=True,
    )
    return path


#: pop-up fixture: card is on screen between these times
POPUP_T_IN, POPUP_T_OUT = 1.0, 3.0
#: ...at this normalized box
POPUP_BOX = (0.52, 0.10, 0.38, 0.22)


def _build_popup_video(path: Path) -> Path:
    """A panning shot with a pasted product card for part of its duration.

    The pan is what makes this a real test: the background moves every frame
    while the card is nailed to the screen, which is exactly the asymmetry the
    detector keys on. The card also appears and disappears mid-shot, so the
    transience signal has something to find.
    """
    import cv2
    import numpy as np

    width, height, seconds = 320, 568, 4
    frames_total = FIXTURE_RATE * seconds
    rng = np.random.default_rng(11)

    # A textured world larger than the frame, so the window can pan across it.
    canvas = np.full((height + 200, width + 200, 3), 60, np.uint8)
    for _ in range(90):
        centre = (int(rng.integers(0, canvas.shape[1])), int(rng.integers(0, canvas.shape[0])))
        colour = tuple(int(c) for c in rng.integers(40, 230, 3))
        if rng.random() < 0.5:
            cv2.circle(canvas, centre, int(rng.integers(12, 45)), colour, -1)
        else:
            corner = (centre[0] + int(rng.integers(20, 70)), centre[1] + int(rng.integers(20, 70)))
            cv2.rectangle(canvas, centre, corner, colour, -1)

    frames_dir = path.parent / "popup_frames"
    frames_dir.mkdir(exist_ok=True)
    px, py, pw, ph = (
        int(POPUP_BOX[0] * width), int(POPUP_BOX[1] * height),
        int(POPUP_BOX[2] * width), int(POPUP_BOX[3] * height),
    )

    for index in range(frames_total):
        # Pan diagonally: ~3 px per frame, i.e. ~15 px between sampled frames.
        ox, oy = int(index * 3) % 190, int(index * 2) % 190
        frame = canvas[oy:oy + height, ox:ox + width].copy()

        t = index / FIXTURE_RATE
        if POPUP_T_IN <= t < POPUP_T_OUT:
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (245, 245, 245), -1)
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (20, 20, 20), 3)
            cv2.circle(frame, (px + pw // 2, py + ph // 2), min(pw, ph) // 4, (30, 90, 220), -1)
            cv2.rectangle(frame, (px + 10, py + ph - 22), (px + pw - 10, py + ph - 10),
                          (40, 40, 40), -1)
        cv2.imwrite(str(frames_dir / f"{index + 1:05d}.png"), frame)

    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-framerate", str(FIXTURE_RATE), "-i", str(frames_dir / "%05d.png"),
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", "-pix_fmt", "yuv420p",
         str(path)],
        check=True, capture_output=True,
    )
    return path


@pytest.fixture(scope="session")
def popup_video() -> Path:
    path = _TMP / "popup.mp4"
    if not path.exists():
        _build_popup_video(path)
    return path


@pytest.fixture(scope="session")
def popup_bytes(popup_video: Path) -> bytes:
    return popup_video.read_bytes()


@pytest.fixture(scope="session")
def caption_video() -> Path:
    path = _TMP / "captions.mp4"
    if not path.exists():
        _build_caption_video(path)
    return path


@pytest.fixture(scope="session")
def caption_bytes(caption_video: Path) -> bytes:
    return caption_video.read_bytes()


@pytest.fixture(scope="session")
def fixture_bytes(fixture_video: Path) -> bytes:
    return fixture_video.read_bytes()


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


def wait_for_terminal(client, job_id: str, timeout: float = 180.0) -> dict:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/v1/jobs/{job_id}").json()
        if body["status"] in {"succeeded", "failed", "cancelled"}:
            return body
        time.sleep(0.25)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")
