"""Generate a UGC-style sample clip to de-edit.

    python scripts/make_sample.py samples/ugc_ad.mp4

Why this exists: trying the app should not depend on finding a TikTok URL that
yt-dlp can still reach. This renders a short-form-shaped clip that contains one
of everything the pipeline looks for --

    * three scenes with hard cuts
    * a panning camera, so the pop-up detector's stability signal has something
      to work with
    * burned-in captions that change per scene
    * a persistent corner watermark
    * a product card that appears mid-shot and leaves again

-- so a single run exercises scene detection, OCR, text tracking, pop-up
detection, masking and inpainting.

Frames are drawn with OpenCV rather than ffmpeg's drawtext filter so the script
does not depend on a font being installed at a known path.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 404, 720          # 9:16-ish; even numbers, which libx264 requires
FPS = 24
SCENE_SECONDS = 3
SCENES = 3

WATERMARK = "@glowlab"
CAPTIONS = [
    "this changed my skin",
    "three drops every morning",
    "link in bio",
]
#: the product card: normalized box, and the window it is on screen
CARD_BOX = (0.52, 0.09, 0.40, 0.24)
CARD_IN, CARD_OUT = 4.0, 7.0


def _world(rng: np.random.Generator) -> np.ndarray:
    """A textured backdrop larger than the frame, so the camera can pan over it."""
    canvas = np.full((HEIGHT + 260, WIDTH + 260, 3), 52, np.uint8)
    for _ in range(120):
        centre = (int(rng.integers(0, canvas.shape[1])), int(rng.integers(0, canvas.shape[0])))
        colour = tuple(int(c) for c in rng.integers(35, 225, 3))
        if rng.random() < 0.55:
            cv2.circle(canvas, centre, int(rng.integers(14, 52)), colour, -1)
        else:
            corner = (centre[0] + int(rng.integers(24, 90)), centre[1] + int(rng.integers(24, 90)))
            cv2.rectangle(canvas, centre, corner, colour, -1)
    return canvas


def _tint(frame: np.ndarray, scene: int) -> np.ndarray:
    """Give each scene its own grade, so the cuts are unambiguous."""
    tints = [(1.15, 0.95, 0.85), (0.85, 1.15, 0.95), (0.95, 0.9, 1.2)]
    return np.clip(frame * np.array(tints[scene % len(tints)]), 0, 255).astype(np.uint8)


def _draw_card(frame: np.ndarray) -> None:
    x = int(CARD_BOX[0] * WIDTH)
    y = int(CARD_BOX[1] * HEIGHT)
    w = int(CARD_BOX[2] * WIDTH)
    h = int(CARD_BOX[3] * HEIGHT)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (242, 242, 240), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (28, 28, 28), 3)
    cv2.circle(frame, (x + w // 2, y + h // 2 - 8), min(w, h) // 5, (60, 90, 210), -1)
    cv2.rectangle(frame, (x + 12, y + h - 30), (x + w - 12, y + h - 14), (45, 45, 45), -1)


def _draw_caption(frame: np.ndarray, text: str) -> None:
    scale, thickness = 0.78, 2
    (tw, _th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    origin = ((WIDTH - tw) // 2, int(HEIGHT * 0.87))
    # Dark outline first, so the caption survives over a bright backdrop -- and
    # so the style extractor has a real stroke to recover.
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (15, 15, 15),
                thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                thickness, cv2.LINE_AA)


def render(destination: Path, *, seed: int = 5) -> Path:
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg is not on PATH; install it first (see the README).")

    rng = np.random.default_rng(seed)
    world = _world(rng)
    total = FPS * SCENE_SECONDS * SCENES
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="lightnote-sample-") as workdir:
        frames_dir = Path(workdir)
        for index in range(total):
            t = index / FPS
            scene = min(SCENES - 1, int(t // SCENE_SECONDS))

            # Pan diagonally; reset the offset at each cut so scenes differ sharply.
            local = index - scene * FPS * SCENE_SECONDS
            ox = min(world.shape[1] - WIDTH - 1, 30 + local * 2)
            oy = min(world.shape[0] - HEIGHT - 1, 20 + local * 2 + scene * 40)
            frame = _tint(world[oy:oy + HEIGHT, ox:ox + WIDTH].copy(), scene)

            cv2.putText(frame, WATERMARK, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (240, 240, 240), 1, cv2.LINE_AA)
            _draw_caption(frame, CAPTIONS[scene])
            if CARD_IN <= t < CARD_OUT:
                _draw_card(frame)

            cv2.imwrite(str(frames_dir / f"{index + 1:05d}.png"), frame)

        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-framerate", str(FPS), "-i", str(frames_dir / "%05d.png"),
             "-f", "lavfi", "-i", f"sine=frequency=320:duration={SCENE_SECONDS * SCENES}",
             "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(destination)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            # Show the encoder's own reason; a bare exit code is not actionable.
            sys.exit(f"ffmpeg failed ({result.returncode}):\n{result.stderr.strip()}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default="samples/ugc_ad.mp4", type=Path)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    path = render(args.output, seed=args.seed)
    size = path.stat().st_size
    print(f"wrote {path}  ({size / 1e6:.1f} MB, {SCENE_SECONDS * SCENES}s, {WIDTH}x{HEIGHT})")
    print("contains: 3 scenes, 3 captions, 1 watermark, 1 product pop-up (4.0-7.0s)")
    print("\nUpload it at http://localhost:5173 and press De-edit.")


if __name__ == "__main__":
    main()
