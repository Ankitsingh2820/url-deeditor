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

-- and, where a speech synthesiser is available, a voice track that *says* the
captions. That last part matters: it is what lets the pipeline demonstrate
classifying text by meaning (the words on screen match the words spoken, so they
are subtitles) rather than by position. Without audio the same captions fall back
to the position heuristic.

So a single run exercises scene detection, OCR, speech transcription, text
classification, pop-up detection, masking and inpainting.

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


def _narrate(text: str, destination: Path) -> bool:
    """Render one line to a WAV with the OS speech synthesiser.

    Windows SAPI via PowerShell -- no extra dependency, and the sample is a
    developer convenience rather than a shipped asset. Returns False on any
    platform or machine without it, and the caller falls back to a tone.
    """
    if sys.platform != "win32":
        return False
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$s.Rate = -1;"
        f"$s.SetOutputToWaveFile('{destination}');"
        f"$s.Speak('{text}');"
        "$s.Dispose()"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and destination.exists() and destination.stat().st_size > 1000


def _mux_audio(video: Path, voices: list[Path | None], destination: Path) -> bool:
    """Lay each spoken line at the start of the scene whose caption it is."""
    tracks = [(i, p) for i, p in enumerate(voices) if p is not None]
    if not tracks:
        return False

    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video)]
    for _, path in tracks:
        args += ["-i", str(path)]

    parts, labels = [], []
    for slot, (scene, _path) in enumerate(tracks, start=1):
        delay = int(scene * SCENE_SECONDS * 1000)
        parts.append(f"[{slot}:a]adelay={delay}|{delay}[a{slot}]")
        labels.append(f"[a{slot}]")
    # apad to the clip's full length: a short audio track is legal, but padding
    # keeps the sample simple to reason about.
    duration = SCENE_SECONDS * SCENES
    parts.append(
        f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0,"
        f"apad,atrim=0:{duration}[a]"
    )

    args += [
        "-filter_complex", ";".join(parts),
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart", str(destination),
    ]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"  (audio mux failed, keeping the silent version: {result.stderr.strip()[:120]})")
        return False
    return True


def render(destination: Path, *, seed: int = 5, narrate: bool = True) -> Path:
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

        silent = Path(workdir) / "silent.mp4"
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-framerate", str(FPS), "-i", str(frames_dir / "%05d.png"),
             "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(silent)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            # Show the encoder's own reason; a bare exit code is not actionable.
            sys.exit(f"ffmpeg failed ({result.returncode}):\n{result.stderr.strip()}")

        voices: list[Path | None] = []
        if narrate:
            for index, line in enumerate(CAPTIONS):
                wav = Path(workdir) / f"voice_{index}.wav"
                voices.append(wav if _narrate(line, wav) else None)
            spoken = sum(1 for v in voices if v)
            print(f"  narrated {spoken}/{len(CAPTIONS)} captions")

        if narrate and _mux_audio(silent, voices, destination):
            return destination

        # No speech synthesiser: fall back to a tone so the clip still has an
        # audio stream, and say so -- the caption/overlay split will then come
        # from position rather than from matching the transcript.
        print("  note: no spoken track; captions will be classified by position,"
              " not by matching the transcript.")
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent),
             "-f", "lavfi", "-i", f"sine=frequency=320:duration={SCENE_SECONDS * SCENES}",
             "-c:v", "copy", "-c:a", "aac", "-shortest",
             "-movflags", "+faststart", str(destination)],
            capture_output=True, text=True, check=False,
        )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default="samples/ugc_ad.mp4", type=Path)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--no-narration", action="store_true",
                        help="skip the spoken track (captions then classify by position)")
    args = parser.parse_args()

    path = render(args.output, seed=args.seed, narrate=not args.no_narration)
    size = path.stat().st_size
    print(f"wrote {path}  ({size / 1e6:.1f} MB, {SCENE_SECONDS * SCENES}s, {WIDTH}x{HEIGHT})")
    print("contains: 3 scenes, 3 captions (spoken), 1 watermark, 1 product pop-up (4.0-7.0s)")
    print("\nUpload it at http://localhost:5173 and press De-edit.")


if __name__ == "__main__":
    main()
