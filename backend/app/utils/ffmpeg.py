"""Thin, typed wrapper around the ffmpeg/ffprobe binaries.

Subprocess rather than a Python binding on purpose: ffmpeg's CLI is the stable,
well-documented interface, it streams instead of loading frames into RAM, and it
keeps heavyweight decode work out of the worker process's address space.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from app.errors import AppError, NoVideoStream

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 600


class FFmpegError(AppError):
    code = "E_FFMPEG"
    status = 500
    title = "ffmpeg failed"


@lru_cache
def binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise FFmpegError(f"'{name}' is not on PATH")
    return path


def run(args: list[str], *, timeout: int = DEFAULT_TIMEOUT, tool: str = "ffmpeg") -> str:
    """Run a tool and return stdout, raising FFmpegError with a useful tail."""
    cmd = [binary(tool), *args]
    log.debug("%s %s", tool, " ".join(args))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"{tool} timed out after {timeout}s") from exc
    if proc.returncode != 0:
        tail = " | ".join((proc.stderr or "").strip().splitlines()[-3:])
        raise FFmpegError(f"{tool} exited {proc.returncode}: {tail or 'no stderr'}")
    return proc.stdout


def ffmpeg(args: list[str], *, timeout: int = DEFAULT_TIMEOUT) -> str:
    # -nostdin: never let ffmpeg swallow the worker's stdin. -y: overwrite.
    return run(["-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args], timeout=timeout)


# ---------------------------------------------------------------- probing


@dataclass
class MediaInfo:
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    has_audio: bool
    rotation: int = 0
    bitrate: int | None = None
    nb_frames: int | None = None
    audio_codec: str | None = None

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_rate(value: str | None) -> float:
    """'30000/1001' -> 29.97"""
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            return float(num) / float(den) if float(den) else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _rotation(stream: dict) -> int:
    """Rotation lives in two places depending on the container/encoder."""
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            return int(round(float(side["rotation"]))) % 360
    tag = (stream.get("tags") or {}).get("rotate")
    if tag:
        try:
            return int(float(tag)) % 360
        except ValueError:
            return 0
    return 0


def probe(path: Path) -> dict:
    raw = run(
        ["-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        tool="ffprobe",
        timeout=60,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"ffprobe returned invalid JSON for {path.name}") from exc


def probe_media(path: Path) -> MediaInfo:
    if not path.exists() or path.stat().st_size == 0:
        raise NoVideoStream(f"{path.name} is missing or empty")

    try:
        data = probe(path)
    except FFmpegError as exc:
        # ffprobe refusing the file means it is not decodable media, which is a
        # bad-input condition, not an internal failure.
        raise NoVideoStream(f"{path.name} is not decodable: {exc.detail}") from exc
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise NoVideoStream(f"{path.name} contains no video stream")

    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    rotation = _rotation(video)
    if rotation in (90, 270):
        # Report *display* dimensions so normalized boxes mean what they say.
        width, height = height, width

    fmt = data.get("format") or {}
    duration = float(fmt.get("duration") or video.get("duration") or 0.0)
    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
    nb_frames = video.get("nb_frames")

    if duration <= 0 and fps and nb_frames:
        duration = int(nb_frames) / fps
    if width <= 0 or height <= 0 or duration <= 0:
        raise NoVideoStream(f"{path.name}: unusable stream ({width}x{height}, {duration}s)")

    return MediaInfo(
        duration=round(duration, 3),
        width=width,
        height=height,
        fps=round(fps or 30.0, 3),
        codec=str(video.get("codec_name") or "unknown"),
        has_audio=audio is not None,
        rotation=rotation,
        bitrate=int(fmt["bit_rate"]) if (fmt.get("bit_rate") or "").isdigit() else None,
        nb_frames=int(nb_frames) if (nb_frames or "").isdigit() else None,
        audio_codec=str(audio.get("codec_name")) if audio else None,
    )


# --------------------------------------------------------------- transforms


def remux(src: Path, dst: Path) -> None:
    """Container swap without touching the streams; falls back to a transcode."""
    try:
        ffmpeg(["-i", str(src), "-c", "copy", "-movflags", "+faststart", str(dst)])
    except FFmpegError:
        log.info("remux failed for %s, transcoding instead", src.name)
        transcode(src, dst)


def transcode(src: Path, dst: Path, *, crf: int = 20) -> None:
    ffmpeg([
        "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(dst),
    ])


def make_proxy(src: Path, dst: Path, *, max_dim: int, fps: float | None = None) -> None:
    """Small, constant-rate analysis copy.

    Every detector runs on this; the final render uses the source. `force_original_
    aspect_ratio=decrease` keeps the aspect, and -2 keeps both sides even for h264.
    """
    filters = [f"scale='min({max_dim},iw)':'min({max_dim},ih)'"
               f":force_original_aspect_ratio=decrease:flags=bicubic",
               "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
    if fps:
        filters.insert(0, f"fps={fps}")
    ffmpeg([
        "-i", str(src),
        "-vf", ",".join(filters),
        "-an",  # analysis never needs audio
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        str(dst),
    ])


def extract_audio(src: Path, dst: Path, *, sample_rate: int = 16000) -> None:
    """16 kHz mono PCM -- what Whisper wants, no resampling later."""
    ffmpeg(["-i", str(src), "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le",
            str(dst)])


def cut(src: Path, dst: Path, t_in: float, t_out: float) -> None:
    """Frame-accurate clip.

    Re-encodes rather than stream-copying: a copy can only cut on keyframes, and
    UGC edits routinely put cuts mid-GOP, which shows up as a frozen first frame.
    The clips are seconds long, so the accuracy is worth the CPU.
    """
    duration = max(0.04, t_out - t_in)
    ffmpeg([
        "-ss", f"{t_in:.3f}", "-i", str(src), "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-avoid_negative_ts", "make_zero", "-movflags", "+faststart",
        str(dst),
    ])


def extract_frame(src: Path, timestamp: float, dst: Path, *, width: int | None = None) -> None:
    args = ["-ss", f"{max(0.0, timestamp):.3f}", "-i", str(src), "-frames:v", "1"]
    if width:
        args += ["-vf", f"scale={width}:-2"]
    ffmpeg([*args, "-q:v", "3", str(dst)])


def extract_frames(src: Path, out_dir: Path, *, fps: float, width: int | None = None,
                   pattern: str = "%05d.jpg") -> list[Path]:
    """Uniform sampling in a single decode pass -- far cheaper than N seeks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    filters = [f"fps={fps}"]
    if width:
        filters.append(f"scale={width}:-2")
    ffmpeg(["-i", str(src), "-vf", ",".join(filters), "-q:v", "3", str(out_dir / pattern)])
    return sorted(out_dir.glob("*.jpg"))


class FrameSink:
    """Encode a stream of BGR frames, optionally muxing audio from another file.

    Piping raw frames into ffmpeg beats cv2.VideoWriter: OpenCV's writer cannot
    carry the original audio, gives no control over the encoder, and picks
    whatever codec the local build happens to have. This keeps encoding
    decisions here and streams frame by frame, so memory stays flat regardless
    of clip length.
    """

    def __init__(self, dst: Path, *, width: int, height: int, fps: float,
                 audio_from: Path | None = None, crf: int = 20) -> None:
        args = [
            binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
            # NOTE: no -nostdin here; stdin *is* the video stream.
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
        ]
        if audio_from is not None:
            args += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a:0?",
                     "-c:a", "aac", "-b:a", "128k", "-shortest"]
        args += [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst),
        ]
        self.dst = dst
        self._proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.PIPE)

    def write(self, frame) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(frame.tobytes())

    def close(self, timeout: int = 300) -> None:
        if self._proc.stdin:
            # A dead encoder gives a broken pipe here; the wait below reports why.
            with contextlib.suppress(BrokenPipeError):
                self._proc.stdin.close()
        try:
            _, stderr = self._proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            self._proc.kill()
            raise FFmpegError(f"encoder timed out writing {self.dst.name}") from exc
        if self._proc.returncode != 0:
            tail = " | ".join((stderr or b"").decode("utf-8", "replace").strip().splitlines()[-3:])
            raise FFmpegError(f"encoder exited {self._proc.returncode}: {tail}")

    def __enter__(self) -> FrameSink:
        return self

    def __exit__(self, exc_type, *_rest) -> None:
        if exc_type is not None:
            self._proc.kill()
            return
        self.close()


def frame_timestamp(index: int, fps: float) -> float:
    """Centre timestamp of the 1-based frame produced by the `fps` filter."""
    return (index - 0.5) / fps if fps > 0 else 0.0


def seconds_to_frames(seconds: float, fps: float) -> int:
    return max(1, int(math.ceil(seconds * fps)))
