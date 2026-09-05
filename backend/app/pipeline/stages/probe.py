"""S2 - inspect and normalise.

Produces the three things every later stage depends on:
  * verified media metadata (and the duration guard)
  * proxy.mp4  - small, constant-rate copy that all detectors run on
  * audio.wav  - 16 kHz mono PCM for ASR

Analyse small, render big: detection cost scales with pixel count, but the final
clean video is rendered from the full-resolution source. Normalized coordinates
make the mapping between the two free.
"""

from __future__ import annotations

from app.config import settings
from app.db.models import JobStatus
from app.errors import TooLong
from app.pipeline.base import Stage
from app.pipeline.context import JobContext
from app.utils import ffmpeg


class ProbeStage(Stage):
    name = "probe"
    status = JobStatus.DOWNLOADING
    weight = 0.6

    def run(self, ctx: JobContext) -> dict:
        ws = ctx.workspace

        ctx.progress(0.1, "probing streams")
        info = ffmpeg.probe_media(ws.source)
        if info.duration > settings.max_duration_s:
            raise TooLong(
                f"Video is {info.duration:.0f}s; the limit is {settings.max_duration_s}s"
            )
        ctx.log(
            f"{info.width}x{info.height} · {info.duration:.1f}s · {info.fps:.2f}fps · "
            f"{info.codec}{' + ' + info.audio_codec if info.audio_codec else ' (silent)'}"
        )

        ctx.progress(0.3, "building analysis proxy")
        ffmpeg.make_proxy(ws.source, ws.proxy, max_dim=settings.proxy_max_dim)
        proxy = ffmpeg.probe_media(ws.proxy)

        result = {
            **info.to_dict(),
            "proxy": {
                "path": ws.rel(ws.proxy),
                "width": proxy.width,
                "height": proxy.height,
                "fps": proxy.fps,
            },
            "audio_path": None,
        }

        if info.has_audio:
            ctx.progress(0.8, "extracting audio")
            try:
                ffmpeg.extract_audio(ws.source, ws.audio)
                result["audio_path"] = ws.rel(ws.audio)
            except ffmpeg.FFmpegError as exc:
                # Non-fatal: only ASR needs it, and ASR is an optional stage.
                ctx.degrade(f"audio_extract_failed:{exc.detail[:60]}")
                result["has_audio"] = False
        else:
            ctx.degrade("no_audio_track")

        return result
