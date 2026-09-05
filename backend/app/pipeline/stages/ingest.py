"""S1 - acquire the source video.

Two inputs, one output: whatever arrives becomes `source.mp4` in the workspace,
so every downstream stage has exactly one thing to open.

URL handling checks metadata *before* downloading, so an over-long video costs
one API round-trip instead of 200 MB of traffic.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from app.config import settings
from app.db.models import JobStatus
from app.errors import DownloadBlocked, NoVideoStream, TooLarge, TooLong
from app.pipeline.base import Stage
from app.pipeline.context import JobContext
from app.utils import ffmpeg

log = logging.getLogger(__name__)

# Prefer a progressive mp4, fall back to muxing the best video+audio, then to
# whatever single file exists (some short-form CDNs only offer one).
YTDLP_FORMAT = "bv*[ext=mp4]+ba[ext=m4a]/bv*+ba/b[ext=mp4]/b"


class _YtdlpLogger:
    """Route yt-dlp output into the app logger.

    yt-dlp writes to stdout/stderr by default, which is wrong inside a worker
    (and blows up when the surrounding process has closed those streams).
    """

    def debug(self, msg: str) -> None:
        log.debug("yt-dlp: %s", msg)

    def info(self, msg: str) -> None:
        log.debug("yt-dlp: %s", msg)

    def warning(self, msg: str) -> None:
        log.warning("yt-dlp: %s", msg)

    def error(self, msg: str) -> None:
        log.error("yt-dlp: %s", msg)


class IngestStage(Stage):
    name = "ingest"
    status = JobStatus.DOWNLOADING
    weight = 2.0

    def run(self, ctx: JobContext) -> dict:
        ws = ctx.workspace
        if ws.source.exists() and ws.source.stat().st_size > 0:
            ctx.log("source.mp4 already present, skipping fetch")
        elif ctx.source_kind == "upload":
            self._from_upload(ctx)
        else:
            self._from_url(ctx)

        ctx.progress(0.9, "hashing source")
        digest = _sha256(ws.source)
        ctx.content_sha256 = digest

        size = ws.source.stat().st_size
        if size > settings.max_upload_mb * 1024 * 1024:
            raise TooLarge(f"Source is {size / 1e6:.0f} MB, limit is {settings.max_upload_mb} MB")

        return {
            "path": ws.rel(ws.source),
            "bytes": size,
            "sha256": digest,
            "origin": ctx.source_ref,
            "kind": ctx.source_kind,
        }

    # ---- upload ---------------------------------------------------------
    def _from_upload(self, ctx: JobContext) -> None:
        ws = ctx.workspace
        rel = ctx.raw_options.get("upload_path")
        raw = ws.path(rel) if rel else next(iter(sorted(ws.root.glob("upload.*"))), None)
        if raw is None or not Path(raw).exists():
            raise NoVideoStream("Uploaded file is missing from the workspace")

        ctx.progress(0.3, f"normalising {Path(raw).name}")
        # Remux into mp4 + faststart so the browser can stream it while the
        # pipeline is still running. Falls back to a transcode for odd codecs.
        try:
            ffmpeg.remux(Path(raw), ws.source)
        except ffmpeg.FFmpegError as exc:
            raise NoVideoStream(f"Could not decode the uploaded file: {exc.detail}") from exc
        Path(raw).unlink(missing_ok=True)

    # ---- url ------------------------------------------------------------
    def _from_url(self, ctx: JobContext) -> None:
        import yt_dlp

        ws = ctx.workspace
        url = ctx.source_ref
        ctx.progress(0.05, "resolving URL")

        def hook(status: dict) -> None:
            if status.get("status") != "downloading":
                return
            total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
            done = status.get("downloaded_bytes") or 0
            if total:
                ctx.progress(0.15 + 0.65 * (done / total), f"downloading {done / 1e6:.1f} MB")

        options = {
            "outtmpl": str(ws.path("download.%(ext)s")),
            "format": YTDLP_FORMAT,
            "merge_output_format": "mp4",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "retries": 2,
            "socket_timeout": 30,
            "max_filesize": settings.max_upload_mb * 1024 * 1024,
            "progress_hooks": [hook],
            "logger": _YtdlpLogger(),
        }

        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                # 1) metadata only -- reject long videos before spending bandwidth
                info = ydl.extract_info(url, download=False)
                if info.get("_type") == "playlist":
                    entries = [e for e in (info.get("entries") or []) if e]
                    if not entries:
                        raise DownloadBlocked("That URL resolved to an empty playlist")
                    info = entries[0]

                duration = float(info.get("duration") or 0)
                if duration and duration > settings.max_duration_s:
                    raise TooLong(
                        f"Video is {duration:.0f}s; the limit is {settings.max_duration_s}s"
                    )
                ctx.log(f"source: {info.get('extractor_key')} · {duration or '?'}s")

                # 2) fetch
                ctx.progress(0.15, "downloading")
                downloaded = ydl.extract_info(url, download=True)

        except (TooLong, TooLarge):
            raise
        except yt_dlp.utils.DownloadError as exc:
            raise DownloadBlocked(_clean_ytdlp_error(str(exc))) from exc
        except Exception as exc:  # noqa: BLE001 - yt-dlp raises a wide variety
            raise DownloadBlocked(f"{type(exc).__name__}: {exc}") from exc

        produced = _downloaded_path(downloaded, ws.root)
        if produced is None:
            raise DownloadBlocked("The download produced no file")

        ctx.progress(0.85, "normalising container")
        if produced.suffix.lower() == ".mp4":
            produced.replace(ws.source)
        else:
            ffmpeg.remux(produced, ws.source)
            produced.unlink(missing_ok=True)


def _downloaded_path(info: dict, root: Path) -> Path | None:
    """yt-dlp reports the final path in a few different places by version/branch."""
    for entry in info.get("requested_downloads") or []:
        for key in ("filepath", "_filename", "filename"):
            if entry.get(key) and Path(entry[key]).exists():
                return Path(entry[key])
    for key in ("filepath", "_filename"):
        if info.get(key) and Path(info[key]).exists():
            return Path(info[key])
    candidates = [p for p in root.glob("download.*") if p.is_file()]
    return max(candidates, key=lambda p: p.stat().st_size) if candidates else None


def _clean_ytdlp_error(message: str) -> str:
    """Strip yt-dlp's ANSI/prefix noise so the API returns something readable."""
    text = message.replace("[0;31mERROR:[0m", "").replace("ERROR:", "").strip()
    return text.split("\n")[0][:300] or "The platform refused the request"


def _sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()
