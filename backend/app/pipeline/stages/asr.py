"""S6 - transcribe the audio.

Whisper is here to do something more useful than produce a transcript: it is
what tells a burned-in *caption* apart from a graphic *text overlay*.

Position cannot do this. A hook line and a subtitle both sit in the lower third
of a UGC ad; a price tag and a caption are both short and centred. The thing
that actually separates them is whether the words on screen are the words being
said. Matching each OCR track against the speech in its own time window answers
that directly, and it is a signal no amount of pixel analysis can produce.

faster-whisper (CTranslate2) rather than openai-whisper: several times realtime
on CPU with int8, which keeps a 30 s clip inside the job's time budget.
"""

from __future__ import annotations

import logging
import threading

from app.config import settings
from app.db.models import JobStatus
from app.pipeline.base import Stage
from app.pipeline.context import JobContext

log = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()


def get_model(name: str, compute_type: str = "int8"):
    """Process-wide Whisper model; loading costs seconds, so do it once."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from faster_whisper import WhisperModel

                _model = WhisperModel(name, device="cpu", compute_type=compute_type)
                log.info("faster-whisper %r loaded", name)
    return _model


class AsrStage(Stage):
    name = "asr"
    status = JobStatus.ANALYZING
    weight = 1.5
    required = False  # a silent video must still produce a full de-edit

    def skip_if(self, ctx: JobContext) -> str | None:
        probe = ctx.get("probe") or {}
        if not probe.get("has_audio") or not probe.get("audio_path"):
            return "no_audio_track"
        if not settings.ai_enabled:
            return "ai_disabled"
        return None

    def run(self, ctx: JobContext) -> dict:
        ws = ctx.workspace
        audio = ws.path(ctx.require("probe")["audio_path"])

        ctx.progress(0.05, f"loading whisper ({settings.whisper_model})")
        model = get_model(settings.whisper_model)

        ctx.progress(0.15, "transcribing")
        segments, info = model.transcribe(
            str(audio),
            beam_size=1,           # greedy: this feeds a fuzzy match, not a subtitle file
            vad_filter=True,       # skip silence rather than hallucinate over it
            word_timestamps=False,
        )

        duration = float(ctx.require("probe")["duration"]) or 1.0
        out: list[dict] = []
        for segment in segments:
            ctx.raise_if_cancelled()
            text = (segment.text or "").strip()
            if not text:
                continue
            out.append({
                "t_in": round(float(segment.start), 3),
                "t_out": round(float(segment.end), 3),
                "text": text,
            })
            ctx.progress(min(0.95, 0.15 + 0.8 * (segment.end / duration)),
                         f"transcribed {segment.end:.1f}s / {duration:.1f}s")

        if not out:
            ctx.degrade("no_speech_detected")
        ctx.log(
            f"{len(out)} speech segments, language={getattr(info, 'language', '?')} "
            f"({getattr(info, 'language_probability', 0):.0%})"
        )
        return {
            "engine": f"faster-whisper:{settings.whisper_model}",
            "language": getattr(info, "language", None),
            "language_probability": round(float(getattr(info, "language_probability", 0.0)), 4),
            "segments": out,
            "count": len(out),
        }
