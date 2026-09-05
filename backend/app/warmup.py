"""Preload the models a worker will need.

The first job on a fresh machine otherwise pays for downloading and loading
Whisper and the OCR weights inside the request -- measured at ~38 s against
~1 s once cached. Warming at boot moves that cost off the user's first job.

Every step is best-effort: a warmup failure must never stop a worker starting,
because the stages that use these models degrade on their own.
"""

from __future__ import annotations

import logging
import threading
import time

from app.config import settings

log = logging.getLogger(__name__)


def preload_models() -> None:
    """Load OCR and ASR models into the process. Safe to call more than once."""
    started = time.perf_counter()

    try:
        from app.pipeline.stages.ocr import get_engine

        get_engine()
    except Exception as exc:  # noqa: BLE001 - warmup is advisory
        log.warning("OCR warmup skipped: %s: %s", type(exc).__name__, exc)

    if settings.ai_enabled:
        try:
            from app.pipeline.stages.asr import get_model

            get_model(settings.whisper_model)
        except Exception as exc:  # noqa: BLE001
            log.warning("ASR warmup skipped: %s: %s", type(exc).__name__, exc)

    log.info("model warmup finished in %.1fs", time.perf_counter() - started)


def preload_in_background() -> threading.Thread:
    """Warm without blocking startup -- the API must answer /healthz immediately."""
    thread = threading.Thread(target=preload_models, name="warmup", daemon=True)
    thread.start()
    return thread
