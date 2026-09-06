"""Liveness and readiness.

/healthz  - is the process up (never touches dependencies)
/readyz   - can it actually do work: db, storage, ffmpeg, broker
"""

from __future__ import annotations

import shutil
import subprocess

from fastapi import APIRouter, Response
from sqlalchemy import text

from app.config import settings
from app.db.session import engine

router = APIRouter(tags=["health"])


def _check_db() -> tuple[bool, str]:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _check_storage() -> tuple[bool, str]:
    try:
        probe = settings.storage_dir / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True, str(settings.storage_dir)
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _check_ffmpeg() -> tuple[bool, str]:
    binary = shutil.which("ffmpeg")
    if not binary:
        return False, "ffmpeg not on PATH"
    try:
        out = subprocess.run(
            [binary, "-version"], capture_output=True, text=True, timeout=10, check=True
        )
        return True, out.stdout.splitlines()[0]
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _check_models() -> tuple[bool, str]:
    """Report which optional model layers are importable.

    Not a readiness failure -- the pipeline degrades without them -- but the
    fastest way for someone who just set the project up to see whether their
    install is complete.
    """
    layers = {
        "ocr": "rapidocr_onnxruntime",
        "asr": "faster_whisper",
        "vlm_gemini": "google.genai",
        "vlm_claude": "anthropic",
        "inpaint_lama": "simple_lama_inpainting",
    }
    present, missing = [], []
    for label, module in layers.items():
        try:
            __import__(module)
            present.append(label)
        except ImportError:
            missing.append(label)

    detail = f"available: {', '.join(present) or 'none'}"
    if missing:
        detail += f" | not installed: {', '.join(missing)}"
    # OCR is the one the core de-edit genuinely needs.
    return "ocr" in present, detail


def _check_broker() -> tuple[bool, str]:
    if settings.queue_mode != "celery":
        return True, "thread pool (no broker required)"
    try:
        import redis

        redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2).ping()
        return True, settings.redis_url
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


@router.get("/healthz", summary="Liveness probe")
def healthz() -> dict:
    return {"status": "ok", "app": settings.app_name, "env": settings.env}


@router.get("/readyz", summary="Readiness probe")
def readyz(response: Response) -> dict:
    checks = {
        "database": _check_db(),
        "storage": _check_storage(),
        "ffmpeg": _check_ffmpeg(),
        "broker": _check_broker(),
        "models": _check_models(),
    }
    ready = all(ok for ok, _ in checks.values())
    response.status_code = 200 if ready else 503
    return {
        "ready": ready,
        "queue_mode": settings.queue_mode,
        "ai_enabled": settings.ai_enabled,
        "vlm": {
            "provider": settings.vlm_provider,
            "key_configured": bool(settings.gemini_api_key or settings.anthropic_api_key),
        },
        "checks": {name: {"ok": ok, "detail": detail} for name, (ok, detail) in checks.items()},
    }
