"""Typed application errors + RFC-7807 (application/problem+json) rendering.

Every failure the user can cause has a stable machine-readable code, so the
frontend can react (e.g. "download blocked -> suggest upload") without string
matching, and so `project.json` diagnostics stay meaningful.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

PROBLEM_JSON = "application/problem+json"


class AppError(Exception):
    """Base class for expected, user-facing failures."""

    code = "E_INTERNAL"
    status = 500
    title = "Internal error"

    def __init__(self, detail: str | None = None, *, hint: str | None = None, **extra):
        self.detail = detail or self.title
        self.hint = hint
        self.extra = extra
        super().__init__(self.detail)

    def to_problem(self, instance: str | None = None) -> dict:
        problem = {
            "type": f"https://lightnote.dev/errors/{self.code.lower()}",
            "title": self.title,
            "status": self.status,
            "code": self.code,
            "detail": self.detail,
        }
        if self.hint:
            problem["hint"] = self.hint
        if instance:
            problem["instance"] = instance
        problem.update(self.extra)
        return problem


# ---- input / validation -------------------------------------------------


class BadInput(AppError):
    code = "E_BAD_INPUT"
    status = 400
    title = "Invalid request"


class UnsupportedMedia(AppError):
    code = "E_UNSUPPORTED_MEDIA"
    status = 415
    title = "Unsupported media type"


class TooLarge(AppError):
    code = "E_TOO_LARGE"
    status = 413
    title = "File too large"


class TooLong(AppError):
    code = "E_TOO_LONG"
    status = 422
    title = "Video too long"


class NotFound(AppError):
    code = "E_NOT_FOUND"
    status = 404
    title = "Not found"


class Conflict(AppError):
    code = "E_CONFLICT"
    status = 409
    title = "Conflicting state"


# ---- pipeline -----------------------------------------------------------


class DownloadBlocked(AppError):
    code = "E_DOWNLOAD_BLOCKED"
    status = 422
    title = "Could not download that URL"

    def __init__(self, detail: str | None = None, **kw):
        kw.setdefault(
            "hint",
            "The platform may require login or block this region. "
            "Upload the video file instead.",
        )
        super().__init__(detail, **kw)


class NoVideoStream(AppError):
    code = "E_NO_VIDEO_STREAM"
    status = 422
    title = "No decodable video stream"


class StageFailed(AppError):
    code = "E_STAGE_FAILED"
    status = 500
    title = "Processing stage failed"

    def __init__(self, stage: str, detail: str | None = None, **kw):
        self.stage = stage
        super().__init__(detail or f"Stage '{stage}' failed", stage=stage, **kw)


class Cancelled(AppError):
    code = "E_CANCELLED"
    status = 499
    title = "Job cancelled"


# ---- FastAPI wiring -----------------------------------------------------


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status,
        content=exc.to_problem(instance=str(request.url.path)),
        media_type=PROBLEM_JSON,
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Never leak a stack trace to the client; the traceback is already logged.
    return JSONResponse(
        status_code=500,
        content=AppError(f"{type(exc).__name__}").to_problem(instance=str(request.url.path)),
        media_type=PROBLEM_JSON,
    )
