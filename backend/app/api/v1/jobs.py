"""Job routes.

POST /jobs accepts *either* representation of a source:
  - application/json      {"url": "...", "options": {...}}
  - multipart/form-data   file=<binary>, options=<json string>
One resource, two encodings, so the client has a single "submit" call.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Header, Query, Request, Response, status
from fastapi.responses import FileResponse
from sqlmodel import Session
from sse_starlette.sse import EventSourceResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.config import settings
from app.db.session import get_session
from app.errors import BadInput, NotFound
from app.events import stream
from app.schemas.job import CreateJobRequest, JobList, JobOptions, JobRead
from app.services import jobs as service
from app.storage import Workspace

router = APIRouter(prefix="/jobs", tags=["jobs"])

CREATE_BODY_DOC = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/CreateJobRequest"},
                "example": {"url": "https://www.tiktok.com/@user/video/123", "options": {}},
            },
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": {"type": "string", "format": "binary"},
                        "options": {"type": "string", "description": "JSON-encoded JobOptions"},
                    },
                }
            },
        },
    }
}


def _parse_options(raw: object) -> JobOptions:
    if raw in (None, ""):
        return JobOptions()
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BadInput(f"options is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise BadInput("options must be a JSON object")
    try:
        return JobOptions(**raw)
    except Exception as exc:  # noqa: BLE001
        raise BadInput(f"invalid options: {exc}") from exc


@router.post(
    "",
    response_model=JobRead,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a video URL or file for de-editing",
    openapi_extra=CREATE_BODY_DOC,
)
async def create_job(
    request: Request,
    session: Session = Depends(get_session),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> JobRead:
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip()

    if content_type == "multipart/form-data":
        form = await request.form()
        upload = form.get("file")
        if not isinstance(upload, StarletteUploadFile):
            raise BadInput("multipart body must include a 'file' part")
        options = _parse_options(form.get("options"))
        job = await service.create_upload_job(session, upload, options, idempotency_key)

    elif content_type == "application/json":
        try:
            payload = CreateJobRequest(**await request.json())
        except BadInput:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BadInput(str(exc)) from exc
        job = service.create_url_job(session, str(payload.url), payload.options, idempotency_key)

    else:
        raise BadInput(
            f"unsupported content-type '{content_type or 'none'}'",
            hint="Send application/json with a url, or multipart/form-data with a file",
        )

    return JobRead.from_job(job, settings.api_prefix)


@router.get("", response_model=JobList, summary="List jobs, newest first")
def list_jobs(
    session: Session = Depends(get_session),
    limit: int = Query(25, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> JobList:
    rows, total = service.list_jobs(session, limit, offset)
    return JobList(
        items=[JobRead.from_job(j, settings.api_prefix) for j in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{job_id}", response_model=JobRead, summary="Job status")
def get_job(job_id: str, session: Session = Depends(get_session)) -> JobRead:
    return JobRead.from_job(service.get_job(session, job_id), settings.api_prefix)


@router.get("/{job_id}/events", summary="Live progress (SSE)")
async def job_events(
    job_id: str,
    session: Session = Depends(get_session),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    after: int = Query(0, ge=0, description="Resume cursor; Last-Event-ID wins if both are sent"),
) -> EventSourceResponse:
    service.get_job(session, job_id)  # 404 before opening a stream
    cursor = int(last_event_id) if (last_event_id or "").isdigit() else after
    return EventSourceResponse(stream(job_id, cursor), ping=15)


@router.get("/{job_id}/project", summary="The editable de-edit result (project.json)")
def get_project(job_id: str, session: Session = Depends(get_session)) -> dict:
    job = service.get_job(session, job_id)
    workspace = Workspace.for_job(job.id)
    if not workspace.project_json.exists():
        raise NotFound(
            "project.json is not ready yet",
            hint=f"Job status is '{job.status.value}'; poll {settings.api_prefix}/jobs/{job.id}",
        )
    return json.loads(workspace.project_json.read_text("utf-8"))


@router.get("/{job_id}/files/{path:path}", summary="Fetch a job artifact")
def get_artifact(job_id: str, path: str, session: Session = Depends(get_session)) -> FileResponse:
    job = service.get_job(session, job_id)
    resolved = Workspace.for_job(job.id).resolve_public(path)
    return FileResponse(resolved, filename=resolved.name)


@router.post("/{job_id}/cancel", response_model=JobRead, summary="Cancel a running job")
def cancel_job(job_id: str, session: Session = Depends(get_session)) -> JobRead:
    return JobRead.from_job(service.cancel_job(session, job_id), settings.api_prefix)


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete job + artifacts",
)
def delete_job(job_id: str, session: Session = Depends(get_session)) -> Response:
    service.delete_job(session, job_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
