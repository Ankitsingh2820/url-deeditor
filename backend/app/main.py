"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from app.api.v1 import health
from app.api.v1.router import api_router
from app.config import settings
from app.db.session import init_db
from app.errors import AppError, app_error_handler, unhandled_error_handler
from app.worker import dispatch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("lightnote")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    log.info(
        "%s starting | env=%s queue=%s storage=%s",
        settings.app_name, settings.env, settings.queue_mode, settings.storage_dir,
    )
    if settings.queue_mode == "thread":
        # In thread mode this process *is* the worker, so warm the models here.
        # Under celery the worker warms itself and the API stays lightweight.
        from app.warmup import preload_in_background

        preload_in_background()
    yield
    dispatch.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        summary="De-edits short-form video into an editable project (scenes, captions, overlays).",
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Last-Event-ID"],
    )

    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)

    app.include_router(health.router)
    app.include_router(api_router, prefix=settings.api_prefix)

    # Direct artifact serving. StaticFiles handles Range requests, which the
    # <video> element needs for seeking -- the /jobs/{id}/files route is the
    # documented API alias over the same tree.
    app.mount("/media", StaticFiles(directory=settings.storage_dir), name="media")

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {"app": settings.app_name, "docs": "/docs", "api": settings.api_prefix}

    return app


app = create_app()
