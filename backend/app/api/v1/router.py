from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import jobs

api_router = APIRouter()
api_router.include_router(jobs.router)
# projects.router (recompose / asset replacement) is added in block I.
