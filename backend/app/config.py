"""Application settings.

Everything tunable lives here so the pipeline never reads os.environ directly.
Values come from environment variables or a local .env file (see .env.example).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Anchored to the package, not the process CWD: otherwise starting uvicorn
    # from the repo root (or a worker from anywhere) would silently skip the
    # .env and fall back to defaults.
    model_config = SettingsConfigDict(
        env_file=(BACKEND_ROOT / ".env", BACKEND_ROOT.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- app -----------------------------------------------------------
    app_name: str = "LightNote De-Editor"
    env: Literal["dev", "prod"] = "dev"
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

    # ---- storage -------------------------------------------------------
    storage_dir: Path = BACKEND_ROOT / "storage"
    database_url: str = "sqlite:///./lightnote.db"

    # ---- queue ---------------------------------------------------------
    # "thread"  -> in-process ThreadPoolExecutor. No Redis needed; ideal for
    #              local dev on Windows where the Celery prefork pool is broken.
    # "celery"  -> real broker + separate worker process (what docker-compose runs).
    queue_mode: Literal["thread", "celery"] = "thread"
    redis_url: str = "redis://localhost:6379/0"
    thread_workers: int = 2

    # ---- ingest limits (enforced before any expensive work) -------------
    max_upload_mb: int = 200
    max_duration_s: int = 180
    allowed_upload_suffixes: list[str] = [".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"]

    # ---- processing defaults (used from block B onwards) ----------------
    proxy_max_dim: int = 720
    sample_fps: float = 3.0
    removal_method: Literal["opencv", "lama", "mask_blur", "auto"] = "auto"
    job_time_budget_s: int = 900

    # ---- AI ------------------------------------------------------------
    ai_enabled: bool = True
    vlm_provider: Literal["gemini", "claude", "noop"] = "gemini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    anthropic_api_key: str | None = None
    whisper_model: str = "base"

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return settings


settings = get_settings()
