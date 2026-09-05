# Python 3.11 on purpose: several media/AI wheels (paddle, some torch extras)
# have no 3.13 builds yet. See plan.md §3.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg is the workhorse for cutting, encoding and muxing.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Dependency layer first so code edits do not invalidate the wheel cache.
COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install "."

COPY app ./app

ENV STORAGE_DIR=/srv/storage \
    DATABASE_URL=sqlite:////srv/data/lightnote.db
RUN mkdir -p /srv/storage /srv/data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
