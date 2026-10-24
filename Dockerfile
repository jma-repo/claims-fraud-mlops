# syntax=docker/dockerfile:1.7
# Scoring API + copilot + dashboard. The image serves the committed artifacts; it never trains.

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock README.md ./
# Runtime dependencies only: no MLflow, no plotting, no test tools.
RUN uv sync --frozen --no-default-groups --no-install-project
COPY src ./src
RUN uv sync --frozen --no-default-groups

FROM python:3.12-slim
# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY src ./src
COPY configs ./configs
COPY sql ./sql
COPY artifacts ./artifacts
COPY site ./site
ENV PATH="/app/.venv/bin:$PATH" \
    FRAUDOPS_REPO=/app \
    FRAUDOPS_HOME=/app \
    PYTHONUNBUFFERED=1 \
    LLM_BASE_URL=http://host.docker.internal:1234/v1
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status != 200)"]
CMD ["uvicorn", "fraudops.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
