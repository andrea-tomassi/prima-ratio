# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# builder: install the pinned SemIf stack + server deps into a standalone venv
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pin: bump deliberately (calibration fingerprints are bound to the served revision)
ARG SEMIF_GIT_REV=b9ba007ee4d2928bbab5b1d8bfe9009c3696b6de
RUN uv pip install \
    "semif-phase1 @ git+https://github.com/TheoLeeCJ/SemIf-OpenJev@${SEMIF_GIT_REV}" \
    fastapi uvicorn flash-linear-attention

# ---------------------------------------------------------------------------
# runtime: slim image, weights live in a mounted HF cache (never baked in)
# ---------------------------------------------------------------------------
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/cache/huggingface \
    SEMIF_MANIFEST=/app/build/calibration-manifest.json

WORKDIR /app
COPY api_server.py .
COPY build/calibration-manifest.json ./build/calibration-manifest.json

EXPOSE 8000
CMD ["python", "api_server.py"]
