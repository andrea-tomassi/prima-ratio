# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# single-stage: the venv is the payload — no builder/runtime duplication
# (keeps the build well under ~10 GB of transient disk on small hosts)
# ---------------------------------------------------------------------------
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git gcc libc6-dev libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/cache/huggingface \
    SEMIF_MANIFEST=/app/build/calibration-manifest.json

WORKDIR /app

# pin: bump deliberately (calibration fingerprints are bound to the served revision)
ARG SEMIF_GIT_REV=23cf1f39fc9534fe81437200959b6dfc7106e45a
RUN git clone https://github.com/TheoLeeCJ/SemIf-OpenJev.git /tmp/semif \
    && git -C /tmp/semif checkout --quiet ${SEMIF_GIT_REV} \
    && uv pip install --system "/tmp/semif" fastapi uvicorn flash-linear-attention \
    && rm -rf /root/.cache/uv /root/.cache/pip /tmp/semif

COPY api_server.py .
COPY build/calibration-manifest.json ./build/calibration-manifest.json

EXPOSE 8000
CMD ["python", "api_server.py"]
