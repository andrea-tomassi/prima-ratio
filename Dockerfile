# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# prima-ratio — one image: decisions + chat + vision on a single GPU.
# Multi-stage: nvcc only in the builder; the runtime carries the CUDA runtime
# libraries plus the finished venv. Engine assets are not baked into the image —
# they download into /cache on first start (mount /cache to keep them).
#
#   docker run --gpus all -p 8000:8000 \
#     -v prima-cache:/cache -v ./build:/app/build \
#     ghcr.io/andrea-tomassi/prima-ratio:latest
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3.10-dev build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# CUDA arch must be pinned explicitly: the build container has no GPU, so the
# "native" detection falls back to a 5.2 PTX target and every kernel is
# JIT-compiled at runtime (~3x slower). Default list covers Turing -> Hopper
# and Blackwell natively (RTX 20/30/40/50, A100, H100) plus a 120 PTX target
# for forward compatibility. Narrow it for a faster build / smaller image.
ARG CUDA_ARCHS="75;80;86;89;90;120;120-virtual"
# GGML_NATIVE=off: CI runners may carry newer instruction sets (e.g. AVX-512) that
# consumer hosts lack — a "native" build crashes with SIGILL there. Build portable.
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} -DGGML_NATIVE=off"
# nvcc is memory-hungry: cap parallel compile jobs on low-RAM builders
ARG BUILD_PARALLEL=8
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_PARALLEL}

RUN uv venv /opt/venv --python python3.10
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml /build/pyproject.toml
COPY LICENSE /build/LICENSE
COPY README.md /build/README.md
COPY prima_ratio /build/prima_ratio
RUN uv pip install --no-cache /build \
    && rm -rf /root/.cache/uv /root/.cache/pip /build

# note: `import llama_cpp` is NOT run at build time — its libraries link against
# libcuda.so.1, which only exists at runtime through the NVIDIA container toolkit.

# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.9.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/cache/huggingface \
    PRIMA_MANIFEST=/app/build/calibration-manifest.json \
    PRIMA_LLAMA_GPU_LAYERS=-1

WORKDIR /app
RUN mkdir -p /cache/engine /cache/huggingface
EXPOSE 8000
CMD ["python", "-m", "prima_ratio"]
