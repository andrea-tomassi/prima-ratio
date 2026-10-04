# prima-ratio 2.0.2 — turn-key System One decision service.
#
# Engine: Clef-Flash nf4 (Qwen3.5-9B backbone + joint schema head + vision tower),
# served by torch + bitsandbytes. The pip torch wheels bundle their own CUDA
# runtime libraries, so a lean Ubuntu base is enough — the NVIDIA container
# runtime injects the driver at run time (`--gpus all`).
FROM ubuntu:24.04

# flash-attention 2 ships a prebuilt wheel for this exact torch/cu126/cp312 combo
# on x86_64 only — arm64 falls back to sdpa (the server detects it at load time).
ARG TARGETARCH=amd64

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE /app/
COPY prima_ratio /app/prima_ratio

# torch first (its own index), then flash-attention 2 (prebuilt wheel for this
# exact torch/cu126/cp312 combo — see the repo's releases), then the package.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126 \
    && pip install --no-cache-dir einops ninja \
    && if [ "$TARGETARCH" = "amd64" ]; then pip install --no-cache-dir "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.10.0/flash_attn-2.6.3%2Bcu126torch2.14-cp312-cp312-linux_x86_64.whl"; fi \
    && pip install --no-cache-dir .

# gcc: bitsandbytes compiles its nf4 kernels through triton at first use
# (runtime need — a late layer keeps the pip layers cached).
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

# The model (~8 GB, nf4 + joint head) is downloaded on first start into the cache:
# mount /cache and it survives restarts and image upgrades.
ENV PRIMA_CACHE=/cache \
    HF_HOME=/cache/huggingface \
    TRITON_CACHE_DIR=/cache/triton \
    PRIMA_PORT=8000 \
    PRIMA_MODEL_NAME=prima-ratio-clef-flash

VOLUME ["/cache"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"

CMD ["python", "-m", "prima_ratio"]
