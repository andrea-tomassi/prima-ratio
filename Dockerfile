# syntax=docker/dockerfile:1

# ---- builder: llama.cpp with the Decision 3.0 (vllm-sr d3) support ----------
# Pinned: the commit that introduced the vllm-d3 decision model support
# (ggml-org/llama.cpp #30383, merged 2026-10-12).
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS builder
ARG LLAMA_CPP_COMMIT=3bb63cbcb
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone https://github.com/ggml-org/llama.cpp.git \
    && cd llama.cpp \
    && git checkout ${LLAMA_CPP_COMMIT}
RUN cd llama.cpp \
    && cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" \
        -DGGML_NATIVE=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_CURL=OFF \
    && cmake --build build -j"$(nproc)" --target llama-server

# ---- runtime -----------------------------------------------------------------
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl tini libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/llama.cpp/build/bin/llama-server /opt/llama/llama-server
COPY --from=builder /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so.12* /opt/cuda-libs/
COPY --from=builder /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12* /opt/cuda-libs/
COPY --from=builder /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublasLt.so.12* /opt/cuda-libs/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libnccl.so.2* /opt/cuda-libs/
COPY entrypoint.sh /usr/local/bin/prima-ratio-entrypoint
RUN chmod +x /usr/local/bin/prima-ratio-entrypoint

# Benchmarked config on a 16 GB card: Q4_K_M weights, 40,960-token context,
# q8_0 KV cache, Flash Attention. The whole state must fit one micro-batch
# (the d3 readout is non-causal), hence -b/-ub pinned to the context size.
ENV LD_LIBRARY_PATH=/opt/cuda-libs \
    PRIMA_CACHE=/cache \
    PRIMA_MODEL_REPO=ggml-org/vllm-d3-mini-GGUF \
    PRIMA_MODEL_REVISION=e7230e2309c186c80182fbeed6fef70680b20c64 \
    PRIMA_MODEL_FILE=vllm-d3-mini-Q4_K_M.gguf \
    PRIMA_MMPROJ_FILE=mmproj-vllm-d3-mini-Q8_0.gguf \
    PRIMA_CTX=40960 \
    PRIMA_KV_TYPE=q8_0 \
    PRIMA_PORT=8000

VOLUME /cache
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PRIMA_PORT}/health" >/dev/null || exit 1

ENTRYPOINT ["tini", "--", "/usr/local/bin/prima-ratio-entrypoint"]
