# Running prima-ratio

## Docker (recommended)

```bash
docker run -d --name prima-ratio --gpus all \
  -p 8000:8000 \
  -v prima-cache:/cache \
  ghcr.io/andrea-tomassi/prima-ratio:latest
```

First start downloads ~4.1 GB of weights (Q4_K_M + Q8_0 vision projector) into
`/cache`. Keep the cache volume mounted: restarts and image upgrades reuse it.

## GPU

Any NVIDIA card with room for the compute buffer: at the default
`PRIMA_CTX=40960` the container uses **~14.3 GB** on a 16 GB card (RTX 4060 Ti).
Only a recent driver is needed (`--gpus all`) — the CUDA runtime ships in the image.

The d3 readout is non-causal, so the whole state is processed in one micro-batch
(`-b`/`-ub` follow `PRIMA_CTX`): lower `PRIMA_CTX` on smaller cards, raise it on
bigger ones until the buffer no longer fits.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `PRIMA_CACHE` | `/cache` | model cache — mount a volume here |
| `PRIMA_MODEL_REPO` | `ggml-org/vllm-d3-mini-GGUF` | weights repo |
| `PRIMA_MODEL_REVISION` | pinned commit | HF revision |
| `PRIMA_MODEL_FILE` | `vllm-d3-mini-Q4_K_M.gguf` | weights (Q4_K_M benchmarked) |
| `PRIMA_MMPROJ_FILE` | `mmproj-vllm-d3-mini-Q8_0.gguf` | vision projector |
| `PRIMA_CTX` | `40960` | context in tokens (`-b`/`-ub` follow it) |
| `PRIMA_KV_TYPE` | `q8_0` | KV cache type (`f16` for maximum fidelity) |
| `PRIMA_PORT` | `8000` | HTTP port |
| `PRIMA_EXTRA_ARGS` | — | extra `llama-server` flags (e.g. `--api-key KEY`) |

## Health and logs

```bash
docker logs -f prima-ratio
curl -s localhost:8000/health
```

The healthcheck probes `/health`; the first start is slower while the weights
download.

## Without Docker (development)

Build llama.cpp at the pinned commit (see `LLAMA_CPP_COMMIT` in the Dockerfile)
with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89"`, download the two
GGUF files, then run the exact command from `entrypoint.sh`.
