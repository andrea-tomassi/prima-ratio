# Running prima-ratio

## Docker (recommended)

```bash
docker run -d --name prima-ratio --gpus all \
  -p 8000:8000 \
  -v prima-cache:/cache \
  ghcr.io/andrea-tomassi/prima-ratio:latest
```

First start downloads the engine (~8 GB: nf4 weights + joint head + vision)
into `/cache`. Keep the cache mounted: restarts and image upgrades reuse it.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `PRIMA_MODEL_NAME` | `prima-ratio-clef-flash` | model id served by `/v1/models` |
| `PRIMA_MODEL_REPO` | `meossistant/clef-flash-4bit` | weights to download on first start |
| `PRIMA_MODEL_PATH` | — | explicit local model directory (skips the cache/download) |
| `PRIMA_CACHE` | `/cache` | model + HF cache root — mount it |
| `PRIMA_DEVICE` | `cuda` | torch device |
| `PRIMA_MAX_LENGTH` | `131072` | decision context, in tokens (model native: 262144) |
| `PRIMA_PREFILL_CHUNK` | `4096` | chunked-prefill slice size — 0 disables it |
| `PRIMA_ATTN` | `auto` | attention backend: `auto`/`flash_attention_2`/`sdpa` |
| `PRIMA_PORT` / `PRIMA_HOST` | `8000` / `0.0.0.0` | HTTP bind |

## Long context

Long states are processed with a **chunked prefill**: the backbone consumes
4K-token slices while the hybrid cache (attention KV + the linear layers'
conv/recurrent states) is carried across them, and the joint head then scores
the whole sequence. Peak memory becomes the slice plus the cache instead of
the full state: **~95K-token decisions fit in ~14.6 GB** on a 16 GB card
(one-shot, without chunking, caps around 20K). Probabilities are invariant to
the slicing — identical to the one-shot path at four decimals.

On a 24 GB card raise `PRIMA_MAX_LENGTH` towards the model's 262144. Requests
that still exceed the card return a clean `507` instead of crashing the server.

## GPU

Any NVIDIA card with ≥ 8 GB VRAM (the model runs in ~8 GB at nf4). The pip
torch wheels bundle their CUDA runtime: no host CUDA toolkit needed, only a
recent driver (`--gpus all`).

## Without Docker

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install .
PRIMA_CACHE=./cache python -m prima_ratio
```
