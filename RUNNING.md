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
| `PRIMA_MAX_LENGTH` | `16384` | decision context, in tokens |
| `PRIMA_PORT` / `PRIMA_HOST` | `8000` / `0.0.0.0` | HTTP bind |

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
