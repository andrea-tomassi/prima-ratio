# 🧭 Other ways to run semif-server

The [Quick Start](README.md) is the base way: one `docker run`, done.
This file collects everything else.

## 🔧 From source

```bash
git clone https://github.com/andrea-tomassi/semif-server && cd semif-server
docker compose up -d          # builds and binds :8000
curl http://localhost:8000/v1/models
```

## 📦 The GGUF model file

llama.cpp does the forward pass on the model's GGUF file — scoring **and** chat
share the loaded weights:

```bash
docker run -d --gpus all -p 8000:8000 --restart unless-stopped \
  -v ~/.cache/huggingface:/cache/huggingface \
  -v /path/to/models:/models:ro \
  -e SEMIF_GGUF=/models/gemma-4-12b-it-UD-Q6_K_XL.gguf \
  ghcr.io/andrea-tomassi/semif-server:gguf
```

At startup the image verifies that the mounted checkpoint is the expected
model — a mismatch stops the server instead of serving quietly wrong
decisions. Full GPU offload is already the default; set
`SEMIF_LLAMA_GPU_LAYERS=0` for CPU-only runs.

## 👁️ Vision — images in chat and decisions

Mount the projector:

```bash
  -v /path/to/models:/models:ro \
  -e SEMIF_MMPROJ=/models/mmproj-F16.gguf
```

Chat understands images through OpenAI-style content parts (base64 data URLs):

```json
{"messages": [{"role": "user", "content": [
  {"type": "text", "text": "What does the image say?"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]}]}
```

`GET /v1/models` reports `"vision": true` on the chat entry when the projector
is loaded.

The same images work in **decisions** (GGUF backend only): give `state` as
content parts and the option logits are read **conditioned on the image** — no
caption step in between:

```json
{"state": [{"type": "text", "text": "Inspect the receipt."},
           {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}],
 "model": "semif-gemma4-12b",
 "questions": {"over": {"type": "choice", "instructions": "Is the total over 100?",
                          "criteria": {"yes": "over 100", "no": "not over 100"}}}}
```

Multiple questions in one request share the image. Calibration works the same
way: labeled rows with image states fit a temperature for the visual workload.
The captcha demo in the [README](README.md) shows it end to end.

## 🧠 Long contexts on a small card

The context is configurable — `SEMIF_MAX_TOKENS` for decisions,
`SEMIF_CHAT_TOKENS` for chat. A 150K-token scoring context fits a 16 GB GPU
alongside the chat slots and vision (verified; retrieval correct at 27K
depth). The settings for that reference point:

```bash
  -e SEMIF_MAX_TOKENS=150000 \
  -e SEMIF_KV_TYPE_K=q8_0 -e SEMIF_KV_TYPE_V=q8_0 \
  -e SEMIF_SWA_FULL=0
```

`SEMIF_SWA_FULL=0` switches sliding-window layers to a window-sized cache (the
upstream full-size default makes KV memory grow with the whole context);
`SEMIF_KV_TYPE_*` halves the remaining KV and enables flash attention.

## 🔀 Concurrency

`SEMIF_PARALLEL` (default `1`) sets how many chat/vision generations run at
once on the shared weights — llama.cpp `--parallel` semantics. The total chat
context (`SEMIF_CHAT_TOKENS`, default 180K) is split evenly across the slots:
one slot gets the whole budget, two slots get half each. A request that
exceeds its slot's share is refused with a **400 that explains the budget**
(never silently truncated). Decisions are single 130 ms forwards and stay
serialized; vision generation queues behind the projector.

## 🏗️ Which GPUs? — custom CUDA builds

The GGUF image compiles llama.cpp's CUDA kernels for **Turing → Blackwell in
one build** (default: `75;80;86;89;90;120` + a `120` PTX target for forward
compatibility — RTX 20/30/40/50, A100, H100 all run native SASS). Rebuild with
a narrower list for a faster build and a smaller image, or extend it for other
targets:

```bash
docker build -f Dockerfile.gguf --build-arg CUDA_ARCHS="86;89" -t my/semif-server:gguf .
```

| GPU family | `CUDA_ARCHS` |
|---|---|
| Turing (RTX 20, T4) | `75` |
| Ampere (RTX 30, A100) | `86` / `80` |
| Ada (RTX 40, L4) | `89` |
| Hopper (H100) | `90` |
| Blackwell (RTX 50) | `120` |
| future GPUs | `120-virtual` (PTX, JIT at load) |

Note: **DGX Spark (GB10) is ARM64** — it needs an `arm64` build of this image
(same Dockerfile, built on/for that machine), not the amd64 image.

About 200 MB of SASS per architecture; the default multi-arch build takes
~40–60 min on 12 cores (cap jobs with `--build-arg BUILD_PARALLEL=8` on
low-RAM builders).

## ⚙️ Configuration

All environment variables — including the projector, KV cache types and
long-context knobs used above — are listed in the environment reference of the
[README](README.md).
