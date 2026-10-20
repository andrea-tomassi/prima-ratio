# 🌳 prima-ratio

**Typed decisions with a probability for every option, in one Docker — powered by
[Decision 3.0 mini](https://huggingface.co/vllm-sr/d3-mini) on [llama.cpp](https://github.com/ggml-org/llama.cpp).**

Give it a state (text or JSON, optionally with images) and typed questions: pick
one of several options (`choice`), yes / no (`noul`), or a level (`score`). It
answers in one pass and returns a probability for every option — no generation,
no parsing, no retry loops.

- 🎚️ **Calibrated out of the box.** The d3 readout is trained for it: ECE ≈ 0.04
  on the public typed-decisions suite, no temperature tuning.
- 👁️ **Images, natively.** PNG/JPEG/WebP up to 1.6 MP, several per request; every
  question sees them. No captioning step.
- 📏 **40K-token states** on a 16 GB consumer card (`PRIMA_CTX`), vision included.
- ⚡ **~100 ms** for a short three-question decision; ~10 s for a 28K-token state.
- 🧰 **One endpoint** — `POST /v1/systemone` — and a 4 GB model download on first
  start. No Python, no torch, no vLLM in the runtime image.

## Engine

| | |
|---|---|
| Model | Decision 3.0 mini — 4.54B (Qwen3.5-4B + 0.33B vision encoder), Apache-2.0 |
| Weights | `vllm-d3-mini-Q4_K_M.gguf` + `mmproj-…-Q8_0.gguf` (ggml-org conversion) |
| Runtime | llama.cpp, pinned commit with the `vllm-d3` decision support |
| Config | 40,960 ctx · `q8_0` KV cache · Flash Attention · all layers on GPU |

The d3 readout uses **non-causal attention**: the whole state must fit one
micro-batch, so `-b`/`-ub` are pinned to the context size. That is what caps the
context on 16 GB (the compute buffer grows ~170 KB/token). Longer contexts need a
bigger card or a smaller d3 (d3-lite 0.8B, d3-nano 2B).

## Benchmarks (RTX 4060 Ti 16 GB, Q4_K_M)

| Suite | Result |
|---|---|
| a144 (144 authored decisions) | **0.9444** acc (0.9299 balanced) |
| Typed-decisions (400 cases) | **0.727** acc · ECE 0.036 · W1 0.956 |
| typesafe102 | **0.8453** |
| Email triage (47-email clean set) | **AUC 0.949** · 6/17 false alarms at 100% recall · 0.82 s |
| Jev Decision Index 0.3.2 (official text board) | **52.6** |
| Latency (3 questions, short text) | **~100 ms** |
| Latency (28.7K-token state) | **10.8 s** |

## Quick start

```bash
docker run -d --name prima-ratio --gpus all -p 8000:8000 \
  -v prima-cache:/cache \
  ghcr.io/andrea-tomassi/prima-ratio:latest
```

First start downloads ~4.1 GB of weights into `/cache` (keep the volume mounted:
restarts and upgrades reuse it).

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "model": "d3-mini",
  "state": "The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.",
  "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this request?",
             "criteria": {"returns": "Refunds, replacements and damaged deliveries",
                          "billing": "Payments, invoices and charges",
                          "technical": "Product setup and faults"}},
    "receipt": {"type": "noul", "instructions": "Does the customer have a receipt?"},
    "urgency": {"type": "score", "instructions": "How urgent is this request?",
                "criteria": ["Routine", "Soon", "Today"]}
  }
}'
```

Images go in `images` as base64 data URLs (or http(s) URLs); every question of
the request sees them:

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' --data-binary @- <<JSON
{"model":"d3-mini","state":"The customer attached the receipt.",
 "images":["data:image/png;base64,$(base64 -w0 receipt.png)"],
 "questions":{"on_receipt":{"type":"noul","instructions":"Does the receipt list a blender?"}}}
JSON
```

## Endpoints

| Method | Path | |
|---|---|---|
| POST | `/v1/systemone` | typed decisions (choice / noul / score), probabilities per option |
| GET | `/v1/models` | the served model and its limits |
| GET | `/health` | readiness probe |

The server is llama.cpp's own; `PRIMA_EXTRA_ARGS="--api-key <key>"` enables
bearer authentication.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `PRIMA_CACHE` | `/cache` | model cache — mount a volume here |
| `PRIMA_MODEL_REPO` | `ggml-org/vllm-d3-mini-GGUF` | weights repo |
| `PRIMA_MODEL_REVISION` | pinned commit | HF revision |
| `PRIMA_MODEL_FILE` | `vllm-d3-mini-Q4_K_M.gguf` | weights (Q4_K_M benchmarked) |
| `PRIMA_MMPROJ_FILE` | `mmproj-vllm-d3-mini-Q8_0.gguf` | vision projector |
| `PRIMA_CTX` | `40960` | context in tokens (`-b`/`-ub` follow it) |
| `PRIMA_KV_TYPE` | `q8_0` | KV cache type |
| `PRIMA_PORT` | `8000` | HTTP port |
| `PRIMA_EXTRA_ARGS` | — | extra `llama-server` flags |

## Security

🔒 LAN-only by default — put the service behind a reverse proxy with auth for any
external exposure, or set `PRIMA_EXTRA_ARGS="--api-key <key>"`.

## Roadmap

- Longer contexts (bigger GPU / smaller d3 sizes / chunked decision variants)
- Video inputs (the transformers runtime supports them; llama.cpp does not yet)
- vLLM Engine-mode server for batched multi-client serving

## Credits & license

Engine: [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT). Model:
[Decision 3.0 mini](https://huggingface.co/vllm-sr/d3-mini) (Apache-2.0) by vLLM
Semantic Router, GGUF conversion by [ggml-org](https://huggingface.co/ggml-org/vllm-d3-mini-GGUF).
See [NOTICE](NOTICE). The packaging is released under the license in [LICENSE](LICENSE).
