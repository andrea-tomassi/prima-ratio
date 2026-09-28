# 🌳 semif-server

A local, single-GPU **System One + Chat Completion endpoint** in a single
package: the [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) engine ships
inside the Docker image (pull, run, done) with **one stock model in VRAM**
serving **typed decisions with probabilities** (yes/no, multiple-choice, scores
read directly from option logits), **image-conditioned decisions** (the same
readout, pointed at a photo — no caption step), **and normal chat
completions**.

Built on [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) (direct option-logit
readout, MIT). Two image variants, same API: **`:latest`** serves
`Qwen/Qwen3.5-4B` bf16 through transformers; **`:gguf`** serves any local
`.gguf` checkpoint through llama.cpp with opt-in full GPU offload and vision
projector support (this is how a 12B Q6_K_XL with image understanding runs
smoothly on a 16 GB card).
API-compatible with TypeSafe's System One / Jev pattern.

Independent project; not affiliated with TypeSafe, Jev, SemIf or Qwen.

![license](https://img.shields.io/badge/license-MIT-0a0a0a) ![GPU](https://img.shields.io/badge/NVIDIA-≥8GB_VRAM-76b900) ![API](https://img.shields.io/badge/API-System_One_/_OpenAI-0a0a0a)

---

## 🎯 Why it's comfortable

One small service does the things people usually glue together:

- ⚡ **Typed decisions instead of prompts.** You describe the state and the
  question; the answer is a choice with probabilities — no answer sentence to
  parse, no JSON repair, no retry loops. The model never generates tokens:
  scoring one decision takes about a tenth of a second.
- 👁️ **Decisions from images.** Mount the vision projector and the same
  option-logit readout works on photos: send an image, ask one or more
  questions, get typed answers with probabilities — **no captioning step in
  between**. Nine CAPTCHA squares in one request, 2.2 s — and visual workloads
  calibrate like any other.
- 🎚️ **Your own confidence.** Feed it a few dozen labeled examples and it
  re-calibrates its probabilities on your workload. Confidence you can put a
  threshold on, with honest out-of-fold numbers to back it.
- 🏷️ **Model variants without MLOps.** Every calibrated workload becomes a
  **model variant**: same API, same server, just a different name after a
  colon. Creating or deleting one is a single HTTP call — no restart, no
  rebuild, no machine-learning expertise required.
- 🎁 **Two APIs, one VRAM footprint.** The ~8 GB that host the decision engine
  also serve a full OpenAI-shaped **chat completions** endpoint — same model,
  same process, zero extra memory. Decisions for your pipelines, a
  conversational fallback for everything else, in one deployment.

---

## 🚀 Quick Start (NVIDIA GPU)

🐳 **Immagine pronta all'uso** su ghcr — niente clone, niente build:

```bash
docker run -d --gpus all -p 8000:8000 --restart unless-stopped \
  -v ~/.cache/huggingface:/cache/huggingface \
  ghcr.io/andrea-tomassi/semif-server:latest
curl http://localhost:8000/v1/models
```

Oppure dal sorgente:

```bash
git clone https://github.com/andrea-tomassi/semif-server && cd semif-server
docker compose up -d          # builds and binds :8000
curl http://localhost:8000/v1/models
```

- ✅ **Requires an NVIDIA GPU** (≥ 8 GB VRAM for the 4B bf16 model) and the
  [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- ⬇️ Model weights download once to the mounted HF cache on first start (~8 GB) —
  they are never baked into the image.
- 🧪 **CPU-only / bigger quantized models**: use the `:gguf` variant below — the
  same image runs without a GPU (point it at a GGUF, keep
  `SEMIF_LLAMA_GPU_LAYERS=0`) and scales to full GPU offload (`-1`) on cards
  that fit the checkpoint.

### 📦 Bigger models — the GGUF variant

Same endpoints, same calibration flow; llama.cpp does the forward pass on a
local GGUF checkpoint (scoring **and** chat share the loaded weights):

```bash
docker run -d --gpus all -p 8000:8000 --restart unless-stopped \
  -v ~/.cache/huggingface:/cache/huggingface \
  -v /path/to/models:/models:ro \
  -e SEMIF_GGUF=/models/your-model.gguf \
  ghcr.io/andrea-tomassi/semif-server:gguf
```

At startup the image checks that the GGUF's vocabulary tokenizes exactly like
the pinned tokenizer (a mismatch stops the server instead of serving quietly
wrong decisions) and records the file's sha256 in every response. GPU offload
is opt-in through the same llama.cpp semantics you already know: `0` = CPU
only, `-1` = all layers, `N` = first N layers.

**Long contexts on a small card** — a 150K-token context on a 16 GB GPU, with
identical decisions to the f16 default (verified) and correct retrieval at 27K
depth:

```bash
  -e SEMIF_MAX_TOKENS=150000 \
  -e SEMIF_KV_TYPE_K=q8_0 -e SEMIF_KV_TYPE_V=q8_0 \
  -e SEMIF_SWA_FULL=0
```

`SEMIF_SWA_FULL=0` switches sliding-window layers to a window-sized cache (the
upstream full-size default makes KV memory grow with the whole context);
`SEMIF_KV_TYPE_*` halves the remaining KV and enables flash attention.

**Vision** — mount the projector and chat understands images:

```bash
  -v /path/to/models:/models:ro \
  -e SEMIF_MMPROJ=/models/mmproj-F16.gguf
```

Then send OpenAI-style content parts with base64 data URLs:

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

#### 👁️ Demo — nine questions, one image, one request

A reCAPTCHA-style challenge (*select all images with traffic lights*), nine
labeled squares, all nine questions in a single request — the image is encoded
once and every option logit is read against it:

| square | P(traffic light) | what it is |
|---|---|---|
| 2 | **0.98** | clear green signal |
| 5 | **0.96** | signals on the overhead arm |
| 8 | **0.79** | hanging yellow signal housing |
| 3 | 0.38 | bicycle + pole — the ambiguous one |
| 1 | 0.13 | signs and poles |
| 4 · 6 · 7 · 9 | 0.001 | clean negatives |

**9/9 against the visual ground truth, 2.2 s for all nine questions.** The
confidence behaves like a human's: sharp where the signal is obvious, hesitant
exactly where a person squints, rock-solid on clean negatives. *(Raw model
output — no visual calibration fitted.)*

![traffic lights captcha demo](assets/vision-traffic-lights.png)

**Which GPUs?** The GGUF image compiles llama.cpp's CUDA kernels for **sm_89
(Ada / RTX 40-series)** by default — the measured-fastest build for the
deployment it was made for. On a different architecture the service still
starts, but the CUDA backend cannot initialise and llama.cpp falls back to CPU;
rebuild for your GPU (one build arg, ~10–25 min):

```bash
docker build -f Dockerfile.gguf --build-arg CUDA_ARCHS=86 -t my/semif-server:gguf .
```

| GPU family | `CUDA_ARCHS` |
|---|---|
| RTX 40 / L4 (Ada) | `89` — the default |
| RTX 30 / A10 (Ampere) | `86` |
| RTX 20 / T4 (Turing) | `75` |
| A100 | `80` |
| H100 | `90` |
| mixed fleet | `"86;89"` (semicolon list) |
| maximum portability | `all-major` (long build, ~+1 GB of SASS) |

The `:latest` (torch) image ships PyTorch kernels for every architecture, so it
runs on any NVIDIA GPU out of the box.

---

## 🔌 Endpoints

| Method | Path | Purpose |
|:---:|---|---|
| `GET`  | `/v1/models` | 📋 model catalog, including your calibrated variants |
| `POST` | `/v1/systemone` | 🎯 typed decisions: `{state, model, questions{id:{type,instructions,criteria}}}` |
| `POST` | `/v1/chat/completions` | 💬 OpenAI-shaped chat (`"thinking": false` to skip reasoning) |
| `POST` | `/v1/calibrate` | 🧪 fit + publish a new model variant from labeled examples |
| `DELETE` | `/v1/calibrate/<variant>` | 🗑️ remove a variant |

---

## 🎯 Decisions in one call

```bash
curl http://localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Entra ID account record: displayName='"'"'Mario Rossi'"'"', userPrincipalName='"'"'m.rossi@example.com'"'"'.",
  "model": "semif-qwen3.5-4b",
  "questions": {"tipo": {"type": "choice", "instructions": "Human or service account?",
    "criteria": {"human": "Real person", "service_account": "Non-human identity"}}}
}'
```

The answer gives you the winning option, the probability of every option and a
confidence score — plus timing and an honest note about whether the
probabilities are calibrated. Ask several questions at once (yes/no, choices,
scores): they share the state, so you pay for reading it only once.

Chat works the same way you always expect:

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100
}'
```

---

## 🏷️ Model variants: your own calibrated confidence

This is the part that makes the server *yours*. A **variant** is the base model
with its confidence re-scaled on a workload you care about — routing tickets,
classifying accounts, judging alerts.

You bring labeled examples (**30–60 are plenty**: the answer you expect for
each case), give the workload a name, and the server does the rest: it scores
your examples, measures how far the model's confidence is from your labels,
computes the correction, validates it on data it hasn't seen, and publishes the
result. **One call, no restart, no machine-learning knowledge.**

```bash
curl http://localhost:8000/v1/calibrate -H 'Content-Type: application/json' -d '{
  "scenario": "support-routing",
  "dataset": [ {"state": "...", "questions": {"q": {"type": "choice",
               "instructions": "...", "criteria": {...}, "label": 1}}}, ... ]
}'
```

From that moment `semif-qwen3.5-4b:support-routing` is just another model name:
it shows up in `GET /v1/models` (with its fit numbers) and answers like any
other. Removing it is one DELETE. The numbers live in
`build/calibration-manifest.json` — **back that file up**, it is your
calibration state and is gitignored on purpose.

> 📐 **Worked example**: [`calibration/examples/mermaid-syntax/`](calibration/examples/mermaid-syntax/)
> runs the full flow on a task with verifiable ground truth — *will this
> Mermaid diagram render?* — including how the labels were verified with a
> headless-browser render oracle and the held-out numbers against a frontier
> SaaS baseline.

### ⚠️ Three honest caveats

- 🎚️ Calibration adjusts **confidence**, not accuracy — if the model gets the
  answer wrong, no temperature will fix it (that's a job for fine-tuning the
  base model).
- 🔖 A variant is valid for the **exact model revision** it was fitted on —
  after an upgrade, re-calibrate: your labeled examples are the whole cost of
  that.
- 🔎 With the GGUF variant, scores are **conditional on the quantized weights**
  (the GGUF checksum is recorded in every response); expect small numeric
  differences from a bf16 run, not different behaviour.

---

## 📦 Releases & Docker image

Prebuilt images on ghcr (public):

```bash
docker pull ghcr.io/andrea-tomassi/semif-server:latest
```

| Tag | Content |
|---|---|
| `latest` | latest stable build — torch backend, Qwen3.5-4B bf16 |
| `gguf` | llama.cpp backend — any local `.gguf`, opt-in full GPU offload |
| `v0.2.1` | vision (chat + image-conditioned decisions) + 150K long context ([release notes](https://github.com/andrea-tomassi/semif-server/releases/tag/v0.2.1)) |
| `v0.2.0` | GGUF variant + worked calibration example ([release notes](https://github.com/andrea-tomassi/semif-server/releases/tag/v0.2.0)) |
| `v0.1.0` | first release — System One + chat + automated calibration |

Notes and changelogs: [Releases](https://github.com/andrea-tomassi/semif-server/releases).

---

## 🔐 Security

🔒 LAN-only by default — put the service behind a reverse proxy with auth for
any external exposure.

---

## ⚙️ Environment reference

| Variable | Default (`latest` / `gguf`) | Purpose |
|---|---|---|
| `SEMIF_BACKEND` | `torch` / `llamacpp` | scoring backend |
| `SEMIF_GGUF` | — / **required** | path to the local `.gguf` checkpoint |
| `SEMIF_MMPROJ` | — | projector `.gguf` — enables vision in chat (base64 data URLs) |
| `SEMIF_LLAMA_GPU_LAYERS` | `0` / `-1` | llama.cpp offload: `0` CPU, `-1` all layers, `N` first N |
| `SEMIF_KV_TYPE_K` / `SEMIF_KV_TYPE_V` | `f16` | KV cache type — `q8_0` halves KV memory and enables flash attention |
| `SEMIF_SWA_FULL` | `1` | `0` = window-sized SWA cache: sliding-window layers stop scaling with the context (needed for 100K+ on consumer GPUs) |
| `SEMIF_MODEL` | `Qwen/Qwen3.5-4B` / `unsloth/gemma-4-12b-it` | HF tokenizer source (template + provenance) |
| `SEMIF_REVISION` | pinned per model | revision recorded in variant fingerprints |
| `SEMIF_MODEL_NAME` | `semif-qwen3.5-4b` / `semif-gemma4-12b` | public model-id base for variants |
| `SEMIF_MANIFEST` | `build/calibration-manifest.json` | calibration state file (mount it, back it up) |
| `SEMIF_MAX_TOKENS` | `4096` | scoring context budget (no truncation — rows that don't fit are refused) |

---

## 🙏 Credits

- [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) (MIT) — the direct-logit
  scoring method, shared-mode execution and calibration tooling
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0) — the served model
- [TypeSafe](https://docs.typesafe.ai/api) — the System One API pattern

---

## 📄 License

MIT
