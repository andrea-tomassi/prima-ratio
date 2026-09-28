# 🌳 semif-server

**Images in, typed decisions out.** A local, single-GPU **System One + Chat
Completion endpoint** in a single package: the
[SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) engine ships inside the
Docker image (pull, run, done) with **one stock model in VRAM** serving
**image-conditioned decisions** (option logits read straight off the model —
no caption step in between), **typed text decisions** (yes/no, multiple-choice,
scores) and **normal chat completions**.

Built on [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) (direct option-logit
readout, MIT). Two image variants, same API: **`:latest`** (transformers, bf16)
and **`:gguf`** (llama.cpp on a local GGUF file — full GPU offload and vision
projector support included; this is how a 12B Q6_K_XL with image understanding
runs smoothly on a 16 GB card).
API-compatible with TypeSafe's System One / Jev pattern.

Independent project; not affiliated with TypeSafe, Jev, SemIf or Qwen.

![license](https://img.shields.io/badge/license-MIT-0a0a0a) ![GPU](https://img.shields.io/badge/NVIDIA-≥8GB_VRAM-76b900) ![API](https://img.shields.io/badge/API-System_One_/_OpenAI-0a0a0a)

---

## 🎯 Why it's comfortable

One small service does the things people usually glue together:

- 👁️ **Decisions from images.** Mount the vision projector and the same
  option-logit readout works on photos: send an image, ask one or more
  questions, get typed answers with probabilities — **no captioning step in
  between**. Nine CAPTCHA squares in one request, 2.2 s — and visual workloads
  calibrate like any other.
- ⚡ **Typed decisions instead of prompts.** You describe the state and the
  question; the answer is a choice with probabilities — no answer sentence to
  parse, no JSON repair, no retry loops. The model never generates tokens:
  scoring one decision takes about a tenth of a second.
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

🐳 **Ready-to-use image** on ghcr — no clone, no build:

```bash
docker run -d --gpus all -p 8000:8000 --restart unless-stopped \
  -v ~/.cache/huggingface:/cache/huggingface \
  ghcr.io/andrea-tomassi/semif-server:latest
curl http://localhost:8000/v1/models
```

- ✅ **Requires an NVIDIA GPU** (≥ 8 GB VRAM for the 4B bf16 model) and the
  [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- ⬇️ Model weights download once to the mounted HF cache on first start (~8 GB) —
  they are never baked into the image.
- 🧭 **Other ways to run it** — GGUF variant, vision projector, long contexts,
  from source, custom CUDA builds: **[RUNNING.md](RUNNING.md)**.

---

## 👁️ Demo — nine questions, one image, one request

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
  "questions": {"account_type": {"type": "choice", "instructions": "Human or service account?",
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
> headless-browser render oracle and the held-out numbers against Jev.

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
| `latest` | latest stable build — torch backend (bf16) |
| `gguf` | llama.cpp backend — the model as a local GGUF file, full GPU offload, vision |
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
| `SEMIF_MODEL_NAME` | `semif-qwen3.5-4b` / `semif-gemma4-12b` | model-id base for variants |
| `SEMIF_MANIFEST` | `build/calibration-manifest.json` | calibration state file (mount it, back it up) |
| `SEMIF_MAX_TOKENS` | `4096` | scoring context budget (no truncation — rows that don't fit are refused) |

---

## 🙏 Credits

- [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) (MIT) — the direct-logit
  scoring method, shared-mode execution and calibration tooling
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0) — served by the `:latest` image
- [gemma-4-12b-it](https://huggingface.co/unsloth/gemma-4-12b-it) (Gemma Terms of Use) — served by the `:gguf` image
- [TypeSafe](https://docs.typesafe.ai/api) — the System One API pattern

---

## 📄 License

MIT
