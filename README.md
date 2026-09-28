# 🌳 prima-ratio

**Text or images in. Text or typed decisions out.**

Both a calibrated **System One** — like Jev, running locally with no external
APIs — and a very capable chat model with tool-calling support, in a single
16 GB VRAM package.

Point it at a message, a record, or a photo, ask your questions, and read the
answers as **typed decisions with probabilities**, scored straight off the
model's logits — no generation, no parsing, no captioning. Confidence you can
classify, verify and gate on.

It ships as **one Docker image, one engine in VRAM**: pull, run, done — the
image sets itself up on first start, serving **image-conditioned decisions**,
**typed text decisions** (yes/no, multiple-choice, scores) and **normal chat
completions** on the same weights.
API-compatible with TypeSafe's System One / Jev pattern.

Independent project; not affiliated with TypeSafe, Jev or SemIf.

![license](https://img.shields.io/badge/license-MIT-0a0a0a) ![GPU](https://img.shields.io/badge/NVIDIA-≥16GB_VRAM-76b900) ![API](https://img.shields.io/badge/API-System_One_/_OpenAI-0a0a0a)

---

## 💪 Strengths

- 👁️ **Image support, natively.** Send a photo instead of a state string: the
  same typed decisions work on images, probabilities read straight off the
  model — **no captioning step in between**. Nine CAPTCHA squares in one
  request, 2.2 s — and visual workloads calibrate like any other.
- 🔑 **Turn-the-key System One.** Pull, run, ask: typed decisions with
  probabilities from the very first call — no prompt engineering, no answer
  parsing, no JSON repair, no retry loops.
- 🎚️ **Super easy custom calibration.** A few dozen labeled examples and one
  HTTP call re-scale the confidence on *your* workload — with honest
  out-of-fold numbers to back it. No machine-learning expertise required.
- 🏷️ **Calibrated systems served automatically (MLOps).** Every calibrated
  workload becomes its own model name at runtime: create, call, and delete
  variants while the server keeps them all served side by side.
- 🎁 **System One + chat on one VRAM budget.** The same resident model answers
  typed decisions *and* OpenAI-shaped chat with tool calling — decisions for
  your pipelines, conversation for everything else, zero extra memory.
- 💻 **Low-spec friendly.** Runs on a mainstream 16 GB NVIDIA card — no need
  for the latest architectures — and eventually on the CPU as well
  *(roadmap: a 4B engine selected by one flag, for fast and CPU inference)*.

---

## 🚀 Quick Start (NVIDIA GPU)

🐳 **Ready-to-use image** on ghcr — no clone, no build:

```bash
docker run -d --gpus all -p 8000:8000 --restart unless-stopped \
  -v prima-cache:/cache \
  ghcr.io/andrea-tomassi/prima-ratio:latest
curl http://localhost:8000/v1/models
```

- 💾 **Persistent cache** *(optional, strongly recommended)*: mount a folder or
  volume at `/cache` as above — the engine (~12 GB) downloads once, on first
  start, and is reused on every restart. Without it, each new container
  re-downloads.
- ✅ **Requires an NVIDIA GPU with 16 GB** and the
  [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- 🧭 **Other ways to run it** — from source, custom CUDA builds, long contexts:
  **[RUNNING.md](RUNNING.md)**.

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

### 🛠️ Use it

Vision is part of the default engine — just send the image as a content part;
chat and decisions both understand it:

```json
{"state": [{"type": "text", "text": "Inspect the receipt."},
           {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}],
 "model": "prima-ratio-gemma4-12b",
 "questions": {"over": {"type": "choice", "instructions": "Is the total over 100?",
                          "criteria": {"yes": "over 100", "no": "not over 100"}}}}
```

The option logits are read **conditioned on the image** — no caption step in
between. Chat content parts, multiple questions per image and visual-workload
calibration: [RUNNING.md](RUNNING.md).

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
  "model": "prima-ratio-gemma4-12b",
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

From that moment `prima-ratio-gemma4-12b:support-routing` is just another model name:
it shows up in `GET /v1/models` (with its fit numbers) and answers like any
other. Removing it is one DELETE. The numbers live in
`build/calibration-manifest.json` — **back that file up**, it is your
calibration state and is gitignored on purpose.

> 📐 **Worked example**: [`calibration/examples/mermaid-syntax/`](calibration/examples/mermaid-syntax/)
> runs the full flow on a task with verifiable ground truth — *will this
> Mermaid diagram render?* — including how the labels were verified with a
> headless-browser render oracle and the held-out numbers against Jev.

### Out of the box: calibrated → yours → uncalibrated (if you insist)

Three confidence levels, no setup required to get started:

| Model id | Temperature | What it is |
|---|---|---|
| **`:calibrated`** | **3.4** | the built-in default for decisions: **one temperature fitted across mixed workloads and validated held-out — it improves NLL *and* ECE on every tested workload, none worsens** |
| `:your-scenario` | fitted | calibrated on **your** labeled examples (30–60 are plenty) — refines the built-in on your workload |
| `:uncalibrated` | 1.0 | raw option logits — systematically overconfident, **not recommended for decisions** |

The built-in value is model-bound (refit if the base model changes) and tunable
via `PRIMA_CALIBRATED_TEMPERATURE`.

### ⚠️ Three honest caveats

- 🎚️ Calibration adjusts **confidence**, not accuracy — if the model gets the
  answer wrong, no temperature will fix it (that's a job for fine-tuning the
  base model).
- 🔖 A variant is valid for the **exact model revision** it was fitted on —
  after an upgrade, re-calibrate: your labeled examples are the whole cost of
  that.
- 🔎 Scores are **conditional on the exact weights shipped** (their sha256 is
  recorded in every response); expect small numeric differences across builds,
  not different behaviour.

---

## 📊 Benchmarks at a glance

| Fixture | Metric | SemIf 4B | SemIf 27B exl3 | **prima-ratio** *(this repo)* | Jev |
|---|---|---|---|---|---|
| `authored144` — 144 labeled rows | family-balanced accuracy | 0.813 | 0.958 | 0.938 | **0.963** |
| `shape777` — 777 decisions | agreement vs 4B majority | 0.991\* | 0.844 | **0.839** | 0.810 |
| `typesafe_public_102` — 102 public cases | modal agreement | 0.845 | — | **0.898** | 0.883 |
| `typesafe_public_102` — 102 public cases | TV distance | 0.177 | — | 0.140 | **0.127** |
| `typed-decisions` — 400 public cases | accuracy (zero-shot) | — | — | **0.702** | 0.727 |
| `typed-decisions` — 400 public cases | ECE after one global temperature | — | — | **0.089** | 0.144 |

\* the pinned 4B *is* the baseline — 0.991 is its own run-to-run consistency.
Jev rows measured **live** via its public endpoint (358–380 ms/call, ~$0.00002–0.00017)
reproducing its published outputs **102/102** before comparison.

777 decisions in **2.4 min** batched (37 requests × 21 questions, prefix reuse) ·
local decisions at **127–144 ms** · full protocol, caveats and reproduction:
**[BENCHMARKS.md](BENCHMARKS.md)**.

---

## 📦 Releases & Docker image

Prebuilt images on ghcr (public):

```bash
docker pull ghcr.io/andrea-tomassi/prima-ratio:latest
```

| Tag | Content |
|---|---|
| **`latest`** | **the single image** — decisions + chat + vision, full GPU offload |
| `v0.3.0` | **prima-ratio**: single image, single engine; renamed built-ins (`:calibrated` / `:uncalibrated`) |
| pre-0.3.0 | historical builds under the old distribution model |
| `v0.2.2` | multi-arch CUDA (Turing → Blackwell) + chat context pool ([release notes](https://github.com/andrea-tomassi/prima-ratio/releases/tag/v0.2.2)) |
| `v0.2.1` | vision (chat + image-conditioned decisions) + 150K long context ([release notes](https://github.com/andrea-tomassi/prima-ratio/releases/tag/v0.2.1)) |
| `v0.2.0` | first local-model distribution + worked calibration example ([release notes](https://github.com/andrea-tomassi/prima-ratio/releases/tag/v0.2.0)) |
| `v0.1.0` | first release — System One + chat + automated calibration |

Notes and changelogs: [Releases](https://github.com/andrea-tomassi/prima-ratio/releases).
Benchmark comparisons (SemIf 4B / 27B exl3 / Jev): [BENCHMARKS.md](BENCHMARKS.md).

---

## 🗺️ Roadmap

- **More engines, one flag** — `PRIMA_ENGINE` will grow: a 12B engine without
  vision, and 4B engines (with and without vision) for fast and CPU inference.
- **12B on a 12 GB card** — a Q4 QAT engine with optional vision, targeting
  12 GB VRAM.

---

## 🔐 Security

🔒 LAN-only by default — put the service behind a reverse proxy with auth for
any external exposure.

---

## ⚙️ Environment reference

The essentials — context sizes, cache types and slot tuning live in
[RUNNING.md](RUNNING.md).

| Variable | Default | Purpose |
|---|---|---|
| `PRIMA_ENGINE` | `12B_VISION` | engine to serve (12B, 4B, 4B_VISION: roadmap) |
| `PRIMA_CACHE` | `/cache` | engine + tokenizer cache folder — mount it to persist |
| `PRIMA_GGUF` / `PRIMA_MMPROJ` | — | explicit asset paths (override the engine) |
| `PRIMA_MAX_TOKENS` | `4096` | scoring context budget |
| `PRIMA_CHAT_TOKENS` | `180000` | total chat/vision context, split across slots |
| `PRIMA_PARALLEL` | `1` | concurrent chat/vision generation slots |
| `PRIMA_CALIBRATED_TEMPERATURE` | `3.4` | built-in `:calibrated` temperature |
| `PRIMA_MANIFEST` | `build/calibration-manifest.json` | calibration state file (mount it, back it up) |

---

## 🙏 Credits

- [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) — the project that
  inspired the core scoring approach. Thank you.
- [TypeSafe](https://docs.typesafe.ai/api) — the System One API pattern

---

## 📄 License

MIT
