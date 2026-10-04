# 🌳 prima-ratio

**Text or images in. Typed decisions out.**

A calibrated **System One** — like Jev, running locally with no external APIs —
in a single 16 GB VRAM package. Point it at a message, a record, or a photo,
ask your questions, and read the answers as **typed decisions with
probabilities**, straight off the model: no generation, no parsing, no
captioning. Confidence you can classify, verify and gate on.

It ships as **one Docker image, one engine in VRAM**: pull, run, done — the
image sets itself up on first start and serves **typed text decisions**
(yes/no, multiple-choice, scores) and **image-conditioned decisions** on the
same weights. API-compatible with TypeSafe's System One / Jev pattern
and its open multimodal extension.

Independent project; not affiliated with Cloudflare, TypeSafe, Jev or SemIf.

![license](https://img.shields.io/badge/license-MIT-0a0a0a) ![GPU](https://img.shields.io/badge/NVIDIA-%E2%89%A516GB_VRAM-76b900) ![API](https://img.shields.io/badge/API-System_One-0a0a0a)

---

## 💪 Strengths

- 🔑 **Turn-the-key System One.** Pull, run, ask: typed decisions with
  probabilities from the very first call — no prompt engineering, no answer
  parsing, no JSON repair, no retry loops.
- 👁️ **Image support, natively.** Send a photo instead of a state string: the
  same typed decisions work on images, probabilities read straight off the
  model — **no captioning step in between**.
- 🎚️ **Calibrated out of the box.** Probabilities come from a decision head
  trained for it: top-label ECE ≈ 0.02 on the public typed-decision suite, with
  no temperature tuning required.
- 🧠 **One forward pass.** No sampling, no reasoning tokens, no generation:
  latency is one prefill — a few hundred milliseconds per decision on a 16 GB
  consumer card.
- 📏 **Long context, unlocked.** A chunked prefill feeds the backbone in 4K
  slices while the hybrid cache is carried across them — **~95K-token states
  on a 16 GB card**, with probabilities identical to the one-shot path.
- 🧰 **Decisions only, on purpose.** One endpoint, one contract: if you need a
  chat model, run a chat model.

## 🚀 Quick Start (NVIDIA GPU)

```bash
docker run -d --name prima-ratio --gpus all \
  -p 8000:8000 \
  -v prima-cache:/cache \
  ghcr.io/andrea-tomassi/prima-ratio:latest
```

First start downloads the engine (~8 GB, nf4 + vision) into the mounted cache;
later restarts and image upgrades reuse it. Then:

```bash
curl -s localhost:8000/v1/models
```

## 🔌 Endpoints

| Endpoint | What |
|---|---|
| `POST /v1/systemone` | The decision endpoint: `state` (string, JSON, or chat messages) + typed `questions` → probabilities for every option. Images ride in `images` or as message parts — see *System One conformance* below. |
| `GET /v1/models` | The served model (OpenAI shape, `meta` carries engine info). |
| `GET /healthz` | Liveness + model id. |

## 🎯 Decisions in one call

```bash
curl -s localhost:8000/v1/systemone -H "Content-Type: application/json" -d '{
  "state": "Our checkout started returning errors and orders are blocked.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle the message?",
      "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this?",
      "criteria": ["Can wait", "This week", "Today"]
    },
    "outage": {"type": "noul", "instructions": "Is a service down?"}
  }
}'
```

## 🧭 System One conformance

`POST /v1/systemone` follows the [TypeSafe System One API](https://docs.typesafe.ai)
— the same contract as Jev — extended with the multimodal conventions of the
[OpenJev Multimodal API](https://jev-skills.github.io/openjev-multimodal/api),
the reference cited by the llama.cpp decision-model docs:

- **`state`** — a string, a JSON object or array; a value that is not a string
  is given to the model as JSON text. Message-shaped states are accepted too:
  an array of chat messages, or an object with a `messages` array (metadata is
  preserved).
- **`images`** *(extension)* — an array of `data:image/...;base64,...` data
  URLs (bare base64 strings are accepted too); `image_url` parts inside message
  contents are read the same way (data URLs only). All images are placed before
  the state in the prompt — the `images` field first, then the parts — and the
  image parts are removed from the state.
- **`questions`** — `choice` (2–255 options, `null` descriptions allowed),
  `score` (2–10 ordered levels), `noul` (optional `true`/`false` criteria);
  `instructions` can be a string, an object or an array. All questions are
  decided together, in one forward pass.
- **`answers`** — `noul`: the probability of `true`; `choice` and `score`: the
  winning option or the expected level, the full distribution, and
  `confidence` — the TypeSafe concentration measure:
  `(n·pmax − 1) / (n − 1)` for choices, `1 − spread / evenSpread` for scores
  (see [Confidence](https://docs.typesafe.ai/confidence)). `0` means the
  options are equally likely.
- **Local extensions** travel in extra fields (`x_prima`: latency and version)
  and can be ignored by standard clients.

Decisions conditioned on an image, the standard way:

```bash
curl -s localhost:8000/v1/systemone -H "Content-Type: application/json" -d '{
  "state": "A document uploaded by a customer.",
  "images": ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUg..."],
  "questions": {
    "kind": {
      "type": "choice",
      "instructions": "What kind of document is this?",
      "criteria": {"invoice": null, "receipt": null, "contract": null, "other": null}
    }
  }
}'
```

## 📊 Benchmarks

Full tables, harnesses and reproduction notes live in
[BENCHMARKS.md](BENCHMARKS.md). Headline results of the current engine
(Clef-Flash nf4, 16 GB card):

| Suite | Result |
|---|---|
| Bespoke-Nimble (13 subsets, 3,880 records) | **76.6 % macro / 77.6 % micro** |
| Typed-decisions (400 cases) | **Acc 0.70 · ECE 0.018 · W1 0.97** |
| Email triage (195 real emails, binary malice) | **AUC 0.993–0.994 · 1.8–2.5 % false alarms at 100 % recall** (stack-dependent — see [BENCHMARKS.md](BENCHMARKS.md)) |
| Mermaid syntax (100 diagrams) | **74 %** |
| Decision latency | **~0.3 s** per record, one forward pass |

## 🗺️ Roadmap

- vLLM backend (native vision, faster kernels)
- Optional per-workload calibration variants
- CPU / Apple-silicon execution paths

## 🔐 Security

🔒 LAN-only by default — put the service behind a reverse proxy with auth for
any external exposure.

## 🙏 Credits

- [Clef](https://huggingface.co/Cloudflare/clef-flash) (Cloudflare, Apache-2.0)
  — the decision model and its joint schema head; vendored verbatim, see
  [VENDORED.md](VENDORED.md) and [NOTICE](NOTICE).
- [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) — the project that
  inspired the original engine (v1.x). Thank you.
- [TypeSafe](https://docs.typesafe.ai/api) — the System One API pattern.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — the `/v1/systemone`
  reference implementation; [OpenJev Multimodal](https://jev-skills.github.io/openjev-multimodal/)
  — the multimodal API reference.

## 📄 License

MIT
