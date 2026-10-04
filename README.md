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
same weights. API-compatible with TypeSafe's System One / Jev pattern.

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
| `POST /v1/systemone` | The decision endpoint: `state` (string, JSON, or content parts with images) + typed `questions` → probabilities for every option. |
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

Each answer carries its probability distribution and a normalized-margin
confidence. Images ride along as OpenAI-style content parts (`image_url` with
base64 data URLs) or as a top-level `images` array of data URLs.

## 📊 Benchmarks

Full tables, harnesses and reproduction notes live in
[BENCHMARKS.md](BENCHMARKS.md). Headline results of the current engine
(Clef-Flash nf4, 16 GB card):

| Suite | Result |
|---|---|
| Bespoke-Nimble (13 subsets, 3,880 records) | **76.6 % macro / 77.6 % micro** |
| Typed-decisions (400 cases) | **Acc 0.70 · ECE 0.018 · W1 0.97** |
| Email triage (195 real emails, binary malice) | **AUC 0.994 · 1.8 % false alarms at 100 % recall** |
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

## 📄 License

MIT
