# 🌳 semif-server

A local, single-GPU **System One endpoint**: typed decisions with probabilities
(yes/no, multiple-choice, scores) read directly from option logits — plus normal
chat completions — from **one stock model loaded once**.

Built on [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) (direct option-logit
readout, MIT) serving `Qwen/Qwen3.5-4B`. API-compatible with TypeSafe's
System One / Jev pattern.

Independent project; not affiliated with TypeSafe, Jev, SemIf or Qwen.

![license](https://img.shields.io/badge/license-MIT-0a0a0a) ![GPU](https://img.shields.io/badge/NVIDIA-≥8GB_VRAM-76b900) ![API](https://img.shields.io/badge/API-System_One_/_OpenAI-0a0a0a)

---

## 🎯 Why it's comfortable

One small service does three things people usually glue together:

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
- 🧪 **CPU-only / other accelerators**: not packaged yet. The scoring core is
  language-model-agnostic (SemIf also runs llama.cpp GGUF and MLX backends) —
  a CPU variant is on the roadmap.

---

## 🔌 Endpoints

| Method | Path | Purpose |
|:---:|---|---|
| `GET`  | `/v1/models` | 📋 model catalog, including your calibrated variants |
| `POST` | `/v1/systemone` | 🎯 typed decisions: `{state, model, questions{id:{type,instructions,criteria}}}` |
| `POST` | `/v1/chat/completions` | 💬 OpenAI-shaped chat (`"thinking": false` to skip reasoning) |
| `POST` | `/v1/calibrate` | 🎓 fit + publish a new model variant from labeled examples |
| `DELETE` | `/v1/calibrate/<variant>` | 🗑️ remove a variant |

---

## 💬 Decisions in one call

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

### ⚠️ Two honest caveats

- 🎚️ Calibration adjusts **confidence**, not accuracy — if the model gets the
  answer wrong, no temperature will fix it (that's a job for fine-tuning the
  base model).
- 🔖 A variant is valid for the **exact model revision** it was fitted on —
  after an upgrade, re-calibrate: your labeled examples are the whole cost of
  that.

---

## 🔐 Security

🔒 LAN-only by default — put the service behind a reverse proxy with auth for
any external exposure.

---

## 🙏 Credits

- [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) (MIT) — the direct-logit
  scoring method, shared-mode execution and calibration tooling
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0) — the served model
- [TypeSafe](https://docs.typesafe.ai/api) — the System One API pattern

---

## 📄 License

MIT
