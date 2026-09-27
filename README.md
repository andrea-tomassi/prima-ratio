# semif-server

A local, single-GPU **System One endpoint**: typed decisions with probabilities
(yes/no, multiple-choice, scores) read directly from option logits — plus normal
chat completions — from **one stock model loaded once**.

Built on [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) (direct option-logit
readout, MIT) serving `Qwen/Qwen3.5-4B`. API-compatible with TypeSafe's
System One / Jev pattern — the same shape used by
[Rizzo Flow](https://github.com/Rizzo-AI-Academy/rizzo-flow) and
[Kev](https://github.com/jaredpalmer/kev).

Independent project; not affiliated with TypeSafe, Jev, SemIf or Qwen.

## Quick Start (NVIDIA GPU)

```bash
git clone https://github.com/andrea-tomassi/semif-server && cd semif-server
docker compose up -d          # builds and binds :8000
curl http://localhost:8000/v1/models
```

- **Requires an NVIDIA GPU** (≥ 8 GB VRAM for the 4B bf16 model) and the
  [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- Model weights download once to the mounted HF cache on first start (~8 GB) —
  they are never baked into the image.
- **CPU-only / other accelerators**: not packaged yet. The scoring core is
  language-model-agnostic (SemIf also runs llama.cpp GGUF and MLX backends) —
  a CPU variant is on the roadmap.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/v1/models` | OpenAI-shaped model list; calibrated scenarios appear as suffixed ids |
| `POST` | `/v1/systemone` | typed decisions: `{state, model, questions{id:{type,instructions,criteria}}}` |
| `POST` | `/v1/chat/completions` | OpenAI-shaped chat on the same in-memory model (`"thinking": false` to skip reasoning) |
| `POST` | `/v1/calibrate` | fit + publish a calibrated scenario (see below) |
| `DELETE` | `/v1/calibrate/<scenario>` | remove a calibrated scenario |

## Basic usage

**Typed decisions** — pick a model, describe the state, ask typed questions:

```bash
curl http://localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Entra ID account record: displayName='"'"'Mario Rossi'"'"', userPrincipalName='"'"'m.rossi@example.com'"'"'.",
  "model": "semif-qwen3.5-4b",
  "questions": {"tipo": {"type": "choice", "instructions": "Human or service account?",
    "criteria": {"human": "Real person", "service_account": "Non-human identity"}}}
}'
# → {"answers": {"tipo": {"choice": "human", "probabilities": {...}}}, ...}
```

Question types: `noul` (yes/no probability), `choice` (options + probabilities +
confidence), `score` (ordered levels → weighted score). Unknown model suffix →
`422` with the list of available ones.

**Chat** — same OpenAI shape as always:

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 100, "thinking": false
}'
```

## Calibrated scenarios (model suffixes)

A *scenario* is a workload where the model's confidence has been re-scaled on
your own labeled examples: it becomes a **model suffix** and is used like any
other model.

```bash
# fit + publish in one call (server scores, fits the temperature, hot-reloads)
curl http://localhost:8000/v1/calibrate -H 'Content-Type: application/json' -d '{
  "scenario": "support-routing",
  "dataset": [ {"state": "...", "questions": {"q": {"type": "choice",
               "instructions": "...", "criteria": {...}, "label": 1}}}, ... ],
  "heldout":  [ ... ]
}'

# then use it — the suffix appears in GET /v1/models too
curl http://localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "...", "model": "semif-qwen3.5-4b:support-routing", "questions": { ... }
}'

# remove it
curl -X DELETE http://localhost:8000/v1/calibrate/support-routing
```

- ≥ 10 labeled rows per question type; the response reports temperature, ECE
  (raw and out-of-fold) and accuracy — argmax never changes, only confidence
- hot-reload: no restart; scenarios live in `build/calibration-manifest.json`
  (survives restarts; **keep this file backed up** — it is gitignored)
- the shipped manifest includes a toy `:test` scenario so the mechanism works
  out of the box; temperature scaling fixes confidence, not accuracy — for that,
  fine-tune the base model
- the manual recipe (scorer + `benchmarks/calibrate.py` + held-out methodology)
  lives in the upstream [SemIf repo](https://github.com/TheoLeeCJ/SemIf-OpenJev):
  see `benchmarks/` and `docs/CALIBRATION.md` there

## Security

LAN-only by default — put the service behind a reverse proxy with auth for any
external exposure.

## Credits

- [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) (MIT) — the direct-logit
  scoring method, shared-mode execution and calibration tooling
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0) — the served model
- [TypeSafe](https://docs.typesafe.ai/api) — the System One API pattern
- Sibling projects worth reading:
  [Rizzo Flow](https://github.com/Rizzo-AI-Academy/rizzo-flow) (fine-tuned local Jev),
  [Kev](https://github.com/jaredpalmer/kev) (trainable Jev-like family)

## License

MIT
