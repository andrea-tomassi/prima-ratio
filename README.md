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
- The shipped manifest includes a toy `:test` scenario so the
  [model-suffix mechanism](#scenario-calibration-via-model-suffixes) works out of the box.
- **CPU-only / other accelerators**: not packaged yet. The scoring core is
  language-model-agnostic (SemIf also runs llama.cpp GGUF and MLX backends) —
  a CPU build-arg variant is on the roadmap.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/v1/models` | model catalog **including scenario suffixes** + usage docs |
| `POST` | `/v1/systemone` | typed decisions: `{state, model, questions{id:{type,instructions,criteria}}}` |
| `POST` | `/v1/chat/completions` | OpenAI-shaped chat on the same in-memory model (`"thinking": false` to skip reasoning) |

## Scenario calibration via model suffixes

The API stays 100% standard: scenarios are **model suffixes** (OpenRouter-style).
A temperature is fitted per workload and applied as `softmax(logits/T)` — the
argmax never changes, only the confidence.

```bash
# raw logits (uncalibrated)
curl http://localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Entra ID account record: displayName='"'"'svc.noreply'"'"', userPrincipalName='"'"'svc.noreply@example.com'"'"'.",
  "model": "semif-qwen3.5-4b",
  "questions": {"tipo": {"type": "choice", "instructions": "Human or service account?",
    "criteria": {"human": "Real person", "service_account": "Non-human identity"}}}
}'

# calibrated with the scenario temperature
#   ... "model": "semif-qwen3.5-4b:user-classification",

# raw, explicit
#   ... "model": "semif-qwen3.5-4b:vanilla",
```

- unknown suffix → `422` with the list of available scenarios
- every response carries `x_semif`: timing, score mode (`shared`/`direct`),
  applied temperature and an honest `probability_status`
- shared-state scoring falls back to per-row direct scoring automatically when
  the tokenized state prefix is not stable (BPE boundary effects)

## Run

```bash
docker compose up -d          # needs NVIDIA container toolkit; weights download on first start
# or local:
uv venv && uv pip install -e '.[test]' fastapi uvicorn flash-linear-attention
python api_server.py          # binds 0.0.0.0:8000
```

The model weights live in a mounted HF cache (see `docker-compose.yml`) — they are
never baked into the image.

## Add a calibrated scenario

1. Label **30–60 examples** for the new workload (SemIf JSONL: `state`, `question`,
   `options`, `label` = winning option index) — see `calibration/examples/`
2. Score them: `semif-score --mode direct --model Qwen/Qwen3.5-4B --revision <pin> --input yours.jsonl --output preds.jsonl`
3. Fit: `python benchmarks/calibrate.py --gold yours.jsonl --predictions preds.jsonl --report build/<name>.json`
   (the report includes group-disjoint out-of-fold ECE — honest, auditable)
4. Add the entry to `build/calibration-manifest.json` and restart

Temperature scaling is a single scalar: it fixes global confidence, not ranking
or accuracy. When accuracy itself must move, fine-tune the base model instead
(see the sibling projects for two different approaches to that).

## Layout

```
api_server.py                     the System One + chat wrapper (this is the product)
build/calibration-manifest.json   committed EXAMPLE manifest (scenario "test")
calibration/examples/             toy labeled set demonstrating the pipeline
calibration/*.[jsonl|json]        YOUR real calibration data — gitignored, stays local
```

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
