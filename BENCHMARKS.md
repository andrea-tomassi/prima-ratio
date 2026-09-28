# 📊 Benchmarks

Independent, reproducible comparisons of the GGUF deployment
(`ghcr.io/andrea-tomassi/semif-server:gguf`) against the published
[SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) baselines — including the one
public comparison point against **Jev**.

Unless noted otherwise, this deployment's numbers were measured on the **same
box and config**:

> RTX 4060 Ti 16 GB · Gemma4-12B-it `UD-Q6_K_XL` · llama.cpp backend ·
> `q8_0` KV cache + window-sized SWA (`SEMIF_SWA_FULL=0`) ·
> `SEMIF_MAX_TOKENS=150000` · decisions are **last-position option logits**, no
> generation.

---

## Fixtures at a glance

| Fixture | Size | What it measures | Metric used here |
|---|---|---|---|
| `authored144` | 144 labeled rows | semantic decision quality | family-balanced accuracy vs gold |
| `shape777` | 777 decisions (37 records × 21 questions) | systems geometry & numerical parity — **no semantic labels by design** | agreement with the committed pinned-4B majority |
| `typesafe_public_102` | 102 public TypeSafe cases (20 case groups) | replication of publicly published decision distributions | equal-case modal agreement + total-variation distance |

Fixtures and reference outputs live in the SemIf repository
(`benchmarks/data/`, `results/raw/predictions/`, `benchmarks/manifests/`).

**Protocol validation.** Every time a published number is reproduced here, the
harness first re-scores the *reference* systems and checks they match the
published values (Jev 0.8831 / 0.1268; committed 4B 0.8453 / 0.1770;
pinned-4B authored144 0.813). Only then are the new rows reported — the axes
are verified, not assumed.

---

## 1. `authored144` — semantic accuracy (144 labeled rows)

Metric: **family-balanced accuracy** against gold labels (mean of per-family
balanced accuracies), the same protocol the SemIf reference uses.

| Implementation | Model | Balanced acc | Row acc | Source |
|---|---|---|---|---|
| SemIf (pinned reference) | Qwen3.5-4B, bf16 | 0.813 | 0.806 | SemIf `results/phase1-summary.json` |
| SemIf exl3 bridge | Qwen3.8-27B, exl3 5.0bpw | **0.9579** | 0.9583 | SemIf `exl3-bridge/` |
| **semif-server** ⋯ *this repo* | **Gemma4-12B, Q6_K_XL** | **0.9378** | **0.9444** | measured — `benchmarks/authored144_api.py` |

Per-family balanced accuracy (this repo): `candidate_selection` 0.944 ·
`evidence_interpretation` 0.956 · `rule_application` 0.913.

Reading: a 12B GGUF on a single 16 GB card lands **+12.5 points over the pinned
4B** and within ~2 points of a 27B exl3 bridge running on much larger hardware.

## 2. `shape777` — systems geometry & numerical parity (777 decisions)

Fixture: 37 synthetic records (~8 KB each) × 21 binary questions. It carries
**no semantic labels by design** — the manifest states *"systems geometry and
numerical equivalence only"* — so the metric is **agreement with the committed
pinned-4B majority** over its three runs, not accuracy.

| Implementation | Model | Agreement vs 4B majority | Flips |
|---|---|---|---|
| SemIf pinned 4B *(self-consistency across its runs)* | Qwen3.5-4B, bf16 | 0.9910 | 7/777 unstable rows |
| SemIf exl3 bridge | Qwen3.8-27B, exl3 5.0bpw | 0.8443 | 121 |
| **semif-server** ⋯ *this repo* | **Gemma4-12B, Q6_K_XL** | **0.8391** | **125** |
| SemIf control | Qwen3-Reranker-4B | 0.4157 | 362 |
| Jev | closed service | — | no public rows for this fixture |

**Batched run**: 777 decisions in **2.4 minutes** — 37 requests × 21 questions,
each record prefilled once and every question read from the restored prefix
identically (5.8× faster than one-full-prompt-per-decision; same readout
semantics). Cross-run stability of this deployment: **99.1 %** decision
agreement between its f16-KV and q8-KV configurations.

## 3. Jev — the one public comparison point (`typesafe_public_102`)

Jev is a closed service: the only fixture where its outputs are public is the
102-case TypeSafe subset. The protocol rebuilds the frozen fixture from the
**public viewer snapshots** (`evals.typesafe.ai`, `*-cases.js` per workflow),
verifying each parsed snapshot against the hashes recorded in the SemIf
selection manifest, then scores every row and recomputes the published metrics
(equal weight per case across 20 case groups).

| | equal-case modal agreement | TV distance |
|---|---|---|
| published: `opus` | 0.9123 | 0.1013 |
| published: `sol` | 0.9065 | 0.1030 |
| **semif-server + Gemma4-12B** *(this repo)* | **0.8978** | **0.1400** |
| published Jev (`typesafe`) | 0.8831 | 0.1268 |
| SemIf 4B (committed) | 0.8453 | 0.1770 |

The harness reproduces the published Jev and committed-4B numbers exactly
before reporting ours, so the axis is verified. Reading: on **modal agreement**
this deployment sits between the published Jev point and the frontier models;
on **TV distance** the published distributions are tighter — a 12B open model
on a 16 GB card reproduces the public decision patterns, with more
distributional slack (which per-workload calibration is designed to close).
All 102 rows, ~2.5 s/row (each row is a distinct state — no prefix reuse
possible).

---

## Operations (same deployment)

- decision latency: **127–144 ms** per decision (typical state + question)
- chat slots: llama.cpp semantics — `SEMIF_CHAT_TOKENS` total ÷ `SEMIF_PARALLEL`
  slots (1×200K or 2×100K both verified on the 16 GB card)
- vision: the same weights serve image-conditioned decisions **and** image chat
- VRAM peak: **15.3 GB** (150K scoring + 199K chat slot + vision loaded)

## Harnesses (in `benchmarks/`)

| Script | Purpose |
|---|---|
| `authored144_api.py` | scores the 144 labeled rows via `/v1/systemone`, computes balanced accuracy |
| `shape777_grouped.py` | scores 777 decisions as 37 batched requests (prefix reuse) and computes baseline agreement |
| `run_typesafe102.py` | scores the rebuilt `typesafe_public_102` fixture and recomputes modal agreement + TV (validates published numbers first) |
| `probe_ctx_max.py` | pushes chat slots to a target context and reports VRAM headroom/stability |

## Caveats

- GGUF scores are **conditional on the quantized weights** (the checkpoint
  checksum is recorded in every response). KV is `q8_0`: f16-KV and q8-KV runs
  agree on **99.1 %** of shape777 decisions — a handful of borderline flips.
- 4B and 27B baselines run different stacks (torch/CUDA, exllamav3) on different
  machines: published reference points, not controlled ablations.
- `shape777` agreement is **procedural parity with the reference baseline**, not
  semantic quality. `authored144` is the semantic axis.
- `typesafe_public_102` measures **agreement with publicly published outputs**
  on public cases — a reproduction check, not a quality verdict.

## Reproduction

```bash
# server (this repo)
docker run --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/cache/huggingface -v /path/to/models:/models:ro \
  -e SEMIF_GGUF=/models/gemma-4-12b-it-UD-Q6_K_XL.gguf \
  -e SEMIF_MMPROJ=/models/mmproj-F16.gguf \
  -e SEMIF_MAX_TOKENS=150000 \
  -e SEMIF_KV_TYPE_K=q8_0 -e SEMIF_KV_TYPE_V=q8_0 -e SEMIF_SWA_FULL=0 \
  ghcr.io/andrea-tomassi/semif-server:gguf

# harnesses (fixtures come from the SemIf repository)
python benchmarks/authored144_api.py  http://localhost:8000 semif-gemma4-12b
python benchmarks/shape777_grouped.py http://localhost:8000 semif-gemma4-12b
python benchmarks/probe_ctx_max.py 90000     # context/VRAM headroom probe

# Jev comparison: rebuild the frozen fixture from the public viewer snapshots
# (evals.typesafe.ai `*-cases.js` per workflow, hash-verified against the SemIf
# selection manifest), then score it:
python build_typesafe.py --source-dir "$SRC" \
  --selection .../source-selection.jsonl --output /tmp/typesafe102.jsonl
python benchmarks/run_typesafe102.py http://localhost:8000 semif-gemma4-12b
```
