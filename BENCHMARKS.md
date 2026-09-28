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
| Jev (`typesafe/jev-1.13`, live) | closed service | **0.9630** | 0.9722 | measured — `benchmarks/jev_bench.py` |

Per-family balanced accuracy (this repo): `candidate_selection` 0.944 ·
`evidence_interpretation` 0.956 · `rule_application` 0.913.

Reading: a 12B GGUF on a single 16 GB card lands **+12.5 points over the pinned
4B** and within ~2 points of a 27B exl3 bridge running on much larger hardware.
Jev — a closed frontier service — still leads the semantic axis (0.963).

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
| Jev (`typesafe/jev-1.13`, live) | closed service | 0.8095 | 148 |

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
| Jev re-run live *(check)* | 0.8831 | 0.1245 |

The harness reproduces the published Jev and committed-4B numbers exactly
before reporting ours, so the axis is verified. As a further check the Jev rows
were **re-run live** against the public endpoint: aggregate metrics match the
published values (0.8831 agreement) and the per-row argmaxes reproduce
**102/102** of the published outputs. Reading: on **modal agreement**
this deployment sits between the published Jev point and the frontier models;
on **TV distance** the published distributions are tighter — a 12B open model
on a 16 GB card reproduces the public decision patterns, with more
distributional slack (which per-workload calibration is designed to close).
All 102 rows, ~2.5 s/row (each row is a distinct state — no prefix reuse
possible).

## 4. `typed-decisions` — public leaderboard (LocalLLaMA, 400 cases / 2,000 decisions)

[HuggingFace `LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
— a public benchmark for typed probabilistic decisions whose rows are **exactly
a `POST /v1/systemone` body**. Generalist zero-shot entry (never saw these
workflows or question schemas).

| System | Mode | Acc | Soft | F1 | KL | TV | Brier | ECE | ScoreMAE | Within-1 | ms/case |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Prior (ignores input) | reference | 0.470 | 0.430 | 0.207 | 0.347 | 0.317 | 0.189 | **0.088** | – | – | 0 |
| **semif-server + Gemma4-12B (raw)** | **generalist, zero-shot** | **0.702** | 0.568 | 0.520 | 4.927 | 0.414 | 0.389 | 0.276 | 0.513 | 0.889 | **691** |
| semif-server + Gemma4-12B (+ global T on own data) | generalist + calibration | 0.702 | 0.568 | 0.520 | 3.660 | 0.351 | 0.288 | 0.199 | 0.418 | 0.935 | 691 |
| semif-server + Gemma4-12B (+ global T on the train split) | generalist + calibration | 0.702 | 0.568 | 0.520 | 3.351 | **0.278** | **0.162** | **0.089** | 0.418 | 0.935 | 691 |
| TypeSafe Jev 1.13.0 | generalist, zero-shot | 0.727 | 0.580 | 0.613 | 1.442 | 0.251 | 0.148 | 0.144 | 0.391 | 0.952 | 710 |
| meraGPT Decider 1 | generalist, zero-shot | **0.768** | **0.608** | **0.641** | **0.096** | **0.149** | **0.052** | 0.180 | **0.219** | **0.984** | 526 |

Reading:

- **Accuracy 0.702 zero-shot** lands between the prior (0.470) and Jev (0.727),
  essentially on the benchmark's *"perfect scenario understanding"* ceiling
  (0.704) and below the teacher self-agreement ceiling (0.735) — i.e. a 12B
  GGUF on one 16 GB card reads these four unseen workflows about as well as
  the labels allow.
- **Raw confidence is overconfident** (KL 4.93): the option logits are sharp,
  the gold is a three-sample teacher average. A **single global temperature**
  — no per-workflow fit, no label-space knowledge — moves TV/Brier/ECE to
  Jev-level (TV 0.278, Brier 0.162, ECE 0.089) with accuracy untouched
  (temperature preserves the argmax). That is this project's thesis in one
  row: the model's decisions are right, its confidence needs one scalar.
- The remaining KL gap sits in the tails: a few decisions where the model
  strongly disagrees with the teacher pin the unbounded KL term.
- **Latency: 691 ms/case** on a local 4060 Ti (Jev 710 ms, meraGPT 526 ms) at
  zero per-call cost.

Protocol notes: metrics were implemented to reproduce the published Jev row —
8 of 9 reproduce within noise (Acc 0.733 vs 0.727, KL 1.440 vs 1.442, F1 0.614
vs 0.613, TV/Brier/ScoreMAE/Within-1 equivalent); the reference ECE definition
is unpublished, so ECE is reported as standard top-label ECE.

Harness: `benchmarks/typed_decisions_bench.py` (replays the parquet test split
against any System One endpoint, captures per-case latency).

---

## Operations (same deployment)

- decision latency: **127–144 ms** per decision locally · Jev via the public
  endpoint: p50 **358–380 ms** per call (network included), ~$0.00002–0.00017/call
  (all Jev rows in this document cost ~$0.085 in total)
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
| `jev_bench.py` | runs any fixture through Jev (OpenRouter alpha `decisions`) capturing latency/tokens/cost per call |
| `typed_decisions_bench.py` | replays the public `LocalLLaMA/typed-decisions` test split against any System One endpoint and computes leaderboard metrics |

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
- Jev rows were measured live against the public OpenRouter alpha endpoint
  (`typesafe/jev-1.13`, 2026-09-28): a moving service — numbers are stamped to
  that run. The endpoint's published values were reproduced before comparison.

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
