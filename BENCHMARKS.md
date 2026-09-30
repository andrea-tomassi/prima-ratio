# 📊 Benchmarks

Independent, reproducible comparisons of the GGUF deployment
(`ghcr.io/andrea-tomassi/prima-ratio:latest`) against the published
[SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) baselines — including the one
public comparison point against **Jev**.

Unless noted otherwise, this deployment's numbers were measured on the **same
box and config**:

> RTX 4060 Ti 16 GB · Gemma4-12B-it `UD-Q6_K_XL` · llama.cpp backend ·
> `q8_0` KV cache + window-sized SWA (`PRIMA_SWA_FULL=0`) ·
> `PRIMA_MAX_TOKENS=150000` · decisions are **last-position option logits**, no
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
| **prima-ratio** ⋯ *this repo* | **Gemma4-12B, Q6_K_XL** | **0.9378** | **0.9444** | measured — `benchmarks/authored144_api.py` |
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
| **prima-ratio** ⋯ *this repo* | **Gemma4-12B, Q6_K_XL** | **0.8391** | **125** |
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
| **prima-ratio + Gemma4-12B** *(this repo)* | **0.8978** | **0.1400** |
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

| | Accuracy ↑ | KL from gold ↓ | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|---:|
| Spark-X2.5-4B, original *(Rizzo AI Academy)* | 0.574 | 2.899 | 0.480 | 0.349 |
| Rizzo Flow 4B, fine-tuned *(Rizzo AI Academy)* | 0.648 | 0.452 | 0.205 | 0.112 |
| **prima-ratio + 12B, default calibration\*** | **0.702** | 0.564 | 0.234 | 0.146 |
| TypeSafe Jev 1.13.0 *(published)* | 0.727 | 1.442 | 0.148 | – |
| meraGPT Decider 1 *(published)* | 0.768 | 0.096 | 0.052 | 0.180 |

Rizzo Flow and Jev rows are those projects' published numbers on this
benchmark; prima-ratio's row was measured live through the production endpoint
(16 GB card). Full metric set of that run: Soft 0.568 · F1 0.520 · TV 0.312 ·
ScoreMAE 0.362 · Within-1 0.955 · p50 700 ms/case.

\* The default calibration was fitted on several real workloads we use
ourselves — the benchmark's test dataset was never used for calibration.

Reading:

- **Accuracy 0.702 zero-shot** lands between Rizzo Flow (0.648) and Jev
  (0.727), essentially on the benchmark's *"perfect scenario understanding"*
  ceiling (0.704) — a 12B GGUF on one 16 GB card reads these four unseen
  workflows about as well as the labels allow.
- **The default calibration is this project's thesis in one number**: raw
  option logits are overconfident (KL 4.93); one shipped scalar (T=3.4 — no
  benchmark knowledge, no per-workflow fitting) brings KL to **0.564**, ahead
  of Jev's published 1.442, with accuracy untouched (temperature preserves
  every argmax).
- Probabilistic shape (Brier 0.234, ECE 0.146) sits next to the fine-tuned 4B
  (0.205 / 0.112); both trail Jev's published Brier 0.148. Zero per-call cost,
  one local process, one 16 GB card.

Protocol notes: metrics were implemented to reproduce the published Jev row —
8 of 9 reproduce within noise (Acc 0.733 vs 0.727, KL 1.440 vs 1.442, F1 0.614
vs 0.613, TV/Brier/ScoreMAE/Within-1 equivalent); the reference ECE definition
is unpublished, so ECE is reported as standard top-label ECE.

Harness: `benchmarks/typed_decisions_bench.py` (replays the parquet test split
against any System One endpoint, captures per-case latency).

## 5. Generic temperature — cross-workload validation

The claim for the built-in `:calibrated` variant: a **single temperature fitted on
some workloads, applied to a different one, never gets worse than raw**. Setup:
NLL/ECE computed on four workloads (decisions only; temperature preserves every
argmax); fit on one, evaluate on the rest.

| Fitted on (T) | `mermaid-fit60` | `mermaid-eval100` | `typed-decisions` | `authored144` |
|---|---|---|---|---|
| `authored144` (2.7) | **−15.9 %** NLL, −0.101 ECE | **−42.2 %**, −0.107 | **−30.3 %**, −0.078 | (fit) |
| `mermaid-fit60` (4.0) | (fit) | −43.3 %, −0.101 | −34.1 %, −0.135 | −4.4 %, −0.019 |
| **shipped mix** (`authored144`+`mermaid-fit60`) → **T = 3.4** | (fit) | **−43.5 %**, −0.113 | **−32.9 %**, −0.111 | (fit) |
| `typed-decisions` (6.7, *sanity*) | −16.0 %, −0.146 | −38.8 %, −0.042 | (fit) | −2.7 %, −0.009 |

Every held-out cell improves **both** NLL and ECE — even fitting on a
completely different domain (`typed-decisions` → `mermaid` −39 %). That is why
`:calibrated` can ship as the default: it is not optimal per workload (per-workload
calibration still wins by up to ~20 % NLL), but it is **never worse than
uncalibrated**.

Shipped value: **T = 3.4** (`PRIMA_CALIBRATED_TEMPERATURE` / `:calibrated`), fitted on the
mixed-workload fit set and validated on the held-out workloads above.

---

## 6. Bespoke-Nimble public suite — cross-domain, human labels

[Bespoke-Nimble `PUBLIC_BENCHMARKS.md`](https://github.com/bespokelabsai/nimble/blob/main/docs/PUBLIC_BENCHMARKS.md)
— 13 datasets converted to this API's exact record shape (`state` + `questions`, one
Noul/Choice/Score per record), 3,880 records with labels produced by people, and the
authors' own runner, validator and scorer. Their published run: Bespoke-Nimble-9B
**74.8% macro / 75.9% micro**, Jev 1.13.0 **76.0% / 77.3%**.

Our run (repo commit `62076b4`): all subsets rebuilt with their converters and verified
**byte-identical** to the committed manifests (`dataset_sha256` + id lists), then scored
through the production endpoint with `:calibrated`. 3,880/3,880 valid answers, zero errors.

| Group | Subsets | prima-ratio macro | Nimble-9B | Jev 1.13.0 | prima-ratio micro | Nimble-9B | Jev 1.13.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **all** | 13 | **77.1%** | 74.8% | 76.0% | **77.7%** | 75.9% | 77.3% |
| choice | 5 | 81.0% | 81.6% | 82.9% | 80.6% | 81.1% | 82.8% |
| noul | 5 | 84.1% | 80.2% | 84.6% | 84.3% | 80.1% | 84.6% |
| score | 3 | **58.8%** | 54.6% | 50.1% | **54.3%** | 51.2% | 45.2% |

Highlights: moderation (Civil Comments 85.7% vs 81.0 / 70.3), answerability
(SQuAD 2 86.3% vs 82.9 / 80.6), summary consistency (86.8% vs 81.2 / 75.7),
rubric MAE (HelpSteer2 0.881 vs 0.967 / 0.962), MASSIVE de-DE 86.6% with an
English↔German gap of **0.3 points** (Nimble-9B: 3.5).

Calibration, same run (raw recovered exactly as `p_raw ∝ p_cal^T`; accuracy is
temperature-invariant): mean ECE **0.114 calibrated vs 0.207 raw**, Brier improves
on 13/13 subsets, and ECE beats both published models on MASSIVE (0.021/0.052),
BoolQ (0.057) and SQuAD 2 (0.058).

- 3,880 decisions in **23.7 min** client-side, median **0.29 s/record** on the 16 GB
  card (Jev's median: 0.66 s/request hosted).
- Same records, same instructions and criteria, same scorer as the published run; the
  comparison is aggregate against their published numbers (no paired McNemar), Jev ran
  "as shipped" and Nimble at T=1.0 in their tables.
- Adapter notes: probabilities arrive rounded to 4 decimals and are renormalized exactly
  as their own `api_probabilities` does; score answers carry `confidence` (product-side
  since this run).
- Two product gaps surfaced and acted on: the 2–16 choice-option cap rejected MASSIVE's
  18 classes (now 2–26, letters A–Z) and score answers lacked `confidence`.

---

## Operations (same deployment)

- decision latency: **127–144 ms** per decision locally · Jev via the public
  endpoint: p50 **358–380 ms** per call (network included), ~$0.00002–0.00017/call
  (all Jev rows in this document cost ~$0.085 in total)
- chat slots: llama.cpp semantics — `PRIMA_CHAT_TOKENS` total ÷ `PRIMA_PARALLEL`
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
| `nimble_public_bench.py` | replays the Bespoke-Nimble public suite (13 subsets, 3,880 human-labeled records) through any System One endpoint with the authors' own validator/scorer; expected numbers in `nimble_public_expected.json` |

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
  -e PRIMA_GGUF=/models/gemma-4-12b-it-UD-Q6_K_XL.gguf \
  -e PRIMA_MMPROJ=/models/mmproj-F16.gguf \
  -e PRIMA_MAX_TOKENS=150000 \
  -e PRIMA_KV_TYPE_K=q8_0 -e PRIMA_KV_TYPE_V=q8_0 -e PRIMA_SWA_FULL=0 \
  ghcr.io/andrea-tomassi/prima-ratio:latest

# harnesses (fixtures come from the SemIf repository)
python benchmarks/authored144_api.py  http://localhost:8000 prima-ratio-gemma4-12b
python benchmarks/shape777_grouped.py http://localhost:8000 prima-ratio-gemma4-12b
python benchmarks/probe_ctx_max.py 90000     # context/VRAM headroom probe

# Jev comparison: rebuild the frozen fixture from the public viewer snapshots
# (evals.typesafe.ai `*-cases.js` per workflow, hash-verified against the SemIf
# selection manifest), then score it:
python build_typesafe.py --source-dir "$SRC" \
  --selection .../source-selection.jsonl --output /tmp/typesafe102.jsonl
python benchmarks/run_typesafe102.py http://localhost:8000 prima-ratio-gemma4-12b

# Bespoke-Nimble public suite (section 6): clone bespokelabsai/nimble at 62076b4,
# rebuild the 13 subsets with their converters (docs/PUBLIC_BENCHMARKS.md) and verify
# them byte-identical against docs/assets/public-benchmarks/subsets/*-manifest.json,
# then score every record through the endpoint (results vs nimble_public_expected.json):
python benchmarks/nimble_public_bench.py \
  --nimble-repo /path/to/nimble --data /path/to/nimble/data/public \
  --output-dir /tmp/nimble-eval --endpoint http://localhost:8000/v1/systemone
```
