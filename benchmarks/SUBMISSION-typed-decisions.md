# Submission — semif-server + Gemma4-12B (generalist, zero-shot)

Suggested discussion title:

> semif-server + Gemma4-12B (generalist, zero-shot): 0.702 acc / KL 4.93 raw — 0.089 ECE after one global temperature

---

## Results (official `test` split, 400 cases / 2,000 decisions)

| Variant | Acc | Soft | F1 | KL | TV | Brier | ECE | ScoreMAE | Within-1 | ms/case |
|---|---|---|---|---|---|---|---|---|---|---|
| raw (zero-shot, uncalibrated) | **0.702** | 0.568 | 0.520 | 4.927 | 0.414 | 0.389 | 0.276 | 0.513 | 0.889 | **691** |
| + one global temperature fitted on **our own** labeled data | 0.702 | 0.568 | 0.520 | 3.660 | 0.351 | 0.288 | 0.199 | 0.418 | 0.935 | 691 |
| + one global temperature fitted on the **train split** | 0.702 | 0.568 | 0.520 | 3.351 | **0.278** | **0.162** | **0.089** | 0.418 | 0.935 | 691 |

Mode: **generalist, zero-shot** for the raw row — the model never saw these four
workflows or any of the twenty question schemas. The calibration rows add a
**single scalar per run** (no per-workflow fitting, no label-space knowledge);
temperature preserves every argmax, which is why accuracy is identical.

## What this is

[semif-server](https://github.com/andrea-tomassi/semif-server) — a single-model
System One endpoint (typed decisions + chat + vision) running a local
Gemma4-12B-it `UD-Q6_K_XL` GGUF through llama.cpp on **one RTX 4060 Ti 16 GB**
(150K scoring context, q8_0 KV + window-sized SWA). Rows replay through
`POST /v1/systemone` unchanged.

## Reading

- Accuracy lands between the prior (0.470) and Jev (0.727), essentially on the
  benchmark's "perfect scenario understanding" ceiling (0.704).
- Raw option logits are **overconfident** against a three-sample teacher gold;
  one global temperature moves TV/Brier/ECE to Jev-level (ECE 0.089) with
  accuracy untouched. The remaining KL gap lives in the tails (a few decisions
  where the model strongly disagrees with the teacher).
- 691 ms/case locally (Jev 710, meraGPT 526) at zero per-call cost.

## Protocol

Metrics were implemented against the published Jev 1.13.0 row before reporting:
Acc 0.733 vs 0.727, KL 1.440 vs 1.442, Macro F1 0.614 vs 0.613, TV/Brier/
ScoreMAE/Within-1 equivalent — 8/9 reproduce within noise. The reference ECE
definition is unpublished, so ECE above is standard top-label ECE (on the same
definition Jev measures 0.031; on the soft-correctness variant closest to the
published 0.144, this run measures 0.135).

Harness and reproduction: `benchmarks/typed_decisions_bench.py` in the
[semif-server repository](https://github.com/andrea-tomassi/semif-server).
