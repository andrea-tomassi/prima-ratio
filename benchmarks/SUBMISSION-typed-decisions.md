# Submission — prima-ratio + 12B (generalist, zero-shot, default calibration)

Suggested discussion title:

> prima-ratio + 12B (generalist, zero-shot, default calibration): 0.702 acc / KL 0.564 / Brier 0.234 / ECE 0.146

---

## Results (official `test` split, 400 cases / 2,000 decisions)

| | Accuracy ↑ | KL from gold ↓ | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|---:|
| Spark-X2.5-4B, original *(Rizzo AI Academy)* | 0.574 | 2.899 | 0.480 | 0.349 |
| Rizzo Flow 4B, fine-tuned *(Rizzo AI Academy)* | 0.648 | 0.452 | 0.205 | 0.112 |
| **prima-ratio + 12B, default calibration\*** | **0.702** | 0.564 | 0.234 | 0.146 |
| TypeSafe Jev 1.13.0 *(published)* | 0.727 | 1.442 | 0.148 | – |
| meraGPT Decider 1 *(published)* | 0.768 | 0.096 | 0.052 | 0.180 |

Full metric set of our run: Soft 0.568 · F1 0.520 · TV 0.312 · ScoreMAE 0.362 ·
Within-1 0.955 · p50 700 ms/case.

## What this is

[prima-ratio](https://github.com/andrea-tomassi/prima-ratio) — a single-model
System One endpoint (typed decisions + chat + vision) running a local GGUF
through llama.cpp on **one RTX 4060 Ti 16 GB**. Rows replay through
`POST /v1/systemone` unchanged.

The entry uses the **default calibration** (`:calibrated`): one shipped scalar
fitted on our own mixed workloads — no benchmark-specific fitting, no
per-workflow tuning — and the model is zero-shot on all four workflows.

\* The fitted scalar comes from several real workloads we use ourselves — the
benchmark's test dataset was never used for calibration.

## Reading

- Accuracy **0.702** lands between Rizzo Flow (0.648) and Jev (0.727),
  essentially on the benchmark's "perfect scenario understanding" ceiling
  (0.704).
- The default calibration carries the number: raw option logits are
  overconfident (KL 4.93); the shipped scalar brings KL to **0.564** with
  accuracy untouched (temperature preserves every argmax).
- Zero per-call cost, ~0.7 s per case locally.

## Protocol

Metrics reproduce the published Jev 1.13.0 row on our harness — 8 of 9 within
noise (Acc 0.733 vs 0.727, KL 1.440 vs 1.442, F1 0.614 vs 0.613). The
reference ECE definition is unpublished, so ECE above is standard top-label
ECE. Harness: `benchmarks/typed_decisions_bench.py` in the
[prima-ratio repository](https://github.com/andrea-tomassi/prima-ratio).
