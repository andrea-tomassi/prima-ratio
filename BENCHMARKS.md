# 📊 Benchmarks

Results of the current engine: **Clef-Flash nf4** (Qwen3.5-9B backbone + joint
schema head, vendored — see [VENDORED.md](VENDORED.md)), served by this repo's
Docker image (`POST /v1/systemone`, one forward pass per request, no
generation).

| Suite | Result |
|---|---|
| Bespoke-Nimble (13 subsets, 3,880 records) | **76.6 % macro / 77.6 % micro** |
| Typed-decisions (400 cases) | **Acc 0.70 · ECE 0.018 · W1 0.97** |
| Email triage (195 real emails, binary malice) | **AUC 0.993–0.994 · 1.8–2.5 % false alarms at 100 % recall** — stack-dependent, see below |
| Mermaid syntax (100 diagrams) | **74 %** |
| Decision latency | **~0.3 s** per record, one forward pass |

## Email triage — the serving stack is part of the operating point

195 real emails (GLM-reviewed ground truth; 32 malicious / 163 non-malicious),
canonical 5-question prompt, vision state (forensic record text + rendered
preview):

| stack | AUC | recall-100 border | FP at 100 % recall |
|---|---|---|---|
| torch nf4 + flash-attention (test server) | 0.9937 | 0.8963 | 1.8 % |
| **torch nf4 + sdpa (product container)** | **0.9928** | **0.8988** | **2.5 %** (3.1 % with the 0.89 belt) |
| llama.cpp Q4_K_M (text-only) | 0.9933 | 0.9013 | 1.8 % |

Same model, same prompt, different backends: mean score difference 0.006
between two nf4 servers, 0.077 between text and vision, 0.098 between
quantizations. Product thresholds are therefore derived **per stack** — on the
product container the borders are 0.89 (alert) / 0.75 (review).

## Reproduction

Harnesses live in `benchmarks/` (fixtures come from the SemIf repository, as
documented per harness). Point them at a local server and pass the model
explicitly (`--model prima-ratio-clef-flash`).
