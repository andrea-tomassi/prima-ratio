# 📊 Benchmarks

Results of the current engine: **Clef-Flash nf4** (Qwen3.5-9B backbone + joint
schema head, vendored — see [VENDORED.md](VENDORED.md)), served by this repo's
Docker image (`POST /v1/systemone`, one forward pass per request, no
generation).

| Suite | Result |
|---|---|
| Bespoke-Nimble (13 subsets, 3,880 records) | **76.6 % macro / 77.6 % micro** |
| Typed-decisions (400 cases) | **Acc 0.70 · ECE 0.018 · W1 0.97** |
| Decision latency | **~0.3 s** per record, one forward pass |

## Reproduction

Harnesses live in `benchmarks/` (fixtures come from the SemIf repository, as
documented per harness). Point them at a local server and pass the model
explicitly (`--model prima-ratio-clef-flash`).
