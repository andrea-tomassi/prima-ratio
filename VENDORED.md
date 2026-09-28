# Vendored code — provenance and divergences

`prima_ratio/engine/` contains code adapted from **SemIf**
(https://github.com/TheoLeeCJ/SemIf-OpenJev, MIT), commit
`23cf1f39fc9534fe81437200959b6dfc7106e45a`. See `NOTICE`.

| prima-ratio module | upstream | what we kept | what we changed |
|---|---|---|---|
| `engine/prompts.py` | `src/semif_phase1/core.py` | `LETTERS`, `DIRECT_SYSTEM`, `validate_row`, `direct_messages`, `softmax`, `digest` — byte-identical prompt contract | Torch paths removed (`resolve_device`, `synchronize`, `load_causal_model`) |
| `engine/encoding.py` | `src/semif_phase1/direct.py` | `PROMPT_VERSION` (`direct-options-v1`), `_slot_ids`, `encode_prompt` (single-token slot + boundary verification) | Torch scoring removed |
| `engine/prefix.py` | `src/semif_phase1/shared.py` | `_state_prefix` (shared-state prefix extraction) | Torch shared-scoring removed |
| `engine/llamacpp.py` | `src/semif_phase1/llamacpp_backend.py` | the whole GGUF engine: prompt verification, state save/restore prefix reuse, full-vocabulary last-position readout | three first-class, env-gated additions: `PRIMA_LLAMA_GPU_LAYERS` (offload), `PRIMA_KV_TYPE_K/_V` (quantized KV + auto flash-attention), `PRIMA_SWA_FULL=0` (window-sized SWA cache for long contexts) |

Behavioural guarantee: the prompt version, slot verification and numeric
readout are unchanged — the benchmark numbers in `BENCHMARKS.md` were produced
by this engine and are re-validated after the vendoring.

Not vendored (dropped): `mlx_backend.py`, `reranker.py`, `serial.py`,
`cli.py`, the Torch scoring paths, and the upstream benchmark fixtures.
