# Vendored code

## `prima_ratio/engine/joint_schema_model.py`

Source: [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash)
(`joint_schema_model.py`, Apache-2.0), kept **verbatim** to guarantee parity
with the reference implementation.

What it provides:
- `encode_record` / `collate_records` — the record encoding: the state, the
  typed questions (choice/score/noul), the option **token spans**, and the
  image/video media handling;
- `JointSchemaHead` (+ `EvidenceRoutingLayer`) — the decision head: it reads the
  backbone's last hidden states, routes evidence over the question/option spans
  and scores every option jointly;
- `load_release_model` — the backbone + head loader (accepts
  `from_pretrained_kwargs`, which is how the nf4 quantization is loaded);
- `systemone` / `systemone_answer` — the SystemOne request/response mapping.

No modifications. The service layer (model caching, HTTP endpoints, image
data-URL decoding) lives in `prima_ratio/clef.py` and `prima_ratio/server.py`;
the System One wire-standard conformance (message-part images, TypeSafe
confidence formulas) is applied in `prima_ratio/standard.py` around the
vendored call.

## History

Up to v1.x the engine was a vendored llama.cpp option-logit scorer adapted from
SemIf (MIT). v2.0.0 replaced it with the Clef joint-head stack: the same
turn-key promise (typed decisions, calibrated probabilities, vision, Jev-shaped
API, one container), a different technology underneath.
