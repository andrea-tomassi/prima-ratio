"""prima-ratio — a local, single-model System One endpoint.

Same request/response shape as Rizzo Flow / Kev / TypeSafe — 100% standard API:
POST /v1/systemone  {state, model, questions{id:{type,instructions,criteria}}}
Scenarios map to MODEL SUFFIXES (OpenRouter-style): the model field carries the variant.

    prima-ratio-gemma4-12b                    raw logits (same as :uncalibrated)
    prima-ratio-gemma4-12b:calibrated         built-in cross-workload temperature (recommended)
    prima-ratio-gemma4-12b:uncalibrated       raw logits, explicit (not recommended)
    prima-ratio-gemma4-12b:your-scenario      softmax(logits/T), T from the calibration manifest

State is prefilled once and all questions are scored in parallel (shared-state
prefix reuse, with per-row fallback when the tokenized prefix is not stable).
`state` may also be a list of OpenAI-style content parts: images (base64 data
URLs) join the decision context through the vision projector and the option
logits are read conditioned on the image (PRIMA_MMPROJ).
Also: POST /v1/chat/completions — chat on the same in-memory weights.
Scenario temperatures live in build/calibration-manifest.json (PRIMA_MANIFEST).

One backend: a local GGUF through llama.cpp (PRIMA_GGUF) — a single model in
VRAM serving decisions, chat and vision. Environment: PRIMA_MODEL /
PRIMA_REVISION / PRIMA_MODEL_NAME / PRIMA_MAX_TOKENS / PRIMA_MMPROJ /
PRIMA_PARALLEL / PRIMA_CHAT_TOKENS / PRIMA_CALIBRATED_TEMPERATURE /
PRIMA_HOST / PRIMA_PORT.
"""
import contextlib
import hashlib
import json
import math
import os
import random
import re
import tempfile
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .engine import llamacpp as engine_backend
from .engine.prompts import LETTERS, validate_row

MODEL_SOURCE = os.environ.get("PRIMA_MODEL", "unsloth/gemma-4-12b-it")
REVISION = os.environ.get("PRIMA_REVISION", "55cdba0740a9765956f49501f689a66b098feda3")
MODEL_BASE = os.environ.get("PRIMA_MODEL_NAME", "prima-ratio-gemma4-12b")
GGUF_PATH = os.environ.get("PRIMA_GGUF")
MMPROJ_PATH = os.environ.get("PRIMA_MMPROJ")
MAX_TOKENS = int(os.environ.get("PRIMA_MAX_TOKENS", "4096"))
CHAT_MODEL = os.environ.get("PRIMA_CHAT_NAME", MODEL_BASE.removeprefix("prima-ratio-") + "-chat")
DEFAULT_TRUE = "Yes. The evidence supports an affirmative answer to the question."
DEFAULT_FALSE = "No. The evidence supports a negative answer to the question."
MANIFEST_PATH = os.environ.get("PRIMA_MANIFEST", "build/calibration-manifest.json")
CALIB_FOLDS = 5
CALIB_SEED = 217
CALIB_MIN_ROWS_PER_TYPE = 10
SCENARIO_RE = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*$")

if not GGUF_PATH or not os.path.isfile(GGUF_PATH):
    raise SystemExit("PRIMA_GGUF must point at a local .gguf file")
from pathlib import Path

model, tokenizer, metadata = engine_backend.load_model(
    MODEL_SOURCE, REVISION, Path(GGUF_PATH), context_tokens=MAX_TOKENS)
_score_direct = engine_backend.score
_score_shared = engine_backend.score_shared


def _softmax(vals, T=1.0):
    m = max(vals)
    e = [math.exp((v - m) / T) for v in vals]
    s = sum(e)
    return [x / s for x in e]


def _load_manifest():
    try:
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"version": 1, "scenarios": {}}


MANIFEST = _load_manifest()
CALIBRATED_TEMPERATURE = float(os.environ.get("PRIMA_CALIBRATED_TEMPERATURE", "3.4"))
BUILTIN_SCENARIOS = {
    "uncalibrated": {
        "temperature": 1.0, "built_in": True,
        "description": ("raw option logits, no temperature scaling — not calibrated: "
                        "systematically overconfident, not recommended for decisions; "
                        "use ':calibrated' or fit your own workload"),
    },
    "calibrated": {
        "temperature": CALIBRATED_TEMPERATURE, "built_in": True,
        "description": ("one temperature fitted across mixed workloads and validated "
                        "held-out (improves every tested workload vs uncalibrated, none "
                        "worsens); per-workload calibration with your own labels refines it"),
    },
}
# legacy built-in names from older releases: ignored when reading manifests and
# reserved against re-creation — NOT accepted as request aliases (use the current names)
LEGACY_BUILTIN_NAMES = {"vanilla", "generic"}

SCENARIOS = {name: entry for name, entry in MANIFEST.get("scenarios", {}).items()
             if name not in BUILTIN_SCENARIOS and name not in LEGACY_BUILTIN_NAMES}

app = FastAPI(title="prima-ratio")

# the one loaded model serves both scoring and chat: serialize all forwards
_GEN_LOCK = threading.Lock()


class Question(BaseModel):
    type: str
    instructions: str = ""
    criteria: Any = None


class Req(BaseModel):
    state: Any
    model: str = MODEL_BASE
    questions: dict[str, Question]


def build_row(qid: str, state: Any, q: Question) -> dict:
    crit = q.criteria or {}
    t = q.type
    if t == "noul":
        options = [
            {"id": "true", "description": crit.get("true") or DEFAULT_TRUE},
            {"id": "false", "description": crit.get("false") or DEFAULT_FALSE},
        ]
    elif t == "choice":
        options = [
            {"id": k, "description": (v if isinstance(v, str) and v.strip() else k)}
            for k, v in crit.items()
        ]
    elif t == "score":
        options = [{"id": str(i), "description": lvl} for i, lvl in enumerate(crit)]
    else:
        raise ValueError(f"unsupported question type: {t}")
    if not options:
        raise ValueError(f"question {qid} has no options")
    return {
        "id": qid,
        "state": state,
        "question": q.instructions.strip() or "Decide based on the state.",
        "options": options,
    }



def _golden_fit(pairs, bounds=(0.05, 20.0), iterations=60):
    """pairs: [(logits, true_index)] — minimize mean NLL over T (golden section; NLL convex in 1/T)."""
    def nll(T):
        tot = 0.0
        for logits, ti in pairs:
            p = _softmax(logits, T)
            tot += -math.log(max(p[ti], 1e-12))
        return tot / len(pairs)
    ratio = (math.sqrt(5) - 1) / 2
    low, high = bounds
    left, right = high - ratio * (high - low), low + ratio * (high - low)
    f_left, f_right = nll(left), nll(right)
    for _ in range(iterations):
        if f_left < f_right:
            high, right, f_right = right, left, f_left
            left = high - ratio * (high - low)
            f_left = nll(left)
        else:
            low, left, f_left = left, right, f_right
            right = low + ratio * (high - low)
            f_right = nll(right)
    return (low + high) / 2


def _ece(items, bins=10):
    """items: [(confidence, correct)] — top-label expected calibration error."""
    if not items:
        return None
    total = 0.0
    for b in range(bins):
        part = [x for x in items if min(bins - 1, int(x[0] * bins)) == b]
        if part:
            total += abs(sum(x[1] for x in part) / len(part) - sum(x[0] for x in part) / len(part)) * len(part) / len(items)
    return round(total, 4)


def _persist_manifest():
    data = {"version": MANIFEST.get("version", 1), "scenarios": SCENARIOS}
    d = os.path.dirname(MANIFEST_PATH) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, MANIFEST_PATH)


@app.get("/v1/models")
def models():
    """OpenAI v1 shape: scenario-calibrated variants are ordinary model ids
    (suffix after ':'); fit metadata rides in the optional 'meta' field."""
    now = int(time.time())
    data = [{"id": MODEL_BASE, "object": "model", "created": now,
             "owned_by": "prima-ratio", "meta": {"variant": "raw logits (uncalibrated default)"}}]
    for name, e in sorted(BUILTIN_SCENARIOS.items()):
        data.append({
            "id": f"{MODEL_BASE}:{name}",
            "object": "model",
            "created": now,
            "owned_by": "prima-ratio",
            "meta": {"temperature": e.get("temperature"), "built_in": True,
                     "description": e.get("description")},
        })
    for name, e in sorted(SCENARIOS.items()):
        data.append({
            "id": f"{MODEL_BASE}:{name}",
            "object": "model",
            "created": now,
            "owned_by": "prima-ratio",
            "meta": {k: e.get(k) for k in ("temperature", "n_fit", "ece_raw", "ece_out_of_fold", "fitted_at", "description") if e.get(k) is not None},
        })
    data.append({"id": CHAT_MODEL, "object": "model", "created": now,
                 "owned_by": "prima-ratio", "meta": {"variant": "normal chat completions (POST /v1/chat/completions)",
                                                         "vision": bool(MMPROJ_PATH),
                                                         "parallel": CHAT_PARALLEL,
                                                         "slot_tokens": CHAT_SLOT_TOKENS or None}})
    return {"object": "list", "data": data}


@app.post("/v1/systemone")
def systemone(req: Req):
    base, _, suffix = req.model.partition(":")
    if suffix and suffix not in SCENARIOS and suffix not in BUILTIN_SCENARIOS:
        raise HTTPException(
            status_code=422,
            detail={
                "error": f"unknown scenario '{suffix}'",
                "available": sorted(list(BUILTIN_SCENARIOS) + list(SCENARIOS)),
                "hint": f'use "{MODEL_BASE}:<scenario>" — see GET /v1/models',
            },
        )
    entry = (BUILTIN_SCENARIOS.get(suffix) or SCENARIOS.get(suffix)) if suffix else None
    scenario_used = suffix or None
    temperature = entry["temperature"] if entry else 1.0

    rows, types, legends = [], {}, {}
    for qid, q in req.questions.items():
        rows.append(build_row(qid, req.state, q))
        types[qid] = q.type
        if q.type == "score":
            legends[qid] = {str(i): lvl for i, lvl in enumerate(q.criteria or [])}
    for r in rows:
        validate_row(r)

    t0 = time.perf_counter()
    with _GEN_LOCK:
        if _state_has_images(req.state):
            results, timing = _gguf_vision_score_rows(rows, MAX_TOKENS)
            score_mode = "vision"
        else:
            try:
                results, timing = _score_shared(model, tokenizer, rows, metadata, MAX_TOKENS)
                score_mode = "shared"
            except ValueError:
                # shared mode requires a stable tokenized state prefix (BPE boundary effects,
                # e.g. a state ending in a quote char); fall back to per-row direct scoring —
                # identical readout, no prefix reuse.
                results = [_score_direct(model, tokenizer, r, metadata, MAX_TOKENS) for r in rows]
                timing = {"mode": "direct-fallback"}
                score_mode = "direct"
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    if entry:
        if entry.get("built_in"):
            status = f"{entry.get('description', 'built-in')} (T={round(temperature, 4)})"
        else:
            status = (
                f"temperature scaled (T={round(temperature, 4)}, scenario '{scenario_used}', "
                f"n_fit={entry.get('n_fit')}, ECE raw->out-of-fold "
                f"{entry.get('ece_raw')}->{entry.get('ece_out_of_fold')}); "
                "requires held-out validation on your own labels"
            )
    else:
        status = ("conditional option score; uncalibrated as decision confidence — use \":calibrated\" "
                  "or fit your own workload with POST /v1/calibrate (see GET /v1/models)")

    answers = {}
    input_tokens = 0
    for r in results:
        qid = r["id"]
        logits = r.get("option_logits")
        probs = _softmax(logits, temperature) if logits else r["probabilities"]
        ids = r["option_ids"]
        input_tokens += r.get("input_tokens", 0)
        t = types[qid]
        if t == "noul":
            answers[qid] = {"type": "noul", "noul": probs[ids.index("true")]}
        elif t == "choice":
            k = len(ids)
            pmax = max(probs)
            conf = 1.0 if k <= 1 else max(0.0, (pmax - 1 / k) / (1 - 1 / k))
            answers[qid] = {
                "type": "choice",
                "choice": ids[probs.index(pmax)],
                "probabilities": {i: round(p, 4) for i, p in zip(ids, probs)},
                "confidence": round(conf, 4),
            }
        else:  # score
            mean = sum(i * p for i, p in enumerate(probs))
            answers[qid] = {
                "type": "score",
                "score": round(mean, 4),
                "legend": legends[qid],
                "probabilities": {i: round(p, 4) for i, p in zip(ids, probs)},
            }
    return {
        "model": req.model,
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        "latency_ms": latency_ms,
        "x_prima": {
            "timing": timing,
            "score_mode": score_mode,
            "scenario": scenario_used,
            "temperature": temperature,
            "probability_status": status,
        },
    }


class ChatReq(BaseModel):
    messages: list[dict]
    model: str = CHAT_MODEL
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 512
    thinking: bool = True
    tools: list[dict] | None = None
    tool_choice: str | None = None


# --- GGUF chat: a pool of generation contexts (model weights shared) ---
CHAT_PARALLEL = max(1, int(os.environ.get("PRIMA_PARALLEL", "1")))
CHAT_TOKENS = int(os.environ.get("PRIMA_CHAT_TOKENS", "180000"))  # total across slots; 0 = unlimited
CHAT_SLOT_TOKENS = CHAT_TOKENS // CHAT_PARALLEL if CHAT_TOKENS else 0
_VISION_LOCK = threading.Lock()
_CHAT_SLOTS: list[dict] = []
_CHAT_SLOTS_LOCK = threading.Lock()
_CHAT_FREE = threading.BoundedSemaphore(CHAT_PARALLEL)


@contextlib.contextmanager
def _chat_context_lease(lib, native_model, need_tokens: int):
    """Exclusive lease on one of the CHAT_PARALLEL generation contexts.

    Contexts share the loaded model weights and each carries its own KV cache,
    so leases run truly concurrently (like llama.cpp `--parallel`). A lease
    grows its context when a request needs more room.
    """
    _CHAT_FREE.acquire()
    slot = None
    try:
        with _CHAT_SLOTS_LOCK:
            for candidate in _CHAT_SLOTS:
                if not candidate["busy"]:
                    candidate["busy"] = True
                    slot = candidate
                    break
            if slot is None:
                slot = {"context": None, "capacity": 0, "busy": True}
                _CHAT_SLOTS.append(slot)
        context = slot["context"]
        if context is None or need_tokens > slot["capacity"]:
            if context is not None:
                lib.llama_free(context)
            params = lib.llama_context_default_params()
            params.n_ctx = max(need_tokens, 2048)
            params.n_seq_max = 1
            params.n_outputs_max = 1
            try:  # vendored llama.cpp patch: KV type / SWA window knobs
                from .engine.llamacpp import _apply_context_env
            except ImportError:
                pass
            else:
                _apply_context_env(lib, params)
            context = lib.llama_init_from_model(native_model, params)
            if not context:
                raise RuntimeError("llama.cpp failed to create a chat context")
            capacity = int(lib.llama_n_ctx(context))
            if need_tokens > capacity:
                lib.llama_free(context)
                raise RuntimeError("chat prompt does not fit the chat context")
            slot["context"] = context
            slot["capacity"] = capacity
        yield slot["context"]
    finally:
        if slot is not None:
            with _CHAT_SLOTS_LOCK:
                slot["busy"] = False
        _CHAT_FREE.release()


def _gguf_decode(lib, context, tokens: list[int], start: int, want_logits: bool):
    """Batch decode on the chat context (mirrors _Engine._decode in the backend)."""
    for offset in range(0, len(tokens), 512):
        chunk = tokens[offset:offset + 512]
        batch = lib.llama_batch_init(len(chunk), 0, 1)
        try:
            for index, token in enumerate(chunk):
                batch.token[index] = token
                batch.pos[index] = start + offset + index
                batch.n_seq_id[index] = 1
                batch.seq_id[index][0] = 0
                batch.logits[index] = int(want_logits and offset + index == len(tokens) - 1)
            batch.n_tokens = len(chunk)
            if lib.llama_decode(context, batch):
                raise RuntimeError("llama_decode failed for the chat context")
        finally:
            lib.llama_batch_free(batch)
    if not want_logits:
        return None
    return _gguf_last_logits(lib, context)


def _gguf_last_logits(lib, context):
    import ctypes

    import numpy

    pointer = lib.llama_get_logits_ith(context, -1)
    if not pointer:
        raise RuntimeError("llama.cpp returned no chat logits")
    return numpy.ctypeslib.as_array(
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_float)), shape=(model.engine.vocab_size,)
    ).copy()


def _sample_token(logits, temperature, top_p, rng):
    import numpy

    if temperature <= 0:
        return int(numpy.argmax(logits))
    values = logits.astype(numpy.float64) / temperature
    values -= values.max()
    probs = numpy.exp(values)
    probs /= probs.sum()
    if 0 < top_p < 1.0:
        order = numpy.argsort(-probs)
        cumulative = numpy.cumsum(probs[order])
        cutoff = int(numpy.searchsorted(cumulative, top_p)) + 1
        keep = order[:cutoff]
        masked = numpy.zeros_like(probs)
        masked[keep] = probs[keep]
        probs = masked / masked.sum()
    return int(rng.choice(len(probs), p=probs))


_GGUF_VISION = {"context": None}


def _gguf_vision_context():
    """Lazy mtmd (vision projector) context bound to the loaded GGUF model."""
    context = _GGUF_VISION["context"]
    if context is not None:
        return context
    if not MMPROJ_PATH or not os.path.isfile(MMPROJ_PATH):
        raise HTTPException(
            status_code=400,
            detail="vision is not enabled on this server (set PRIMA_MMPROJ to a projector .gguf)",
        )
    import llama_cpp.mtmd_cpp as mtmd

    context = mtmd.mtmd_init_from_file(
        MMPROJ_PATH.encode("utf-8"), model.engine.model, mtmd.mtmd_context_params_default())
    if context is None:
        raise HTTPException(status_code=500, detail="failed to load the vision projector")
    _GGUF_VISION["context"] = context
    return context


def _gguf_split_images(messages: list[dict]):
    """Swap image parts for the multimodal marker and collect their bytes.

    Images must arrive as base64 data URLs (OpenAI-style content parts).
    Returns (messages_for_template, [image_bytes, ...]).
    """
    import base64

    import llama_cpp.mtmd_cpp as mtmd

    marker = mtmd.mtmd_default_marker().decode("utf-8")
    converted, blobs = [], []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            converted.append(message)
            continue
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url")
                url = url.get("url") if isinstance(url, dict) else url
                if not isinstance(url, str) or not url.startswith("data:") or ";base64" not in url:
                    raise HTTPException(status_code=400, detail="images must be base64 data URLs")
                try:
                    blobs.append(base64.b64decode(url.partition(",")[2]))
                except Exception as error:
                    raise HTTPException(status_code=400, detail="invalid base64 image payload") from error
                parts.append({"type": "text", "text": marker})
            else:
                parts.append(part)
        converted.append({**message, "content": parts})
    return converted, blobs


_STOP_MARKERS = ("<turn|>", "<end_of_turn>", "<eos>", "</s>")

_TOOL_CALL_RE = re.compile(r"<\|tool_call>(.*?)<tool_call\|>", re.DOTALL)
_QUOTE_TOKEN = '<|"|>'


def _split_gemma_items(body: str) -> list[str]:
    """Split a gemma-rendered argument body on top-level commas."""
    items, current, depth, in_string = [], [], 0, False
    i = 0
    while i < len(body):
        if body.startswith(_QUOTE_TOKEN, i):
            in_string = not in_string
            current.append(_QUOTE_TOKEN)
            i += len(_QUOTE_TOKEN)
            continue
        char = body[i]
        if not in_string:
            if char in "{[":
                depth += 1
            elif char in "}]":
                depth -= 1
            elif char == "," and depth == 0:
                items.append("".join(current))
                current = []
                i += 1
                continue
        current.append(char)
        i += 1
    if current:
        items.append("".join(current))
    return [item.strip() for item in items if item.strip()]


def _parse_gemma_scalar(raw: str):
    raw = raw.strip()
    if raw.startswith(_QUOTE_TOKEN) and raw.endswith(_QUOTE_TOKEN) and len(raw) >= 2 * len(_QUOTE_TOKEN):
        return raw[len(_QUOTE_TOKEN):-len(_QUOTE_TOKEN)]
    if raw.startswith("{") and raw.endswith("}"):
        return _parse_gemma_pairs(raw[1:-1])
    if raw.startswith("[") and raw.endswith("]"):
        return [_parse_gemma_scalar(item) for item in _split_gemma_items(raw[1:-1])]
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _parse_gemma_pairs(body: str) -> dict:
    pairs = {}
    for item in _split_gemma_items(body):
        key, sep, value = item.partition(":")
        if not sep:
            continue
        pairs[key.strip()] = _parse_gemma_scalar(value)
    return pairs


def _parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Extract native gemma tool calls; returns (text before the first call, calls)."""
    calls, first = [], None
    for match in _TOOL_CALL_RE.finditer(text):
        if first is None:
            first = match.start()
        body = match.group(1).strip()
        if not body.startswith("call:"):
            continue
        name, brace, args = body[len("call:"):].partition("{")
        if not brace or not args.endswith("}"):
            continue
        calls.append({
            "id": "call_" + hashlib.sha1(f"{name}{args}{time.time()}".encode()).hexdigest()[:16],
            "type": "function",
            "function": {
                "name": name.strip(),
                "arguments": json.dumps(_parse_gemma_pairs(args[:-1])),
            },
        })
    return (text[:first] if calls else text), calls


def _gguf_vision_prepare(prompt: str, images: list[bytes], add_special: bool):
    """mtmd: bitmaps + input chunks + prompt position count (mtmd context shared).

    Callers hold _VISION_LOCK around this and the matching evaluation.
    Returns (vision, chunks, bitmaps, prompt_tokens).
    """
    import ctypes

    import llama_cpp.mtmd_cpp as mtmd

    vision = _gguf_vision_context()
    bitmaps, chunks = [], None
    try:
        for blob in images:
            buffer = (ctypes.c_uint8 * len(blob)).from_buffer(bytearray(blob))
            bitmap = mtmd.mtmd_helper_bitmap_init_from_buf(vision, buffer, len(blob), False)
            if bitmap is None:
                raise HTTPException(status_code=400, detail="unsupported or corrupt image payload")
            bitmaps.append(bitmap)
        text = mtmd.mtmd_input_text()
        payload = prompt.encode("utf-8")
        text.text = payload
        text.text_len = len(payload)
        text.add_special = add_special
        text.parse_special = True
        chunks = mtmd.mtmd_input_chunks_init()
        if chunks is None:
            raise RuntimeError("mtmd_input_chunks_init returned NULL")
        bitmap_array = (mtmd.mtmd_bitmap_p_ctypes * len(bitmaps))(*bitmaps)
        status = mtmd.mtmd_tokenize(vision, chunks, ctypes.byref(text), bitmap_array, len(bitmaps))
        if status != 0:
            raise HTTPException(status_code=400, detail=f"multimodal tokenization failed (code {status})")
        prompt_tokens = int(mtmd.mtmd_helper_get_n_pos(chunks))
        if prompt_tokens <= 0:
            raise HTTPException(status_code=400, detail="empty multimodal prompt")
    except BaseException:
        _gguf_vision_release(chunks, bitmaps)
        raise
    return vision, chunks, bitmaps, prompt_tokens


def _gguf_vision_release(chunks, bitmaps) -> None:
    import llama_cpp.mtmd_cpp as mtmd

    if chunks is not None:
        mtmd.mtmd_input_chunks_free(chunks)
    for bitmap in bitmaps:
        mtmd.mtmd_bitmap_free(bitmap)


def _check_chat_budget(prompt_tokens: int, max_tokens: int) -> int:
    """Refuse requests above the per-slot share of PRIMA_CHAT_TOKENS (no truncation).

    llama.cpp semantics: PRIMA_CHAT_TOKENS is the TOTAL context and each of the
    PRIMA_PARALLEL slots gets total / parallel (e.g. 180000 with parallel=2 ->
    90000 per slot).
    """
    need_tokens = prompt_tokens + max_tokens + 8
    if CHAT_SLOT_TOKENS and need_tokens > CHAT_SLOT_TOKENS:
        raise HTTPException(status_code=400, detail={
            "error": "chat request exceeds the per-slot context budget",
            "need_tokens": need_tokens,
            "slot_tokens": CHAT_SLOT_TOKENS,
            "total_tokens": CHAT_TOKENS,
            "parallel": CHAT_PARALLEL,
            "hint": "lower max_tokens / shorten the prompt, raise PRIMA_CHAT_TOKENS, or lower PRIMA_PARALLEL",
        })
    return need_tokens


def _gguf_generate(lib, vocab, context, logits, position: int, req: ChatReq, eos_ids, rng):
    """Autoregressive generation on a leased context; returns (pieces, generated)."""
    from .engine.llamacpp import _gguf_piece

    pieces = bytearray()
    generated = 0
    for _ in range(max(0, req.max_tokens)):
        token = _sample_token(logits, req.temperature, req.top_p, rng)
        if token in eos_ids:
            break
        pieces += _gguf_piece(lib, vocab, token)
        generated += 1
        seen = pieces.decode("utf-8", errors="ignore")
        if any(marker in seen for marker in _STOP_MARKERS):
            break
        logits = _gguf_decode(lib, context, [token], position, True)
        position += 1
    return pieces, generated


# --- vision decisions: option logits read with an image in the context ---
VISION_PROMPT_VERSION = "direct-options-vision-v1"
VISION_ANSWER_CUE = "Answer:"


def _state_has_images(state) -> bool:
    return isinstance(state, list) and any(
        isinstance(part, dict) and part.get("type") == "image_url" for part in state)


def _state_text_and_images(state) -> tuple[str, list[bytes]]:
    """Normalize a SystemOne state into (text, image blobs).

    A plain string is text; a list carries OpenAI-style content parts — text
    parts concatenate, image_url parts (base64 data URLs) become blobs and
    leave the multimodal marker in the text at their position.
    """
    if isinstance(state, str):
        return state, []
    if isinstance(state, dict):
        return json.dumps(state, ensure_ascii=False), []
    import base64

    import llama_cpp.mtmd_cpp as mtmd

    marker = mtmd.mtmd_default_marker().decode("utf-8")
    chunks, blobs = [], []
    for part in state:
        if not isinstance(part, dict):
            raise HTTPException(status_code=400, detail="each state part must be an object")
        kind = part.get("type")
        if kind == "text":
            chunks.append(str(part.get("text", "")))
        elif kind == "image_url":
            url = part.get("image_url")
            url = url.get("url") if isinstance(url, dict) else url
            if not isinstance(url, str) or not url.startswith("data:") or ";base64" not in url:
                raise HTTPException(status_code=400, detail="state images must be base64 data URLs")
            try:
                blobs.append(base64.b64decode(url.partition(",")[2]))
            except Exception as error:
                raise HTTPException(status_code=400, detail="invalid base64 image payload") from error
            chunks.append(marker)
        else:
            raise HTTPException(status_code=400, detail=f"unsupported state part type: {kind!r}")
    return "\n".join(chunks), blobs


def _vision_prompt(state_text: str, row: dict) -> str:
    options = " ".join(
        f"{LETTERS[index]}) {option['description']}" for index, option in enumerate(row["options"]))
    return (
        "<bos><|turn>user\n"
        f"{state_text}\n\n{row['question']}\nOptions: {options}\n"
        "Answer with a single letter, nothing else.\n<turn|>\n<|turn>model\n" + VISION_ANSWER_CUE
    )


def _gguf_vision_score_rows(rows: list[dict], max_tokens: int):
    """Score rows whose state carries images: mtmd prompt + option-slot readout.

    One mtmd evaluation per row — [image(s)] + text prompt — then the option
    token logits are read at the last position: the same quantity as the text
    path (full-vocabulary logits restricted to the declared options).
    """
    import ctypes

    import llama_cpp.mtmd_cpp as mtmd

    lib = model.engine.lib
    vision = _gguf_vision_context()
    results = []
    for row in rows:
        state_text, blobs = _state_text_and_images(row.get("state"))
        if not blobs:
            raise HTTPException(status_code=400, detail="vision rows must carry at least one image")
        letters = LETTERS[: len(row["options"])]
        slots = {}
        for letter in letters:
            token_ids = tokenizer.encode(letter, add_special_tokens=False)
            if len(token_ids) != 1:
                raise HTTPException(status_code=400, detail=f"option letter {letter!r} is not a single token")
            slots[letter] = token_ids[0]
            if tokenizer.encode(VISION_ANSWER_CUE + letter, add_special_tokens=False) != \
                    tokenizer.encode(VISION_ANSWER_CUE, add_special_tokens=False) + [slots[letter]]:
                raise HTTPException(status_code=400,
                                    detail=f"answer boundary shifts tokenization for option {letter!r}")
        prompt = _vision_prompt(state_text, row)
        with _VISION_LOCK:
            vision, chunks, bitmaps, prompt_tokens = _gguf_vision_prepare(prompt, blobs, add_special=False)
            try:
                with _chat_context_lease(lib, model.engine.model, prompt_tokens + 8) as context:
                    lib.llama_memory_clear(lib.llama_get_memory(context), True)
                    next_position = ctypes.c_int32(0)
                    status = mtmd.mtmd_helper_eval_chunks(
                        vision, context, chunks, 0, 0, 512, True, ctypes.byref(next_position))
                    if status != 0:
                        raise HTTPException(status_code=500,
                                            detail=f"multimodal evaluation failed (code {status})")
                    logits = _gguf_last_logits(lib, context)
            finally:
                _gguf_vision_release(chunks, bitmaps)
        raw = [float(logits[slots[letter]]) for letter in letters]
        results.append({
            "id": row["id"],
            "option_ids": [option["id"] for option in row["options"]],
            "probabilities": _softmax(raw, 1.0),
            "option_logits": raw,
            "answer_token_ids": [slots[letter] for letter in letters],
            "input_tokens": prompt_tokens,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8") + b"".join(blobs)).hexdigest(),
            "prompt_version": VISION_PROMPT_VERSION,
            "readout": "vision-options-v1",
        })
    return results, {"mode": "vision-direct", "rows": len(rows)}


def _score_row(row: dict, metadata: dict):
    """Score one row: multimodal when its state carries images, text otherwise."""
    if _state_has_images(row.get("state")):
        results, _ = _gguf_vision_score_rows([row], MAX_TOKENS)
        return results[0]
    return _score_direct(model, tokenizer, row, metadata, MAX_TOKENS)


def _gguf_chat_completions(req: ChatReq):
    from .engine.llamacpp import _gguf_tokenize

    lib = model.engine.lib
    vocab = model.vocab
    messages, images = _gguf_split_images(req.messages)
    template_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if req.tools:
        template_kwargs["tools"] = req.tools
        if req.tool_choice:
            template_kwargs["tool_choice"] = req.tool_choice
    try:
        prompt = tokenizer.apply_chat_template(
            messages, enable_thinking=req.thinking, **template_kwargs)
    except TypeError:
        try:
            prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("tools", None)
            template_kwargs.pop("tool_choice", None)
            prompt = tokenizer.apply_chat_template(messages, **template_kwargs)

    t0 = time.perf_counter()
    import numpy

    eos_ids = set()
    try:
        eos_ids.add(int(lib.llama_vocab_eos(vocab)))
    except (AttributeError, TypeError):
        pass
    rng = numpy.random.default_rng()

    if images:
        import ctypes

        import llama_cpp.mtmd_cpp as mtmd

        with _VISION_LOCK:
            vision, chunks, bitmaps, prompt_tokens = _gguf_vision_prepare(prompt, images, add_special=True)
            try:
                need_tokens = _check_chat_budget(prompt_tokens, req.max_tokens)
                with _chat_context_lease(lib, model.engine.model, need_tokens) as context:
                    lib.llama_memory_clear(lib.llama_get_memory(context), True)
                    next_position = ctypes.c_int32(0)
                    status = mtmd.mtmd_helper_eval_chunks(
                        vision, context, chunks, 0, 0, 512, True, ctypes.byref(next_position))
                    if status != 0:
                        raise HTTPException(status_code=500,
                                            detail=f"multimodal evaluation failed (code {status})")
                    logits = _gguf_last_logits(lib, context)
                    pieces, generated = _gguf_generate(
                        lib, vocab, context, logits, int(next_position.value), req, eos_ids, rng)
            finally:
                _gguf_vision_release(chunks, bitmaps)
    else:
        ids = _gguf_tokenize(lib, vocab, prompt)
        if not ids:
            raise HTTPException(status_code=400, detail="empty chat prompt")
        prompt_tokens = len(ids)
        need_tokens = _check_chat_budget(prompt_tokens, req.max_tokens)
        with _chat_context_lease(lib, model.engine.model, need_tokens) as context:
            lib.llama_memory_clear(lib.llama_get_memory(context), True)
            logits = _gguf_decode(lib, context, ids, 0, True)
            pieces, generated = _gguf_generate(
                lib, vocab, context, logits, prompt_tokens, req, eos_ids, rng)
    dt = time.perf_counter() - t0

    text = pieces.decode("utf-8", errors="ignore")
    for marker in _STOP_MARKERS:
        text = text.replace(marker, "")
    reasoning = None
    if "<|channel>thought" in text:
        # Gemma-style thought channel: <|channel>thought ... <channel|> then the answer
        _, _, after = text.partition("<|channel>thought")
        thought, closed, rest = after.partition("<channel|>")
        reasoning = thought.strip() or None
        text = rest if closed else ""  # truncated inside the thought: no answer yet
    if "</think>" in text:
        think, _, text = text.partition("</think>")
        reasoning = think.replace("<think>", "").strip() or reasoning
    content = text.replace("<channel|>", "").replace("<|channel>thought", "").strip()

    tool_calls: list[dict] = []
    if req.tools:
        content, tool_calls = _parse_tool_calls(content)
        content = content.strip()

    message = {"role": "assistant", "content": (content or None) if tool_calls else content}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{CHAT_MODEL}",
        "object": "chat.completion",
        "model": req.model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": generated,
            "total_tokens": prompt_tokens + generated,
        },
        "timings": {
            "generation_seconds": round(dt, 2),
            "tokens_per_second": round(generated / dt, 1) if dt > 0 and generated else None,
        },
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatReq):
    """OpenAI-compatible chat on the same in-memory weights."""
    return _gguf_chat_completions(req)


class CalibrateReq(BaseModel):
    scenario: str
    dataset: list[dict] | None = None
    dataset_path: str | None = None
    heldout: list[dict] | None = None
    heldout_path: str | None = None
    overwrite: bool = False
    description: str = ""


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise HTTPException(status_code=400, detail=f"file not found: {path}")
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise HTTPException(status_code=400, detail=f"empty dataset: {path}")
    return rows


def _expand_labeled(row: dict, idx: int):
    """Accept System One rows ({state, questions{id:{type,instructions,criteria,label}}})
    or direct gold rows ({id, state, question, options, label}).
    Returns [(gold_row, primitive_type, label)]."""
    out = []
    if "questions" in row:
        state = row.get("state")
        for qid, q in row["questions"].items():
            crit = q.get("criteria") or {}
            t = q.get("type", "choice")
            if t == "noul":
                options = [
                    {"id": "true", "description": crit.get("true") or DEFAULT_TRUE},
                    {"id": "false", "description": crit.get("false") or DEFAULT_FALSE},
                ]
            elif t == "choice":
                options = [
                    {"id": k, "description": (v if isinstance(v, str) and v.strip() else k)}
                    for k, v in crit.items()
                ]
            else:
                options = [{"id": str(i), "description": lvl} for i, lvl in enumerate(crit)]
            rid = str(row.get("id", idx))
            if len(row["questions"]) > 1:
                rid = f"{rid}-{qid}"
            out.append((
                {"id": rid, "state": state, "question": (q.get("instructions") or "").strip() or "Decide based on the state.", "options": options},
                t,
                q.get("label"),
            ))
    else:
        out.append((
            {"id": str(row.get("id", f"row{idx}")), "state": row.get("state"),
             "question": (row.get("question") or "Decide based on the state.").strip(),
             "options": row.get("options", [])},
            row.get("type", "gold"),
            row.get("label"),
        ))
    return out


@app.post("/v1/calibrate")
def calibrate(req: CalibrateReq):
    """Automated scenario pipeline: score labeled rows, fit the temperature,
    validate out-of-fold, publish into the manifest and hot-reload. No restart."""
    if not SCENARIO_RE.fullmatch(req.scenario):
        raise HTTPException(status_code=400, detail="scenario name must match [a-z0-9]+(-[a-z0-9]+)*")
    if req.scenario in BUILTIN_SCENARIOS or req.scenario in LEGACY_BUILTIN_NAMES:
        raise HTTPException(status_code=400,
                            detail=f"'{req.scenario}' is reserved (built-in scenario: {BUILTIN_SCENARIOS[req.scenario]['description']})")
    if req.scenario in SCENARIOS and not req.overwrite:
        raise HTTPException(status_code=409, detail={
            "error": f"scenario '{req.scenario}' already exists",
            "hint": "pass overwrite=true to replace it",
            "current": SCENARIOS[req.scenario],
        })

    def load(part, path):
        if part is not None:
            return part
        if path:
            return _load_jsonl(path)
        return None

    fit_raw = load(req.dataset, req.dataset_path)
    if not fit_raw:
        raise HTTPException(status_code=400, detail="no dataset: pass 'dataset' (list) or 'dataset_path'")
    held_raw = load(req.heldout, req.heldout_path)

    rows = []  # (gold_row, primitive_type, label)
    for idx, src in enumerate(fit_raw):
        rows.extend(_expand_labeled(src, idx))
    per_type = {}
    for _, t, _ in rows:
        per_type[t] = per_type.get(t, 0) + 1
    short = {t: c for t, c in per_type.items() if c < CALIB_MIN_ROWS_PER_TYPE}
    if short:
        raise HTTPException(status_code=400, detail={
            "error": f"at least {CALIB_MIN_ROWS_PER_TYPE} rows per primitive type required",
            "rows_per_type": per_type,
        })
    for gold, _, label in rows:
        validate_row(gold)
        ids = [o["id"] for o in gold["options"]]
        if not isinstance(label, int) or isinstance(label, bool) or not 0 <= label < len(ids):
            raise HTTPException(status_code=400, detail=f"{gold['id']}: label must be an option index (0..{len(ids)-1})")
    if len({g["id"] for g, _, _ in rows}) != len(rows):
        raise HTTPException(status_code=400, detail="duplicate row ids in dataset")

    # score (direct, serialized with chat/scoring lock)
    pairs, conf_items, input_tokens = [], [], 0
    t0 = time.perf_counter()
    with _GEN_LOCK:
        for gold, _, label in rows:
            res = _score_row(gold, metadata)
            logits = res["option_logits"]
            ti = res["option_ids"].index(gold["options"][label]["id"])
            pairs.append((logits, ti))
            probs = _softmax(logits)
            conf_items.append((max(probs), int(probs.index(max(probs)) == ti)))
            input_tokens += res.get("input_tokens", 0)
    score_seconds = round(time.perf_counter() - t0, 2)

    temperature = round(_golden_fit(pairs), 6)

    # honest out-of-fold ECE: group-disjoint folds, T fitted per fold
    groups = sorted({g["id"] for g, _, _ in rows})
    rng = random.Random(CALIB_SEED)
    rng.shuffle(groups)
    fold_of = {g: i % CALIB_FOLDS for i, g in enumerate(groups)}
    ood = []
    fold_temperatures = []
    for fold in range(CALIB_FOLDS):
        train_pairs = [p for p, (gold, _, _l) in zip(pairs, rows) if fold_of[gold["id"]] != fold]
        if not train_pairs:
            continue
        t_f = _golden_fit(train_pairs)
        fold_temperatures.append(round(t_f, 4))
        for p, (gold, _, _l) in zip(pairs, rows):
            if fold_of[gold["id"]] == fold:
                pr = _softmax(p[0], t_f)
                ood.append((max(pr), int(pr.index(max(pr)) == p[1])))
    ece_raw = _ece(conf_items)
    ece_ood = _ece(ood)
    accuracy = round(sum(ok for _, ok in conf_items) / len(conf_items), 4)

    fingerprint = {"model": MODEL_SOURCE, "revision": REVISION, "backend": "llamacpp"}
    entry = {
        "temperature": temperature,
        "n_fit": len(pairs),
        "rows_per_type": per_type,
        "ece_raw": ece_raw,
        "ece_out_of_fold": ece_ood,
        "accuracy_unchanged": accuracy,
        "fingerprint": fingerprint,
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": req.description or f"calibrated via /v1/calibrate on {len(pairs)} rows",
    }
    SCENARIOS[req.scenario] = entry
    _persist_manifest()

    heldout_metrics = None
    if held_raw:
        hrows = []
        for idx, src in enumerate(held_raw):
            hrows.extend(_expand_labeled(src, idx))
        hpairs, hitems = [], []
        with _GEN_LOCK:
            for gold, _, label in hrows:
                validate_row(gold)
                res = _score_row(gold, metadata)
                logits = res["option_logits"]
                ti = res["option_ids"].index(gold["options"][label]["id"])
                hpairs.append((logits, ti))
                pr = _softmax(logits, temperature)
                hitems.append((max(pr), int(pr.index(max(pr)) == ti)))
        hacc = round(sum(ok for _, ok in hitems) / len(hitems), 4) if hitems else None
        hnll = round(sum(-math.log(max(_softmax(l, temperature)[t], 1e-12)) for l, t in hpairs) / len(hpairs), 4) if hpairs else None
        heldout_metrics = {"rows": len(hpairs), "accuracy": hacc, "ece_held_out": _ece(hitems), "nll_held_out": hnll}

    return {
        "scenario": req.scenario,
        "temperature": temperature,
        "n_fit": len(pairs),
        "ece_raw": ece_raw,
        "ece_out_of_fold": ece_ood,
        "accuracy_unchanged": accuracy,
        "fingerprint": fingerprint,
        "status": f"published + hot-reloaded (use model suffix {MODEL_BASE}:{req.scenario}); requires held-out validation",
        "score_seconds": score_seconds,
        "fold_temperatures": fold_temperatures,
        "heldout": heldout_metrics,
        "input_tokens": input_tokens,
    }


def _do_delete_scenario(scenario: str):
    if scenario in BUILTIN_SCENARIOS or scenario in LEGACY_BUILTIN_NAMES:
        raise HTTPException(status_code=400,
                            detail=f"'{scenario}' is built-in and cannot be deleted")
    if scenario not in SCENARIOS:
        raise HTTPException(status_code=404, detail={
            "error": f"scenario '{scenario}' not found",
            "available": sorted(SCENARIOS),
        })
    SCENARIOS.pop(scenario)
    _persist_manifest()
    return {
        "deleted": scenario,
        "scenarios": sorted(SCENARIOS),
        "status": "removed + hot-reloaded (the model suffix now returns 422 'unknown scenario')",
    }


@app.delete("/v1/calibrate/{scenario}")
def delete_calibrate(scenario: str):
    return _do_delete_scenario(scenario)


class DeleteReq(BaseModel):
    scenario: str


@app.post("/v1/calibrate/delete")
def delete_calibrate_post(req: DeleteReq):
    return _do_delete_scenario(req.scenario)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
