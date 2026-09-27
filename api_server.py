"""System One-compatible HTTP wrapper around SemIf direct-logit scoring.

Same request/response shape as Rizzo Flow / Kev / TypeSafe — 100% standard API:
POST /v1/systemone  {state, model, questions{id:{type,instructions,criteria}}}
Scenarios map to MODEL SUFFIXES (OpenRouter-style): the model field carries the variant.

    semif-qwen3.5-4b                     raw logits (uncalibrated)
    semif-qwen3.5-4b:vanilla             raw logits, explicit
    semif-qwen3.5-4b:support-routing softmax(logits/T), T from the calibration manifest

State is prefilled once and all questions are scored in parallel (SemIf shared mode,
with per-row direct fallback when the tokenized state prefix is not stable).
Also: POST /v1/chat/completions — normal chat on the same in-memory model.
Scenario temperatures live in build/calibration-manifest.json.

Backends (env-driven, see README):
    SEMIF_BACKEND=torch      HF transformers bf16 checkpoint (default; Qwen3.5-4B)
    SEMIF_BACKEND=llamacpp   local GGUF through llama.cpp (SEMIF_GGUF=path,
                             SEMIF_LLAMA_GPU_LAYERS=-1 for full GPU offload;
                             scoring and chat share the loaded weights)
Environment: SEMIF_MODEL / SEMIF_REVISION / SEMIF_MODEL_NAME / SEMIF_MANIFEST /
SEMIF_MAX_TOKENS also override their defaults.
"""
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

from semif_phase1 import direct as direct_module
from semif_phase1.core import load_causal_model, validate_row
from semif_phase1.shared import score_shared

BACKEND = os.environ.get("SEMIF_BACKEND", "torch").strip().lower()
if BACKEND not in {"torch", "llamacpp"}:
    raise SystemExit(f"SEMIF_BACKEND must be 'torch' or 'llamacpp', got {BACKEND!r}")
_DEFAULTS = {
    "torch": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "semif-qwen3.5-4b"),
    "llamacpp": ("unsloth/gemma-4-12b-it", "55cdba0740a9765956f49501f689a66b098feda3", "semif-gemma4-12b"),
}
_DEFAULT_SOURCE, _DEFAULT_REVISION, _DEFAULT_NAME = _DEFAULTS[BACKEND]
MODEL_SOURCE = os.environ.get("SEMIF_MODEL", _DEFAULT_SOURCE)
REVISION = os.environ.get("SEMIF_REVISION", _DEFAULT_REVISION)
MODEL_BASE = os.environ.get("SEMIF_MODEL_NAME", _DEFAULT_NAME)
GGUF_PATH = os.environ.get("SEMIF_GGUF")
MAX_TOKENS = int(os.environ.get("SEMIF_MAX_TOKENS", "4096"))
CHAT_MODEL = os.environ.get("SEMIF_CHAT_NAME", MODEL_BASE.removeprefix("semif-") + "-chat")
DEFAULT_TRUE = "Yes. The evidence supports an affirmative answer to the question."
DEFAULT_FALSE = "No. The evidence supports a negative answer to the question."
MANIFEST_PATH = os.environ.get("SEMIF_MANIFEST", "build/calibration-manifest.json")
CALIB_FOLDS = 5
CALIB_SEED = 217
CALIB_MIN_ROWS_PER_TYPE = 10
SCENARIO_RE = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*$")

if BACKEND == "llamacpp":
    if not GGUF_PATH or not os.path.isfile(GGUF_PATH):
        raise SystemExit("SEMIF_BACKEND=llamacpp requires SEMIF_GGUF pointing at a local .gguf file")
    from pathlib import Path

    from semif_phase1 import llamacpp_backend

    model, tokenizer, metadata = llamacpp_backend.load_model(
        MODEL_SOURCE, REVISION, Path(GGUF_PATH), context_tokens=MAX_TOKENS)
    _score_direct = llamacpp_backend.score
    _score_shared = llamacpp_backend.score_shared
else:
    model, tokenizer, metadata = load_causal_model(MODEL_SOURCE, REVISION, "auto", "bfloat16")
    _score_direct = direct_module.score
    _score_shared = score_shared


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
SCENARIOS = MANIFEST.get("scenarios", {})

app = FastAPI(title="semif-systemone")

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
             "owned_by": "semif-server", "meta": {"variant": "raw logits (uncalibrated default)"}}]
    for name, e in sorted(SCENARIOS.items()):
        data.append({
            "id": f"{MODEL_BASE}:{name}",
            "object": "model",
            "created": now,
            "owned_by": "semif-server",
            "meta": {k: e.get(k) for k in ("temperature", "n_fit", "ece_raw", "ece_out_of_fold", "fitted_at", "description") if e.get(k) is not None},
        })
    data.append({"id": CHAT_MODEL, "object": "model", "created": now,
                 "owned_by": "semif-server", "meta": {"variant": "normal chat completions (POST /v1/chat/completions)"}})
    return {"object": "list", "data": data}


@app.post("/v1/systemone")
def systemone(req: Req):
    base, _, suffix = req.model.partition(":")
    if suffix and suffix not in SCENARIOS:
        raise HTTPException(
            status_code=422,
            detail={
                "error": f"unknown scenario '{suffix}'",
                "available": sorted(SCENARIOS),
                "hint": f'use "{MODEL_BASE}:<scenario>" — see GET /v1/models',
            },
        )
    entry = SCENARIOS.get(suffix) if suffix else None
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
        try:
            results, timing = _score_shared(model, tokenizer, rows, metadata)
            score_mode = "shared"
        except ValueError:
            # shared mode requires a stable tokenized state prefix (BPE boundary effects,
            # e.g. a state ending in a quote char); fall back to per-row direct scoring —
            # identical readout, no prefix reuse.
            results = [_score_direct(model, tokenizer, r, metadata) for r in rows]
            timing = {"mode": "direct-fallback"}
            score_mode = "direct"
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    if entry:
        if scenario_used == "vanilla":
            status = "raw logits (vanilla, T=1.0); uncalibrated as decision confidence"
        else:
            status = (
                f"temperature scaled (T={round(temperature, 4)}, scenario '{scenario_used}', "
                f"n_fit={entry.get('n_fit')}, ECE raw->out-of-fold "
                f"{entry.get('ece_raw')}->{entry.get('ece_out_of_fold')}); "
                "requires held-out validation on your own labels"
            )
    else:
        status = "conditional option score; uncalibrated as decision confidence (use a model suffix — see GET /v1/models)"

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
        "x_semif": {
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


# --- GGUF chat: generation on a secondary llama.cpp context (weights shared) ---
_GGUF_CHAT = {"context": None, "capacity": 0}
_CHAT_LOCK = threading.Lock()


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
    import ctypes

    import numpy

    pointer = lib.llama_get_logits_ith(context, -1)
    if not pointer:
        raise RuntimeError("llama.cpp returned no chat logits")
    return numpy.ctypeslib.as_array(
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_float)), shape=(model.engine.vocab_size,)
    ).copy()


def _gguf_chat_context(lib, native_model, need_tokens: int):
    """Lazily create a dedicated chat context on the already-loaded model."""
    context = _GGUF_CHAT["context"]
    if context is not None and need_tokens <= _GGUF_CHAT["capacity"]:
        return context
    if context is not None:
        lib.llama_free(context)
        _GGUF_CHAT["context"] = None
    params = lib.llama_context_default_params()
    params.n_ctx = max(need_tokens, 2048)
    params.n_seq_max = 1
    params.n_outputs_max = 1
    context = lib.llama_init_from_model(native_model, params)
    if not context:
        raise RuntimeError("llama.cpp failed to create the chat context")
    capacity = int(lib.llama_n_ctx(context))
    if need_tokens > capacity:
        lib.llama_free(context)
        raise RuntimeError("chat prompt does not fit the chat context")
    _GGUF_CHAT["context"] = context
    _GGUF_CHAT["capacity"] = capacity
    return context


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


def _gguf_chat_completions(req: ChatReq):
    from semif_phase1.llamacpp_backend import _gguf_piece, _gguf_tokenize

    lib = model.engine.lib
    vocab = model.vocab
    try:
        prompt = tokenizer.apply_chat_template(
            req.messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=req.thinking)
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            req.messages, tokenize=False, add_generation_prompt=True)
    ids = _gguf_tokenize(lib, vocab, prompt)
    if not ids:
        raise HTTPException(status_code=400, detail="empty chat prompt")

    t0 = time.perf_counter()
    import numpy

    with _CHAT_LOCK, _GEN_LOCK:
        context = _gguf_chat_context(lib, model.engine.model, len(ids) + req.max_tokens + 8)
        lib.llama_memory_clear(lib.llama_get_memory(context), True)
        logits = _gguf_decode(lib, context, ids, 0, True)
        eos_ids = set()
        try:
            eos_ids.add(int(lib.llama_vocab_eos(vocab)))
        except (AttributeError, TypeError):
            pass
        rng = numpy.random.default_rng()
        pieces = bytearray()
        generated = 0
        position = len(ids)
        stop_markers = ("<turn|>", "<end_of_turn>", "<eos>", "</s>")
        for _ in range(max(0, req.max_tokens)):
            token = _sample_token(logits, req.temperature, req.top_p, rng)
            if token in eos_ids:
                break
            pieces += _gguf_piece(lib, vocab, token)
            generated += 1
            seen = pieces.decode("utf-8", errors="ignore")
            if any(marker in seen for marker in stop_markers):
                break
            logits = _gguf_decode(lib, context, [token], position, True)
            position += 1
    dt = time.perf_counter() - t0

    text = pieces.decode("utf-8", errors="ignore")
    for marker in stop_markers:
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

    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    return {
        "id": f"chatcmpl-{CHAT_MODEL}",
        "object": "chat.completion",
        "model": req.model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": len(ids),
            "completion_tokens": generated,
            "total_tokens": len(ids) + generated,
        },
        "timings": {
            "generation_seconds": round(dt, 2),
            "tokens_per_second": round(generated / dt, 1) if dt > 0 and generated else None,
        },
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatReq):
    """OpenAI-compatible chat on the same in-memory model."""
    if BACKEND == "llamacpp":
        return _gguf_chat_completions(req)
    import torch

    try:
        prompt = tokenizer.apply_chat_template(
            req.messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=req.thinking)
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            req.messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    n_prompt = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with _GEN_LOCK, torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=req.temperature > 0,
            temperature=req.temperature if req.temperature > 0 else None,
            top_p=req.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    dt = time.perf_counter() - t0
    gen_ids = out[0][n_prompt:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    reasoning = None
    if "</think>" in text:
        think, _, content = text.partition("</think>")
        reasoning = think.replace("<think>", "").strip()
        content = content.strip()
    else:
        content = text.strip()

    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    return {
        "id": f"chatcmpl-{CHAT_MODEL}",
        "object": "chat.completion",
        "model": req.model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": n_prompt,
            "completion_tokens": int(gen_ids.shape[0]),
            "total_tokens": n_prompt + int(gen_ids.shape[0]),
        },
        "timings": {
            "generation_seconds": round(dt, 2),
            "tokens_per_second": round(gen_ids.shape[0] / dt, 1) if dt > 0 else None,
        },
    }


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
    if req.scenario == "vanilla":
        raise HTTPException(status_code=400, detail="'vanilla' is reserved (built-in raw scenario)")
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
            res = _score_direct(model, tokenizer, gold, metadata)
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

    fingerprint = {"model": MODEL_SOURCE, "revision": REVISION, "backend": BACKEND}
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
                res = _score_direct(model, tokenizer, gold, metadata)
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
    if scenario == "vanilla":
        raise HTTPException(status_code=400, detail="'vanilla' is built-in (raw logits) and cannot be deleted")
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
