"""System One-compatible HTTP wrapper around SemIf direct-logit scoring.

Same request/response shape as Rizzo Flow / Kev / TypeSafe — 100% standard API:
POST /v1/systemone  {state, model, questions{id:{type,instructions,criteria}}}
Scenarios map to MODEL SUFFIXES (OpenRouter-style): the model field carries the variant.

    semif-qwen3.5-4b                     raw logits (uncalibrated)
    semif-qwen3.5-4b:vanilla             raw logits, explicit
    semif-qwen3.5-4b:user-classification softmax(logits/T), T from the calibration manifest

State is prefilled once and all questions are scored in parallel (SemIf shared mode,
with per-row direct fallback when the tokenized state prefix is not stable).
Also: POST /v1/chat/completions — normal chat on the same in-memory model.
Scenario temperatures live in build/calibration-manifest.json.
"""
import json
import math
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from semif_phase1 import direct as direct_module
from semif_phase1.core import load_causal_model, validate_row
from semif_phase1.shared import score_shared

MODEL_BASE = "semif-qwen3.5-4b"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DEFAULT_TRUE = "Yes. The evidence supports an affirmative answer to the question."
DEFAULT_FALSE = "No. The evidence supports a negative answer to the question."
MANIFEST_PATH = "build/calibration-manifest.json"

model, tokenizer, metadata = load_causal_model(MODEL_BASE.split("-gguf")[0] if False else "Qwen/Qwen3.5-4B", REVISION, "auto", "bfloat16")


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


class ChatReq(BaseModel):
    messages: list[dict]
    model: str = "qwen3.5-4b-chat"
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 512
    thinking: bool = True


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


def _model_catalog():
    entries = [{"name": MODEL_BASE, "description": "raw logits, uncalibrated (default)"}]
    for name, e in sorted(SCENARIOS.items()):
        desc = f"T={e.get('temperature')}"
        if name == "vanilla":
            desc = "raw logits, explicit (built-in)"
        else:
            desc += f"; ECE raw->out-of-fold {e.get('ece_raw')}->{e.get('ece_out_of_fold')}" if e.get("ece_raw") is not None else ""
        entries.append({"name": f"{MODEL_BASE}:{name}", "description": (e.get("description") or "") + " — " + desc})
    entries.append({
        "name": "qwen3.5-4b-chat",
        "description": "Same in-memory model, normal chat completions (POST /v1/chat/completions; OpenAI-shaped; optional 'thinking': false, default on)",
    })
    return entries


def _suffix_instructions() -> dict:
    return {
        "how": "put the scenario after a ':' in the model field — softmax(logits/T) is applied; argmax never changes; ':vanilla' = raw logits, explicit; no suffix = raw (uncalibrated)",
        "examples": [
            {"model": MODEL_BASE, "note": "raw"},
            {"model": f"{MODEL_BASE}:vanilla", "note": "raw, explicit"},
            {"model": f"{MODEL_BASE}:user-classification", "note": "calibrated with the scenario temperature"},
        ],
        "unknown_suffix": "422 listing the available scenarios",
        "add_new_scenario": "label 30-60 examples (SemIf JSONL with label index) -> semif-score --mode direct -> benchmarks/calibrate.py --gold ... --predictions ... --report build/<name>.json -> add entry to build/calibration-manifest.json -> restart",
        "manifest": MANIFEST_PATH,
    }


@app.get("/v1/models")
def models():
    return {
        "models": _model_catalog(),
        "model_suffixes": _suffix_instructions(),
    }


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
            results, timing = score_shared(model, tokenizer, rows, metadata)
            score_mode = "shared"
        except ValueError:
            # shared mode requires a stable tokenized state prefix (BPE boundary effects,
            # e.g. a state ending in a quote char); fall back to per-row direct scoring —
            # identical readout, no prefix reuse.
            results = [direct_module.score(model, tokenizer, r, metadata) for r in rows]
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
    model: str = "qwen3.5-4b-chat"
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 512
    thinking: bool = True


@app.post("/v1/chat/completions")
def chat_completions(req: ChatReq):
    """OpenAI-compatible chat on the same in-memory stock Qwen3.5-4B."""
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
        "id": "chatcmpl-semif-4b",
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
