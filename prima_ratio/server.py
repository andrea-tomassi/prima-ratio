"""prima-ratio — a turn-key System One decision service.

One endpoint (`POST /v1/systemone`), Jev-shaped: a state (text/JSON, plus optional
images) and typed questions in, a calibrated probability for every option out — one
forward pass per request, no generation, no chat.

Engine: Clef-Flash (Qwen3.5-9B backbone + joint schema head, nf4) with its vision
lens. The model is already calibrated (raw ECE ≈ 0.02 on the public decision
suites); probabilities are returned as-is. Wire-standard conformance
(message-part images, confidence formulas) is applied by
`prima_ratio.standard` around the vendored engine call.

Two surfaces are served:

- `/v1/systemone` — the flat Jev/System One body (`model`, `answers`, `usage`).
- `/client/v4/accounts/{account_id}/ai/run[/@cf/cloudflare/clef-flash]` — a
  drop-in for the hosted Cloudflare Workers AI endpoint: same request, the
  response wrapped in the CF envelope (`result`, `success`, `errors`, `messages`).
"""
from __future__ import annotations

import base64
import io
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from . import standard

MODEL_ID = os.environ.get("PRIMA_MODEL_NAME", "prima-ratio-clef-flash")
# 131072: the practical ceiling on a 16 GB card with the chunked prefill
# (~95K measured at 14.6 GB); bigger cards can raise it up to the model's 262144.
MAX_LENGTH = int(os.environ.get("PRIMA_MAX_LENGTH", "131072"))
VERSION = "2.0.2"

app = FastAPI(title="prima-ratio", version=VERSION, docs_url=None, redoc_url=None)


def _decode_images(raw: list) -> list:
    """data-URLs (or bare base64) → PIL images, as `systemone()` expects."""
    from PIL import Image

    images = []
    for item in raw:
        try:
            data = item.partition(",")[2] if isinstance(item, str) and item.startswith("data:") else item
            images.append(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))
        except Exception as error:  # noqa: BLE001 - surfaced to the client as 400
            raise HTTPException(status_code=400, detail=f"invalid image payload: {error}") from error
    return images


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "model": MODEL_ID, "version": VERSION}


@app.get("/v1/models")
def models() -> dict:
    now = int(time.time())
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": now,
            "owned_by": "prima-ratio",
            "meta": {
                "engine": "clef-flash nf4 (Qwen3.5-9B + joint schema head)",
                "vision": True,
                "calibration": "native (head probabilities, no temperature scaling)",
            },
        }],
    }


def _execute(request: dict) -> dict:
    """Run one System One request: normalize, decode images, decide, normalize back."""
    from .clef import load
    from .engine.joint_schema_model import systemone

    body = dict(request)
    if not isinstance(body.get("model"), str):
        body["model"] = MODEL_ID
    try:
        body = standard.normalize_request(body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    raw = body.get("images") or []
    if raw:
        body["images"] = _decode_images(raw)
    try:
        started = time.perf_counter()
        response = systemone(load()[0], load()[1], body, max_length=MAX_LENGTH)
        response = standard.normalize_response(response)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - best-effort recovery, then report
                pass
            raise HTTPException(
                status_code=507,
                detail="insufficient GPU memory for this state; lower PRIMA_MAX_LENGTH "
                "or PRIMA_PREFILL_CHUNK, or shorten the state",
            ) from error
        raise
    response["x_prima"] = {"latency_s": round(time.perf_counter() - started, 3), "version": VERSION}
    return response


@app.post("/v1/systemone")
def systemone_endpoint(request: dict) -> dict:
    return _execute(request)


# ---- Cloudflare Workers AI-compatible surface --------------------------------
# Drop-in for the hosted `@cf/cloudflare/clef-flash` endpoint: same request
# body, same core, response wrapped in the CF envelope
# {"result": {...}, "success": true, "errors": [], "messages": []}.


def _cf_wrap(request: dict, *, universal: bool = False):
    body = dict(request)
    if universal:
        # The universal route nests the model schema under `input` and uses the
        # full model id — same contract as the hosted endpoint (400 otherwise).
        schema = body.get("input")
        if not isinstance(schema, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "result": None,
                    "success": False,
                    "errors": [{"code": 7000, "message": "Invalid request body: input"}],
                    "messages": [],
                },
            )
        body = dict(schema)
        if not isinstance(body.get("model"), str):
            outer = request.get("model")
            body["model"] = outer if isinstance(outer, str) else "clef-flash"
    model = body.get("model")
    if isinstance(model, str) and model.startswith("@cf/"):
        body["model"] = model.rsplit("/", 1)[-1]
    if not isinstance(body.get("model"), str):
        body["model"] = "clef-flash"
    try:
        result = _execute(body)
    except HTTPException as error:
        return JSONResponse(
            status_code=error.status_code,
            content={
                "result": None,
                "success": False,
                "errors": [{"code": error.status_code, "message": str(error.detail)}],
                "messages": [],
            },
        )
    return {"result": result, "success": True, "errors": [], "messages": []}


@app.post("/client/v4/accounts/{account_id}/ai/run/@cf/cloudflare/clef-flash")
def cf_run_clef_flash(account_id: str, request: dict):
    return _cf_wrap(request)


@app.post("/client/v4/accounts/{account_id}/ai/run")
def cf_run_universal(account_id: str, request: dict):
    return _cf_wrap(request, universal=True)
