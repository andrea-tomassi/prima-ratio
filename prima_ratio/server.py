"""prima-ratio — a turn-key System One decision service.

One endpoint (`POST /v1/systemone`), Jev-shaped: a state (text/JSON, plus optional
images) and typed questions in, a calibrated probability for every option out — one
forward pass per request, no generation, no chat.

Engine: Clef-Flash (Qwen3.5-9B backbone + joint schema head, nf4) with its vision
tower. The model is already calibrated (raw ECE ≈ 0.02 on the public decision
suites); probabilities are returned as-is.
"""
from __future__ import annotations

import base64
import io
import os
import time

from fastapi import FastAPI, HTTPException

MODEL_ID = os.environ.get("PRIMA_MODEL_NAME", "prima-ratio-clef-flash")
MAX_LENGTH = int(os.environ.get("PRIMA_MAX_LENGTH", "16384"))
VERSION = "2.0.0"

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


@app.post("/v1/systemone")
def systemone_endpoint(request: dict) -> dict:
    from .clef import load
    from .engine.joint_schema_model import systemone

    body = dict(request)
    if not isinstance(body.get("model"), str):
        body["model"] = MODEL_ID
    raw = body.get("images") or []
    if raw:
        body["images"] = _decode_images(raw)
    try:
        started = time.perf_counter()
        response = systemone(load()[0], load()[1], body, max_length=MAX_LENGTH)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    response["x_prima"] = {"latency_s": round(time.perf_counter() - started, 3), "version": VERSION}
    return response
