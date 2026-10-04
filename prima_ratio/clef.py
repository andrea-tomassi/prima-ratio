"""Model loading: the Clef-Flash nf4 checkpoint, cached and mounted.

Resolution order for the model:
  1. `PRIMA_MODEL_PATH` — an explicit directory (must contain the release files);
  2. the cache: `<PRIMA_CACHE>/models/<repo slug>` — reused across restarts;
  3. download from `PRIMA_MODEL_REPO` (default: the nf4 release with the joint head)
     into the cache on first start.

The HF download honours `HF_HOME` — mount `<PRIMA_CACHE>` and everything survives
image upgrades.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

MODEL_REPO = os.environ.get("PRIMA_MODEL_REPO", "meossistant/clef-flash-4bit")
CACHE_DIR = Path(os.environ.get("PRIMA_CACHE", "/cache"))
DEVICE = os.environ.get("PRIMA_DEVICE", "cuda")

_model: Any = None
_processor: Any = None


def resolve_model_path() -> str:
    explicit = os.environ.get("PRIMA_MODEL_PATH")
    if explicit:
        if not Path(explicit).is_dir():
            raise SystemExit(f"PRIMA_MODEL_PATH is not a directory: {explicit}")
        return explicit
    target = CACHE_DIR / "models" / MODEL_REPO.replace("/", "--")
    required = ("config.json", "joint_head.safetensors", "joint_schema_model.py")
    if all((target / name).is_file() for name in required):
        return str(target)
    print(f"[prima-ratio] model not found in the cache — downloading {MODEL_REPO} "
          f"to {target} (first start; ~8 GB, this can take a while)...", flush=True)
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=MODEL_REPO, local_dir=str(target))
    return str(target)


def _attn_implementation() -> str:
    """Prefer flash-attention 2 (memory + speed on long states), fall back to sdpa."""
    requested = os.environ.get("PRIMA_ATTN", "auto")
    if requested != "auto":
        return requested
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def load() -> tuple[Any, Any]:
    """Load (once) and return (model, processor)."""
    global _model, _processor
    if _model is None:
        from .engine.joint_schema_model import load_release_model

        path = resolve_model_path()
        attn = _attn_implementation()
        print(f"[prima-ratio] loading {path} on {DEVICE} (nf4, {attn})…", flush=True)
        _model, _processor = load_release_model(path, device=DEVICE, attn_implementation=attn)
        print("[prima-ratio] model ready", flush=True)
    return _model, _processor
