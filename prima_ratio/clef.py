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
# Chunked prefill: long states are fed to the backbone in slices of this many
# tokens (0 disables it). Peak memory becomes the slice plus the cache instead
# of the whole state, which is what unlocks ~100K-token decisions on 16 GB.
PREFILL_CHUNK = int(os.environ.get("PRIMA_PREFILL_CHUNK", "4096"))

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


def _install_chunked_prefill(model: Any, chunk: int) -> None:
    """Route long states through a chunked prefill, leaving the vendored engine untouched.

    The backbone consumes `chunk`-token slices with the hybrid cache (attention KV +
    the linear layers' conv/recurrent states) carried across slices, and the per-slice
    hidden states are kept — they are exactly what the joint head reads. The head then
    scores the whole sequence as usual. The decisions are invariant to the slicing
    (verified: identical probabilities at 4 decimals between one-shot and chunked).

    Slices containing image tokens go through the vision-aware wrapper, which derives
    its own mrope positions; later slices are plain text with explicit positions.
    """
    if chunk <= 0:
        return

    import torch
    from transformers.cache_utils import DynamicCache

    original_forward = model.forward

    def forward(batch: dict) -> Any:
        input_ids = batch["input_ids"]
        length = input_ids.shape[1]
        if length <= chunk:
            return original_forward(batch)

        base_model = (
            model.language_model.get_base_model()
            if hasattr(model.language_model, "get_base_model")
            else model.language_model
        )
        media = batch.get("media") or {}
        vision_model = base_model.model
        text_model = getattr(vision_model, "language_model", vision_model)

        # The first slice must swallow every image/video token so the vision-aware
        # wrapper sees the complete media span; later slices are text only.
        first_end = chunk
        if media:
            token_types = media.get("mm_token_type_ids")
            if token_types is not None and token_types.numel():
                media_positions = (token_types[0] != 0).nonzero()
                if media_positions.numel():
                    first_end = max(chunk, int(media_positions[-1].item()) + 1)

        cache = DynamicCache(config=model.language_model.config)
        hidden_parts = []
        start = 0
        while start < length:
            end = min(first_end if start == 0 else start + chunk, length)
            slice_ids = input_ids[:, start:end]
            if start == 0 and media:
                slice_media = {
                    key: (value[:, :end] if key == "mm_token_type_ids" else value)
                    for key, value in media.items()
                }
                outputs = vision_model(
                    input_ids=slice_ids,
                    attention_mask=batch["attention_mask"][:, :end],
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                    **slice_media,
                )
            else:
                position_ids = torch.arange(start, end, device=input_ids.device).unsqueeze(0)
                outputs = text_model(
                    input_ids=slice_ids,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
            hidden_parts.append(outputs.last_hidden_state)
            start = end

        return model.head(
            torch.cat(hidden_parts, dim=1),
            input_ids,
            batch["attention_mask"],
            batch["records"],
            base_model.get_output_embeddings().weight,
        )

    model.forward = forward


def load() -> tuple[Any, Any]:
    """Load (once) and return (model, processor)."""
    global _model, _processor
    if _model is None:
        from .engine.joint_schema_model import load_release_model

        path = resolve_model_path()
        attn = _attn_implementation()
        print(f"[prima-ratio] loading {path} on {DEVICE} (nf4, {attn}, "
              f"prefill chunk {PREFILL_CHUNK or 'off'})…", flush=True)
        _model, _processor = load_release_model(path, device=DEVICE, attn_implementation=attn)
        _install_chunked_prefill(_model, PREFILL_CHUNK)
        print("[prima-ratio] model ready", flush=True)
    return _model, _processor
