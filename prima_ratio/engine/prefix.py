# Vendored from SemIf (https://github.com/TheoLeeCJ/SemIf-OpenJev), MIT licence,
# commit 23cf1f39fc9534fe81437200959b6dfc7106e45a — adapted for prima-ratio:
# the Torch model paths were removed (GGUF/llama.cpp only). Algorithm, prompt
# contract and numeric behaviour are unchanged; see NOTICE and VENDORED.md.
"""Shared-state prefix extraction for one-prefill multi-decision scoring."""

from __future__ import annotations

import json

from .prompts import direct_messages


def _state_prefix(tokenizer, state) -> list[int]:
    row = {
        "id": "prefix-only",
        "state": state,
        # This value occurs after the extracted evidence boundary.
        "question": "prefix boundary placeholder",
        "options": [
            {"id": "yes", "description": "Yes"},
            {"id": "no", "description": "No"},
        ],
    }
    turns = direct_messages(row)
    prompt = tokenizer.apply_chat_template(
        turns, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    payload = turns[-1]["content"]
    if prompt.count(payload) != 1:
        raise ValueError("Cannot locate the unmodified evidence payload in the chat template")
    evidence = json.dumps({"evidence": state}, ensure_ascii=False)[:-1]
    if not payload.startswith(evidence):
        raise ValueError("Evidence serialization changed")
    text = prompt[: prompt.index(payload)] + evidence
    # Appending JSON punctuation can merge with the final boundary token.
    return tokenizer.encode(text, add_special_tokens=False)[:-1]
