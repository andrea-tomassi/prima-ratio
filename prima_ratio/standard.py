"""System One wire-standard conformance for the HTTP layer.

Two public references define the contract applied here:

- TypeSafe's System One API (https://docs.typesafe.ai): ``state`` is a string
  or a structured JSON value, and answers carry probabilities.
- The multimodal extension — the reference cited by the llama.cpp
  ``/v1/systemone`` docs (https://jev-skills.github.io/openjev-multimodal/api):
  images travel either in a top-level ``images`` array of data URLs and/or as
  ``image_url`` parts inside a chat-message state; all images are read before
  the state in the prompt, the ``images`` field first, and the parts are
  removed from the state.

``confidence`` follows the Cloudflare Workers AI clef-flash definition:
normalized concentration (Gini purity) ``(n·Σpᵢ² − 1)/(n − 1)``, verified
against the hosted endpoint to 4 decimals.

The vendored engine file stays verbatim (see VENDORED.md): everything the
engine does not implement of the wire contract is applied here, around the
``systemone`` call — request normalization in, response normalization out.
"""
from __future__ import annotations

from typing import Any


def _is_message(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("role"), str)
        and "content" in value
    )


def _messages_of(state: Any) -> list | None:
    """Return the message list when ``state`` follows the chat-message convention.

    The convention (OpenJev multimodal reference): an array of chat messages, or
    an object with a ``messages`` array. Anything else is an ordinary JSON state.
    """
    if isinstance(state, list) and all(_is_message(message) for message in state):
        return state
    if isinstance(state, dict):
        messages = state.get("messages")
        if isinstance(messages, list) and all(_is_message(message) for message in messages):
            return messages
    return None


def _part_image_url(part: Any) -> str | None:
    """Data URL of an ``image_url`` part; ``None`` for any other part.

    Parts follow the chat-completions format: ``{"type": "image_url",
    "image_url": {"url": "data:..."}}`` (a bare ``"image_url": "data:..."``
    string is accepted too). Only data URLs are valid.
    """
    if not isinstance(part, dict) or str(part.get("type") or "") != "image_url":
        return None
    image_url = part.get("image_url")
    url = image_url.get("url") if isinstance(image_url, dict) else image_url
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ValueError("image_url parts must use data URLs (data:image/...;base64,...)")
    return url


def extract_message_images(state: Any) -> tuple[Any, list[str]]:
    """Split a message-shaped state into (clean state, image data URLs).

    Image parts are removed, left to right; everything else is preserved —
    including metadata fields of a ``{"messages": [...]}`` object. States that
    do not follow the message convention are returned unchanged.
    """
    messages = _messages_of(state)
    if messages is None:
        return state, []
    images: list[str] = []
    cleaned_messages = []
    for message in messages:
        message = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            cleaned_parts = []
            for part in content:
                url = _part_image_url(part)
                if url is None:
                    cleaned_parts.append(part)
                else:
                    images.append(url)
            message["content"] = cleaned_parts
        cleaned_messages.append(message)
    if isinstance(state, list):
        cleaned_state: Any = cleaned_messages
    else:
        cleaned_state = dict(state)
        cleaned_state["messages"] = cleaned_messages
    return cleaned_state, images


def normalize_request(body: dict) -> dict:
    """Hoist message-part images into the ``images`` field, standard order.

    Order: the ``images`` field first, then the parts, left to right
    (llama.cpp docs: "the ones from ``images`` first"). The state keeps
    everything else, its image parts removed.
    """
    body = dict(body)
    cleaned_state, part_urls = extract_message_images(body.get("state"))
    if part_urls:
        field_images = body.get("images") or []
        if not isinstance(field_images, list):
            raise ValueError("images must be an array of data URLs")
        body["state"] = cleaned_state
        body["images"] = list(field_images) + part_urls
    return body


def _concentration(probabilities: dict[str, float]) -> float:
    """Normalized concentration (Gini purity): ``(n·Σpᵢ² − 1) / (n − 1)``.

    ``0`` means the options are equally likely, ``1`` means all mass on one.
    This is the ``confidence`` definition of the Cloudflare Workers AI
    clef-flash endpoint — verified against the hosted API to 4 decimals on 48
    choice/score answers (2026-10-05). TypeSafe's margin/spread formulas and
    OpenJev's entropy formula are the other definitions in the wild.
    """
    values = list(probabilities.values())
    count = len(values)
    if count < 2:
        return 1.0
    purity = sum(value * value for value in values)
    return max(0.0, min(1.0, (count * purity - 1.0) / (count - 1.0)))


def choice_confidence(probabilities: dict[str, float]) -> float:
    """``confidence`` for a choice answer (see :func:`_concentration`)."""
    return _concentration(probabilities)


def score_confidence(probabilities: dict[str, float]) -> float:
    """``confidence`` for a score answer (see :func:`_concentration`)."""
    return _concentration(probabilities)


def normalize_response(response: dict) -> dict:
    """Rewrite ``confidence`` on choice/score answers with the standard formula.

    The engine reports the chosen option's probability; the wire contract defines
    confidence as normalized concentration instead (Gini purity, matching the
    Cloudflare Workers AI clef-flash endpoint). Probability distributions and
    every other field are left untouched; ``noul`` answers have no confidence
    field, as per the standard.
    """
    answers = response.get("answers")
    if not isinstance(answers, dict):
        return response
    for answer in answers.values():
        if not isinstance(answer, dict):
            continue
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or not probabilities:
            continue
        if answer.get("type") == "choice":
            answer["confidence"] = round(choice_confidence(probabilities), 4)
        elif answer.get("type") == "score":
            answer["confidence"] = round(score_confidence(probabilities), 4)
    return response
