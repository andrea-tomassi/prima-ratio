"""System One wire-standard conformance for the HTTP layer.

Two public references define the contract applied here:

- TypeSafe's System One API (https://docs.typesafe.ai): ``state`` is a string
  or a structured JSON value, answers carry probabilities, and ``confidence``
  is a 0-1 concentration measure defined per question type
  (https://docs.typesafe.ai/confidence).
- The multimodal extension — the reference cited by the llama.cpp
  ``/v1/systemone`` docs (https://jev-skills.github.io/openjev-multimodal/api):
  images travel either in a top-level ``images`` array of data URLs and/or as
  ``image_url`` parts inside a chat-message state; all images are read before
  the state in the prompt, the ``images`` field first, and the parts are
  removed from the state.

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


def choice_confidence(probabilities: dict[str, float]) -> float:
    """TypeSafe Choice confidence: ``(n·pmax − 1) / (n − 1)``, clamped to 0-1.

    ``0`` means all options are equally likely, ``1`` means a single peak.
    """
    values = list(probabilities.values())
    count = len(values)
    if count < 2:
        return 1.0
    peak = max(values)
    return max(0.0, min(1.0, (count * peak - 1.0) / (count - 1.0)))


def score_confidence(probabilities: dict[str, float]) -> float:
    """TypeSafe Score confidence: ``1 − spread/evenSpread`` (docs.typesafe.ai/confidence).

    ``spread`` is the probability mass weighted by its distance from the peak
    level; ``evenSpread`` is the spread of a flat distribution over the levels.
    ``0`` means the levels are equally likely, ``1`` means all mass on the peak.
    """
    count = len(probabilities)
    try:
        values = [probabilities[str(index)] for index in range(count)]
    except KeyError:  # levels not keyed 0..n-1: keep the reported order
        values = list(probabilities.values())
    if count < 2:
        return 1.0
    peak = max(range(count), key=values.__getitem__)
    spread = sum(value * abs(index - peak) for index, value in enumerate(values))
    even_spread = sum(abs(index - (count - 1) / 2) for index in range(count)) / count
    if even_spread == 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - spread / even_spread))


def normalize_response(response: dict) -> dict:
    """Rewrite ``confidence`` on choice/score answers with the standard formulas.

    The engine reports the chosen option's probability; the wire standard defines
    confidence as a concentration measure instead (docs.typesafe.ai/confidence).
    Probability distributions and every other field are left untouched; ``noul``
    answers have no confidence field, as per the standard.
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
