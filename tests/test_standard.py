"""Tests for the System One wire-standard layer — no GPU, no model, no torch.

Run with pytest, or directly: `python3 tests/test_standard.py`.

The confidence expectations match the Cloudflare Workers AI clef-flash
definition: normalized concentration (Gini purity) ``(n·Σpᵢ² − 1)/(n − 1)``,
verified against the hosted endpoint to 4 decimals (2026-10-05).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prima_ratio import standard

PNG = "data:image/png;base64,QUJD"


def message(content):
    return {"role": "user", "content": content}


def image_part(url):
    return {"type": "image_url", "image_url": {"url": url}}


def test_extract_array_of_messages():
    state = [message([{"type": "text", "text": "Look at this."}, image_part(PNG)])]
    cleaned, images = standard.extract_message_images(state)
    assert images == [PNG]
    assert cleaned[0]["content"] == [{"type": "text", "text": "Look at this."}]
    # the input is never mutated
    assert state[0]["content"][1]["type"] == "image_url"


def test_extract_messages_object_preserves_metadata():
    state = {"messages": [message("plain text")], "meta": {"source": "ticket-7"}}
    cleaned, images = standard.extract_message_images(state)
    assert images == []
    assert cleaned["messages"][0]["content"] == "plain text"
    assert cleaned["meta"] == {"source": "ticket-7"}


def test_extract_images_left_to_right_and_string_form():
    state = [
        message([
            image_part("data:image/jpeg;base64,QQ=="),
            {"type": "text", "text": "and"},
            {"type": "image_url", "image_url": "data:image/jpeg;base64,Qg=="},
        ])
    ]
    cleaned, images = standard.extract_message_images(state)
    assert images == ["data:image/jpeg;base64,QQ==", "data:image/jpeg;base64,Qg=="]
    assert cleaned[0]["content"] == [{"type": "text", "text": "and"}]


def test_non_message_states_pass_through():
    for state in ["plain", {"invoice": {"id": 1}}, [1, 2, 3], [{"nope": True}], []]:
        cleaned, images = standard.extract_message_images(state)
        assert cleaned == state and images == []


def test_image_part_without_data_url_raises():
    state = [message([image_part("https://example.com/a.png")])]
    try:
        standard.extract_message_images(state)
    except ValueError:
        pass
    else:
        raise AssertionError("a non data URL image_url part must raise ValueError")


def test_normalize_request_field_first_then_parts():
    field = "data:image/jpeg;base64,RklFTEQ="
    body = {
        "model": "m",
        "state": [message([image_part(PNG)])],
        "images": [field],
        "questions": {},
    }
    out = standard.normalize_request(body)
    assert out["images"] == [field, PNG]
    assert out["state"][0]["content"] == []
    # the input request is never mutated
    assert body["images"] == [field]
    assert body["state"][0]["content"][0]["type"] == "image_url"


def test_normalize_request_rejects_non_list_images_when_merging():
    body = {"model": "m", "state": [message([image_part(PNG)])], "images": "data:image/png;base64,QQ==", "questions": {}}
    try:
        standard.normalize_request(body)
    except ValueError:
        pass
    else:
        raise AssertionError("a non-list images field must raise ValueError when merging parts")


def test_normalize_request_plain_states_unchanged():
    body = {"model": "m", "state": "hello", "images": ["data:image/png;base64,QQ=="], "questions": {}}
    out = standard.normalize_request(body)
    assert out["state"] == "hello"
    assert out["images"] == ["data:image/png;base64,QQ=="]


def test_confidence_matches_cloudflare_workers_ai():
    # Verified against the hosted @cf/cloudflare/clef-flash endpoint (2026-10-05).
    # choice, 3 options (their docs example): reported 0.817
    assert abs(standard.choice_confidence({"billing": 0.0505, "technical": 0.9355, "sales": 0.014}) - 0.817) < 1e-3
    # choice, 3 options (corpus sample): reported 0.3036
    assert abs(standard.choice_confidence({"bulk": 0.0072, "company": 0.6429, "personal": 0.3499}) - 0.3036) < 1e-3
    # choice, 7 options (corpus sample): reported 0.6044
    seven = {"phishing_credentials": 0.0095, "bec": 0.0385, "brand_spoofing": 0.8062, "payment_lure": 0.0055,
             "advance_fee_scam": 0.0041, "malware_attachment": 0.0683, "qr_harvesting": 0.0679}
    assert abs(standard.choice_confidence(seven) - 0.6044) < 1e-3
    # score, 4 levels (their docs example): reported 0.5005
    assert abs(standard.score_confidence({"0": 0.0151, "1": 0.0144, "2": 0.2077, "3": 0.7628}) - 0.5005) < 1e-3
    # edges: uniform -> 0, one-hot -> 1, two options -> (2p-1)^2
    assert standard.choice_confidence({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}) == 0.0
    assert standard.choice_confidence({"a": 1.0, "b": 0.0, "c": 0.0}) == 1.0
    assert abs(standard.choice_confidence({"a": 0.9, "b": 0.1}) - 0.64) < 1e-9
    # score edges
    assert standard.score_confidence({"0": 0.5, "1": 0.5}) == 0.0
    assert standard.score_confidence({"0": 0.0, "1": 0.0, "2": 1.0, "3": 0.0}) == 1.0
    assert standard.score_confidence({"0": 0.0, "1": 1.0}) == 1.0


def test_normalize_response_rewrites_only_confidence():
    response = {
        "model": "m",
        "answers": {
            "malevola": {"type": "noul", "noul": 0.9},
            "route": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.9049,  # engine value (pmax) must be replaced
                "probabilities": {"billing": 0.9049, "shipping": 0.0275, "technical": 0.0676},
            },
            "urgency": {
                "type": "score",
                "score": 2.2821,
                "confidence": 0.5478,
                "legend": {"0": "a", "1": "b", "2": "c", "3": "d"},
                "probabilities": {"0": 0.036, "1": 0.1937, "2": 0.2225, "3": 0.5478},
            },
        },
        "usage": {"input_tokens": 130, "output_tokens": 0},
    }
    out = standard.normalize_response(response)
    assert out["answers"]["malevola"] == {"type": "noul", "noul": 0.9}
    assert out["answers"]["route"]["confidence"] == 0.7363
    assert out["answers"]["route"]["choice"] == "billing"
    assert out["answers"]["route"]["probabilities"] == {"billing": 0.9049, "shipping": 0.0275, "technical": 0.0676}
    assert out["answers"]["urgency"]["confidence"] == 0.1845
    assert out["answers"]["urgency"]["score"] == 2.2821
    assert out["usage"] == {"input_tokens": 130, "output_tokens": 0}


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"ok   {name}")
            except Exception as error:  # noqa: BLE001 - test runner reports everything
                failures += 1
                print(f"FAIL {name}: {error!r}")
    print(f"\n{'FAILED' if failures else 'PASSED'} — {failures} failure(s)")
    sys.exit(1 if failures else 0)
