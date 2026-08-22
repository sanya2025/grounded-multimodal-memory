"""Tests for the Ollama VLM adapter (mocked endpoint; no server needed)."""

from __future__ import annotations

from unittest.mock import patch

from grounded_memory.models.ollama_vlm import OllamaVLM


def _entry(**overrides) -> dict:
    return {"ollama_model": "qwen3-vl:8b", **overrides}


def test_generate_uses_response_field_when_present():
    resp = {
        "response": "a cat on a mat",
        "thinking": "reasoning that should be ignored",
        "done_reason": "stop",
        "eval_count": 5,
        "prompt_eval_count": 10,
    }
    with patch("grounded_memory.models.ollama_vlm.ollama_client.post_json", return_value=resp):
        out = OllamaVLM(_entry()).generate(None, "describe")
    assert out.text == "a cat on a mat"
    assert out.metadata["thinking_fallback"] is False


def test_generate_falls_back_to_thinking_when_response_empty():
    resp = {
        "response": "",
        "thinking": 'reasoning...\n{"scene_summary": "a room"}',
        "done_reason": "stop",
        "eval_count": 900,
        "prompt_eval_count": 10,
    }
    with patch("grounded_memory.models.ollama_vlm.ollama_client.post_json", return_value=resp):
        out = OllamaVLM(_entry()).generate(None, "describe")
    assert out.text == resp["thinking"]
    assert out.metadata["thinking_fallback"] is True


def test_generate_empty_response_and_empty_thinking_stays_empty():
    resp = {"response": "", "thinking": "", "done_reason": "length", "eval_count": 512}
    with patch("grounded_memory.models.ollama_vlm.ollama_client.post_json", return_value=resp):
        out = OllamaVLM(_entry()).generate(None, "describe")
    assert out.text == ""
    assert out.metadata["thinking_fallback"] is False
    assert out.metadata["done_reason"] == "length"


def test_generate_no_thinking_field_at_all_is_backward_compatible():
    resp = {"response": "plain answer", "eval_count": 5, "prompt_eval_count": 10}
    with patch("grounded_memory.models.ollama_vlm.ollama_client.post_json", return_value=resp):
        out = OllamaVLM(_entry()).generate(None, "describe")
    assert out.text == "plain answer"
    assert out.metadata["thinking_fallback"] is False


def test_generate_sends_format_json_only_for_json_prompts():
    resp = {"response": "ok"}
    with patch(
        "grounded_memory.models.ollama_vlm.ollama_client.post_json", return_value=resp
    ) as mock_post:
        OllamaVLM(_entry()).generate(None, "return a JSON object")
    payload = mock_post.call_args.args[1]
    assert payload["format"] == "json"

    with patch(
        "grounded_memory.models.ollama_vlm.ollama_client.post_json", return_value=resp
    ) as mock_post:
        OllamaVLM(_entry()).generate(None, "describe the image in prose")
    payload = mock_post.call_args.args[1]
    assert "format" not in payload
