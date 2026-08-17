"""Ollama VLM adapter — run local vision models (qwen2.5vl, llava, ...) via Ollama.

Implements the common ``VisionLanguageModel`` interface by calling a local Ollama
server's ``/api/generate`` endpoint with the image base64-encoded. Great on Apple
Silicon: Metal-accelerated, pre-quantized, no Hugging Face downloads, no MPS
`device_map` juggling, no bitsandbytes.

Notes for this project's experiments:
- Determinism: temperature is forwarded and a fixed ``seed`` is set.
- Structured (Condition C): when the prompt asks for JSON, ``format="json"`` is
  set so Ollama constrains output to valid JSON.
- Timings come from Ollama's own counters (nanoseconds): prompt-eval, generation,
  load, total. Ollama does NOT expose a separate vision-encoding time, so the
  E3.4 latency breakdown is coarser than a raw Transformers forward pass.
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any

from grounded_memory import ollama_client
from grounded_memory.models.base import GenerationOutput, VisionLanguageModel


class OllamaVLM(VisionLanguageModel):
    def __init__(self, entry: dict[str, Any], defaults: dict[str, Any] | None = None) -> None:
        defaults = defaults or {}
        self.name = entry.get("name", entry.get("ollama_model", "ollama_vlm").replace(":", "_"))
        self.ollama_model = entry["ollama_model"]        # e.g. "qwen2.5vl:latest"
        self.model_id = f"ollama/{self.ollama_model}"
        self.revision = entry.get("revision", "")         # optional: pin a digest string
        self.endpoint = entry.get("endpoint") or ollama_client.default_endpoint()
        self.seed = int(entry.get("seed", defaults.get("seed", 0)))
        self.timeout = float(entry.get("timeout", 300.0))

    @staticmethod
    def _encode_image(image: Any) -> str | None:
        """Return base64 of an image given a path, bytes, or PIL image; None if no image."""
        if image is None:
            return None
        if isinstance(image, str):
            data = Path(image).read_bytes()
        elif isinstance(image, bytes):
            data = image
        else:  # assume PIL.Image
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            data = buf.getvalue()
        return base64.b64encode(data).decode("utf-8")

    @staticmethod
    def _wants_json(prompt: str) -> bool:
        # The structured (Condition C) prompt is the only one that requests JSON.
        return "json" in prompt.lower()

    def generate(
        self,
        image: Any,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> GenerationOutput:
        payload: dict[str, Any] = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "seed": self.seed,
                "num_predict": max_new_tokens,
            },
        }
        img_b64 = self._encode_image(image)
        if img_b64 is not None:
            payload["images"] = [img_b64]
        if self._wants_json(prompt):
            payload["format"] = "json"

        t0 = time.perf_counter()
        resp = ollama_client.post_json(
            "/api/generate", payload, endpoint=self.endpoint, timeout=self.timeout
        )
        wall = time.perf_counter() - t0

        text = resp.get("response", "")
        ns = 1e9
        comp = {
            "load": resp.get("load_duration", 0) / ns,
            "prompt_eval": resp.get("prompt_eval_duration", 0) / ns,
            "generation": resp.get("eval_duration", 0) / ns,
            "total": resp.get("total_duration", 0) / ns,
        }
        latency = comp["total"] or wall
        return GenerationOutput(
            text=text,
            input_tokens=resp.get("prompt_eval_count"),
            output_tokens=resp.get("eval_count"),
            latency_s=latency,
            component_latencies=comp,
            metadata={"backend": "ollama", "model": self.ollama_model},
        )


__all__ = ["OllamaVLM"]
