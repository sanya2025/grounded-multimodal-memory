"""LLaVA-NeXT adapter (skeleton).

Requires the optional ``models`` extra (torch + transformers) and a GPU in
practice. Same lazy-load pattern as the Qwen adapter.

Docs:
- https://github.com/LLaVA-VL/LLaVA-NeXT
- https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf
"""

from __future__ import annotations

import time
from typing import Any

from grounded_memory.models.base import GenerationOutput, VisionLanguageModel


class LlavaModel(VisionLanguageModel):
    def __init__(self, entry: dict[str, Any], defaults: dict[str, Any]) -> None:
        self.name = "llava_next_7b"
        self.model_id = entry["hf_id"]
        self.revision = entry.get("revision", "main")
        self.dtype = entry.get("dtype", defaults.get("dtype", "float16"))
        self.device = entry.get("device", defaults.get("device", "cpu"))
        self.quantization = entry.get("quantization")
        self._model = None
        self._processor = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        # TODO(hardware): validated on GPU only.
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        dtype = _torch_dtype(self.dtype)
        self._model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=dtype, revision=self.revision, device_map=self.device
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)

    def generate(
        self,
        image: Any,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> GenerationOutput:
        self._lazy_load()
        import torch

        pil_image = _to_pil(image)
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        text = self._processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self._processor(images=pil_image, text=text, return_tensors="pt").to(
            self._model.device
        )
        t0 = time.perf_counter()
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )
        latency = time.perf_counter() - t0
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        out_text = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return GenerationOutput(
            text=out_text,
            input_tokens=int(inputs["input_ids"].shape[1]),
            output_tokens=int(trimmed.shape[1]),
            latency_s=latency,
            component_latencies={"generation": latency},
        )


def _torch_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
        name, torch.float32
    )


def _to_pil(image: Any):
    from PIL import Image

    if image is None:
        raise ValueError("LLaVA-NeXT requires an image.")
    if isinstance(image, (str, bytes)):
        return Image.open(image).convert("RGB")
    return image


__all__ = ["LlavaModel"]
