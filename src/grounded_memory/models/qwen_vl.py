"""Qwen2.5-VL adapter (skeleton).

Requires the optional ``models`` extra (torch + transformers) and, in practice,
a GPU. Kept import-light: heavy imports happen inside ``_lazy_load`` so the class
can be constructed and unit-tested for config wiring without downloading weights.

Docs:
- https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl
- https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
"""

from __future__ import annotations

import time
from typing import Any

from grounded_memory.models.base import GenerationOutput, VisionLanguageModel


class QwenVLModel(VisionLanguageModel):
    def __init__(self, entry: dict[str, Any], defaults: dict[str, Any]) -> None:
        self.name = "qwen2_5_vl_7b"
        self.model_id = entry["hf_id"]
        self.revision = entry.get("revision", "main")
        self.dtype = entry.get("dtype", defaults.get("dtype", "bfloat16"))
        self.device = entry.get("device", defaults.get("device", "cpu"))
        self.quantization = entry.get("quantization")
        self._model = None
        self._processor = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        # TODO(hardware): validated on GPU only. Loading 7B weights on CPU is slow.
        import torch  # noqa: F401
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        dtype = _torch_dtype(self.dtype)
        load_kwargs: dict[str, Any] = {"torch_dtype": dtype, "revision": self.revision}
        if self.quantization in ("int8", "int4"):
            # TODO(E3.2): guard by hardware; requires bitsandbytes on supported GPUs.
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=self.quantization == "int8",
                load_in_4bit=self.quantization == "int4",
            )
        else:
            load_kwargs["device_map"] = self.device
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, **load_kwargs
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
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], images=[pil_image], return_tensors="pt")
        inputs = inputs.to(self._model.device)
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
        raise ValueError("Qwen2.5-VL requires an image.")
    if isinstance(image, (str, bytes)):
        return Image.open(image).convert("RGB")
    return image


__all__ = ["QwenVLModel"]
