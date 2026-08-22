"""Vision-Language model interface, adapters, and a mock model for tests."""

from __future__ import annotations

from grounded_memory.models.base import GenerationOutput, VisionLanguageModel
from grounded_memory.models.mock import MockVLM

__all__ = ["VisionLanguageModel", "GenerationOutput", "MockVLM", "load_model"]


def load_model(name: str, models_cfg: dict | None = None) -> VisionLanguageModel:
    """Instantiate a VLM adapter by registry name (see configs/models.yaml).

    Heavy adapters are imported lazily so that importing this package never
    requires torch/transformers.
    """
    from grounded_memory.config import load_config

    models_cfg = models_cfg or load_config("models")
    entry = models_cfg.get("models", {}).get(name)
    if entry is None:
        raise KeyError(f"Model {name!r} not found in configs/models.yaml")
    adapter = entry["adapter"]
    defaults = models_cfg.get("defaults", {})
    if adapter == "qwen_vl":
        from grounded_memory.models.qwen_vl import QwenVLModel

        return QwenVLModel(entry, defaults)
    if adapter == "llava":
        from grounded_memory.models.llava import LlavaModel

        return LlavaModel(entry, defaults)
    if adapter == "mock":
        return MockVLM()
    if adapter == "ollama":
        from grounded_memory.models.ollama_vlm import OllamaVLM

        return OllamaVLM(entry, defaults)
    raise ValueError(f"Unknown adapter {adapter!r} for model {name!r}")
