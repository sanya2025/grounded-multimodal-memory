"""Common Vision-Language model interface.

All experiments call this abstraction so models are swappable and injectable.
``GenerationOutput`` carries token counts and latency so efficiency experiments
(E3) can measure without special-casing each adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationOutput:
    """Result of one generation call."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_s: float | None = None
    component_latencies: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class VisionLanguageModel(ABC):
    """Abstract base for image+text -> text models.

    Implementations MUST be deterministic when ``temperature == 0``.
    """

    #: Human-readable id, e.g. "qwen2_5_vl_7b".
    name: str = "vlm"
    #: Exact HF id / revision, recorded with every prediction.
    model_id: str = ""
    revision: str = ""

    @abstractmethod
    def generate(
        self,
        image: Any,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> GenerationOutput:
        """Generate text conditioned on an image and prompt.

        ``image`` accepts a filesystem path, a PIL image, or None (text-only).
        """
        raise NotImplementedError

    def describe(self) -> dict[str, str]:
        """Provenance fields for experiment tracking."""
        return {"name": self.name, "model_id": self.model_id, "revision": self.revision}


__all__ = ["VisionLanguageModel", "GenerationOutput"]
