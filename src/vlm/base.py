from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VLMGenerationResult:
    text: str
    skipped: bool = False
    skip_reason: str | None = None


class VLMRunner(ABC):
    backend: str

    @abstractmethod
    def generate(self, image_path: str | None, prompt: str, **kwargs) -> str:
        """Generate a field answer from an optional image and text prompt."""

    def generate_result(self, image_path: str | None, prompt: str, **kwargs) -> VLMGenerationResult:
        return VLMGenerationResult(text=self.generate(image_path, prompt, **kwargs))
