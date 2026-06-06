from __future__ import annotations

from abc import ABC, abstractmethod


class VLMRunner(ABC):
    backend: str

    @abstractmethod
    def generate(self, image_path: str | None, prompt: str, **kwargs) -> str:
        """Generate a field answer from an optional image and text prompt."""
