from __future__ import annotations

from src.vlm.base import VLMRunner


class DummyVLMRunner(VLMRunner):
    backend = "dummy"
    skipped = True
    skip_reason = "dummy backend does not run a real VLM"

    def generate(self, image_path: str | None, prompt: str, **kwargs) -> str:
        return ""
