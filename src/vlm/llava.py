from __future__ import annotations

from pathlib import Path

from src.vlm.base import VLMGenerationResult, VLMRunner


class LLaVARunner(VLMRunner):
    backend = "llava"

    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf", device: str = "auto") -> None:
        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "LLaVA dependencies are not installed. Install torch and transformers outside the repository "
                "before running this backend."
            ) from exc
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_name)
        if device != "auto":
            self.model = self.model.to(device)

    def generate(self, image_path: str | None, prompt: str, **kwargs) -> str:
        return self.generate_result(image_path, prompt, **kwargs).text

    def generate_result(self, image_path: str | None, prompt: str, **kwargs) -> VLMGenerationResult:
        if not image_path:
            return VLMGenerationResult(text="", skipped=True, skip_reason="LLaVA backend requires image_path for inference.")
        path = Path(image_path)
        if not path.exists():
            path = Path.cwd().joinpath(path)
        if not path.exists():
            return VLMGenerationResult(text="", skipped=True, skip_reason=f"Image not found for LLaVA inference: {image_path}")
        try:
            from PIL import Image
            image = Image.open(path).convert("RGB")
            llava_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
            inputs = self.processor(text=llava_prompt, images=image, return_tensors="pt")
            if hasattr(self.model, "device"):
                inputs = inputs.to(self.model.device)
            output = self.model.generate(**inputs, max_new_tokens=kwargs.get("max_new_tokens", 64))
            text = self.processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
            return VLMGenerationResult(text=text)
        except Exception as exc:
            return VLMGenerationResult(text="", skipped=True, skip_reason=str(exc))
