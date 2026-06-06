from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.vlm.base import VLMRunner


class Qwen25VLRunner(VLMRunner):
    backend = "qwen2_5_vl"

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = "auto") -> None:
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen2.5-VL dependencies are not installed. Install torch, transformers, and qwen-vl-utils "
                "outside the repository before running this backend."
            ) from exc
        self.torch = torch
        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(model_name)
        device_map = "auto" if device == "auto" else None
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, device_map=device_map)
        if device != "auto":
            self.model = self.model.to(device)

    def generate(self, image_path: str | None, prompt: str, **kwargs) -> str:
        content: list[dict[str, str]] = []
        if image_path:
            path = Path(image_path)
            if not path.exists():
                path = Path.cwd().joinpath(path)
            if path.exists():
                content.append({"type": "image", "image": Image.open(path).convert("RGB")})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        if hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=kwargs.get("max_new_tokens", 64))
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
