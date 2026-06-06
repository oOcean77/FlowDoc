from __future__ import annotations

from pathlib import Path

from src.vlm.base import VLMGenerationResult, VLMRunner


def _looks_like_local_path(model_name: str) -> bool:
    return (
        model_name.startswith(".")
        or model_name.startswith("/")
        or ":\\" in model_name
        or model_name.startswith("\\\\")
    )


class Qwen25VLRunner(VLMRunner):
    backend = "qwen2_5_vl"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "auto",
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        do_sample: bool = False,
        lora_adapter: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.lora_adapter = lora_adapter
        is_local_path = _looks_like_local_path(model_name)
        if is_local_path and not Path(model_name).exists():
            raise FileNotFoundError(f"Local Qwen2.5-VL model path does not exist: {model_name}")
        if lora_adapter and not Path(lora_adapter).exists():
            raise FileNotFoundError(f"LoRA adapter path does not exist: {lora_adapter}")
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen2.5-VL dependencies are not installed. Install torch, transformers, pillow, and qwen-vl-utils "
                "outside the repository before running this backend."
            ) from exc
        self.torch = torch
        self.process_vision_info = process_vision_info
        try:
            local_files_only = not is_local_path
            self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)
            device_map = "auto" if device == "auto" else None
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                device_map=device_map,
                local_files_only=local_files_only,
            )
            if lora_adapter:
                try:
                    from peft import PeftModel
                except ImportError as exc:
                    raise ImportError("PEFT is required to load a LoRA adapter. Install peft before adapter evaluation.") from exc
                self.model = PeftModel.from_pretrained(self.model, lora_adapter)
            if device != "auto":
                self.model = self.model.to(device)
        except Exception as exc:
            adapter_note = f" Adapter path: {lora_adapter}." if lora_adapter else ""
            raise RuntimeError(
                f"Unable to load Qwen2.5-VL model '{model_name}'. If this is a Hugging Face repo id, "
                "download or cache it outside the repository first, then rerun the smoke test. If it is a local path, "
                f"confirm the directory contains a valid Qwen2.5-VL checkpoint.{adapter_note}"
            ) from exc

    def _resolve_image(self, image_path: str | None):
        if not image_path:
            return None
        path = Path(image_path)
        if not path.exists():
            path = Path.cwd().joinpath(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found for Qwen2.5-VL inference: {image_path}")
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required to load image inputs for Qwen2.5-VL.") from exc
        try:
            return Image.open(path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Unable to read image for Qwen2.5-VL inference: {image_path}") from exc

    def generate(self, image_path: str | None, prompt: str, **kwargs) -> str:
        return self.generate_result(image_path, prompt, **kwargs).text

    def generate_result(self, image_path: str | None, prompt: str, **kwargs) -> VLMGenerationResult:
        try:
            content: list[dict] = []
            image = self._resolve_image(image_path)
            if image is not None:
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = self.process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            if hasattr(self.model, "device"):
                inputs = inputs.to(self.model.device)
            max_new_tokens = int(kwargs.get("max_new_tokens", self.max_new_tokens))
            do_sample = bool(kwargs.get("do_sample", self.do_sample))
            temperature = float(kwargs.get("temperature", self.temperature))
            generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
            if do_sample:
                generate_kwargs["temperature"] = temperature
            generated_ids = self.model.generate(**inputs, **generate_kwargs)
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=False)
            ]
            text_output = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            return VLMGenerationResult(text=text_output)
        except Exception as exc:
            return VLMGenerationResult(text="", skipped=True, skip_reason=str(exc))
