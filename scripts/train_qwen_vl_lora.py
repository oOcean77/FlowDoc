from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = "outputs/lora/qwen2_5_vl_flowdoc_dryrun"
DEFAULT_LOG_PATH = "outputs/metrics/lora_dryrun_train_log.json"
LABEL_IGNORE_INDEX = -100
VISION_LABEL_MASK_LIMITATION = (
    "Answer-only label mask is implemented for tokenized prompt/answer text. "
    "Qwen2.5-VL image token masking is treated as best-effort and must be audited before formal SFT."
)


@dataclass
class TrainConfig:
    model_name: str
    train_file: str
    eval_file: str
    output_dir: str
    max_train_samples: int
    max_eval_samples: int
    max_steps: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    batch_size: int
    gradient_accumulation_steps: int
    bf16: bool
    fp16: bool
    dry_run: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5-VL LoRA SFT dry-run scaffold for FlowDoc-VLM.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--train-file", default="data/processed/train_instructions.jsonl")
    parser.add_argument("--eval-file", default="data/processed/eval_instructions.jsonl")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-train-samples", type=int, default=20)
    parser.add_argument("--max-eval-samples", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        model_name=args.model_name,
        train_file=args.train_file,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        dry_run=args.dry_run,
    )


def load_instruction_jsonl(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Instruction JSONL not found: {target}")
    rows = []
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def extract_user_text(sample: dict[str, Any]) -> str:
    messages = sample.get("messages", [])
    if not messages:
        return ""
    content = messages[0].get("content", [])
    return "\n".join(item.get("text", "") for item in content if item.get("type") == "text")


def build_answer_only_labels(input_ids: list[int], prompt_token_count: int, ignore_index: int = LABEL_IGNORE_INDEX) -> list[int]:
    if prompt_token_count < 0:
        raise ValueError("prompt_token_count must be non-negative")
    labels = list(input_ids)
    mask_until = min(prompt_token_count, len(labels))
    for index in range(mask_until):
        labels[index] = ignore_index
    return labels


def build_text_training_example(sample: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    prompt = extract_user_text(sample)
    answer = str(sample.get("answer", "")).strip()
    full_text = f"{prompt}\n{answer}"
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full = tokenizer(full_text, add_special_tokens=True)
    input_ids = full["input_ids"]
    labels = build_answer_only_labels(input_ids, len(prompt_ids))
    return {"input_ids": input_ids, "attention_mask": full.get("attention_mask", [1] * len(input_ids)), "labels": labels}


def _base_log(config: TrainConfig, train_samples: int, eval_samples: int) -> dict[str, Any]:
    return {
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "max_steps": config.max_steps,
        "final_loss": None,
        "adapter_saved": False,
        "skipped": False,
        "skip_reason": "",
        "model_name": config.model_name,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "learning_rate": config.learning_rate,
        "label_mask_limitation": VISION_LABEL_MASK_LIMITATION,
        "config": asdict(config),
    }


def write_train_log(log: dict[str, Any], path: str | Path = DEFAULT_LOG_PATH) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def run_dry_run(config: TrainConfig) -> dict[str, Any]:
    train_rows = load_instruction_jsonl(config.train_file, config.max_train_samples)
    eval_rows = load_instruction_jsonl(config.eval_file, config.max_eval_samples)
    log = _base_log(config, len(train_rows), len(eval_rows))
    log["skipped"] = True
    log["skip_reason"] = "dry-run requested; Qwen model is not loaded and no optimizer step is executed"
    log["adapter_saved"] = False
    return log


def _resolve_training_imports() -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments
    except ImportError as exc:
        raise ImportError(
            "Training dependencies missing. Install torch, transformers, qwen-vl-utils, peft, accelerate, and bitsandbytes on AutoDL."
        ) from exc
    return {
        "torch": torch,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "process_vision_info": process_vision_info,
        "AutoProcessor": AutoProcessor,
        "Qwen2_5_VLForConditionalGeneration": Qwen2_5_VLForConditionalGeneration,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def run_training(config: TrainConfig) -> dict[str, Any]:
    train_rows = load_instruction_jsonl(config.train_file, config.max_train_samples)
    eval_rows = load_instruction_jsonl(config.eval_file, config.max_eval_samples)
    log = _base_log(config, len(train_rows), len(eval_rows))
    try:
        deps = _resolve_training_imports()
        processor = deps["AutoProcessor"].from_pretrained(config.model_name)
        tokenizer = getattr(processor, "tokenizer", processor)
        model = deps["Qwen2_5_VLForConditionalGeneration"].from_pretrained(config.model_name, device_map="auto")
        lora_config = deps["LoraConfig"](
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = deps["get_peft_model"](model, lora_config)
        train_dataset = [build_text_training_example(sample, tokenizer) for sample in train_rows]
        eval_dataset = [build_text_training_example(sample, tokenizer) for sample in eval_rows]

        def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
            return tokenizer.pad(batch, padding=True, return_tensors="pt")

        args = deps["TrainingArguments"](
            output_dir=config.output_dir,
            max_steps=config.max_steps,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            bf16=config.bf16,
            fp16=config.fp16,
            logging_steps=1,
            save_steps=config.max_steps,
            report_to=[],
            remove_unused_columns=False,
        )
        trainer = deps["Trainer"](model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=collate)
        result = trainer.train()
        model.save_pretrained(config.output_dir)
        log["final_loss"] = float(result.training_loss) if result.training_loss is not None else None
        log["adapter_saved"] = True
    except Exception as exc:
        log["skipped"] = True
        log["skip_reason"] = str(exc)
    return log


def main(argv: list[str] | None = None) -> int:
    config = config_from_args(parse_args(argv))
    log = run_dry_run(config) if config.dry_run else run_training(config)
    path = write_train_log(log)
    print(f"Saved LoRA dry-run/train log to {path}")
    print(f"skipped={str(log['skipped']).lower()} reason={log['skip_reason']}")
    if log["label_mask_limitation"]:
        print(f"limitation={log['label_mask_limitation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
