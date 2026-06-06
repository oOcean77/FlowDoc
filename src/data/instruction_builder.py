from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.dataset_adapter import load_local_csv


STRATEGIES = {"ocr_only", "image_only", "image_ocr"}


def _path_exists(path_value: object) -> bool:
    path_text = "" if path_value is None else str(path_value).strip()
    if not path_text:
        return False
    path = Path(path_text)
    return path.exists() or Path.cwd().joinpath(path).exists()


def _text_prompt(row: dict[str, Any], strategy: str) -> str:
    question = row.get("question", "")
    if strategy == "ocr_only":
        return f"OCR text: {row.get('ocr_text', '')}\nQuestion: {question}\nAnswer with only the field value."
    if strategy == "image_only":
        return f"Question: {question}\nAnswer with only the field value."
    return f"OCR text: {row.get('ocr_text', '')}\nQuestion: {question}\nAnswer with only the field value."


def build_instruction_sample(row: dict[str, Any], strategy: str) -> dict[str, Any]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unsupported strategy '{strategy}'. Expected one of {sorted(STRATEGIES)}.")
    image_path = str(row.get("image_path", ""))
    needs_image = strategy in {"image_only", "image_ocr"}
    image_exists = bool(row.get("image_exists", True)) and _path_exists(image_path)
    unavailable = needs_image and not image_exists
    content: list[dict[str, str]] = []
    if needs_image and image_exists:
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": _text_prompt(row, strategy)})
    sample = {
        "sample_id": row.get("sample_id", ""),
        "doc_id": row.get("doc_id", ""),
        "doc_type": row.get("doc_type", ""),
        "image_path": image_path,
        "strategy": strategy,
        "messages": [{"role": "user", "content": content}],
        "answer": row.get("answer", ""),
        "field_name": row.get("field_name", ""),
        "field_type": row.get("field_type", ""),
        "source_dataset": row.get("source_dataset", ""),
        "unavailable": unavailable,
        "skip_reason": "image_path not found" if unavailable else "",
    }
    return sample


def build_instruction_samples(df: pd.DataFrame, strategy: str) -> list[dict[str, Any]]:
    return [build_instruction_sample(row, strategy) for row in df.to_dict(orient="records")]


def write_instruction_jsonl(input_path: str | Path, strategy: str, output_path: str | Path) -> Path:
    df = load_local_csv(input_path)
    samples = build_instruction_samples(df, strategy)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return target
