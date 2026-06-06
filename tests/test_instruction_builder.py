from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.instruction_builder import build_instruction_sample, write_instruction_jsonl


def _row(image_path: str = "missing.png") -> dict:
    return {
        "sample_id": "s1",
        "doc_id": "d1",
        "doc_type": "receipt",
        "image_path": image_path,
        "question": "What is total?",
        "answer": "123",
        "field_name": "total_amount",
        "field_type": "amount",
        "ocr_text": "Total: 123",
        "bbox": None,
        "source_dataset": "mock",
        "image_exists": Path(image_path).exists(),
    }


def test_instruction_strategies() -> None:
    image = Path("outputs/test_artifacts/instruction_image.png")
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"not a real image but it exists")

    ocr_only = build_instruction_sample(_row(str(image)), "ocr_only")
    image_only = build_instruction_sample(_row(str(image)), "image_only")
    image_ocr = build_instruction_sample(_row(str(image)), "image_ocr")

    assert ocr_only["messages"][0]["content"][0]["type"] == "text"
    assert "OCR text" not in image_only["messages"][0]["content"][-1]["text"]
    assert image_only["messages"][0]["content"][0]["type"] == "image"
    assert "OCR text: Total: 123" in image_ocr["messages"][0]["content"][-1]["text"]


def test_instruction_missing_image_is_unavailable() -> None:
    sample = build_instruction_sample(_row("outputs/test_artifacts/no_such_image.png"), "image_ocr")
    assert sample["unavailable"] is True
    assert sample["skip_reason"] == "image_path not found"


def test_instruction_jsonl_output() -> None:
    root = Path("outputs/test_artifacts/instructions")
    root.mkdir(parents=True, exist_ok=True)
    input_path = root / "qa.csv"
    output_path = root / "instructions.jsonl"
    pd.DataFrame([_row()]).to_csv(input_path, index=False)

    write_instruction_jsonl(input_path, "ocr_only", output_path)

    line = output_path.read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert payload["strategy"] == "ocr_only"
    assert payload["answer"] == "123"
