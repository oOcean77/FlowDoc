from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.split_instruction_data import split_instruction_file, split_instruction_rows


def _rows(count: int) -> list[dict]:
    return [
        {
            "sample_id": f"s{index}",
            "doc_id": f"d{index}",
            "strategy": "image_ocr",
            "image_path": "data/samples/mock.png",
            "answer": str(index),
        }
        for index in range(count)
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_split_instruction_file_train_eval() -> None:
    root = Path("outputs/test_artifacts/split")
    input_path = root / "instructions.jsonl"
    train_path = root / "train.jsonl"
    eval_path = root / "eval.jsonl"
    _write_jsonl(input_path, _rows(10))

    result = split_instruction_file(input_path, train_path, eval_path, eval_ratio=0.2, seed=42)

    assert result["num_train"] == 8
    assert result["num_eval"] == 2
    train_ids = {json.loads(line)["sample_id"] for line in train_path.read_text(encoding="utf-8").splitlines()}
    eval_ids = {json.loads(line)["sample_id"] for line in eval_path.read_text(encoding="utf-8").splitlines()}
    assert train_ids.isdisjoint(eval_ids)


def test_split_instruction_seed_is_reproducible() -> None:
    train_a, eval_a, _ = split_instruction_rows(_rows(12), eval_ratio=0.25, seed=7)
    train_b, eval_b, _ = split_instruction_rows(_rows(12), eval_ratio=0.25, seed=7)

    assert [row["sample_id"] for row in train_a] == [row["sample_id"] for row in train_b]
    assert [row["sample_id"] for row in eval_a] == [row["sample_id"] for row in eval_b]


def test_split_duplicate_sample_id_fails() -> None:
    rows = _rows(2)
    rows[1]["sample_id"] = rows[0]["sample_id"]

    with pytest.raises(ValueError, match="Duplicate sample_id"):
        split_instruction_rows(rows)


def test_split_tiny_dataset_warns() -> None:
    train_rows, eval_rows, warnings = split_instruction_rows(_rows(3), eval_ratio=0.2, seed=42)

    assert len(train_rows) + len(eval_rows) == 3
    assert any("too small" in warning for warning in warnings)
