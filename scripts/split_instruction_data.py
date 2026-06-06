from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
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
    return rows


def split_instruction_rows(rows: list[dict[str, Any]], eval_ratio: float = 0.2, seed: int = 42) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    if not 0 < eval_ratio < 1:
        raise ValueError("--eval-ratio must be between 0 and 1")
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    duplicates = sorted({sample_id for sample_id in sample_ids if sample_ids.count(sample_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate sample_id values found: {', '.join(duplicates[:10])}")
    warnings = []
    if len(rows) < 10:
        warnings.append(f"Only {len(rows)} samples found; split is valid but too small for training.")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    eval_count = max(1, round(len(shuffled) * eval_ratio)) if len(shuffled) > 1 else 0
    eval_rows = shuffled[:eval_count]
    train_rows = shuffled[eval_count:]
    if not train_rows:
        warnings.append("No train samples after split.")
    if not eval_rows:
        warnings.append("No eval samples after split.")
    return train_rows, eval_rows, warnings


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target


def split_instruction_file(
    input_path: str | Path,
    train_output: str | Path,
    eval_output: str | Path,
    eval_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    rows = load_jsonl(input_path)
    train_rows, eval_rows, warnings = split_instruction_rows(rows, eval_ratio=eval_ratio, seed=seed)
    write_jsonl(train_output, train_rows)
    write_jsonl(eval_output, eval_rows)
    return {
        "input": str(input_path),
        "train_output": str(train_output),
        "eval_output": str(eval_output),
        "num_samples": len(rows),
        "num_train": len(train_rows),
        "num_eval": len(eval_rows),
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split FlowDoc-VLM instruction JSONL into train and eval files.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = split_instruction_file(args.input, args.train_output, args.eval_output, args.eval_ratio, args.seed)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {result['num_train']} train samples to {result['train_output']}")
    print(f"Saved {result['num_eval']} eval samples to {result['eval_output']}")
    for warning in result["warnings"]:
        print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
