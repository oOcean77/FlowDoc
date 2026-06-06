from __future__ import annotations

import json
from pathlib import Path

from scripts.train_qwen_vl_lora import (
    LABEL_IGNORE_INDEX,
    build_answer_only_labels,
    config_from_args,
    load_instruction_jsonl,
    main,
    parse_args,
    run_dry_run,
)


def _instruction(sample_id: str = "s1") -> dict:
    return {
        "sample_id": sample_id,
        "doc_id": "d1",
        "doc_type": "receipt",
        "image_path": "data/samples/mock.png",
        "strategy": "image_ocr",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "OCR text: Total 123\nQuestion: What is total?\nAnswer with only the field value."}],
            }
        ],
        "answer": "123",
        "field_name": "total_amount",
        "field_type": "amount",
        "source_dataset": "mock",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_train_arg_defaults() -> None:
    args = parse_args(["--model-name", "Qwen/Qwen2.5-VL-3B-Instruct"])
    config = config_from_args(args)

    assert config.max_steps == 10
    assert config.batch_size == 1
    assert config.gradient_accumulation_steps == 4
    assert config.lora_r == 8
    assert config.lora_alpha == 16
    assert config.learning_rate == 1e-4
    assert config.max_train_samples == 20
    assert config.max_eval_samples == 8


def test_instruction_jsonl_reading() -> None:
    path = Path("outputs/test_artifacts/train_lora/instructions.jsonl")
    _write_jsonl(path, [_instruction("s1"), _instruction("s2")])

    rows = load_instruction_jsonl(path, max_samples=1)

    assert len(rows) == 1
    assert rows[0]["sample_id"] == "s1"


def test_answer_only_label_mask() -> None:
    labels = build_answer_only_labels([10, 11, 12, 13], prompt_token_count=2)

    assert labels == [LABEL_IGNORE_INDEX, LABEL_IGNORE_INDEX, 12, 13]


def test_dry_run_does_not_load_model_or_train() -> None:
    root = Path("outputs/test_artifacts/train_lora")
    train_path = root / "train.jsonl"
    eval_path = root / "eval.jsonl"
    _write_jsonl(train_path, [_instruction("train_1")])
    _write_jsonl(eval_path, [_instruction("eval_1")])
    config = config_from_args(
        parse_args(
            [
                "--model-name",
                "Qwen/Qwen2.5-VL-3B-Instruct",
                "--train-file",
                str(train_path),
                "--eval-file",
                str(eval_path),
                "--dry-run",
            ]
        )
    )

    log = run_dry_run(config)

    assert log["skipped"] is True
    assert log["adapter_saved"] is False
    assert log["train_samples"] == 1
    assert "dry-run requested" in log["skip_reason"]


def test_dry_run_main_writes_log() -> None:
    root = Path("outputs/test_artifacts/train_lora_main")
    train_path = root / "train.jsonl"
    eval_path = root / "eval.jsonl"
    _write_jsonl(train_path, [_instruction("train_1")])
    _write_jsonl(eval_path, [_instruction("eval_1")])

    exit_code = main(
        [
            "--model-name",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "--train-file",
            str(train_path),
            "--eval-file",
            str(eval_path),
            "--output-dir",
            str(root / "adapter"),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    log_path = Path("outputs/metrics/lora_dryrun_train_log.json")
    assert log_path.exists()
    log = json.loads(log_path.read_text(encoding="utf-8"))
    assert log["skipped"] is True
