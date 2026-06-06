from __future__ import annotations

import json
from pathlib import Path

from scripts.check_sft_readiness import check_sft_readiness, write_readiness_report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_sft_readiness_missing_instruction_file() -> None:
    report = check_sft_readiness(
        instruction_file="outputs/test_artifacts/missing_instructions.jsonl",
        model_name="C:\\models\\missing-qwen",
        baseline_metrics="outputs/test_artifacts/missing_metrics.json",
        train_file="outputs/test_artifacts/missing_train.jsonl",
        eval_file="outputs/test_artifacts/missing_eval.jsonl",
        min_samples=2,
    )

    assert report["ready"] is False
    assert any("Instruction JSONL not found" in issue for issue in report["blocking_issues"])


def test_sft_readiness_missing_baseline_metrics() -> None:
    root = Path("outputs/test_artifacts/sft_readiness_missing_baseline")
    instruction = root / "instructions.jsonl"
    train = root / "train.jsonl"
    eval_file = root / "eval.jsonl"
    rows = [
        {
            "sample_id": "s1",
            "strategy": "image_ocr",
            "image_path": str(root / "image.png"),
            "answer": "123",
        },
        {
            "sample_id": "s2",
            "strategy": "image_ocr",
            "image_path": str(root / "image.png"),
            "answer": "456",
        },
    ]
    (root / "image.png").parent.mkdir(parents=True, exist_ok=True)
    (root / "image.png").write_bytes(b"exists")
    _write_jsonl(instruction, rows)
    _write_jsonl(train, rows[:1])
    _write_jsonl(eval_file, rows[1:])

    report = check_sft_readiness(instruction, "Qwen/Qwen2.5-VL-3B-Instruct", root / "missing.json", train, eval_file, min_samples=2)

    assert report["ready"] is False
    assert any("Baseline metrics file not found" in issue for issue in report["blocking_issues"])


def test_sft_readiness_report_generation_for_mock_instruction() -> None:
    root = Path("outputs/test_artifacts/sft_readiness")
    image = root / "image.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"exists")
    instruction = root / "instructions.jsonl"
    train = root / "train.jsonl"
    eval_file = root / "eval.jsonl"
    metrics = root / "qwen_metrics.json"
    rows = [
        {
            "sample_id": f"s{index}",
            "strategy": "image_ocr",
            "image_path": str(image),
            "answer": "123",
        }
        for index in range(3)
    ]
    _write_jsonl(instruction, rows)
    _write_jsonl(train, rows[:2])
    _write_jsonl(eval_file, rows[2:])
    metrics.write_text(json.dumps({"skipped": False, "num_evaluated": 3}), encoding="utf-8")

    report = check_sft_readiness(instruction, str(root / "missing_model"), metrics, train, eval_file, min_samples=2)
    json_path, md_path = write_readiness_report(report, root / "report.json", root / "report.md")

    assert report["ready"] is False
    assert json_path.exists()
    assert md_path.exists()
    assert "SFT Readiness Report" in md_path.read_text(encoding="utf-8")
