from __future__ import annotations

import json
from pathlib import Path

from scripts.report_benchmark import build_benchmark_report


def test_benchmark_report_generation() -> None:
    root = Path("outputs/test_artifacts/benchmark_report")
    root.mkdir(parents=True, exist_ok=True)
    (root / "field_eval_results.json").write_text(
        json.dumps({"dataset": "mock-hard", "baseline_family": "ocr-only", "num_samples": 39, "field_level_accuracy": 0.769}),
        encoding="utf-8",
    )
    (root / "sroie_qwen_image_ocr_50.json").write_text(
        json.dumps(
            {
                "backend": "qwen2_5_vl",
                "strategy": "image_ocr",
                "num_evaluated": 50,
                "skipped": False,
                "field_level_accuracy": 0.7,
            }
        ),
        encoding="utf-8",
    )
    (root / "vlm_baseline_qwen_lora_image_ocr_full.json").write_text(
        json.dumps(
            {
                "backend": "qwen2_5_vl",
                "strategy": "image_ocr",
                "lora_adapter": "outputs/lora/adapter",
                "num_evaluated": 39,
                "skipped": False,
                "field_level_accuracy": 0.718,
            }
        ),
        encoding="utf-8",
    )

    output = build_benchmark_report(root, root / "benchmark_report.md")
    text = output.read_text(encoding="utf-8")

    assert "mock-hard" in text
    assert "sroie-real" in text
    assert "zero-shot" in text
    assert "LoRA" in text
    assert "evaluated" in text
