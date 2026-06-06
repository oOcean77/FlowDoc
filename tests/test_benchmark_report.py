from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.report_benchmark import build_benchmark_report, collect_benchmark_rows, write_skipped_samples_summary


def test_benchmark_report_generation() -> None:
    root = Path("outputs/test_artifacts/benchmark_report")
    root.mkdir(parents=True, exist_ok=True)
    (root / "field_eval_results.json").write_text(
        json.dumps({"dataset": "mock-hard", "baseline_family": "ocr-only", "num_samples": 39, "field_level_accuracy": 0.769}),
        encoding="utf-8",
    )
    (root / "vlm_env_report.json").write_text(json.dumps({"python": "3.12"}), encoding="utf-8")
    (root / "lora_dryrun_train_log.json").write_text(json.dumps({"final_loss": 0.3}), encoding="utf-8")
    (root / "lora_dryrun_train_log_step10_backup.json").write_text(json.dumps({"final_loss": 0.2}), encoding="utf-8")
    (root / "sft_readiness_report.json").write_text(json.dumps({"ready": False}), encoding="utf-8")
    (root / "prediction_change_summary.json").write_text(json.dumps({"changed": 1}), encoding="utf-8")
    (root / "skipped_samples_summary.json").write_text(json.dumps({"num_skipped_rows": 1}), encoding="utf-8")
    (root / "sroie_qwen_image_ocr_100.json").write_text(
        json.dumps(
            {
                "backend": "qwen2_5_vl",
                "strategy": "image_ocr",
                "num_samples": 100,
                "num_evaluated": 92,
                "num_skipped": 8,
                "skipped": False,
                "field_level_accuracy": 0.761,
                "per_field_accuracy": {"address": 0.217, "total_amount": 0.9},
            }
        ),
        encoding="utf-8",
    )
    (root / "sroie_ocr_field_eval_100.json").write_text(
        json.dumps(
            {
                "dataset": "sroie-real",
                "baseline_family": "ocr-only",
                "strategy": "ocr_text",
                "num_samples": 100,
                "field_level_accuracy": 0.32,
                "per_field_accuracy": {"address": 0.1, "total_amount": 0.8},
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

    rows = collect_benchmark_rows(root)
    predictions_dir = root / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "field_name": "address",
                "skipped": True,
                "skip_reason": "image_path not found",
            }
        ]
    ).to_csv(predictions_dir / "sroie_qwen_image_ocr_100_predictions.csv", index=False)
    output = build_benchmark_report(root, root / "benchmark_report.md", predictions_dir)
    text = output.read_text(encoding="utf-8")

    assert all("report" not in row["file"] and "train_log" not in row["file"] for row in rows)
    assert "lora_dryrun_train_log_step10_backup.json" not in text
    assert "prediction_change_summary.json" not in text
    assert "skipped_samples_summary.json" not in text
    assert "mock-hard" in text
    assert "sroie-real-100" in text
    assert "zero-shot" in text
    assert "LoRA" in text
    assert "evaluated" in text
    assert "Skipped Samples Summary" in text
    assert "SROIE Same-Subset Comparison" in text
    assert "| SROIE real | Qwen2.5-VL zero-shot | ocr_only | unavailable | unavailable | unavailable | unavailable | missing `sroie_qwen_ocr_only_100.json` |" in text
    assert "SROIE Per-Field Comparison" in text
    assert "address" in text
    skipped_summary = json.loads((root / "skipped_samples_summary.json").read_text(encoding="utf-8"))
    assert skipped_summary["skipped_samples"][0]["sample_id"] == "s1"


def test_skipped_samples_summary_outputs_json_and_md() -> None:
    root = Path("outputs/test_artifacts/skipped_summary")
    predictions_dir = root / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"sample_id": "s1", "field_name": "company", "skipped": False, "skip_reason": ""},
            {"sample_id": "s2", "field_name": "address", "skipped": True, "skip_reason": "image decode failed"},
        ]
    ).to_csv(predictions_dir / "sroie_qwen_image_ocr_100_predictions.csv", index=False)
    pd.DataFrame(
        [
            {"sample_id": "old", "field_name": "address", "skipped": True, "skip_reason": "legacy skipped"},
        ]
    ).to_csv(predictions_dir / "qwen2_5_vl_image_ocr_predictions.csv", index=False)

    summary = write_skipped_samples_summary(
        predictions_dir,
        root / "skipped_samples_summary.json",
        root / "skipped_samples_summary.md",
    )

    assert summary["num_skipped_rows"] == 1
    assert summary["skip_reason_counts"] == {"image decode failed": 1}
    assert summary["legacy_prediction_files_ignored"]
    assert (root / "skipped_samples_summary.json").exists()
    md_text = (root / "skipped_samples_summary.md").read_text(encoding="utf-8")
    assert "image decode failed" in md_text
    assert "legacy skipped" not in md_text
