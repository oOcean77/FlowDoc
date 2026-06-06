from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.compare_baselines import build_report
from scripts.run_vlm_baseline import parse_args
from src.eval.vlm_baseline import run_vlm_baseline


def _dataset(path: Path, count: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "sample_id": f"s{index}",
                "doc_id": "d1",
                "doc_type": "receipt",
                "image_path": "missing.png",
                "question": "What is total?",
                "answer": "123",
                "field_name": "total_amount",
                "field_type": "amount",
                "ocr_text": "Total: 123",
                "bbox": None,
                "source_dataset": "mock",
            }
            for index in range(count)
        ]
    ).to_csv(path, index=False)
    return path


def test_dummy_vlm_baseline_skips_and_writes_predictions() -> None:
    root = Path("outputs/test_artifacts/vlm_baseline")
    dataset = _dataset(root / "qa.csv")
    metrics_path = root / "vlm_baseline_dummy_image_ocr.json"
    predictions_dir = root / "predictions"

    metrics = run_vlm_baseline(dataset, "image_ocr", "dummy", metrics_path, predictions_dir=predictions_dir)

    assert metrics["skipped"] is True
    assert metrics["field_level_accuracy"] is None
    pred_path = predictions_dir / "dummy_image_ocr_predictions.csv"
    assert pred_path.exists()
    pred_df = pd.read_csv(pred_path)
    assert bool(pred_df.iloc[0]["skipped"]) == True


def test_smoke_test_limits_to_three_samples() -> None:
    root = Path("outputs/test_artifacts/vlm_smoke")
    dataset = _dataset(root / "qa.csv", count=5)
    metrics = run_vlm_baseline(
        dataset,
        "image_ocr",
        "dummy",
        root / "metrics.json",
        smoke_test=True,
        predictions_dir=root / "predictions",
    )

    assert metrics["num_samples"] == 3
    assert len(metrics["smoke_logs"]) == 3
    pred_df = pd.read_csv(root / "predictions" / "dummy_image_ocr_predictions.csv")
    assert len(pred_df) == 3


def test_lora_adapter_arg_parse() -> None:
    args = parse_args(
        [
            "--strategy",
            "image_ocr",
            "--backend",
            "qwen2_5_vl",
            "--output",
            "outputs/metrics/test.json",
            "--lora-adapter",
            "outputs/lora/adapter",
        ]
    )

    assert args.lora_adapter == "outputs/lora/adapter"


def test_missing_lora_adapter_skips_and_marks_metrics() -> None:
    root = Path("outputs/test_artifacts/vlm_lora_missing")
    dataset = _dataset(root / "qa.csv")
    metrics_path = root / "metrics.json"
    predictions_dir = root / "predictions"

    metrics = run_vlm_baseline(
        dataset,
        "image_ocr",
        "qwen2_5_vl",
        metrics_path,
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        lora_adapter=str(root / "missing_adapter"),
        predictions_dir=predictions_dir,
    )

    assert metrics["skipped"] is True
    assert metrics["lora_adapter"].endswith("missing_adapter")
    assert "LoRA adapter path does not exist" in metrics["skip_reason"]
    assert (predictions_dir / "qwen2_5_vl_lora_image_ocr_predictions.csv").exists()


def test_compare_baselines_report() -> None:
    root = Path("outputs/test_artifacts/comparison")
    root.mkdir(parents=True, exist_ok=True)
    (root / "field_eval_results.json").write_text(
        json.dumps(
            {
                "num_samples": 1,
                "field_level_accuracy": 1.0,
                "missing_field_rate": 0.0,
                "multi_value_conflict_rate": 0.0,
            }
        ),
        encoding="utf-8",
    )
    (root / "vlm_baseline_dummy_image_ocr.json").write_text(
        json.dumps(
            {
                "backend": "dummy",
                "strategy": "image_ocr",
                "num_evaluated": 0,
                "num_skipped": 1,
                "skipped": True,
                "skip_reason": "dummy backend does not run a real VLM",
                "field_level_accuracy": None,
            }
        ),
        encoding="utf-8",
    )
    (root / "vlm_baseline_qwen_lora_image_ocr_full.json").write_text(
        json.dumps(
            {
                "backend": "qwen2_5_vl",
                "strategy": "image_ocr",
                "lora_adapter": "outputs/lora/qwen2_5_vl_flowdoc_dryrun",
                "num_evaluated": 1,
                "skipped": False,
                "skip_reason": "",
                "field_level_accuracy": 0.5,
            }
        ),
        encoding="utf-8",
    )
    output = build_report(root, root / "baseline_comparison.md")
    text = output.read_text(encoding="utf-8")
    assert "| dummy | image_ocr | zero-shot | skipped | 0 | unavailable | dummy backend does not run a real VLM |" in text
    assert "| qwen2_5_vl | image_ocr | lora | evaluated | 1 | 0.500 |  |" in text
    assert "field_level_accuracy: 1.000" in text
