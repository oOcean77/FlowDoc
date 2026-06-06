from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.analyze_sroie_errors import analyze_predictions


def test_analyze_sroie_predictions_outputs_wrong_and_skipped() -> None:
    root = Path("outputs/test_artifacts/sroie_errors")
    predictions = root / "predictions.csv"
    output_dir = root / "analysis"
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "doc_id": "d1",
                "field_name": "total_amount",
                "gold_answer": "88.50",
                "pred_answer": "$88.50",
                "strategy": "image_ocr",
                "backend": "qwen2_5_vl",
                "skipped": False,
                "skip_reason": "",
            },
            {
                "sample_id": "s2",
                "doc_id": "d1",
                "field_name": "address",
                "gold_answer": "1 Main Road",
                "pred_answer": "Main Road",
                "strategy": "image_ocr",
                "backend": "qwen2_5_vl",
                "skipped": False,
                "skip_reason": "",
            },
            {
                "sample_id": "s3",
                "doc_id": "d2",
                "field_name": "company",
                "gold_answer": "Shop Co",
                "pred_answer": "",
                "strategy": "image_ocr",
                "backend": "qwen2_5_vl",
                "skipped": True,
                "skip_reason": "image_path not found",
            },
        ]
    ).to_csv(predictions, index=False)

    summary = analyze_predictions(predictions, output_dir)

    assert summary["num_skipped"] == 1
    assert summary["num_wrong"] == 1
    assert summary["per_field_wrong_cases"] == {"address": 1}
    assert (output_dir / "skipped_samples.csv").exists()
    assert (output_dir / "wrong_cases.csv").exists()
    assert (output_dir / "address_wrong_examples.csv").exists()
    assert Path("outputs/metrics/skipped_samples_summary.json").exists()
    saved = json.loads((output_dir / "sroie_error_summary.json").read_text(encoding="utf-8"))
    assert saved["num_evaluated"] == 2
    skipped_summary = json.loads(Path("outputs/metrics/skipped_samples_summary.json").read_text(encoding="utf-8"))
    assert skipped_summary["skipped_samples"][0]["sample_id"] == "s3"
