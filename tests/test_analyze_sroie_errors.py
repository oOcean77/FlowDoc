from __future__ import annotations

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
                "cleaned_pred_answer": "88.50",
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
                "pred_answer": "Warehouse Lane",
                "cleaned_pred_answer": "1 Main Road",
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
                "cleaned_pred_answer": "",
                "strategy": "image_ocr",
                "backend": "qwen2_5_vl",
                "skipped": True,
                "skip_reason": "image_path not found",
            },
        ]
    ).to_csv(predictions, index=False)

    summary = analyze_predictions(predictions, output_dir)

    assert summary["skipped"] == 1
    assert summary["evaluated"] == 2
    assert summary["num_wrong"] == 1
    assert summary["raw_accuracy"] == 0.0
    assert summary["normalized_accuracy"] == 0.5
    assert summary["relaxed_accuracy"] == 0.5
    assert summary["postprocess_available"] is True
    assert summary["cleaned_normalized_accuracy"] == 1.0
    assert summary["cleaned_relaxed_accuracy"] == 1.0
    assert summary["fixed_by_postprocess"] == 1
    assert summary["broken_by_postprocess"] == 0
    assert summary["per_field_normalized_accuracy"]["total_amount"] == 1.0
    assert "relaxed_accuracy" in summary
    assert "per_field_relaxed_accuracy" in summary
    assert "per_field_avg_token_f1" in summary
    assert "per_field_avg_char_similarity" in summary
    assert summary["per_field_relaxed_accuracy"]["address"] == 0.0
    assert 0.0 <= summary["address_token_f1"] <= 1.0
    assert 0.0 <= summary["address_char_similarity"] <= 1.0
    assert summary["per_field_wrong_cases"] == {"address": 1}
    assert summary["top_error_fields"] == [{"field_name": "address", "wrong_count": 1}]
    assert (output_dir / "skipped_samples.csv").exists()
    assert (output_dir / "wrong_cases.csv").exists()
    assert (output_dir / "relaxed_wrong_cases.csv").exists()
    assert (output_dir / "address_wrong_cases.csv").exists()
    assert (output_dir / "cleaned_wrong_cases.csv").exists()
    assert (output_dir / "fixed_by_postprocess.csv").exists()
    assert (output_dir / "broken_by_postprocess.csv").exists()
    assert (output_dir / "error_summary.json").exists()
