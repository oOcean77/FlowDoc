from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.evaluate_cleaned_predictions import evaluate_cleaned_predictions


def test_evaluate_cleaned_predictions_counts_fixed_and_broken() -> None:
    root = Path("outputs/test_artifacts/evaluate_cleaned")
    predictions = root / "cleaned_predictions.csv"
    output = root / "cleaned_eval.json"
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "sample_id": "fixed",
                "field_name": "company",
                "pred_answer": "Merchant Pine Street Deli",
                "cleaned_pred_answer": "Pine Street Deli",
                "gold_answer": "Pine Street Deli",
                "postprocess_changed": True,
                "skipped": False,
            },
            {
                "sample_id": "broken",
                "field_name": "total_amount",
                "pred_answer": "88.50",
                "cleaned_pred_answer": "8850",
                "gold_answer": "88.50",
                "postprocess_changed": True,
                "skipped": False,
            },
            {
                "sample_id": "skipped",
                "field_name": "address",
                "pred_answer": "",
                "cleaned_pred_answer": "",
                "gold_answer": "12 Main Road",
                "postprocess_changed": False,
                "skipped": True,
            },
        ]
    ).to_csv(predictions, index=False)

    summary = evaluate_cleaned_predictions(predictions, "cleaned_pred_answer", output)

    assert summary["evaluated"] == 2
    assert summary["raw_pred_normalized_accuracy"] == 0.5
    assert summary["cleaned_pred_normalized_accuracy"] == 0.5
    assert summary["changed_and_fixed_count"] == 1
    assert summary["changed_and_broke_count"] == 1
    assert summary["changed_but_wrong_count"] == 1
    assert "company" in summary["per_field_delta"]
    assert output.exists()
