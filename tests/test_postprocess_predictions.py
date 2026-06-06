from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.postprocess_predictions import postprocess_predictions


def test_postprocess_predictions_adds_cleaned_columns() -> None:
    root = Path("outputs/test_artifacts/postprocess_predictions")
    input_csv = root / "predictions.csv"
    output_csv = root / "cleaned_predictions.csv"
    summary_json = root / "summary.json"
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "field_name": "company",
                "pred_answer": "Merchant Pine Street Deli",
                "gold_answer": "Pine Street Deli",
                "skipped": False,
            },
            {
                "sample_id": "s2",
                "field_name": "total_amount",
                "pred_answer": "$88.50",
                "gold_answer": "88.50",
                "skipped": False,
            },
        ]
    ).to_csv(input_csv, index=False)

    summary = postprocess_predictions(input_csv, output_csv, summary_json)

    df = pd.read_csv(output_csv)
    assert "cleaned_pred_answer" in df.columns
    assert df.loc[0, "cleaned_pred_answer"] == "Pine Street Deli"
    assert df.loc[1, "cleaned_pred_answer"] == "88.50"
    assert summary["changed"] == 2
    assert json.loads(summary_json.read_text(encoding="utf-8"))["rules_count"]["remove_company_prefix"] == 1
