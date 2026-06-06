from __future__ import annotations

import pandas as pd

from src.baselines.ocr_only import predict_from_ocr
from src.data.dataset_adapter import load_mock_dataset
from src.eval.field_metrics import evaluate_rows


def run_ocr_eval(dataset_path: str = "data/processed/mock_qa.csv") -> dict:
    df = load_mock_dataset(dataset_path)
    rows = []
    for row in df.to_dict(orient="records"):
        pred = predict_from_ocr(row.get("ocr_text", ""), row["field_name"], row["field_type"])
        rows.append({**row, "pred_answer": pred.answer, "confidence": pred.confidence, "error_type": pred.error_type})
    return evaluate_rows(rows)
