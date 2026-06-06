from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eval.run_eval import run_ocr_eval


ERROR_COLUMNS = ["sample_id", "doc_type", "field_name", "question", "gold_answer", "pred_answer", "ocr_text", "error_type"]


def export_error_cases(dataset_path: str = "data/processed/mock_qa.csv", output_path: str = "outputs/error_cases/error_cases.csv") -> pd.DataFrame:
    results = run_ocr_eval(dataset_path)
    rows = []
    for row in results["rows"]:
        if not row["correct"]:
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "doc_type": row["doc_type"],
                    "field_name": row["field_name"],
                    "question": row["question"],
                    "gold_answer": row["answer"],
                    "pred_answer": row.get("pred_answer", ""),
                    "ocr_text": row.get("ocr_text", ""),
                    "error_type": row.get("error_type") or "normalization_error",
                }
            )
    df = pd.DataFrame(rows, columns=ERROR_COLUMNS)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return df
