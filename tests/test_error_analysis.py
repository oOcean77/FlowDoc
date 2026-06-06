from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.eval.error_analysis import ERROR_COLUMNS, export_error_cases


def test_error_case_export() -> None:
    root = Path("outputs/test_artifacts")
    root.mkdir(parents=True, exist_ok=True)
    dataset = root / "mock.csv"
    output = root / "errors.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "doc_id": "d1",
                "doc_type": "receipt",
                "image_path": "x.png",
                "question": "What is applicant?",
                "answer": "u_001",
                "field_name": "applicant",
                "field_type": "person",
                "ocr_text": "No applicant here",
                "bbox": None,
                "source_dataset": "mock",
            }
        ]
    ).to_csv(dataset, index=False)
    df = export_error_cases(str(dataset), str(output))
    assert output.exists()
    assert list(df.columns) == ERROR_COLUMNS
