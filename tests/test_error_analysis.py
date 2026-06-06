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
    report = root / "analysis.md"
    df = export_error_cases(str(dataset), str(output), str(report))
    assert output.exists()
    assert report.exists()
    assert list(df.columns) == ERROR_COLUMNS
    assert "Errors by Error Type" in report.read_text(encoding="utf-8")


def test_error_type_export() -> None:
    root = Path("outputs/test_artifacts")
    root.mkdir(parents=True, exist_ok=True)
    dataset = root / "ambiguous.csv"
    output = root / "ambiguous_errors.csv"
    report = root / "ambiguous_analysis.md"
    pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "doc_id": "d1",
                "doc_type": "invoice",
                "image_path": "x.png",
                "question": "What is the date?",
                "answer": "2026-05-30",
                "field_name": "date",
                "field_type": "date",
                "ocr_text": "Due Date: 2026-06-29\nInvoice Date: 2026-05-30",
                "bbox": None,
                "source_dataset": "mock",
            }
        ]
    ).to_csv(dataset, index=False)
    df = export_error_cases(str(dataset), str(output), str(report))
    assert df.iloc[0]["error_type"] == "ambiguous_context"
    assert "| ambiguous_context | 1 |" in report.read_text(encoding="utf-8")
