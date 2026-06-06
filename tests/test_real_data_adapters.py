from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_sroie_data import prepare_sroie_dataframe, records_from_sroie_raw, sroie_stats
from src.data.dataset_adapter import convert_csv, load_local_csv, write_processed_csv


def test_local_csv_required_columns_validation() -> None:
    path = Path("outputs/test_artifacts/bad_local.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"sample_id": "s1"}]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_local_csv(path)


def test_missing_input_file_reports_error() -> None:
    with pytest.raises(FileNotFoundError, match="Input data file not found"):
        convert_csv("funsd", "outputs/test_artifacts/no_such_raw.csv")


def test_funsd_sroie_docvqa_like_csv_to_unified_schema() -> None:
    root = Path("outputs/test_artifacts/adapters")
    root.mkdir(parents=True, exist_ok=True)

    funsd = root / "funsd.csv"
    pd.DataFrame(
        [{"id": "f1", "document_id": "doc1", "label": "applicant", "text": "u_001", "image_path": "missing.png"}]
    ).to_csv(funsd, index=False)
    funsd_df = convert_csv("funsd", funsd)
    assert funsd_df.iloc[0]["source_dataset"] == "funsd"
    assert funsd_df.iloc[0]["image_exists"] == False

    sroie = root / "sroie.csv"
    pd.DataFrame([{"doc_id": "r1", "company": "Shop Co", "date": "2026-06-01", "total": "9.99"}]).to_csv(sroie, index=False)
    sroie_df = convert_csv("sroie", sroie)
    assert set(sroie_df["field_name"]) == {"company", "date", "total_amount"}

    docvqa = root / "docvqa.csv"
    pd.DataFrame([{"questionId": "q1", "image": "doc.png", "question": "What is total?", "answer": "9.99"}]).to_csv(docvqa, index=False)
    docvqa_df = convert_csv("docvqa", docvqa)
    assert docvqa_df.iloc[0]["sample_id"] == "q1"
    assert "image_exists" in docvqa_df.columns


def test_write_processed_csv() -> None:
    root = Path("outputs/test_artifacts/adapters")
    root.mkdir(parents=True, exist_ok=True)
    source = root / "docvqa.csv"
    output = root / "processed_docvqa.csv"
    pd.DataFrame([{"questionId": "q1", "image": "doc.png", "question": "What is total?", "answer": "9.99"}]).to_csv(source, index=False)
    df = convert_csv("docvqa", source)
    write_processed_csv(df, output)
    assert output.exists()


def test_sroie_like_csv_preparation() -> None:
    root = Path("outputs/test_artifacts/sroie_prepare")
    image = root / "receipt.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"fake-image")
    csv_path = root / "sroie_like.csv"
    pd.DataFrame(
        [
            {
                "doc_id": "r1",
                "image_path": str(image),
                "ocr_text": "Shop Co\nTotal 9.99",
                "company": "Shop Co",
                "date": "2026-06-01",
                "address": "1 Main Road",
                "total": "9.99",
            }
        ]
    ).to_csv(csv_path, index=False)

    df = prepare_sroie_dataframe(input_csv=csv_path, max_docs=10)
    stats = sroie_stats(df)

    assert set(df["field_name"]) == {"company", "date", "address", "total_amount"}
    assert stats["num_docs"] == 1
    assert stats["num_qa_samples"] == 4
    assert stats["missing_image_count"] == 0


def test_sroie_raw_missing_files_error() -> None:
    with pytest.raises(FileNotFoundError, match="SROIE raw directory not found"):
        records_from_sroie_raw("outputs/test_artifacts/no_sroie_raw")
