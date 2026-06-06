from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.data.dataset_adapter import load_local_csv
from src.data.schema import DocumentQASample


def test_schema_required_fields() -> None:
    sample = DocumentQASample(
        sample_id="doc_001_q_001",
        doc_id="doc_001",
        doc_type="receipt",
        image_path="data/samples/doc_001.png",
        question="What is the total amount?",
        answer="123.45",
        field_name="total_amount",
        field_type="amount",
        ocr_text="Total: 123.45",
        bbox=None,
        source_dataset="mock",
    )
    assert sample.sample_id == "doc_001_q_001"


def test_local_csv_adapter_schema() -> None:
    root = Path("outputs/test_artifacts")
    root.mkdir(parents=True, exist_ok=True)
    path = root / "sample.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "doc_id": "d1",
                "doc_type": "receipt",
                "image_path": "x.png",
                "question": "What is total?",
                "answer": "123",
                "field_name": "total_amount",
                "field_type": "amount",
                "ocr_text": "Total: 123",
                "source_dataset": "mock",
            }
        ]
    ).to_csv(path, index=False)
    df = load_local_csv(path)
    assert list(df.columns)[0] == "sample_id"
    assert df.iloc[0]["field_type"] == "amount"
