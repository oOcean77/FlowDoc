from __future__ import annotations

from pathlib import Path

from src.data.mock_docs import generate_mock_dataset


def test_hard_mock_samples_are_generated() -> None:
    root = Path("outputs/test_artifacts/mock_docs")
    df = generate_mock_dataset(root / "mock_qa.csv", root / "samples")

    assert 30 <= len(df) <= 40
    assert {"receipt", "invoice", "access_request_form"}.issubset(set(df["doc_type"]))
    ocr_text = "\n".join(df["ocr_text"].unique())
    assert "Due Date: 2026-06-29" in ocr_text
    assert "Subtotal 600.00" in ocr_text
    assert "Access needed: read-only access to billing exports" in ocr_text
    assert "Reason: investigate delayed orders" in ocr_text
