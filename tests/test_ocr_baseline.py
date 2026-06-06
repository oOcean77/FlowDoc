from __future__ import annotations

from src.baselines.ocr_only import predict_from_ocr


def test_ocr_baseline_extract_total_amount() -> None:
    pred = predict_from_ocr("Total: RMB 123.00", "total_amount", "amount")
    assert pred.answer == "RMB 123.00"


def test_ocr_baseline_multi_amount_prefers_total() -> None:
    text = "Subtotal: 100.00\nTax: 23.00\nTotal: RMB 123.00"
    pred = predict_from_ocr(text, "total_amount", "amount")
    assert pred.answer == "RMB 123.00"
    assert pred.error_type == "multi_value_conflict"
