from __future__ import annotations

from src.eval.field_metrics import evaluate_rows, normalized_exact_match, regex_match
from src.utils.text_norm import normalize_text


def test_normalized_exact_match() -> None:
    assert normalized_exact_match(" RMB 123.00 ", "123")


def test_regex_amount_match() -> None:
    assert regex_match("Subtotal 100 Tax 23 Total RMB 123.00", "123", "amount")


def test_date_normalization() -> None:
    assert normalize_text("2026年6月5日") == "2026-06-05"
    assert normalize_text("06/05/2026") == "2026-06-05"


def test_multi_value_conflict_rate() -> None:
    results = evaluate_rows(
        [
            {"answer": "123", "pred_answer": "123", "field_name": "total_amount", "field_type": "amount", "error_type": "multi_value_conflict"},
            {"answer": "u_001", "pred_answer": "", "field_name": "applicant", "field_type": "person", "error_type": "missing_field"},
        ]
    )
    assert results["multi_value_conflict_rate"] == 0.5
    assert results["missing_field_rate"] == 0.5
