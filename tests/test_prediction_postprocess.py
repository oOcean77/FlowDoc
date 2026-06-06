from __future__ import annotations

from src.eval.prediction_postprocess import postprocess_prediction, strip_answer_prefix


def test_strip_english_answer_prefix() -> None:
    assert strip_answer_prefix("The answer is RMB 123.00.") == "RMB 123.00"


def test_strip_chinese_answer_prefix() -> None:
    assert strip_answer_prefix("答案是：2026-06-05。") == "2026-06-05"


def test_amount_field_extraction() -> None:
    assert postprocess_prediction("The answer is USD 1,200.00 total.", "amount") == "USD 1,200.00"


def test_date_field_extraction() -> None:
    assert postprocess_prediction("字段值为：invoice date 2026-06-05", "date") == "2026-06-05"
