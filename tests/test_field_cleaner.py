from __future__ import annotations

from src.postprocess.field_cleaner import clean_prediction


def test_total_amount_removes_currency_without_guessing() -> None:
    assert clean_prediction("$88.50", "total_amount")["cleaned"] == "88.50"
    assert clean_prediction("120", "total_amount")["cleaned"] == "120"


def test_company_removes_safe_prefix() -> None:
    result = clean_prediction("Merchant Pine Street Deli", "company")

    assert result["cleaned"] == "Pine Street Deli"
    assert result["changed"] is True
    assert "remove_company_prefix" in result["rules_applied"]


def test_address_merges_lines_and_prefix() -> None:
    result = clean_prediction("Address:\n12 Jalan ABC,\nKuala Lumpur", "address")

    assert result["cleaned"] == "12 Jalan ABC, Kuala Lumpur"
    assert result["confidence"] == "medium"


def test_reason_light_prefix_only() -> None:
    result = clean_prediction("Reason: investigate delayed orders", "reason")

    assert result["cleaned"] == "investigate delayed orders"
    assert result["rules_applied"] == ["remove_light_prefix"]


def test_short_address_low_confidence() -> None:
    result = clean_prediction("Addr: ABC", "address")

    assert result["cleaned"] == "ABC"
    assert result["confidence"] == "low"
