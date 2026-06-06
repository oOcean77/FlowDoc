from __future__ import annotations

from src.eval.relaxed_field_metrics import char_similarity, evaluate_field_relaxed, token_f1


def test_amount_uses_strict_policy_without_relaxed_override() -> None:
    result = evaluate_field_relaxed("$88.50", "88.50", "total_amount")

    assert result["match_policy"] == "strict"
    assert result["relaxed_correct"] is True
    assert result["normalized_exact_match"] is True


def test_company_allows_relaxed_prefix_difference() -> None:
    result = evaluate_field_relaxed("Merchant Pine Street Deli", "Pine Street Deli", "company")

    assert result["match_policy"] == "relaxed"
    assert result["token_f1"] >= 0.85
    assert result["relaxed_correct"] is True


def test_address_punctuation_and_newline_have_high_char_similarity() -> None:
    score = char_similarity("12 Main St.\nSuite 5", "12 main st suite 5")

    assert score >= 0.90


def test_total_amount_must_use_strict_policy() -> None:
    result = evaluate_field_relaxed("Total eighty eight dollars", "88.00", "total_amount")

    assert result["match_policy"] == "strict"
    assert result["relaxed_correct"] is False


def test_address_must_use_relaxed_policy() -> None:
    result = evaluate_field_relaxed("12 Main Street, Suite 5", "12 Main Street Suite 5", "address")

    assert result["match_policy"] == "relaxed"
    assert token_f1("12 Main Street, Suite 5", "12 Main Street Suite 5") == 1.0
    assert result["relaxed_correct"] is True
