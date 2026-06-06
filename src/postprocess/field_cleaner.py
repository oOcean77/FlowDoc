from __future__ import annotations

import re
from typing import Any


_AMOUNT_FIELDS = {"total_amount", "subtotal", "tax"}
_DATE_FIELDS = {"date", "deadline"}
_COMPANY_FIELDS = {"company", "merchant", "vendor"}
_LIGHT_PREFIX_FIELDS = {"permission_scope", "reason", "deadline", "security_group"}

_COMPANY_PREFIX_RE = re.compile(r"^\s*(?:merchant|company|vendor|store|shop)\s*:?\s*", flags=re.I)
_ADDRESS_PREFIX_RE = re.compile(r"^\s*(?:address|addr|location)\s*:?\s*", flags=re.I)
_LIGHT_PREFIX_RE = re.compile(r"^\s*(?:reason|permission|scope|deadline|group)\s*:?\s*", flags=re.I)
_DATE_PREFIX_RE = re.compile(r"^\s*(?:date|invoice date|receipt date|日期)\s*:?\s*", flags=re.I)
_NOISE_SUFFIX_RE = re.compile(r"\s+(?:AP)\s*$", flags=re.I)
_AMOUNT_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _collapse_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_repeated_punctuation(text: str) -> str:
    text = re.sub(r"([,.;:])\1+", r"\1", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([,.;:])(?=\S)", r"\1 ", text)
    return _collapse_space(text)


def _confidence(cleaned: str, field_name: str, changed: bool) -> str:
    tokens = cleaned.split()
    if not cleaned:
        return "low"
    if field_name == "address" and (len(cleaned) < 8 or len(tokens) < 3):
        return "low"
    if changed:
        return "medium"
    return "high"


def _clean_amount(text: str) -> tuple[str, list[str]]:
    rules: list[str] = []
    candidates = _AMOUNT_RE.findall(text)
    if candidates:
        cleaned = candidates[-1].replace(",", "")
        rules.append("extract_amount_candidate")
        if "," in candidates[-1]:
            rules.append("remove_thousands_separator")
        if re.search(r"(?i)(?:\$|rm|rmb|cny|usd|楼)", text):
            rules.append("remove_currency_marker")
        return cleaned, rules
    cleaned = _collapse_space(re.sub(r"(?i)(?:\$|rm|rmb|cny|usd|楼)", "", text))
    if cleaned != text:
        rules.append("remove_currency_marker")
    return cleaned, rules


def _clean_date(text: str) -> tuple[str, list[str]]:
    cleaned = _DATE_PREFIX_RE.sub("", text)
    rules = ["remove_date_prefix"] if cleaned != text else []
    collapsed = _collapse_space(cleaned)
    if collapsed != cleaned:
        rules.append("collapse_whitespace")
    return collapsed, rules


def _clean_company(text: str) -> tuple[str, list[str]]:
    rules: list[str] = []
    cleaned = _COMPANY_PREFIX_RE.sub("", text)
    if cleaned != text:
        rules.append("remove_company_prefix")
    without_suffix = _NOISE_SUFFIX_RE.sub("", cleaned)
    if without_suffix != cleaned:
        rules.append("remove_noise_suffix")
    cleaned = _clean_repeated_punctuation(without_suffix.replace("\n", " "))
    if cleaned != without_suffix:
        rules.append("normalize_punctuation_whitespace")
    return cleaned, rules


def _clean_address(text: str) -> tuple[str, list[str]]:
    rules: list[str] = []
    cleaned = _ADDRESS_PREFIX_RE.sub("", text)
    if cleaned != text:
        rules.append("remove_address_prefix")
    merged = cleaned.replace("\r", " ").replace("\n", " ")
    if merged != cleaned:
        rules.append("merge_address_lines")
    cleaned = _clean_repeated_punctuation(merged)
    if cleaned != merged:
        rules.append("normalize_punctuation_whitespace")
    return cleaned, rules


def _clean_light_prefix(text: str) -> tuple[str, list[str]]:
    cleaned = _LIGHT_PREFIX_RE.sub("", text)
    rules = ["remove_light_prefix"] if cleaned != text else []
    collapsed = _collapse_space(cleaned)
    if collapsed != cleaned:
        rules.append("collapse_whitespace")
    return collapsed, rules


def clean_prediction(pred: str, field_name: str, ocr_text: str | None = None) -> dict[str, Any]:
    raw = "" if pred is None else str(pred)
    field = str(field_name)
    stripped = raw.strip()
    rules: list[str] = []
    if stripped != raw:
        rules.append("strip")

    if field in _AMOUNT_FIELDS:
        cleaned, field_rules = _clean_amount(stripped)
    elif field in _DATE_FIELDS:
        cleaned, field_rules = _clean_date(stripped)
    elif field in _COMPANY_FIELDS:
        cleaned, field_rules = _clean_company(stripped)
    elif field == "address":
        cleaned, field_rules = _clean_address(stripped)
    elif field in _LIGHT_PREFIX_FIELDS:
        cleaned, field_rules = _clean_light_prefix(stripped)
    else:
        cleaned = _collapse_space(stripped)
        field_rules = ["collapse_whitespace"] if cleaned != stripped else []

    rules.extend(field_rules)
    changed = cleaned != raw
    return {
        "raw": raw,
        "cleaned": cleaned,
        "field_name": field,
        "rules_applied": rules,
        "changed": changed,
        "confidence": _confidence(cleaned, field, changed),
    }
