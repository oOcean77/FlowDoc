from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class BaselinePrediction:
    answer: str
    confidence: float
    error_type: str | None = None


AMOUNT_RE = re.compile(r"(?i)(?:rmb|cny|usd|eur|gbp|\$)?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?")
DATE_RE = re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}/\d{1,2}/\d{4}")
ID_RE = re.compile(r"\b[A-Z]{1,5}-\d{3,}(?:-\d{3})?\b|\bINV-\d{4}-\d{3}\b", re.I)


def _lines(ocr_text: str) -> list[str]:
    return [line.strip() for line in ocr_text.splitlines() if line.strip()]


def _value_after_label(line: str) -> str:
    parts = re.split(r":", line, maxsplit=1)
    return parts[1].strip() if len(parts) == 2 else line.strip()


def _find_labeled(ocr_text: str, labels: list[str]) -> str:
    for line in _lines(ocr_text):
        lower = line.lower()
        if any(label.lower() in lower for label in labels):
            return _value_after_label(line)
    return ""


def _extract_amount(ocr_text: str) -> BaselinePrediction:
    lines = _lines(ocr_text)
    preferred = [r"\btotal amount\b", r"\bamount due\b", r"^total\b", r"\bgrand total\b"]
    weak_total_labels = [r"\bbalance paid\b", r"\bplease pay\b", r"\btotal due\b"]
    all_amounts = [match.strip() for match in AMOUNT_RE.findall(ocr_text)]
    for line in lines:
        if any(re.search(label, line.lower()) for label in preferred):
            matches = [match.strip() for match in AMOUNT_RE.findall(line)]
            if matches:
                error = "multi_value_conflict" if len(all_amounts) > 1 else None
                return BaselinePrediction(matches[-1], 0.8, error)
    for line in lines:
        if any(re.search(label, line.lower()) for label in weak_total_labels):
            matches = [match.strip() for match in AMOUNT_RE.findall(line)]
            if matches:
                return BaselinePrediction(matches[-1], 0.45, "ambiguous_context")
    if not all_amounts:
        return BaselinePrediction("", 0.0, "missing_field")
    return BaselinePrediction(all_amounts[-1], 0.3, "wrong_candidate")


def _extract_date(ocr_text: str, field_name: str, field_type: str) -> BaselinePrediction:
    labels = ["deadline", "valid until", "ends"] if field_name == "deadline" or field_type == "deadline" else ["date"]
    labeled_value = _find_labeled(ocr_text, labels)
    all_dates = DATE_RE.findall(ocr_text)
    if labeled_value:
        match = DATE_RE.search(labeled_value)
        value = match.group(0) if match else labeled_value
        error = "ambiguous_context" if len(set(all_dates)) > 1 and labels == ["date"] else None
        return BaselinePrediction(value, 0.7, error)
    if all_dates:
        error = "ambiguous_context" if len(set(all_dates)) > 1 else "wrong_candidate"
        return BaselinePrediction(all_dates[0], 0.45, error)
    return BaselinePrediction("", 0.0, "missing_field")


def _extract_company(ocr_text: str) -> BaselinePrediction:
    vendor = _find_labeled(ocr_text, ["vendor", "merchant", "company name", "company"])
    if vendor:
        return BaselinePrediction(vendor, 0.6, None)
    lines = _lines(ocr_text)
    company_like = [line for line in lines if re.search(r"\b(?:ltd|labs|market|cafe|deli|books|services|trading)\b", line, re.I)]
    if len(company_like) > 1:
        return BaselinePrediction(company_like[0], 0.35, "ambiguous_context")
    if company_like:
        return BaselinePrediction(company_like[0], 0.45, "wrong_candidate")
    return BaselinePrediction(lines[0] if lines else "", 0.35 if lines else 0.0, None if lines else "missing_field")


def _extract_id(ocr_text: str, field_name: str) -> BaselinePrediction:
    labels = ["invoice no", "invoice id", "doc number"] if field_name == "invoice_id" else [field_name.replace("_", " "), "ticket"]
    value = _find_labeled(ocr_text, labels)
    if value:
        match = ID_RE.search(value)
        return BaselinePrediction(match.group(0) if match else value, 0.7, None)
    match = ID_RE.search(ocr_text)
    if match:
        return BaselinePrediction(match.group(0), 0.35, "wrong_candidate")
    return BaselinePrediction("", 0.0, "missing_field" if field_name == "invoice_id" else "unsupported_field")


def predict_from_ocr(ocr_text: str, field_name: str, field_type: str) -> BaselinePrediction:
    name = field_name.lower()
    kind = field_type.lower()
    if kind == "amount" or "amount" in name or "total" in name:
        return _extract_amount(ocr_text)
    if kind in {"date", "deadline"} or name in {"date", "deadline"}:
        return _extract_date(ocr_text, name, kind)
    if "company" in name or kind == "company":
        return _extract_company(ocr_text)
    if name in {"invoice_id", "document_id", "approval_code"} or kind == "id":
        return _extract_id(ocr_text, name)
    if name == "applicant":
        value = _find_labeled(ocr_text, ["applicant", "requester", "user"])
        if value:
            match = re.search(r"\bu_\d+\b", value)
            return BaselinePrediction(match.group(0) if match else value, 0.7, None)
        match = re.search(r"\bu_\d+\b", ocr_text)
        return BaselinePrediction(match.group(0) if match else "", 0.4 if match else 0.0, "wrong_candidate" if match else "missing_field")
    if name == "permission_scope" or kind == "permission":
        value = _find_labeled(ocr_text, ["permission scope", "permission", "scope"])
        if value:
            return BaselinePrediction(value, 0.55, "ambiguous_context" if " " in value else None)
        natural = _find_labeled(ocr_text, ["access needed", "scope requested in words"])
        if natural:
            return BaselinePrediction(natural, 0.35, "ambiguous_context")
        return BaselinePrediction("", 0.0, "missing_field")
    value = _find_labeled(ocr_text, [field_name.replace("_", " ")])
    return BaselinePrediction(value, 0.4 if value else 0.0, None if value else "unsupported_field")
