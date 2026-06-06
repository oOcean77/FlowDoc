from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class BaselinePrediction:
    answer: str
    confidence: float
    error_type: str | None = None


AMOUNT_RE = re.compile(r"(?i)(?:rmb|cny|usd|\$|¥|￥)?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?")
DATE_RE = re.compile(r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日?|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}/\d{1,2}/\d{4}")


def _lines(ocr_text: str) -> list[str]:
    return [line.strip() for line in ocr_text.splitlines() if line.strip()]


def _value_after_label(line: str) -> str:
    parts = re.split(r"[:：]", line, maxsplit=1)
    return parts[1].strip() if len(parts) == 2 else line.strip()


def _find_labeled(ocr_text: str, labels: list[str]) -> str:
    for line in _lines(ocr_text):
        lower = line.lower()
        if any(label.lower() in lower for label in labels):
            return _value_after_label(line)
    return ""


def _extract_amount(ocr_text: str) -> BaselinePrediction:
    lines = _lines(ocr_text)
    preferred = [r"\btotal amount\b", r"\bamount due\b", r"^total\b", r"合计", r"总计"]
    all_amounts = AMOUNT_RE.findall(ocr_text)
    for line in lines:
        if any(re.search(label, line.lower()) for label in preferred):
            matches = AMOUNT_RE.findall(line)
            if matches:
                return BaselinePrediction(matches[-1].strip(), 0.8, "multi_value_conflict" if len(all_amounts) > 1 else None)
    return BaselinePrediction(all_amounts[-1].strip() if all_amounts else "", 0.3, "missing_field" if not all_amounts else "wrong_candidate")


def predict_from_ocr(ocr_text: str, field_name: str, field_type: str) -> BaselinePrediction:
    name = field_name.lower()
    kind = field_type.lower()
    if kind == "amount" or "amount" in name or "total" in name:
        return _extract_amount(ocr_text)
    if kind in {"date", "deadline"} or name in {"date", "deadline"}:
        if name == "deadline" or kind == "deadline":
            value = _find_labeled(ocr_text, ["deadline", "due", "截止", "有效期"])
            match = DATE_RE.search(value)
            return BaselinePrediction(match.group(0) if match else value, 0.7 if value else 0.0, None if value else "missing_field")
        match = DATE_RE.search(ocr_text)
        return BaselinePrediction(match.group(0) if match else "", 0.7 if match else 0.0, None if match else "missing_field")
    if "company" in name or kind == "company":
        value = _find_labeled(ocr_text, ["merchant", "company name", "company", "vendor", "单位", "公司"])
        if not value:
            value = _lines(ocr_text)[0] if _lines(ocr_text) else ""
        return BaselinePrediction(value, 0.6 if value else 0.0, None if value else "missing_field")
    if name in {"invoice_id", "document_id"} or kind == "id":
        value = _find_labeled(ocr_text, ["invoice no", "receipt no", "id", "编号", "单号"])
        return BaselinePrediction(value, 0.7 if value else 0.0, None if value else "missing_field")
    if name == "applicant":
        value = _find_labeled(ocr_text, ["applicant", "申请人"])
        return BaselinePrediction(value, 0.7 if value else 0.0, None if value else "missing_field")
    if name == "permission_scope" or kind == "permission":
        value = _find_labeled(ocr_text, ["permission scope", "permission", "scope", "权限范围", "权限"])
        return BaselinePrediction(value, 0.7 if value else 0.0, None if value else "missing_field")
    value = _find_labeled(ocr_text, [field_name.replace("_", " ")])
    return BaselinePrediction(value, 0.4 if value else 0.0, None if value else "unsupported_field")
