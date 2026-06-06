from __future__ import annotations

import re


ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:the\s+answer\s+is|answer\s*:|field\s+value\s*:|字段值为[:：]?|答案是[:：]?|答[:：])\s*",
    re.I,
)
AMOUNT_RE = re.compile(r"(?i)(?:rmb|cny|usd|eur|gbp|\$)?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?")
DATE_RE = re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}/\d{1,2}/\d{4}")
ID_RE = re.compile(r"\b[A-Z]{1,8}-\d{2,}(?:-\d{2,})?\b|\bu_\d+\b", re.I)


def strip_answer_prefix(text: object) -> str:
    value = "" if text is None else str(text).strip()
    value = ANSWER_PREFIX_RE.sub("", value).strip()
    return value.strip(" \t\r\n。.")


def postprocess_prediction(text: object, field_type: str = "other") -> str:
    value = strip_answer_prefix(text)
    kind = field_type.lower()
    if kind == "amount":
        match = AMOUNT_RE.search(value)
        return match.group(0).strip() if match else value
    if kind in {"date", "deadline"}:
        match = DATE_RE.search(value)
        return match.group(0).strip() if match else value
    if kind in {"id", "person"}:
        match = ID_RE.search(value)
        return match.group(0).strip() if match else value
    return value
