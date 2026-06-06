from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation


_PUNCT_RE = re.compile(r"[,:;!?\[\]{}()\"'`]")
_CURRENCY_RE = re.compile(r"(?i)\b(?:rmb|cny|usd|eur|gbp)\b|[$¥￥]")
_DATE_PATTERNS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y.%m.%d",
]


def normalize_amount(text: str) -> str | None:
    cleaned = _CURRENCY_RE.sub("", text).replace(",", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        value = Decimal(match.group(0)).normalize()
    except InvalidOperation:
        return None
    if value == value.to_integral():
        return str(int(value))
    return format(value, "f").rstrip("0").rstrip(".")


def normalize_date(text: str) -> str | None:
    raw = text.strip()
    cn_match = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?", raw)
    if cn_match:
        y, m, d = [int(x) for x in cn_match.groups()]
        return f"{y:04d}-{m:02d}-{d:02d}"

    candidate = re.search(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}/\d{1,2}/\d{4}", raw)
    if not candidate:
        return None
    value = candidate.group(0)
    for pattern in _DATE_PATTERNS:
        try:
            return datetime.strptime(value, pattern).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    value = str(text).lower().strip()
    amount = normalize_amount(value)
    date = normalize_date(value)
    if date and re.search(r"\d{4}\s*年|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}/\d{1,2}/\d{4}", value):
        return date
    if amount and re.search(r"(?i)(?:rmb|cny|usd|eur|gbp|[$¥￥]|\d+\.\d+)", value):
        return amount
    value = _CURRENCY_RE.sub("", value)
    value = _PUNCT_RE.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value
