from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from src.utils.text_norm import normalize_text


def exact_match(pred: object, gold: object) -> bool:
    return str(pred) == str(gold)


def normalized_exact_match(pred: object, gold: object) -> bool:
    return normalize_text(pred) == normalize_text(gold)


def regex_match(pred: object, gold: object, field_type: str | None = None) -> bool:
    pred_text = str(pred)
    gold_norm = normalize_text(gold)
    if field_type == "amount":
        values = [normalize_text(match.group(0)) for match in re.finditer(r"(?:rmb|cny|usd|\$|¥|￥)?\s*\d+(?:\.\d+)?", pred_text, flags=re.I)]
        return gold_norm in values
    if field_type in {"date", "deadline"}:
        return gold_norm == normalize_text(pred_text)
    return normalized_exact_match(pred, gold)


def evaluate_rows(rows: Iterable[dict]) -> dict:
    items = list(rows)
    total = len(items)
    correct = 0
    missing = 0
    conflicts = 0
    per_field_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    row_results = []
    for row in items:
        pred = row.get("pred_answer", "")
        gold = row.get("answer", "")
        field = row.get("field_name", "unknown")
        field_type = row.get("field_type", "other")
        is_correct = normalized_exact_match(pred, gold) or regex_match(pred, gold, field_type)
        correct += int(is_correct)
        missing += int(pred in (None, ""))
        conflicts += int(row.get("error_type") == "multi_value_conflict")
        per_field_counts[field][0] += int(is_correct)
        per_field_counts[field][1] += 1
        row_results.append({**row, "correct": is_correct})
    per_field_accuracy = {field: hits / count if count else 0.0 for field, (hits, count) in per_field_counts.items()}
    return {
        "num_samples": total,
        "field_level_accuracy": correct / total if total else 0.0,
        "per_field_accuracy": per_field_accuracy,
        "missing_field_rate": missing / total if total else 0.0,
        "multi_value_conflict_rate": conflicts / total if total else 0.0,
        "rows": row_results,
    }
