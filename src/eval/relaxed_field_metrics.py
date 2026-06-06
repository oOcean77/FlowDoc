from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from src.eval.field_metrics import normalized_exact_match


STRICT_FIELDS = {
    "total_amount",
    "subtotal",
    "tax",
    "date",
    "invoice_id",
    "receipt_id",
    "document_id",
}

RELAXED_FIELDS = {
    "address",
    "company",
    "merchant",
    "vendor",
}

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_relaxed_text(text: object) -> str:
    value = "" if text is None else str(text)
    value = value.lower()
    value = _PUNCT_RE.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _tokens(text: object) -> list[str]:
    normalized = normalize_relaxed_text(text)
    return normalized.split() if normalized else []


def token_f1(pred: object, gold: object) -> float:
    pred_tokens = _tokens(pred)
    gold_tokens = _tokens(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    overlap = sum(min(pred_counts.get(token, 0), gold_counts.get(token, 0)) for token in pred_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def char_similarity(pred: object, gold: object) -> float:
    pred_norm = normalize_relaxed_text(pred)
    gold_norm = normalize_relaxed_text(gold)
    if not pred_norm and not gold_norm:
        return 1.0
    if not pred_norm or not gold_norm:
        return 0.0
    return SequenceMatcher(None, pred_norm, gold_norm).ratio()


def evaluate_field_relaxed(pred: object, gold: object, field_name: str) -> dict[str, Any]:
    field = str(field_name)
    is_normalized_exact = normalized_exact_match(pred, gold)
    f1 = token_f1(pred, gold)
    similarity = char_similarity(pred, gold)
    if field in RELAXED_FIELDS:
        match_policy = "relaxed"
        relaxed_correct = is_normalized_exact or f1 >= 0.85 or similarity >= 0.90
    else:
        match_policy = "strict"
        relaxed_correct = is_normalized_exact
    return {
        "normalized_exact_match": bool(is_normalized_exact),
        "token_f1": float(f1),
        "char_similarity": float(similarity),
        "relaxed_correct": bool(relaxed_correct),
        "match_policy": match_policy,
    }
