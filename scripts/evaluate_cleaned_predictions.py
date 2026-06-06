from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.field_metrics import normalized_exact_match, regex_match
from src.eval.relaxed_field_metrics import evaluate_field_relaxed


def _field_type(row: dict[str, Any]) -> str:
    field_name = str(row.get("field_name", ""))
    return str(row.get("field_type") or ("amount" if field_name in {"total_amount", "subtotal", "tax"} else ("date" if field_name == "date" else "other")))


def _normalized_correct(row: dict[str, Any], answer_column: str) -> bool:
    return normalized_exact_match(row.get(answer_column, ""), row.get("gold_answer", "")) or regex_match(
        row.get(answer_column, ""),
        row.get("gold_answer", ""),
        _field_type(row),
    )


def _relaxed_correct(row: dict[str, Any], answer_column: str) -> bool:
    return bool(evaluate_field_relaxed(row.get(answer_column, ""), row.get("gold_answer", ""), str(row.get("field_name", "")))["relaxed_correct"])


def _accuracy(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _per_field_accuracy(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    counts: dict[str, list[int]] = {}
    for row in rows:
        field = str(row.get("field_name", "unknown"))
        counts.setdefault(field, [0, 0])
        counts[field][0] += int(bool(row[key]))
        counts[field][1] += 1
    return {field: hits / total if total else 0.0 for field, (hits, total) in sorted(counts.items())}


def evaluate_cleaned_predictions(predictions: str | Path, answer_column: str = "cleaned_pred_answer", output: str | Path = "outputs/metrics/cleaned_eval.json") -> dict[str, Any]:
    path = Path(predictions)
    if not path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {path}")
    df = pd.read_csv(path).fillna("")
    if answer_column not in df.columns:
        raise ValueError(f"Missing answer column: {answer_column}")
    skipped_mask = df["skipped"].astype(str).str.lower().isin({"true", "1"}) if "skipped" in df.columns else pd.Series(False, index=df.index)
    evaluated = df.drop(df[skipped_mask].index)
    rows = []
    for row in evaluated.to_dict(orient="records"):
        raw_norm = _normalized_correct(row, "pred_answer")
        cleaned_norm = _normalized_correct(row, answer_column)
        raw_relaxed = _relaxed_correct(row, "pred_answer")
        cleaned_relaxed = _relaxed_correct(row, answer_column)
        changed = str(row.get("postprocess_changed", "")).lower() in {"true", "1"} or row.get("pred_answer", "") != row.get(answer_column, "")
        rows.append(
            {
                **row,
                "raw_normalized_correct": raw_norm,
                "cleaned_normalized_correct": cleaned_norm,
                "raw_relaxed_correct": raw_relaxed,
                "cleaned_relaxed_correct": cleaned_relaxed,
                "postprocess_changed_bool": changed,
            }
        )
    raw_norm_values = [bool(row["raw_normalized_correct"]) for row in rows]
    cleaned_norm_values = [bool(row["cleaned_normalized_correct"]) for row in rows]
    raw_relaxed_values = [bool(row["raw_relaxed_correct"]) for row in rows]
    cleaned_relaxed_values = [bool(row["cleaned_relaxed_correct"]) for row in rows]
    per_field_raw = _per_field_accuracy(rows, "raw_normalized_correct")
    per_field_cleaned = _per_field_accuracy(rows, "cleaned_normalized_correct")
    fields = sorted(set(per_field_raw) | set(per_field_cleaned))
    changed_rows = [row for row in rows if row["postprocess_changed_bool"]]
    summary = {
        "predictions": str(path),
        "answer_column": answer_column,
        "total": int(len(df)),
        "evaluated": int(len(rows)),
        "skipped": int(len(df) - len(rows)),
        "raw_pred_normalized_accuracy": _accuracy(raw_norm_values),
        "cleaned_pred_normalized_accuracy": _accuracy(cleaned_norm_values),
        "raw_pred_relaxed_accuracy": _accuracy(raw_relaxed_values),
        "cleaned_pred_relaxed_accuracy": _accuracy(cleaned_relaxed_values),
        "per_field_cleaned_accuracy": per_field_cleaned,
        "per_field_delta": {field: per_field_cleaned.get(field, 0.0) - per_field_raw.get(field, 0.0) for field in fields},
        "changed_but_wrong_count": int(sum(1 for row in changed_rows if not row["cleaned_normalized_correct"])),
        "changed_and_fixed_count": int(sum(1 for row in changed_rows if not row["raw_normalized_correct"] and row["cleaned_normalized_correct"])),
        "changed_and_broke_count": int(sum(1 for row in changed_rows if row["raw_normalized_correct"] and not row["cleaned_normalized_correct"])),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cleaned prediction columns without replacing raw prediction metrics.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--answer-column", default="cleaned_pred_answer")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = evaluate_cleaned_predictions(args.predictions, args.answer_column, args.output)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
