from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.field_metrics import exact_match, normalized_exact_match, regex_match
from src.eval.relaxed_field_metrics import evaluate_field_relaxed


def _field_type(row: dict[str, Any]) -> str:
    field_name = str(row.get("field_name", ""))
    return str(row.get("field_type") or ("amount" if field_name == "total_amount" else ("date" if field_name == "date" else "other")))


def _is_normalized_correct(row: dict[str, Any]) -> bool:
    return normalized_exact_match(row.get("pred_answer", ""), row.get("gold_answer", "")) or regex_match(
        row.get("pred_answer", ""),
        row.get("gold_answer", ""),
        _field_type(row),
    )


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


def _per_field_average(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    counts: dict[str, list[float]] = {}
    for row in rows:
        field = str(row.get("field_name", "unknown"))
        counts.setdefault(field, [0.0, 0.0])
        counts[field][0] += float(row[key])
        counts[field][1] += 1.0
    return {field: total / count if count else 0.0 for field, (total, count) in sorted(counts.items())}


def _field_average(rows: list[dict[str, Any]], field_name: str, key: str) -> float:
    values = [float(row[key]) for row in rows if str(row.get("field_name", "")) == field_name]
    return sum(values) / len(values) if values else 0.0


def analyze_predictions(predictions_csv: str | Path, output_dir: str | Path = "outputs/analysis/sroie") -> dict[str, Any]:
    path = Path(predictions_csv)
    if not path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {path}")
    df = pd.read_csv(path).fillna("")
    skipped_mask = df["skipped"].astype(str).str.lower().isin({"true", "1"}) if "skipped" in df.columns else pd.Series(False, index=df.index)
    skipped = df[skipped_mask].copy()
    evaluated = df.drop(skipped.index)
    evaluated_rows = []
    wrong_rows = []
    relaxed_wrong_rows = []
    for row in evaluated.to_dict(orient="records"):
        raw_correct = exact_match(row.get("pred_answer", ""), row.get("gold_answer", ""))
        normalized_correct = _is_normalized_correct(row)
        relaxed = evaluate_field_relaxed(row.get("pred_answer", ""), row.get("gold_answer", ""), str(row.get("field_name", "")))
        enriched = {
            **row,
            "raw_correct": raw_correct,
            "normalized_correct": normalized_correct,
            "relaxed_normalized_exact_match": relaxed["normalized_exact_match"],
            "token_f1": relaxed["token_f1"],
            "char_similarity": relaxed["char_similarity"],
            "relaxed_correct": relaxed["relaxed_correct"],
            "match_policy": relaxed["match_policy"],
        }
        evaluated_rows.append(enriched)
        if not normalized_correct:
            wrong_rows.append(enriched)
        if not relaxed["relaxed_correct"]:
            relaxed_wrong_rows.append(enriched)
    wrong = pd.DataFrame(wrong_rows)
    relaxed_wrong = pd.DataFrame(relaxed_wrong_rows)
    field_counts = Counter(wrong["field_name"].tolist()) if not wrong.empty and "field_name" in wrong.columns else Counter()
    address_wrong = wrong[wrong["field_name"] == "address"].copy() if not wrong.empty and "field_name" in wrong.columns else pd.DataFrame()

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    skipped_path = target / "skipped_samples.csv"
    wrong_path = target / "wrong_cases.csv"
    per_field_path = target / "per_field_wrong_cases.csv"
    address_path = target / "address_wrong_cases.csv"
    relaxed_wrong_path = target / "relaxed_wrong_cases.csv"
    summary_path = target / "error_summary.json"
    skipped.to_csv(skipped_path, index=False)
    wrong.to_csv(wrong_path, index=False)
    relaxed_wrong.to_csv(relaxed_wrong_path, index=False)
    pd.DataFrame([{"field_name": key, "wrong_count": value} for key, value in sorted(field_counts.items())]).to_csv(per_field_path, index=False)
    address_wrong.to_csv(address_path, index=False)
    raw_values = [bool(row["raw_correct"]) for row in evaluated_rows]
    normalized_values = [bool(row["normalized_correct"]) for row in evaluated_rows]
    relaxed_values = [bool(row["relaxed_correct"]) for row in evaluated_rows]
    summary = {
        "predictions_csv": str(path),
        "total": int(len(df)),
        "evaluated": int(len(evaluated_rows)),
        "skipped": int(len(skipped)),
        "raw_accuracy": _accuracy(raw_values),
        "normalized_accuracy": _accuracy(normalized_values),
        "relaxed_accuracy": _accuracy(relaxed_values),
        "per_field_raw_accuracy": _per_field_accuracy(evaluated_rows, "raw_correct"),
        "per_field_normalized_accuracy": _per_field_accuracy(evaluated_rows, "normalized_correct"),
        "per_field_relaxed_accuracy": _per_field_accuracy(evaluated_rows, "relaxed_correct"),
        "per_field_avg_token_f1": _per_field_average(evaluated_rows, "token_f1"),
        "per_field_avg_char_similarity": _per_field_average(evaluated_rows, "char_similarity"),
        "address_token_f1": _field_average(evaluated_rows, "address", "token_f1"),
        "address_char_similarity": _field_average(evaluated_rows, "address", "char_similarity"),
        "top_error_fields": [{"field_name": key, "wrong_count": value} for key, value in field_counts.most_common()],
        "num_wrong": int(len(wrong)),
        "per_field_wrong_cases": dict(sorted(field_counts.items())),
        "outputs": {
            "skipped_samples": str(skipped_path),
            "wrong_cases": str(wrong_path),
            "relaxed_wrong_cases": str(relaxed_wrong_path),
            "per_field_wrong_cases": str(per_field_path),
            "address_wrong_cases": str(address_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SROIE prediction CSV wrong cases and skipped samples.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default="outputs/error_cases/sroie")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = analyze_predictions(args.predictions, args.output_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
