from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.field_metrics import normalized_exact_match, regex_match


def _is_correct(row: dict[str, Any]) -> bool:
    field_name = str(row.get("field_name", ""))
    field_type = row.get("field_type") or ("amount" if field_name == "total_amount" else ("date" if field_name == "date" else "other"))
    return normalized_exact_match(row.get("pred_answer", ""), row.get("gold_answer", "")) or regex_match(
        row.get("pred_answer", ""),
        row.get("gold_answer", ""),
        field_type,
    )


def analyze_predictions(predictions_csv: str | Path, output_dir: str | Path = "outputs/error_cases/sroie") -> dict[str, Any]:
    path = Path(predictions_csv)
    if not path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {path}")
    df = pd.read_csv(path).fillna("")
    skipped_mask = df["skipped"].astype(str).str.lower().isin({"true", "1"}) if "skipped" in df.columns else pd.Series(False, index=df.index)
    skipped = df[skipped_mask].copy()
    evaluated = df.drop(skipped.index)
    wrong_rows = []
    for row in evaluated.to_dict(orient="records"):
        if not _is_correct(row):
            wrong_rows.append(row)
    wrong = pd.DataFrame(wrong_rows)
    field_counts = Counter(wrong["field_name"].tolist()) if not wrong.empty and "field_name" in wrong.columns else Counter()
    address_wrong = wrong[wrong["field_name"] == "address"].copy() if not wrong.empty and "field_name" in wrong.columns else pd.DataFrame()

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    skipped_path = target / "skipped_samples.csv"
    wrong_path = target / "wrong_cases.csv"
    per_field_path = target / "per_field_wrong_cases.csv"
    address_path = target / "address_wrong_examples.csv"
    summary_path = target / "sroie_error_summary.json"
    skipped_summary_path = Path("outputs/metrics/skipped_samples_summary.json")
    skipped.to_csv(skipped_path, index=False)
    wrong.to_csv(wrong_path, index=False)
    pd.DataFrame([{"field_name": key, "wrong_count": value} for key, value in sorted(field_counts.items())]).to_csv(per_field_path, index=False)
    address_wrong.head(20).to_csv(address_path, index=False)
    summary = {
        "predictions_csv": str(path),
        "num_rows": int(len(df)),
        "num_skipped": int(len(skipped)),
        "num_evaluated": int(len(evaluated)),
        "num_wrong": int(len(wrong)),
        "per_field_wrong_cases": dict(sorted(field_counts.items())),
        "outputs": {
            "skipped_samples": str(skipped_path),
            "wrong_cases": str(wrong_path),
            "per_field_wrong_cases": str(per_field_path),
            "address_wrong_examples": str(address_path),
        },
    }
    skipped_summary_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_summary = {
        "predictions_csv": str(path),
        "num_skipped": int(len(skipped)),
        "skipped_samples": skipped[[column for column in ["sample_id", "field_name", "skip_reason"] if column in skipped.columns]].to_dict(orient="records"),
    }
    summary["outputs"]["skipped_samples_summary"] = str(skipped_summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    skipped_summary_path.write_text(json.dumps(skipped_summary, ensure_ascii=False, indent=2), encoding="utf-8")
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
