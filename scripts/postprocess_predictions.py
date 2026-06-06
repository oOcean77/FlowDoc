from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.postprocess.field_cleaner import clean_prediction


def postprocess_predictions(input_predictions: str | Path, output_predictions: str | Path, summary_output: str | Path) -> dict[str, Any]:
    input_path = Path(input_predictions)
    if not input_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {input_path}")
    df = pd.read_csv(input_path).fillna("")
    rows = []
    rules_count: Counter[str] = Counter()
    per_field_changed: Counter[str] = Counter()
    low_confidence_count = 0
    changed_count = 0
    for row in df.to_dict(orient="records"):
        result = clean_prediction(
            str(row.get("pred_answer", "")),
            str(row.get("field_name", "")),
            str(row.get("ocr_text", "")) if "ocr_text" in row else None,
        )
        changed_count += int(result["changed"])
        per_field_changed[str(row.get("field_name", "unknown"))] += int(result["changed"])
        low_confidence_count += int(result["confidence"] == "low")
        rules_count.update(result["rules_applied"])
        rows.append(
            {
                **row,
                "cleaned_pred_answer": result["cleaned"],
                "postprocess_changed": result["changed"],
                "postprocess_rules": ";".join(result["rules_applied"]),
                "postprocess_confidence": result["confidence"],
            }
        )
    output_path = Path(output_predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    total = int(len(rows))
    summary = {
        "input_predictions": str(input_path),
        "output_predictions": str(output_path),
        "total": total,
        "changed": int(changed_count),
        "changed_rate": changed_count / total if total else 0.0,
        "per_field_changed": dict(sorted(per_field_changed.items())),
        "rules_count": dict(sorted(rules_count.items())),
        "low_confidence_count": int(low_confidence_count),
    }
    summary_path = Path(summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean prediction CSV outputs with field-aware rules.")
    parser.add_argument("--input-predictions", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--summary-output", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = postprocess_predictions(args.input_predictions, args.output_predictions, args.summary_output)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
