from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.run_eval import run_ocr_eval
from src.utils.io import write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR-only field-level evaluation.")
    parser.add_argument("--input", default="data/processed/mock_qa.csv")
    parser.add_argument("--output", default="outputs/metrics/field_eval_results.json")
    return parser.parse_args(argv)


def dataset_name(input_path: str) -> str:
    lowered = input_path.replace("\\", "/").lower()
    if "sroie" in lowered:
        return "sroie-real"
    return "mock-hard"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = run_ocr_eval(args.input)
    rows = results.pop("rows")
    results["dataset"] = dataset_name(args.input)
    results["baseline_family"] = "ocr-only"
    results["tuning"] = "rule"
    output = write_json(args.output, results)
    print(f"field_level_accuracy={results['field_level_accuracy']:.3f}")
    print(f"per_field_accuracy={results['per_field_accuracy']}")
    print(f"Saved metrics to {output}")
    results["rows"] = rows
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
