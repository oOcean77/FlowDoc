from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.run_eval import run_ocr_eval
from src.utils.io import write_json


if __name__ == "__main__":
    results = run_ocr_eval()
    rows = results.pop("rows")
    output = write_json("outputs/metrics/field_eval_results.json", results)
    print(f"field_level_accuracy={results['field_level_accuracy']:.3f}")
    print(f"per_field_accuracy={results['per_field_accuracy']}")
    print(f"Saved metrics to {output}")
    results["rows"] = rows
