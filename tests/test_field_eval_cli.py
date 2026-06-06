from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.run_field_eval import main


def test_run_field_eval_cli_input_output() -> None:
    root = Path("outputs/test_artifacts/field_eval_cli")
    root.mkdir(parents=True, exist_ok=True)
    dataset = root / "sroie_qa.csv"
    output = root / "sroie_metrics.json"
    pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "doc_id": "d1",
                "doc_type": "receipt",
                "image_path": "missing.jpg",
                "question": "What is the total_amount?",
                "answer": "123",
                "field_name": "total_amount",
                "field_type": "amount",
                "ocr_text": "Total: 123",
                "bbox": None,
                "source_dataset": "sroie",
            }
        ]
    ).to_csv(dataset, index=False)

    assert main(["--input", str(dataset), "--output", str(output)]) == 0
    metrics = json.loads(output.read_text(encoding="utf-8"))
    assert metrics["dataset"] == "sroie-real"
    assert metrics["field_level_accuracy"] == 1.0
