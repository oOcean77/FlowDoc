from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.eval.vlm_baseline import run_vlm_baseline
from src.vlm.qwen2_5_vl import Qwen25VLRunner


def test_qwen_runner_missing_local_path_raises_clear_error() -> None:
    with pytest.raises(FileNotFoundError, match="Local Qwen2.5-VL model path does not exist"):
        Qwen25VLRunner(model_name="C:\\models\\missing-qwen")


def test_qwen_baseline_missing_local_path_skips() -> None:
    root = Path("outputs/test_artifacts/qwen_skip")
    root.mkdir(parents=True, exist_ok=True)
    dataset = root / "qa.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "doc_id": "d1",
                "doc_type": "receipt",
                "image_path": "missing.png",
                "question": "What is total?",
                "answer": "123",
                "field_name": "total_amount",
                "field_type": "amount",
                "ocr_text": "Total: 123",
                "bbox": None,
                "source_dataset": "mock",
            }
        ]
    ).to_csv(dataset, index=False)

    metrics = run_vlm_baseline(
        dataset,
        "image_ocr",
        "qwen2_5_vl",
        root / "qwen_metrics.json",
        model_name="C:\\models\\missing-qwen",
        smoke_test=True,
        predictions_dir=root / "predictions",
    )

    assert metrics["skipped"] is True
    assert "Local Qwen2.5-VL model path does not exist" in metrics["skip_reason"]
    pred_path = Path(metrics["predictions_path"])
    assert pred_path.name == "qwen_metrics_predictions.csv"
    assert pred_path.exists()
