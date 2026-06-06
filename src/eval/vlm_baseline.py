from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.dataset_adapter import load_local_csv
from src.data.instruction_builder import STRATEGIES, build_instruction_sample
from src.eval.field_metrics import evaluate_rows
from src.utils.io import write_json
from src.vlm.dummy import DummyVLMRunner
from src.vlm.llava import LLaVARunner
from src.vlm.qwen2_5_vl import Qwen25VLRunner


PREDICTION_COLUMNS = [
    "sample_id",
    "doc_id",
    "field_name",
    "gold_answer",
    "pred_answer",
    "strategy",
    "backend",
    "skipped",
    "skip_reason",
]


def _make_runner(backend: str, model_name: str | None = None, device: str = "auto"):
    if backend == "dummy":
        return DummyVLMRunner()
    if backend == "qwen2_5_vl":
        return Qwen25VLRunner(model_name=model_name or "Qwen/Qwen2.5-VL-3B-Instruct", device=device)
    if backend == "llava":
        return LLaVARunner(model_name=model_name or "llava-hf/llava-1.5-7b-hf", device=device)
    raise ValueError(f"Unsupported backend '{backend}'.")


def _prompt_from_instruction(sample: dict[str, Any]) -> str:
    for item in sample["messages"][0]["content"]:
        if item["type"] == "text":
            return item["text"]
    return ""


def _prediction_path(backend: str, strategy: str, predictions_dir: str | Path = "outputs/predictions") -> Path:
    return Path(predictions_dir) / f"{backend}_{strategy}_predictions.csv"


def _skipped_metrics(backend: str, strategy: str, num_samples: int, reason: str) -> dict[str, Any]:
    return {
        "backend": backend,
        "strategy": strategy,
        "num_samples": num_samples,
        "num_evaluated": 0,
        "skipped": True,
        "skip_reason": reason,
        "field_level_accuracy": None,
        "missing_field_rate": None,
        "multi_value_conflict_rate": None,
    }


def run_vlm_baseline(
    input_path: str | Path,
    strategy: str,
    backend: str,
    output_path: str | Path,
    model_name: str | None = None,
    max_samples: int | None = None,
    device: str = "auto",
    predictions_dir: str | Path = "outputs/predictions",
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unsupported strategy '{strategy}'.")
    df = load_local_csv(input_path)
    if max_samples is not None:
        df = df.head(max_samples)
    rows = df.to_dict(orient="records")
    pred_path = _prediction_path(backend, strategy, predictions_dir)
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        runner = _make_runner(backend, model_name=model_name, device=device)
    except ImportError as exc:
        reason = str(exc)
        predictions = [
            {
                "sample_id": row["sample_id"],
                "doc_id": row["doc_id"],
                "field_name": row["field_name"],
                "gold_answer": row["answer"],
                "pred_answer": "",
                "strategy": strategy,
                "backend": backend,
                "skipped": True,
                "skip_reason": reason,
            }
            for row in rows
        ]
        pd.DataFrame(predictions, columns=PREDICTION_COLUMNS).to_csv(pred_path, index=False)
        metrics = _skipped_metrics(backend, strategy, len(rows), reason)
        write_json(output_path, metrics)
        return metrics

    if isinstance(runner, DummyVLMRunner):
        reason = runner.skip_reason
        predictions = [
            {
                "sample_id": row["sample_id"],
                "doc_id": row["doc_id"],
                "field_name": row["field_name"],
                "gold_answer": row["answer"],
                "pred_answer": "",
                "strategy": strategy,
                "backend": backend,
                "skipped": True,
                "skip_reason": reason,
            }
            for row in rows
        ]
        pd.DataFrame(predictions, columns=PREDICTION_COLUMNS).to_csv(pred_path, index=False)
        metrics = _skipped_metrics(backend, strategy, len(rows), reason)
        write_json(output_path, metrics)
        return metrics

    predictions = []
    eval_rows = []
    for row in rows:
        sample = build_instruction_sample(row, strategy)
        if sample["unavailable"]:
            predictions.append(
                {
                    "sample_id": row["sample_id"],
                    "doc_id": row["doc_id"],
                    "field_name": row["field_name"],
                    "gold_answer": row["answer"],
                    "pred_answer": "",
                    "strategy": strategy,
                    "backend": backend,
                    "skipped": True,
                    "skip_reason": sample["skip_reason"],
                }
            )
            continue
        pred = runner.generate(row.get("image_path") if strategy != "ocr_only" else None, _prompt_from_instruction(sample), max_new_tokens=max_new_tokens)
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "doc_id": row["doc_id"],
                "field_name": row["field_name"],
                "gold_answer": row["answer"],
                "pred_answer": pred,
                "strategy": strategy,
                "backend": backend,
                "skipped": False,
                "skip_reason": "",
            }
        )
        eval_rows.append({**row, "pred_answer": pred, "error_type": None})

    pd.DataFrame(predictions, columns=PREDICTION_COLUMNS).to_csv(pred_path, index=False)
    if not eval_rows:
        metrics = _skipped_metrics(backend, strategy, len(rows), "no evaluable samples")
    else:
        evaluated = evaluate_rows(eval_rows)
        metrics = {
            "backend": backend,
            "strategy": strategy,
            "num_samples": len(rows),
            "num_evaluated": len(eval_rows),
            "skipped": False,
            "skip_reason": "",
            "field_level_accuracy": evaluated["field_level_accuracy"],
            "missing_field_rate": evaluated["missing_field_rate"],
            "multi_value_conflict_rate": evaluated["multi_value_conflict_rate"],
            "per_field_accuracy": evaluated["per_field_accuracy"],
        }
    write_json(output_path, metrics)
    return metrics
