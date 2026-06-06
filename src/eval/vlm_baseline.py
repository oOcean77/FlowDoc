from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.dataset_adapter import load_local_csv
from src.data.instruction_builder import STRATEGIES, build_instruction_sample
from src.eval.field_metrics import evaluate_rows
from src.eval.prediction_postprocess import postprocess_prediction
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
    "lora_adapter",
    "skipped",
    "skip_reason",
]


def _make_runner(
    backend: str,
    model_name: str | None = None,
    device: str = "auto",
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    do_sample: bool = False,
    lora_adapter: str | None = None,
):
    if backend == "dummy":
        return DummyVLMRunner()
    if backend == "qwen2_5_vl":
        return Qwen25VLRunner(
            model_name=model_name or "Qwen/Qwen2.5-VL-3B-Instruct",
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            lora_adapter=lora_adapter,
        )
    if backend == "llava":
        return LLaVARunner(model_name=model_name or "llava-hf/llava-1.5-7b-hf", device=device)
    raise ValueError(f"Unsupported backend '{backend}'.")


def _prompt_from_instruction(sample: dict[str, Any]) -> str:
    for item in sample["messages"][0]["content"]:
        if item["type"] == "text":
            return item["text"]
    return ""


def _prediction_path(backend: str, strategy: str, predictions_dir: str | Path = "outputs/predictions", lora_adapter: str | None = None) -> Path:
    adapter_tag = "_lora" if lora_adapter else ""
    return Path(predictions_dir) / f"{backend}{adapter_tag}_{strategy}_predictions.csv"


def _skipped_metrics(backend: str, strategy: str, num_samples: int, reason: str, lora_adapter: str | None = None) -> dict[str, Any]:
    return {
        "backend": backend,
        "strategy": strategy,
        "lora_adapter": lora_adapter,
        "num_samples": num_samples,
        "num_evaluated": 0,
        "num_skipped": num_samples,
        "skipped": True,
        "skip_reason": reason,
        "field_level_accuracy": None,
        "missing_field_rate": None,
        "multi_value_conflict_rate": None,
    }


def _skipped_predictions(rows: list[dict[str, Any]], strategy: str, backend: str, reason: str, lora_adapter: str | None = None) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": row["sample_id"],
            "doc_id": row["doc_id"],
            "field_name": row["field_name"],
            "gold_answer": row["answer"],
            "pred_answer": "",
            "strategy": strategy,
            "backend": backend,
            "lora_adapter": lora_adapter or "",
            "skipped": True,
            "skip_reason": reason,
        }
        for row in rows
    ]


def _smoke_logs_from_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": row["sample_id"],
            "field_name": row["field_name"],
            "strategy": row["strategy"],
            "skipped": bool(row["skipped"]),
            "skip_reason": row["skip_reason"],
            "pred_answer_preview": str(row["pred_answer"])[:100],
        }
        for row in predictions
    ]


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
    temperature: float = 0.0,
    do_sample: bool = False,
    smoke_test: bool = False,
    lora_adapter: str | None = None,
) -> dict[str, Any]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unsupported strategy '{strategy}'.")
    df = load_local_csv(input_path)
    effective_max_samples = 3 if smoke_test and max_samples is None else max_samples
    if effective_max_samples is not None:
        df = df.head(effective_max_samples)
    rows = df.to_dict(orient="records")
    pred_path = _prediction_path(backend, strategy, predictions_dir, lora_adapter=lora_adapter)
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        runner = _make_runner(
            backend,
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            lora_adapter=lora_adapter,
        )
    except (ImportError, FileNotFoundError, RuntimeError, OSError, ValueError) as exc:
        reason = str(exc)
        predictions = _skipped_predictions(rows, strategy, backend, reason, lora_adapter=lora_adapter)
        pd.DataFrame(predictions, columns=PREDICTION_COLUMNS).to_csv(pred_path, index=False)
        metrics = _skipped_metrics(backend, strategy, len(rows), reason, lora_adapter=lora_adapter)
        metrics["smoke_test"] = smoke_test
        metrics["generation"] = {"max_new_tokens": max_new_tokens, "temperature": temperature, "do_sample": do_sample}
        if smoke_test:
            metrics["smoke_logs"] = _smoke_logs_from_predictions(predictions)
        write_json(output_path, metrics)
        return metrics

    if isinstance(runner, DummyVLMRunner):
        reason = runner.skip_reason
        predictions = _skipped_predictions(rows, strategy, backend, reason, lora_adapter=lora_adapter)
        pd.DataFrame(predictions, columns=PREDICTION_COLUMNS).to_csv(pred_path, index=False)
        metrics = _skipped_metrics(backend, strategy, len(rows), reason, lora_adapter=lora_adapter)
        metrics["smoke_test"] = smoke_test
        metrics["generation"] = {"max_new_tokens": max_new_tokens, "temperature": temperature, "do_sample": do_sample}
        if smoke_test:
            metrics["smoke_logs"] = _smoke_logs_from_predictions(predictions)
        write_json(output_path, metrics)
        return metrics

    predictions = []
    eval_rows = []
    smoke_logs = []
    for row in rows:
        sample = build_instruction_sample(row, strategy)
        if sample["unavailable"]:
            pred_row = {
                "sample_id": row["sample_id"],
                "doc_id": row["doc_id"],
                "field_name": row["field_name"],
                "gold_answer": row["answer"],
                "pred_answer": "",
                "strategy": strategy,
                "backend": backend,
                "lora_adapter": lora_adapter or "",
                "skipped": True,
                "skip_reason": sample["skip_reason"],
            }
            predictions.append(pred_row)
            smoke_logs.append(pred_row)
            continue
        result = runner.generate_result(
            row.get("image_path") if strategy != "ocr_only" else None,
            _prompt_from_instruction(sample),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        pred = postprocess_prediction(result.text, row.get("field_type", "other"))
        pred_row = {
            "sample_id": row["sample_id"],
            "doc_id": row["doc_id"],
            "field_name": row["field_name"],
            "gold_answer": row["answer"],
            "pred_answer": pred,
            "strategy": strategy,
            "backend": backend,
            "lora_adapter": lora_adapter or "",
            "skipped": result.skipped,
            "skip_reason": result.skip_reason or "",
        }
        predictions.append(pred_row)
        smoke_logs.append(pred_row)
        if result.skipped:
            continue
        eval_rows.append({**row, "pred_answer": pred, "error_type": None})

    pd.DataFrame(predictions, columns=PREDICTION_COLUMNS).to_csv(pred_path, index=False)
    if not eval_rows:
        reasons = sorted({row["skip_reason"] for row in predictions if row["skip_reason"]})
        metrics = _skipped_metrics(backend, strategy, len(rows), "; ".join(reasons) or "no evaluable samples", lora_adapter=lora_adapter)
    else:
        evaluated = evaluate_rows(eval_rows)
        metrics = {
            "backend": backend,
            "strategy": strategy,
            "lora_adapter": lora_adapter,
            "num_samples": len(rows),
            "num_evaluated": len(eval_rows),
            "num_skipped": len(rows) - len(eval_rows),
            "skipped": False,
            "skip_reason": "",
            "field_level_accuracy": evaluated["field_level_accuracy"],
            "missing_field_rate": evaluated["missing_field_rate"],
            "multi_value_conflict_rate": evaluated["multi_value_conflict_rate"],
            "per_field_accuracy": evaluated["per_field_accuracy"],
        }
    metrics["smoke_test"] = smoke_test
    metrics["generation"] = {"max_new_tokens": max_new_tokens, "temperature": temperature, "do_sample": do_sample}
    if smoke_test:
        metrics["smoke_logs"] = _smoke_logs_from_predictions(smoke_logs)
    write_json(output_path, metrics)
    return metrics
