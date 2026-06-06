from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


EXCLUDED_NAME_PARTS = [
    "train_log",
    "env_report",
    "readiness_report",
    "sft_readiness",
    "lora_dryrun",
    "prediction_change_summary",
    "skipped_samples_summary",
]

ALLOWED_EXACT_NAMES = {"field_eval_results.json"}
ALLOWED_PREFIXES = (
    "vlm_baseline",
    "sroie_qwen",
    "sroie_ocr_field_eval",
)

LEGACY_PREDICTION_FILENAMES = {
    "dummy_image_ocr_predictions.csv",
    "qwen2_5_vl_image_only_predictions.csv",
    "qwen2_5_vl_image_ocr_predictions.csv",
    "qwen2_5_vl_ocr_only_predictions.csv",
    "qwen2_5_vl_lora_image_ocr_predictions.csv",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_benchmark_metrics_file(path: Path, metrics: dict[str, Any]) -> bool:
    name = path.name.lower()
    if any(part in name for part in EXCLUDED_NAME_PARTS):
        return False
    allowed_name = name in ALLOWED_EXACT_NAMES or name.startswith(ALLOWED_PREFIXES) or "_field_eval" in name
    if not allowed_name:
        return False
    has_accuracy = "field_level_accuracy" in metrics
    skipped_benchmark = bool(metrics.get("skipped")) and ("backend" in metrics or "baseline_family" in metrics)
    return has_accuracy or skipped_benchmark


def _infer_dataset(path: Path, metrics: dict[str, Any]) -> str:
    if metrics.get("dataset"):
        return str(metrics["dataset"])
    name = path.name.lower()
    if "sroie" in name:
        return "sroie-real"
    return "mock-hard"


def _infer_subset(path: Path, metrics: dict[str, Any], dataset: str) -> str:
    if metrics.get("benchmark_subset"):
        return str(metrics["benchmark_subset"])
    name = path.stem.lower()
    match = re.search(r"(?:^|_)(\d+)(?:_|$)", name)
    if match and dataset == "sroie-real":
        return f"{dataset}-{match.group(1)}"
    if "full" in name:
        return f"{dataset}-full"
    return dataset


def _infer_tuning(metrics: dict[str, Any]) -> str:
    if metrics.get("lora_adapter"):
        return "LoRA"
    if metrics.get("backend") in {"qwen2_5_vl", "llava", "dummy"}:
        return "zero-shot"
    return str(metrics.get("tuning", "rule"))


def _status(metrics: dict[str, Any]) -> str:
    if metrics.get("skipped"):
        return "skipped"
    if metrics.get("field_level_accuracy") is None:
        return "unavailable"
    return "evaluated"


def _fmt(value: object) -> str:
    return "null" if value is None else f"{float(value):.3f}"


def collect_benchmark_rows(metrics_dir: str | Path = "outputs/metrics") -> list[dict[str, Any]]:
    root = Path(metrics_dir)
    rows = []
    for path in sorted(root.glob("*.json")):
        metrics = _load_json(path)
        if not is_benchmark_metrics_file(path, metrics):
            continue
        dataset = _infer_dataset(path, metrics)
        subset = _infer_subset(path, metrics, dataset)
        rows.append(
            {
                "file": path.name,
                "dataset": dataset,
                "subset": subset,
                "backend": metrics.get("backend", metrics.get("baseline_family", "ocr-only")),
                "strategy": metrics.get("strategy", "ocr_text"),
                "tuning": _infer_tuning(metrics),
                "status": _status(metrics),
                "num_samples": metrics.get("num_samples", 0),
                "num_evaluated": metrics.get("num_evaluated", metrics.get("num_samples", 0) if not metrics.get("skipped") else 0),
                "num_skipped": metrics.get("num_skipped", 0),
                "accuracy": metrics.get("field_level_accuracy"),
                "reason": metrics.get("skip_reason", ""),
                "per_field_accuracy": metrics.get("per_field_accuracy", {}),
            }
        )
    return rows


def _run_label(row: dict[str, Any]) -> str:
    backend = row["backend"]
    strategy = row["strategy"]
    tuning = row["tuning"]
    if backend == "ocr-only":
        return "OCR rule"
    return f"{backend} {strategy} {tuning}"


def _append_main_table(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            "| Dataset | Subset | Backend | Strategy | Tuning | Status | Num Evaluated | Num Skipped | Accuracy | Source | Reason |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    if not rows:
        lines.append("| none | none | none | none | none | unavailable | 0 | 0 | null | none | no benchmark metrics found |")
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['subset']} | {row['backend']} | {row['strategy']} | {row['tuning']} | {row['status']} | "
            f"{row['num_evaluated']} | {row['num_skipped']} | {_fmt(row['accuracy'])} | {row['file']} | {row['reason']} |"
        )


def _append_skipped_summary(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines.extend(["", "## Skipped Samples Summary", "", "| Run | Num Skipped | Reason |", "| --- | ---: | --- |"])
    skipped_rows = [row for row in rows if row["num_skipped"] or row["status"] == "skipped"]
    if not skipped_rows:
        lines.append("| none | 0 | no skipped benchmark samples found |")
    for row in skipped_rows:
        reason = row["reason"] or ("partial skip" if row["num_skipped"] else "")
        lines.append(f"| {_run_label(row)} ({row['subset']}) | {row['num_skipped']} | {reason} |")


def collect_skipped_prediction_rows(predictions_dir: str | Path = "outputs/predictions") -> dict[str, Any]:
    root = Path(predictions_dir)
    rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    files_seen = 0
    legacy_files_ignored: list[str] = []
    if root.exists():
        for path in sorted(root.glob("*_predictions.csv")):
            if path.name in LEGACY_PREDICTION_FILENAMES:
                legacy_files_ignored.append(str(path))
                continue
            files_seen += 1
            df = pd.read_csv(path).fillna("")
            if "skipped" not in df.columns:
                continue
            skipped_mask = df["skipped"].astype(str).str.lower().isin({"true", "1"})
            for item in df[skipped_mask].to_dict(orient="records"):
                reason = str(item.get("skip_reason", ""))
                reason_counts[reason or "unknown"] += 1
                rows.append(
                    {
                        "source_prediction_file": str(path),
                        "sample_id": item.get("sample_id", ""),
                        "field_name": item.get("field_name", ""),
                        "skip_reason": reason,
                    }
                )
    return {
        "predictions_dir": str(root),
        "prediction_files_seen": files_seen,
        "legacy_prediction_files_ignored": legacy_files_ignored,
        "num_skipped_rows": len(rows),
        "skip_reason_counts": dict(sorted(reason_counts.items())),
        "skipped_samples": rows,
        "trace_warning": ""
        if files_seen
        else "Skipped rows cannot be traced because the corresponding prediction CSV was overwritten or missing.",
    }


def write_skipped_samples_summary(
    predictions_dir: str | Path = "outputs/predictions",
    output_json: str | Path = "outputs/metrics/skipped_samples_summary.json",
    output_md: str | Path = "outputs/metrics/skipped_samples_summary.md",
) -> dict[str, Any]:
    summary = collect_skipped_prediction_rows(predictions_dir)
    json_path = Path(output_json)
    md_path = Path(output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Skipped Samples Summary",
        "",
        f"Prediction files seen: `{summary['prediction_files_seen']}`",
        f"Skipped rows: `{summary['num_skipped_rows']}`",
        "",
    ]
    if summary["legacy_prediction_files_ignored"]:
        lines.extend(
            [
                "Legacy prediction files ignored to avoid double-counting overwritten runs:",
                "",
                *[f"- `{path}`" for path in summary["legacy_prediction_files_ignored"]],
                "",
            ]
        )
    if summary["trace_warning"]:
        lines.extend([summary["trace_warning"], ""])
    lines.extend(["## Skip Reasons", "", "| Skip Reason | Count |", "| --- | ---: |"])
    if not summary["skip_reason_counts"]:
        lines.append("| none | 0 |")
    for reason, count in summary["skip_reason_counts"].items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(["", "## Rows", "", "| Source Prediction File | Sample ID | Field Name | Skip Reason |", "| --- | --- | --- | --- |"])
    if not summary["skipped_samples"]:
        lines.append("| none | none | none | none |")
    for row in summary["skipped_samples"]:
        lines.append(
            f"| {row['source_prediction_file']} | {row['sample_id']} | {row['field_name']} | {row['skip_reason']} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return summary


def _append_sroie_per_field(lines: list[str], rows: list[dict[str, Any]]) -> None:
    sroie_rows = [row for row in rows if row["dataset"] == "sroie-real" and row["per_field_accuracy"]]
    lines.extend(["", "## SROIE Per-Field Comparison", ""])
    if not sroie_rows:
        lines.append("No SROIE per-field metrics found yet. Run SROIE OCR-only and Qwen baselines first.")
        return
    fields = sorted({field for row in sroie_rows for field in row["per_field_accuracy"]})
    labels = [_run_label(row) + f" [{row['subset']}]" for row in sroie_rows]
    lines.append("| Field | " + " | ".join(labels) + " |")
    lines.append("| --- | " + " | ".join(["---:"] * len(labels)) + " |")
    for field in fields:
        values = [_fmt(row["per_field_accuracy"].get(field)) for row in sroie_rows]
        lines.append("| " + field + " | " + " | ".join(values) + " |")


def _append_sroie_same_subset(lines: list[str], rows: list[dict[str, Any]]) -> None:
    by_file = {row["file"]: row for row in rows}
    expected = [
        ("sroie_ocr_field_eval_100.json", "SROIE real", "OCR-only rule", "ocr_text", "same-subset rule OCR baseline"),
        ("sroie_qwen_ocr_only_100.json", "SROIE real", "Qwen2.5-VL zero-shot", "ocr_only", "same-subset VLM OCR-only baseline"),
        ("sroie_qwen_image_ocr_100.json", "SROIE real", "Qwen2.5-VL zero-shot", "image_ocr", "100 requested; skipped rows require tracing"),
    ]
    lines.extend(["", "## SROIE Same-Subset Comparison", ""])
    lines.extend(
        [
            "| Dataset | Method | Strategy | Requested | Evaluated | Skipped | Accuracy | Notes |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for filename, dataset, method, strategy, note in expected:
        row = by_file.get(filename)
        if row is None:
            lines.append(f"| {dataset} | {method} | {strategy} | unavailable | unavailable | unavailable | unavailable | missing `{filename}` |")
            continue
        requested = row["num_samples"] or row["num_evaluated"] + row["num_skipped"]
        lines.append(
            f"| {dataset} | {method} | {strategy} | {requested} | {row['num_evaluated']} | {row['num_skipped']} | {_fmt(row['accuracy'])} | {note} |"
        )
    lines.extend(
        [
            "",
            "OCR-only full 399 and Qwen 100 are not fully fair one-to-one comparisons.",
            "The same-subset 100 rows are the preferred horizontal comparison when all expected files are present.",
            "Qwen image+OCR 100 currently has partial evaluation when `Evaluated` is lower than `Requested`; skipped rows should be inspected.",
        ]
    )


def build_benchmark_report(
    metrics_dir: str | Path = "outputs/metrics",
    output_path: str | Path = "outputs/metrics/benchmark_report.md",
    predictions_dir: str | Path = "outputs/predictions",
) -> Path:
    rows = collect_benchmark_rows(metrics_dir)
    skipped_summary = write_skipped_samples_summary(
        predictions_dir=predictions_dir,
        output_json=Path(metrics_dir) / "skipped_samples_summary.json",
        output_md=Path(metrics_dir) / "skipped_samples_summary.md",
    )
    lines = [
        "# FlowDoc-VLM Benchmark Report",
        "",
        "This report summarizes benchmark metric JSON files only. It excludes train logs, environment reports, and readiness reports, and it does not fabricate missing real-data results.",
        "",
    ]
    _append_main_table(lines, rows)
    _append_skipped_summary(lines, rows)
    lines.extend(
        [
            "",
            "## Prediction CSV Skipped Trace",
            "",
            f"Prediction files seen: `{skipped_summary['prediction_files_seen']}`. Skipped rows traced: `{skipped_summary['num_skipped_rows']}`.",
        ]
    )
    if skipped_summary["legacy_prediction_files_ignored"]:
        lines.append("Legacy prediction files were ignored to avoid double-counting old overwritten outputs.")
    if skipped_summary["trace_warning"]:
        lines.append(skipped_summary["trace_warning"])
    _append_sroie_same_subset(lines, rows)
    _append_sroie_per_field(lines, rows)
    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `mock-hard` is the deliberately hard synthetic benchmark used for pipeline validation.",
            "- `sroie-real-100` means SROIE metrics computed on the same 100-sample subset when matching files are present.",
            "- `zero-shot` rows use the base VLM without LoRA adapter.",
            "- `LoRA` rows must be compared only against zero-shot rows with the same dataset, strategy, and evaluation set.",
            "- `skipped` rows are environment, data, or image availability states and are not model metrics.",
        ]
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a markdown benchmark report from outputs/metrics/*.json.")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--predictions-dir", default="outputs/predictions")
    parser.add_argument("--output", default="outputs/metrics/benchmark_report.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = build_benchmark_report(args.metrics_dir, args.output, args.predictions_dir)
    print(f"Saved benchmark report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
