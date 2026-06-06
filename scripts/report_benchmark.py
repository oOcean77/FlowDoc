from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


EXCLUDED_NAME_PARTS = [
    "train_log",
    "env_report",
    "readiness_report",
    "sft_readiness",
    "lora_dryrun",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_benchmark_metrics_file(path: Path, metrics: dict[str, Any]) -> bool:
    name = path.name.lower()
    if any(part in name for part in EXCLUDED_NAME_PARTS):
        return False
    return "field_level_accuracy" in metrics or "per_field_accuracy" in metrics


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
    subset_rows = [row for row in rows if row["subset"] == "sroie-real-100"]
    lines.extend(["", "## SROIE Same-Subset 100 Comparison", ""])
    if not subset_rows:
        lines.append("No `sroie-real-100` metrics found. Expected files include `sroie_ocr_field_eval_100.json`, `sroie_qwen_ocr_only_100.json`, and `sroie_qwen_image_ocr_100.json`.")
        return
    lines.extend(["| Run | Requested | Evaluated | Skipped | Accuracy | Source |", "| --- | ---: | ---: | ---: | ---: | --- |"])
    for row in subset_rows:
        requested = row["num_samples"] or 100
        lines.append(f"| {_run_label(row)} | {requested} | {row['num_evaluated']} | {row['num_skipped']} | {_fmt(row['accuracy'])} | {row['file']} |")


def build_benchmark_report(metrics_dir: str | Path = "outputs/metrics", output_path: str | Path = "outputs/metrics/benchmark_report.md") -> Path:
    rows = collect_benchmark_rows(metrics_dir)
    lines = [
        "# FlowDoc-VLM Benchmark Report",
        "",
        "This report summarizes benchmark metric JSON files only. It excludes train logs, environment reports, and readiness reports, and it does not fabricate missing real-data results.",
        "",
    ]
    _append_main_table(lines, rows)
    _append_skipped_summary(lines, rows)
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
    parser.add_argument("--output", default="outputs/metrics/benchmark_report.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = build_benchmark_report(args.metrics_dir, args.output)
    print(f"Saved benchmark report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
