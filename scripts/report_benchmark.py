from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_dataset(path: Path, metrics: dict[str, Any]) -> str:
    if metrics.get("dataset"):
        return str(metrics["dataset"])
    name = path.name.lower()
    if "sroie" in name:
        return "sroie-real"
    return "mock-hard"


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


def build_benchmark_report(metrics_dir: str | Path = "outputs/metrics", output_path: str | Path = "outputs/metrics/benchmark_report.md") -> Path:
    root = Path(metrics_dir)
    rows = []
    for path in sorted(root.glob("*.json")):
        if path.name.endswith("_train_log.json") or path.name.endswith("_report.json"):
            continue
        metrics = _load_json(path)
        rows.append(
            {
                "file": path.name,
                "dataset": _infer_dataset(path, metrics),
                "backend": metrics.get("backend", metrics.get("baseline_family", "ocr-only")),
                "strategy": metrics.get("strategy", "ocr_text"),
                "tuning": _infer_tuning(metrics),
                "status": _status(metrics),
                "num_evaluated": metrics.get("num_evaluated", metrics.get("num_samples", 0) if not metrics.get("skipped") else 0),
                "accuracy": metrics.get("field_level_accuracy"),
                "reason": metrics.get("skip_reason", ""),
            }
        )
    lines = [
        "# FlowDoc-VLM Benchmark Report",
        "",
        "This report summarizes available metric JSON files. It does not fabricate missing real-data results.",
        "",
        "| Dataset | Backend | Strategy | Tuning | Status | Num Evaluated | Accuracy | Source | Reason |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    if not rows:
        lines.append("| none | none | none | none | unavailable | 0 | null | none | no metrics found |")
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['backend']} | {row['strategy']} | {row['tuning']} | {row['status']} | "
            f"{row['num_evaluated']} | {_fmt(row['accuracy'])} | {row['file']} | {row['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `mock-hard` is the deliberately hard synthetic benchmark used for pipeline validation.",
            "- `sroie-real` is reserved for user-provided SROIE/SROIE-like receipt data.",
            "- `zero-shot` rows use the base VLM without LoRA adapter.",
            "- `LoRA` rows must be compared only against zero-shot rows with the same dataset, strategy, and evaluation set.",
            "- `skipped` rows are environment or data availability states and are not model metrics.",
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
