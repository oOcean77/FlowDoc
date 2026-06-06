from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


METRICS_DIR = Path("outputs/metrics")
OUTPUT_PATH = METRICS_DIR / "baseline_comparison.md"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_metric(value: object) -> str:
    return "unavailable" if value is None else f"{float(value):.3f}"


def build_report(metrics_dir: Path = METRICS_DIR, output_path: Path = OUTPUT_PATH) -> Path:
    lines = [
        "# Baseline Comparison",
        "",
        "This report compares available FlowDoc-VLM baselines. Skipped VLM runs are reported as skipped and do not receive fabricated metrics.",
        "",
        "## OCR-only",
        "",
    ]
    ocr_path = metrics_dir / "field_eval_results.json"
    if ocr_path.exists():
        ocr = _load_json(ocr_path)
        lines.extend(
            [
                f"- num_samples: {ocr.get('num_samples')}",
                f"- field_level_accuracy: {_format_metric(ocr.get('field_level_accuracy'))}",
                f"- missing_field_rate: {_format_metric(ocr.get('missing_field_rate'))}",
                f"- multi_value_conflict_rate: {_format_metric(ocr.get('multi_value_conflict_rate'))}",
                "",
            ]
        )
    else:
        lines.extend(["- OCR metrics unavailable. Run `python scripts/run_field_eval.py` first.", ""])

    lines.extend(["## VLM Baselines", "", "| Backend | Strategy | Status | Num Evaluated | Accuracy | Reason |", "| --- | --- | --- | ---: | ---: | --- |"])
    vlm_paths = sorted(metrics_dir.glob("vlm_baseline_*.json"))
    if not vlm_paths:
        lines.append("| none | none | unavailable | 0 | unavailable | no VLM baseline metrics found |")
    for path in vlm_paths:
        metrics = _load_json(path)
        status = "skipped" if metrics.get("skipped") else "available"
        reason = metrics.get("skip_reason") or ""
        lines.append(
            f"| {metrics.get('backend')} | {metrics.get('strategy')} | {status} | "
            f"{metrics.get('num_evaluated', 0)} | {_format_metric(metrics.get('field_level_accuracy'))} | {reason} |"
        )
    lines.extend(
        [
            "",
            "## Strategy Availability",
            "",
            "- OCR-only is available through the rule baseline and uses OCR text only.",
            "- Image-only is available only when a real VLM backend and image files are available; otherwise it remains skipped or placeholder.",
            "- Image+OCR is available only when a real VLM backend is available; dummy runs intentionally skip metrics.",
            "",
            "## Recommendations",
            "",
            "- If VLM inference is available, next run a small real-data evaluation with `--max-samples` before any training.",
            "- If VLM inference is unavailable, prepare the local Qwen2.5-VL or LLaVA environment first.",
            "- Keep LoRA SFT for week 3 or later; this week is inference scaffolding and data preparation only.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    output = build_report()
    print(f"Saved baseline comparison report to {output}")
