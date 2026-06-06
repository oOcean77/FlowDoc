from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Instruction JSONL not found: {target}")
    rows = []
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
    return rows


def _path_exists(path_value: object) -> bool:
    path_text = "" if path_value is None else str(path_value).strip()
    if not path_text:
        return False
    path = Path(path_text)
    return path.exists() or Path.cwd().joinpath(path).exists()


def _is_local_model_path(model_name: str) -> bool:
    return model_name.startswith(".") or model_name.startswith("/") or ":\\" in model_name or model_name.startswith("\\\\")


def _installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _cuda_available() -> bool:
    if not _installed("torch"):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def check_sft_readiness(
    instruction_file: str | Path,
    model_name: str,
    baseline_metrics: str | Path,
    train_file: str | Path = "data/processed/train_instructions.jsonl",
    eval_file: str | Path = "data/processed/eval_instructions.jsonl",
    min_samples: int = 100,
) -> dict[str, Any]:
    blocking: list[str] = []
    warnings: list[str] = []
    next_steps: list[str] = []
    rows: list[dict[str, Any]] = []

    try:
        rows = _load_jsonl(instruction_file)
    except (FileNotFoundError, ValueError) as exc:
        blocking.append(str(exc))

    if rows:
        if len(rows) < min_samples:
            blocking.append(f"Instruction sample count {len(rows)} is below recommended minimum {min_samples}.")
        missing_image_path = [row.get("sample_id", "") for row in rows if not row.get("image_path")]
        if missing_image_path:
            blocking.append(f"{len(missing_image_path)} samples are missing image_path.")
        missing_images = [row.get("sample_id", "") for row in rows if row.get("strategy") in {"image_ocr", "image_only"} and not _path_exists(row.get("image_path"))]
        if missing_images:
            blocking.append(f"{len(missing_images)} image-based samples reference missing image files.")
        empty_answers = [row.get("sample_id", "") for row in rows if not str(row.get("answer", "")).strip()]
        if empty_answers:
            blocking.append(f"{len(empty_answers)} samples have empty answers.")
        unsupported_strategy = sorted({str(row.get("strategy", "")) for row in rows if row.get("strategy") not in {"image_ocr", "image_only"}})
        if unsupported_strategy:
            blocking.append(f"SFT expects image_ocr or image_only samples; found strategies: {', '.join(unsupported_strategy)}.")

    if not Path(train_file).exists() or not Path(eval_file).exists():
        blocking.append("Train/eval split files are missing. Run scripts/split_instruction_data.py first.")

    metrics = _load_json(baseline_metrics)
    if metrics is None:
        blocking.append(f"Baseline metrics file not found: {baseline_metrics}")
    elif metrics.get("skipped"):
        blocking.append(f"Qwen baseline metrics are skipped: {metrics.get('skip_reason')}")
    elif metrics.get("num_evaluated", 0) <= 0:
        blocking.append("Baseline metrics contain no evaluated samples.")

    if _is_local_model_path(model_name) and not Path(model_name).exists():
        blocking.append(f"Local model path does not exist: {model_name}")
    elif not _is_local_model_path(model_name):
        warnings.append("Model name is a Hugging Face repo id; training requires network/cache/model permission readiness.")

    if not _cuda_available():
        blocking.append("CUDA is not available.")
    for module in ["peft", "accelerate", "bitsandbytes"]:
        if not _installed(module):
            blocking.append(f"Required training package is not installed: {module}")

    if blocking:
        next_steps.extend(
            [
                "Run Qwen2.5-VL smoke inference successfully before training.",
                "Create train/eval instruction splits with scripts/split_instruction_data.py.",
                "Use a valid local model path outside the Git repository on AutoDL.",
                "Install peft, accelerate, bitsandbytes, and a CUDA-enabled torch build.",
            ]
        )
    else:
        next_steps.append("Ready for a small controlled LoRA SFT dry run in the next phase.")

    return {
        "ready": not blocking,
        "instruction_file": str(instruction_file),
        "num_instruction_samples": len(rows),
        "train_file": str(train_file),
        "eval_file": str(eval_file),
        "baseline_metrics": str(baseline_metrics),
        "model_name": model_name,
        "blocking_issues": blocking,
        "warnings": warnings,
        "recommended_next_steps": next_steps,
    }


def write_readiness_report(
    report: dict[str, Any],
    json_output: str | Path = "outputs/metrics/sft_readiness_report.json",
    md_output: str | Path = "outputs/metrics/sft_readiness_report.md",
) -> tuple[Path, Path]:
    json_target = Path(json_output)
    md_target = Path(md_output)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    md_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# SFT Readiness Report",
        "",
        f"- ready: {str(report['ready']).lower()}",
        f"- instruction samples: {report['num_instruction_samples']}",
        f"- instruction file: {report['instruction_file']}",
        f"- model_name: {report['model_name']}",
        "",
        "## Blocking Issues",
        "",
    ]
    if report["blocking_issues"]:
        lines.extend(f"- {item}" for item in report["blocking_issues"])
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend(f"- {item}" for item in report["warnings"])
    else:
        lines.append("- none")
    lines.extend(["", "## Recommended Next Steps", ""])
    lines.extend(f"- {item}" for item in report["recommended_next_steps"])
    md_target.write_text("\n".join(lines), encoding="utf-8")
    return json_target, md_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether FlowDoc-VLM instruction data and environment are ready for LoRA SFT.")
    parser.add_argument("--instruction-file", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--train-file", default="data/processed/train_instructions.jsonl")
    parser.add_argument("--eval-file", default="data/processed/eval_instructions.jsonl")
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--json-output", default="outputs/metrics/sft_readiness_report.json")
    parser.add_argument("--md-output", default="outputs/metrics/sft_readiness_report.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = check_sft_readiness(
        instruction_file=args.instruction_file,
        model_name=args.model_name,
        baseline_metrics=args.baseline_metrics,
        train_file=args.train_file,
        eval_file=args.eval_file,
        min_samples=args.min_samples,
    )
    json_path, md_path = write_readiness_report(report, args.json_output, args.md_output)
    print(f"Saved SFT readiness JSON report to {json_path}")
    print(f"Saved SFT readiness Markdown report to {md_path}")
    print(f"ready={str(report['ready']).lower()}")
    if report["blocking_issues"]:
        print("Blocking issues:")
        for issue in report["blocking_issues"]:
            print(f"- {issue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
