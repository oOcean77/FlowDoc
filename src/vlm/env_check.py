from __future__ import annotations

import importlib.util
import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _local_model_status(model_name: str) -> dict[str, Any]:
    looks_local = (
        model_name.startswith(".")
        or model_name.startswith("/")
        or ":\\" in model_name
        or model_name.startswith("\\\\")
    )
    if not looks_local:
        return {
            "is_local_path": False,
            "exists": None,
            "note": "Hugging Face repo id; running inference may require network access, cached files, and model permissions.",
        }
    path = Path(model_name)
    return {
        "is_local_path": True,
        "exists": path.exists(),
        "note": "Local path exists." if path.exists() else "Local model path does not exist.",
    }


def collect_vlm_env_report(model_name: str = DEFAULT_MODEL_NAME) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": {"version": sys.version.split()[0], "platform": platform.platform()},
        "model_name": model_name,
        "packages": {
            "torch": {"installed": _installed("torch"), "version": _version("torch")},
            "transformers": {"installed": _installed("transformers"), "version": _version("transformers")},
            "pillow": {"installed": _installed("PIL"), "version": _version("Pillow")},
            "qwen_vl_utils": {"installed": _installed("qwen_vl_utils"), "version": _version("qwen-vl-utils")},
        },
        "cuda": {"available": False, "device_count": 0, "devices": []},
        "model": _local_model_status(model_name),
        "can_attempt_qwen_inference": False,
        "recommendations": [],
    }
    if report["packages"]["torch"]["installed"]:
        try:
            import torch

            report["cuda"]["available"] = bool(torch.cuda.is_available())
            report["cuda"]["device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
            devices = []
            for index in range(report["cuda"]["device_count"]):
                props = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "memory_gb": round(props.total_memory / (1024**3), 2),
                    }
                )
            report["cuda"]["devices"] = devices
        except Exception as exc:
            report["cuda"]["warning"] = str(exc)
    missing = [name for name, item in report["packages"].items() if not item["installed"]]
    if missing:
        report["recommendations"].append("Install missing packages: pip install \"transformers>=4.49.0\" pillow qwen-vl-utils")
    if not report["packages"]["torch"]["installed"]:
        report["recommendations"].append("Install a torch build appropriate for your CPU or CUDA environment.")
    elif not report["cuda"]["available"]:
        report["recommendations"].append("CUDA is not available. CPU fallback may work for very small smoke tests but can be very slow.")
    if report["model"]["is_local_path"] and not report["model"]["exists"]:
        report["recommendations"].append("Use an existing local model path or a Hugging Face repo id with cache/network access.")
    if not report["model"]["is_local_path"]:
        report["recommendations"].append("The default model is a Hugging Face repo id; this check does not download it.")
    report["recommendations"].append("Start with --max-samples 3 --smoke-test before any larger evaluation.")
    report["can_attempt_qwen_inference"] = (
        all(item["installed"] for item in report["packages"].values())
        and (not report["model"]["is_local_path"] or bool(report["model"]["exists"]))
    )
    return report


def write_vlm_env_report(
    model_name: str = DEFAULT_MODEL_NAME,
    json_path: str | Path = "outputs/metrics/vlm_env_report.json",
    md_path: str | Path = "outputs/metrics/vlm_env_report.md",
) -> tuple[Path, Path, dict[str, Any]]:
    report = collect_vlm_env_report(model_name)
    json_target = Path(json_path)
    md_target = Path(md_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    md_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# VLM Environment Report",
        "",
        f"- Python: {report['python']['version']}",
        f"- Platform: {report['python']['platform']}",
        f"- Model name: {report['model_name']}",
        f"- Model path status: {report['model']['note']}",
        f"- CUDA available: {report['cuda']['available']}",
        f"- CUDA device count: {report['cuda']['device_count']}",
        "",
        "## Packages",
        "",
        "| Package | Installed | Version |",
        "| --- | --- | --- |",
    ]
    for name, item in report["packages"].items():
        lines.append(f"| {name} | {item['installed']} | {item['version'] or ''} |")
    lines.extend(["", "## CUDA Devices", ""])
    if report["cuda"]["devices"]:
        for device in report["cuda"]["devices"]:
            lines.append(f"- {device['index']}: {device['name']} ({device['memory_gb']} GB)")
    else:
        lines.append("- none detected")
    lines.extend(["", "## Recommendations", ""])
    for item in report["recommendations"]:
        lines.append(f"- {item}")
    md_target.write_text("\n".join(lines), encoding="utf-8")
    return json_target, md_target, report
