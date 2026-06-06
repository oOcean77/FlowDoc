from __future__ import annotations

import json
from pathlib import Path

from src.vlm.env_check import collect_vlm_env_report, write_vlm_env_report


def test_vlm_env_report_outputs_json_and_markdown() -> None:
    root = Path("outputs/test_artifacts/vlm_env")
    json_path = root / "vlm_env_report.json"
    md_path = root / "vlm_env_report.md"

    json_out, md_out, report = write_vlm_env_report("Qwen/Qwen2.5-VL-3B-Instruct", json_path, md_path)

    assert json_out.exists()
    assert md_out.exists()
    assert json.loads(json_out.read_text(encoding="utf-8"))["model_name"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert "VLM Environment Report" in md_out.read_text(encoding="utf-8")
    assert "packages" in report


def test_local_missing_model_path_is_reported() -> None:
    report = collect_vlm_env_report("C:\\models\\missing-qwen")
    assert report["model"]["is_local_path"] is True
    assert report["model"]["exists"] is False
