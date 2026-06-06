from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vlm.env_check import DEFAULT_MODEL_NAME, write_vlm_env_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether the current environment can attempt Qwen2.5-VL inference.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--json-output", default="outputs/metrics/vlm_env_report.json")
    parser.add_argument("--md-output", default="outputs/metrics/vlm_env_report.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    json_path, md_path, report = write_vlm_env_report(args.model_name, args.json_output, args.md_output)
    print(f"Saved VLM environment JSON report to {json_path}")
    print(f"Saved VLM environment Markdown report to {md_path}")
    print(f"can_attempt_qwen_inference={report['can_attempt_qwen_inference']}")
    if report["recommendations"]:
        print("Recommendations:")
        for item in report["recommendations"]:
            print(f"- {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
