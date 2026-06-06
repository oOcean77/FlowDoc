from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.instruction_builder import STRATEGIES, write_instruction_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VLM instruction-answer JSONL data from FlowDoc-VLM QA CSV.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--strategy", required=True, choices=sorted(STRATEGIES))
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output = write_instruction_jsonl(args.input, args.strategy, args.output)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved instruction data to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
