from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_adapter import convert_csv, write_processed_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert real-data-like CSV files into the unified FlowDoc-VLM QA schema.")
    parser.add_argument("--source", required=True, choices=["funsd", "sroie", "docvqa", "local"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        df = convert_csv(args.source, args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    output = write_processed_csv(df, args.output)
    missing_images = int((~df["image_exists"]).sum()) if "image_exists" in df.columns else 0
    print(f"Saved {len(df)} {args.source} QA samples to {output}")
    if missing_images:
        print(f"Warning: {missing_images} rows reference missing image files; image_exists=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
