from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_adapter import convert_sroie_like, write_processed_csv
from src.data.mock_docs import generate_mock_dataset


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
FIELD_ALIASES = {
    "company": "company",
    "date": "date",
    "address": "address",
    "total": "total",
    "total_amount": "total_amount",
}


def _find_image(images_dir: Path, doc_id: str) -> str:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{doc_id}{extension}"
        if candidate.exists():
            return str(candidate)
    return ""


def _load_entity_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "company" in data:
        return data
    if isinstance(data, dict) and "fields" in data and isinstance(data["fields"], dict):
        return data["fields"]
    return data if isinstance(data, dict) else {}


def _normalize_entity_fields(entities: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in entities.items():
        target = FIELD_ALIASES.get(str(key).lower())
        if target and value not in (None, ""):
            normalized[target] = value
    return normalized


def records_from_sroie_raw(raw_dir: str | Path, max_docs: int | None = None) -> list[dict[str, Any]]:
    root = Path(raw_dir)
    images_dir = root / "images"
    ocr_dir = root / "ocr"
    entities_dir = root / "entities"
    if not root.exists():
        raise FileNotFoundError(f"SROIE raw directory not found: {root}")
    if not entities_dir.exists():
        raise FileNotFoundError(f"SROIE entities directory not found: {entities_dir}")
    entity_files = sorted(entities_dir.glob("*.json"))
    if not entity_files:
        raise FileNotFoundError(f"No SROIE entity JSON files found under {entities_dir}")
    records = []
    for entity_path in entity_files[:max_docs]:
        doc_id = entity_path.stem
        fields = _normalize_entity_fields(_load_entity_json(entity_path))
        if not fields:
            continue
        ocr_path = ocr_dir / f"{doc_id}.txt"
        record = {
            "doc_id": doc_id,
            "image_path": _find_image(images_dir, doc_id),
            "ocr_text": ocr_path.read_text(encoding="utf-8") if ocr_path.exists() else "",
            **fields,
        }
        records.append(record)
    if not records:
        raise ValueError(f"No usable SROIE records found under {root}")
    return records


def prepare_sroie_dataframe(
    raw_dir: str | Path = "data/raw/sroie",
    input_csv: str | Path | None = None,
    max_docs: int | None = 100,
    use_mock_fallback: bool = False,
) -> pd.DataFrame:
    if input_csv:
        path = Path(input_csv)
        if not path.exists():
            raise FileNotFoundError(f"SROIE-like CSV not found: {path}")
        frame = pd.read_csv(path)
        records = frame.where(pd.notna(frame), None).to_dict(orient="records")
        if max_docs is not None:
            records = records[:max_docs]
        return convert_sroie_like(records)
    try:
        records = records_from_sroie_raw(raw_dir, max_docs=max_docs)
        return convert_sroie_like(records)
    except (FileNotFoundError, ValueError):
        if use_mock_fallback:
            return generate_mock_dataset(output_csv=Path("outputs/test_artifacts/sroie_mock_fallback.csv"))
        raise


def sroie_stats(df: pd.DataFrame) -> dict[str, Any]:
    per_field = Counter(df["field_name"].tolist()) if not df.empty else Counter()
    return {
        "num_docs": int(df["doc_id"].nunique()) if "doc_id" in df.columns else 0,
        "num_qa_samples": int(len(df)),
        "per_field_count": dict(sorted(per_field.items())),
        "missing_image_count": int((~df["image_exists"]).sum()) if "image_exists" in df.columns else 0,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SROIE or SROIE-like receipt data into FlowDoc-VLM QA CSV.")
    parser.add_argument("--raw-dir", default="data/raw/sroie")
    parser.add_argument("--input-csv")
    parser.add_argument("--output", default="data/processed/sroie_qa.csv")
    parser.add_argument("--max-docs", type=int, default=100)
    parser.add_argument("--use-mock-fallback", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        df = prepare_sroie_dataframe(args.raw_dir, args.input_csv, args.max_docs, args.use_mock_fallback)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Provide data/raw/sroie/{images,ocr,entities} or --input-csv data/raw/sroie_like.csv. Mock fallback requires --use-mock-fallback.", file=sys.stderr)
        return 1
    output = write_processed_csv(df, args.output)
    stats = sroie_stats(df)
    print(f"Saved SROIE QA data to {output}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
