from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.data.schema import REQUIRED_COLUMNS, validate_records


BASE_COLUMNS = [column for column in REQUIRED_COLUMNS if column != "image_exists"]


def _path_exists(path_value: object) -> bool:
    path_text = "" if path_value is None else str(path_value).strip()
    if not path_text:
        return False
    path = Path(path_text)
    return path.exists() or Path.cwd().joinpath(path).exists()


def _normalize_frame(df: pd.DataFrame, source_dataset: str | None = None) -> pd.DataFrame:
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            if column == "bbox":
                df[column] = None
            elif column == "image_exists":
                df[column] = df["image_path"].map(_path_exists) if "image_path" in df.columns else False
            elif column == "source_dataset" and source_dataset:
                df[column] = source_dataset
            else:
                df[column] = ""
    df = df[REQUIRED_COLUMNS].copy()
    df["bbox"] = df["bbox"].where(pd.notna(df["bbox"]), None)
    df["image_exists"] = df["image_exists"].map(lambda value: bool(value) if not isinstance(value, str) else value.lower() == "true")
    missing_image_mask = ~df["image_path"].fillna("").map(_path_exists)
    df.loc[missing_image_mask, "image_exists"] = False
    for column in REQUIRED_COLUMNS:
        if column not in {"bbox", "image_exists"}:
            df[column] = df[column].fillna("").astype(str)
    validate_records(df.to_dict(orient="records"))
    return df


def _read_csv(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Input data file not found: {target}")
    return pd.read_csv(target)


def _records_from_frame(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.where(pd.notna(df), None).to_dict(orient="records")


def _first_present(record: dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return default


def load_local_csv(path: str | Path) -> pd.DataFrame:
    df = _read_csv(path)
    missing = [column for column in BASE_COLUMNS if column not in df.columns and column != "bbox"]
    if missing:
        raise ValueError(f"Missing required columns in local CSV: {', '.join(missing)}")
    return _normalize_frame(df)


def load_mock_dataset(path: str | Path = "data/processed/mock_qa.csv") -> pd.DataFrame:
    return load_local_csv(path)


def convert_funsd_like(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for index, record in enumerate(records):
        field_name = _first_present(record, ["field_name", "label", "entity_label"], "other")
        text = _first_present(record, ["answer", "text", "entity_text"], "")
        rows.append(
            {
                "sample_id": _first_present(record, ["sample_id", "id"], f"funsd_{index:04d}"),
                "doc_id": _first_present(record, ["doc_id", "document_id", "image_id"], f"funsd_doc_{index:04d}"),
                "doc_type": _first_present(record, ["doc_type"], "form"),
                "image_path": _first_present(record, ["image_path", "image"], ""),
                "question": _first_present(record, ["question"], f"What is the {field_name}?"),
                "answer": text,
                "field_name": field_name,
                "field_type": _first_present(record, ["field_type"], "other"),
                "ocr_text": _first_present(record, ["ocr_text", "document_text", "text"], text),
                "bbox": record.get("bbox"),
                "source_dataset": "funsd",
            }
        )
    return _normalize_frame(pd.DataFrame(rows), source_dataset="funsd")


def convert_sroie_like(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    field_map = {
        "company": ("company", "company"),
        "date": ("date", "date"),
        "total": ("total_amount", "amount"),
        "total_amount": ("total_amount", "amount"),
        "address": ("address", "address"),
    }
    for index, record in enumerate(records):
        doc_id = _first_present(record, ["doc_id", "image_id"], f"sroie_doc_{index:04d}")
        image_path = _first_present(record, ["image_path", "image"], "")
        ocr_text = _first_present(record, ["ocr_text", "text"], "")
        entities = record.get("entities")
        if isinstance(entities, str) and entities:
            try:
                import json

                entities = json.loads(entities)
            except ValueError:
                entities = None
        merged_record = {**record, **entities} if isinstance(entities, dict) else record
        for raw_key, (field_name, field_type) in field_map.items():
            if raw_key not in merged_record or merged_record.get(raw_key) in (None, ""):
                continue
            rows.append(
                {
                    "sample_id": f"{doc_id}_{field_name}",
                    "doc_id": doc_id,
                    "doc_type": "receipt",
                    "image_path": image_path,
                    "question": f"What is the {field_name}?",
                    "answer": merged_record.get(raw_key, ""),
                    "field_name": field_name,
                    "field_type": field_type,
                    "ocr_text": ocr_text,
                    "bbox": record.get("bbox"),
                    "source_dataset": "sroie",
                }
            )
    return _normalize_frame(pd.DataFrame(rows), source_dataset="sroie")


def convert_docvqa_like(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for index, record in enumerate(records):
        answers = record.get("answers")
        answer = record.get("answer", answers[0] if isinstance(answers, list) and answers else "")
        rows.append(
            {
                "sample_id": _first_present(record, ["sample_id", "questionId", "question_id"], f"docvqa_{index:04d}"),
                "doc_id": _first_present(record, ["doc_id", "image_id", "image"], f"docvqa_doc_{index:04d}"),
                "doc_type": _first_present(record, ["doc_type"], "other"),
                "image_path": _first_present(record, ["image_path", "image"], ""),
                "question": _first_present(record, ["question"], ""),
                "answer": answer,
                "field_name": _first_present(record, ["field_name"], "other"),
                "field_type": _first_present(record, ["field_type"], "other"),
                "ocr_text": _first_present(record, ["ocr_text", "document_text"], ""),
                "bbox": record.get("bbox"),
                "source_dataset": "docvqa",
            }
        )
    return _normalize_frame(pd.DataFrame(rows), source_dataset="docvqa")


def convert_csv(source: str, input_path: str | Path) -> pd.DataFrame:
    records = _records_from_frame(_read_csv(input_path))
    if source == "funsd":
        return convert_funsd_like(records)
    if source == "sroie":
        return convert_sroie_like(records)
    if source == "docvqa":
        return convert_docvqa_like(records)
    if source == "local":
        return load_local_csv(input_path)
    raise ValueError(f"Unsupported source '{source}'. Expected funsd, sroie, docvqa, or local.")


def write_processed_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return target
