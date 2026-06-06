from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.schema import REQUIRED_COLUMNS, validate_records


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = None if column == "bbox" else ""
    df = df[REQUIRED_COLUMNS].copy()
    df["bbox"] = df["bbox"].where(pd.notna(df["bbox"]), None)
    for column in REQUIRED_COLUMNS:
        if column != "bbox":
            df[column] = df[column].fillna("").astype(str)
    validate_records(df.to_dict(orient="records"))
    return df


def load_local_csv(path: str | Path) -> pd.DataFrame:
    return _normalize_frame(pd.read_csv(path))


def load_mock_dataset(path: str | Path = "data/processed/mock_qa.csv") -> pd.DataFrame:
    return load_local_csv(path)


def convert_funsd_like(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for index, record in enumerate(records):
        # TODO: FUNSD relation/entity formats vary by preprocessing pipeline.
        rows.append(
            {
                "sample_id": record.get("sample_id", f"funsd_{index:04d}"),
                "doc_id": record.get("doc_id", f"funsd_doc_{index:04d}"),
                "doc_type": record.get("doc_type", "form"),
                "image_path": record.get("image_path", ""),
                "question": record.get("question", f"What is {record.get('field_name', 'the field')}?"),
                "answer": record.get("answer", record.get("text", "")),
                "field_name": record.get("field_name", "other"),
                "field_type": record.get("field_type", "other"),
                "ocr_text": record.get("ocr_text", record.get("text", "")),
                "bbox": record.get("bbox"),
                "source_dataset": "funsd",
            }
        )
    return _normalize_frame(pd.DataFrame(rows))


def convert_sroie_like(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    field_map = {"total": "total_amount", "date": "date", "company": "company", "address": "address"}
    for index, record in enumerate(records):
        # TODO: SROIE labels may arrive as nested JSON or separate OCR transcripts.
        for key, field_name in field_map.items():
            if key in record:
                rows.append(
                    {
                        "sample_id": f"sroie_{index:04d}_{field_name}",
                        "doc_id": record.get("doc_id", f"sroie_doc_{index:04d}"),
                        "doc_type": "receipt",
                        "image_path": record.get("image_path", ""),
                        "question": f"What is the {field_name}?",
                        "answer": record.get(key, ""),
                        "field_name": field_name,
                        "field_type": "amount" if key == "total" else ("date" if key == "date" else "company"),
                        "ocr_text": record.get("ocr_text", ""),
                        "bbox": None,
                        "source_dataset": "sroie",
                    }
                )
    return _normalize_frame(pd.DataFrame(rows))


def convert_docvqa_like(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for index, record in enumerate(records):
        # TODO: DocVQA answer lists and image ids differ across releases.
        answer = record.get("answer", record.get("answers", [""])[0] if record.get("answers") else "")
        rows.append(
            {
                "sample_id": record.get("questionId", record.get("sample_id", f"docvqa_{index:04d}")),
                "doc_id": record.get("doc_id", record.get("image", f"docvqa_doc_{index:04d}")),
                "doc_type": record.get("doc_type", "other"),
                "image_path": record.get("image_path", record.get("image", "")),
                "question": record.get("question", ""),
                "answer": answer,
                "field_name": record.get("field_name", "other"),
                "field_type": record.get("field_type", "other"),
                "ocr_text": record.get("ocr_text", ""),
                "bbox": record.get("bbox"),
                "source_dataset": "docvqa",
            }
        )
    return _normalize_frame(pd.DataFrame(rows))
