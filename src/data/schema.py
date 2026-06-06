from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocType(str, Enum):
    receipt = "receipt"
    form = "form"
    invoice = "invoice"
    contract = "contract"
    access_request_form = "access_request_form"
    other = "other"


class FieldType(str, Enum):
    amount = "amount"
    date = "date"
    company = "company"
    address = "address"
    id = "id"
    person = "person"
    permission = "permission"
    deadline = "deadline"
    other = "other"


class DocumentQASample(BaseModel):
    sample_id: str
    doc_id: str
    doc_type: DocType
    image_path: str
    question: str
    answer: str
    field_name: str
    field_type: FieldType
    ocr_text: str = ""
    bbox: Any | None = None
    source_dataset: str = "mock"


REQUIRED_COLUMNS = list(DocumentQASample.model_fields.keys())


def validate_sample(row: dict[str, Any]) -> DocumentQASample:
    return DocumentQASample(**row)


def validate_records(records: list[dict[str, Any]]) -> list[DocumentQASample]:
    return [validate_sample(record) for record in records]
