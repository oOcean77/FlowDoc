from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from src.utils.seed import set_seed


def _draw_doc(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (760, 520), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 740, 500), outline="black", width=2)
    draw.text((42, 44), title, fill="black")
    y = 92
    for line in lines:
        draw.text((48, y), line, fill="black")
        y += 34
    image.save(path)


def generate_mock_dataset(output_csv: str | Path = "data/processed/mock_qa.csv", sample_dir: str | Path = "data/samples") -> pd.DataFrame:
    set_seed(42)
    sample_root = Path(sample_dir)
    docs = [
        {
            "doc_id": "receipt_001",
            "doc_type": "receipt",
            "title": "Receipt",
            "fields": {"company": "Northwind Market", "date": "2026-06-05", "total_amount": "123.00"},
            "ocr": "Northwind Market\nSubtotal: 100.00\nTax: 23.00\nTotal: RMB 123.00\nDate: 2026-06-05\nReceipt No: R-001",
        },
        {
            "doc_id": "receipt_002",
            "doc_type": "receipt",
            "title": "Receipt",
            "fields": {"company": "Blue Harbor Cafe", "date": "06/05/2026", "total_amount": "88.50"},
            "ocr": "Blue Harbor Cafe\nAmount Due: $88.50\nSubtotal: $80.00\nTax: $8.50\nDate: 06/05/2026\nReceipt No: R-002",
        },
        {
            "doc_id": "invoice_001",
            "doc_type": "invoice",
            "title": "Invoice",
            "fields": {"company": "Acme Software Ltd", "invoice_id": "INV-2026-001", "total_amount": "980"},
            "ocr": "Company Name: Acme Software Ltd\nInvoice No: INV-2026-001\nSubtotal 900\nTax 80\nTotal Amount: 980",
        },
        {
            "doc_id": "invoice_002",
            "doc_type": "invoice",
            "title": "Invoice",
            "fields": {"company": "Contoso Services", "invoice_id": "INV-2026-002", "date": "2026年6月5日"},
            "ocr": "Vendor: Contoso Services\n编号: INV-2026-002\n开票日期: 2026年6月5日\nAmount Due: CNY 456.00",
        },
        {
            "doc_id": "access_001",
            "doc_type": "access_request_form",
            "title": "Access Request",
            "fields": {
                "applicant": "u_001",
                "permission_scope": "server_root",
                "reason": "troubleshoot payment failure",
                "deadline": "2026-06-05 22:00",
            },
            "ocr": "Applicant: u_001\nPermission Scope: server_root\nReason: troubleshoot payment failure\nDeadline: 2026-06-05 22:00",
        },
        {
            "doc_id": "access_002",
            "doc_type": "access_request_form",
            "title": "Access Request",
            "fields": {"applicant": "u_002", "permission_scope": "billing_read", "deadline": "2026-06-12"},
            "ocr": "申请人: u_002\n权限范围: billing_read\n有效期: 2026-06-12\nReason: audit support",
        },
    ]
    field_types = {
        "company": "company",
        "date": "date",
        "total_amount": "amount",
        "invoice_id": "id",
        "applicant": "person",
        "permission_scope": "permission",
        "deadline": "deadline",
        "reason": "other",
    }
    rows = []
    for doc in docs:
        image_path = sample_root / f"{doc['doc_id']}.png"
        _draw_doc(image_path, doc["title"], doc["ocr"].splitlines())
        for field_name, answer in doc["fields"].items():
            rows.append(
                {
                    "sample_id": f"{doc['doc_id']}_{field_name}",
                    "doc_id": doc["doc_id"],
                    "doc_type": doc["doc_type"],
                    "image_path": str(image_path).replace("\\", "/"),
                    "question": f"What is the {field_name}?",
                    "answer": answer,
                    "field_name": field_name,
                    "field_type": field_types.get(field_name, "other"),
                    "ocr_text": doc["ocr"],
                    "bbox": None,
                    "source_dataset": "mock",
                }
            )
    df = pd.DataFrame(rows)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
