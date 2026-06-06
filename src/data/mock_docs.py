from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from src.utils.seed import set_seed


def _draw_doc(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (760, 620), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 740, 600), outline="black", width=2)
    draw.text((42, 44), title, fill="black")
    y = 92
    for line in lines:
        draw.text((48, y), line[:96], fill="black")
        y += 30
    image.save(path)


def _doc(doc_id: str, doc_type: str, title: str, fields: dict[str, str], ocr: str, hard_case: str = "") -> dict:
    return {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "title": title,
        "fields": fields,
        "ocr": ocr,
        "hard_case": hard_case,
    }


def _mock_docs() -> list[dict]:
    return [
        _doc(
            "receipt_001",
            "receipt",
            "Receipt",
            {"company": "Northwind Market", "date": "2026-06-05", "total_amount": "123.00"},
            "Northwind Market\nSubtotal: 100.00\nTax: 23.00\nTotal: RMB 123.00\nDate: 2026-06-05\nReceipt No: R-001",
            "subtotal_tax_total",
        ),
        _doc(
            "receipt_002",
            "receipt",
            "Receipt",
            {"company": "Blue Harbor Cafe", "date": "06/05/2026", "total_amount": "88.50"},
            "Blue Harbor Cafe\nAmount Due: $88.50\nSubtotal: $80.00\nTax: $8.50\nDate: 06/05/2026\nReceipt No: R-002",
            "subtotal_tax_total",
        ),
        _doc(
            "receipt_003",
            "receipt",
            "Receipt",
            {"company": "Green Finch Books", "date": "2026-06-03", "total_amount": "47.20"},
            "Tax 3.20\nGreen Finch Books\nPaid by card\nDate 2026-06-03\nSubtotal 44.00\nGrand Total 47.20",
            "ocr_order_jumbled",
        ),
        _doc(
            "receipt_004",
            "receipt",
            "Receipt",
            {"company": "Pine Street Deli", "date": "2026-06-04", "total_amount": "31.90"},
            "Merchant: Pine Street Deli\nCashier: Sam\nItems total 29.00\nVAT 2.90\nBalance paid 31.90\nTxn day unavailable",
            "missing_date_label",
        ),
        _doc(
            "invoice_001",
            "invoice",
            "Invoice",
            {"company": "Acme Software Ltd", "invoice_id": "INV-2026-001", "total_amount": "980"},
            "Company Name: Acme Software Ltd\nInvoice No: INV-2026-001\nSubtotal 900\nTax 80\nTotal Amount: 908",
            "ocr_digit_error_with_multi_amounts",
        ),
        _doc(
            "invoice_002",
            "invoice",
            "Invoice",
            {"company": "Contoso Services", "invoice_id": "INV-2026-002", "date": "2026-06-08"},
            "Vendor: Contoso Services\nInvoice No: INV-2026-002\nInvoice Date: 2026-06-08\nDue Date: 2026-07-08\nAmount Due: CNY 456.00",
            "invoice_date_due_date",
        ),
        _doc(
            "invoice_003",
            "invoice",
            "Invoice",
            {"company": "Globex Trading", "invoice_id": "GBX-7781", "date": "2026-05-30", "total_amount": "1200.00"},
            "Bill To: Globex Trading AP\nVendor: Eastline Parts Co\nInvoice ID: GBX-7781\nDue Date: 2026-06-29\nInvoice Date: 2026-05-30\nTotal Due USD 1,200.00",
            "vendor_company_conflict",
        ),
        _doc(
            "invoice_004",
            "invoice",
            "Invoice",
            {"company": "Helio Labs", "invoice_id": "H-402", "total_amount": "642.18"},
            "Remit To: Helio Labz\nClient: Helio Labs\nDoc number H-402\nSubtotal 600.00\nSales tax 42.18\nPlease pay 642.18",
            "ocr_typo_and_non_standard_labels",
        ),
        _doc(
            "access_001",
            "access_request_form",
            "Access Request",
            {
                "applicant": "u_001",
                "permission_scope": "server_root",
                "reason": "troubleshoot payment failure",
                "deadline": "2026-06-05 22:00",
            },
            "Applicant: u_001\nPermission Scope: server_root\nReason: troubleshoot payment failure\nDeadline: 2026-06-05 22:00",
        ),
        _doc(
            "access_002",
            "access_request_form",
            "Access Request",
            {"applicant": "u_002", "permission_scope": "billing_read", "deadline": "2026-06-12"},
            "Requester: u_002\nAccess needed: read-only access to billing exports for June audit\nValid until 2026-06-12\nReason: audit support",
            "natural_language_permission",
        ),
        _doc(
            "access_003",
            "access_request_form",
            "Access Request",
            {
                "applicant": "u_003",
                "permission_scope": "prod_db_read",
                "reason": "investigate delayed orders",
                "deadline": "2026-06-09",
            },
            "Reason: investigate delayed orders\nEnds: 2026-06-09\nScope requested in words: allow read queries on production order database only\nUser u_003",
            "missing_standard_labels",
        ),
        _doc(
            "access_004",
            "access_request_form",
            "Access Request",
            {"applicant": "u_004", "permission_scope": "crm_export", "security_group": "crm_exporters"},
            "Applicant: u_004\nManager approval: pending\nScope: export customer CRM records for migration window\nTicket: AR-944",
            "unsupported_field",
        ),
    ]


def generate_mock_dataset(
    output_csv: str | Path = "data/processed/mock_qa.csv",
    sample_dir: str | Path = "data/samples",
) -> pd.DataFrame:
    set_seed(42)
    sample_root = Path(sample_dir)
    docs = _mock_docs()
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
