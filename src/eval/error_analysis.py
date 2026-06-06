from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eval.run_eval import run_ocr_eval


ERROR_COLUMNS = ["sample_id", "doc_type", "field_name", "question", "gold_answer", "pred_answer", "ocr_text", "error_type"]
REPORT_PATH = "outputs/error_cases/analysis_report.md"


def _markdown_counts(title: str, counts: pd.Series) -> list[str]:
    lines = [f"## {title}", "", "| Name | Error Count |", "| --- | ---: |"]
    if counts.empty:
        lines.append("| none | 0 |")
    else:
        for name, count in counts.items():
            lines.append(f"| {name} | {int(count)} |")
    lines.append("")
    return lines


def write_analysis_report(error_df: pd.DataFrame, output_path: str = REPORT_PATH, total_samples: int | None = None) -> Path:
    field_counts = error_df["field_name"].value_counts() if not error_df.empty else pd.Series(dtype="int64")
    type_counts = error_df["error_type"].value_counts() if not error_df.empty else pd.Series(dtype="int64")
    total_errors = int(len(error_df))
    denominator = total_samples if total_samples is not None else total_errors
    error_rate = total_errors / denominator if denominator else 0.0

    lines = [
        "# OCR-only Baseline Error Analysis",
        "",
        "This report is generated from the mock evaluation set. The baseline uses OCR text and hand-written rules only; it does not call a VLM or train any large model.",
        "",
        "## Summary",
        "",
        f"- Total evaluated QA samples: {denominator}",
        f"- Exported error cases: {total_errors}",
        f"- Error rate on this mock set: {error_rate:.3f}",
        "",
    ]
    lines.extend(_markdown_counts("Errors by Field", field_counts))
    lines.extend(_markdown_counts("Errors by Error Type", type_counts))
    lines.extend(
        [
            "## Main OCR-only Limitations",
            "",
            "- Multiple values with similar surface forms are hard to disambiguate, especially subtotal/tax/total and invoice date/due date.",
            "- Company extraction depends on labels and line order, so vendor, merchant, client, and company contexts can be confused.",
            "- Natural-language permission scopes are not stable key-value fields, so rule matching often returns a verbose phrase instead of the canonical scope.",
            "- OCR text order can be different from the visual layout, which makes first-match and last-match rules brittle.",
            "- Missing or non-standard labels expose unsupported fields and weak fallbacks rather than true document understanding.",
            "",
            "## Next Improvement Direction",
            "",
            "- Image+OCR prompting should use both visual layout and OCR text to resolve field ownership when multiple candidates are present.",
            "- A VLM-SFT stage can learn canonical field answers from instruction-style samples, including natural-language permission descriptions.",
            "- Evaluation should compare OCR-only, image-only placeholder, image+OCR, and later VLM-SFT without fabricating VLM metrics before the model exists.",
            "",
        ]
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def export_error_cases(
    dataset_path: str = "data/processed/mock_qa.csv",
    output_path: str = "outputs/error_cases/error_cases.csv",
    report_path: str = REPORT_PATH,
) -> pd.DataFrame:
    results = run_ocr_eval(dataset_path)
    rows = []
    for row in results["rows"]:
        if not row["correct"]:
            error_type = row.get("error_type") or "normalization_error"
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "doc_type": row["doc_type"],
                    "field_name": row["field_name"],
                    "question": row["question"],
                    "gold_answer": row["answer"],
                    "pred_answer": row.get("pred_answer", ""),
                    "ocr_text": row.get("ocr_text", ""),
                    "error_type": error_type,
                }
            )
    df = pd.DataFrame(rows, columns=ERROR_COLUMNS)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    write_analysis_report(df, report_path, total_samples=results["num_samples"])
    return df
