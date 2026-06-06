from __future__ import annotations


def build_image_ocr_prompt(image_path: str, ocr_text: str, question: str) -> str:
    return (
        "You are extracting enterprise document fields.\n"
        f"Image path: {image_path}\n"
        "OCR text:\n"
        f"{ocr_text}\n\n"
        f"Question: {question}\n"
        "Return only the answer value."
    )
