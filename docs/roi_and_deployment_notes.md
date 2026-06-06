# ROI and Deployment Notes

FlowDoc-VLM is not production-ready yet, but the current benchmark already suggests a practical deployment direction: use cheap deterministic extraction where it works, and reserve VLM inference for hard or low-confidence fields.

## Cost and Accuracy Tradeoff

Rule OCR is low cost and easy to run, but it underperforms on real SROIE receipts:

- SROIE rule OCR same-subset 100: `0.320`
- SROIE full rule OCR: `0.298 / 399`

Qwen2.5-VL is more accurate on real receipts, but it requires GPU resources and robust inference handling:

- Qwen OCR-only 100: `0.710`
- Qwen image+OCR 100: `0.761 / 92 evaluated / 8 skipped`

The 8 skipped rows show that GPU memory and image handling need engineering fallback before production use.

## Suggested Industrial Strategy

Use a tiered extraction policy:

- Easy fields use rule OCR first: compact values such as obvious dates, totals, or IDs can often be extracted cheaply.
- Hard fields use Qwen: long addresses, ambiguous company names, noisy layouts, and low-confidence OCR outputs should route to VLM inference.
- Low-confidence outputs trigger escalation: when rule OCR returns empty, conflicting, or weakly matched candidates, route to Qwen OCR-only or image+OCR.
- OOM or GPU unavailable falls back to OCR-only: fallback should be explicit in the output payload and benchmark report.
- Human review remains available for high-value or compliance-sensitive documents.

## Deployment Shape

A production-like service could expose:

- input: image, OCR text, target fields, optional confidence thresholds
- dispatcher: rule OCR first, Qwen fallback for hard cases
- post-processing: field-aware normalization and JSON validation
- observability: model path, strategy, skipped/fallback reason, latency, and per-field confidence
- reporting: rolling accuracy and wrong-case sampling when ground truth is available

## Current Non-goals

- Do not claim production-grade reliability from the current 100-document SROIE benchmark.
- Do not claim LoRA accuracy gains; smoke training did not improve mock-hard results.
- Do not hide skipped rows; OOM fallback must be implemented before image+OCR can be treated as a complete 100-row run.

## Practical ROI Hypothesis

The likely ROI is not replacing OCR everywhere. It is reducing manual review for hard fields while keeping deterministic extraction for cheap and easy fields. The next validation step is a larger SROIE train/test benchmark with OOM fallback and address-aware evaluation.
