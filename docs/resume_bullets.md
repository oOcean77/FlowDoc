# FlowDoc-VLM Resume Bullets

## Conservative Version

- Built FlowDoc-VLM, a document-image field extraction evaluation pipeline with OCR-only rules, Qwen2.5-VL zero-shot baselines, SROIE-style real-data ingestion, field-level metrics, and skipped-sample/error-case reporting.
- Integrated a SROIE train-subset benchmark with 100 documents and 399 QA samples; compared OCR-only rules against Qwen2.5-VL OCR-only and image+OCR baselines on same-subset evaluation.
- Implemented reproducible reporting for raw versus normalized accuracy, per-field accuracy, prediction CSV traceability, and skipped-row analysis; maintained CPU-safe pytest coverage with 54 passing tests.

## Engineering-Strengthened Version

- Designed a multimodal document extraction benchmark pipeline covering unified schemas, mock-hard data, SROIE real-data adapters, OCR-only baselines, Qwen2.5-VL zero-shot inference, LoRA smoke-training scaffolding, adapter evaluation, and benchmark reports.
- Added same-subset SROIE comparison and robust prediction artifact naming, preventing mock/SROIE/LoRA runs from overwriting prediction CSVs and enabling skipped-sample tracing by `sample_id`, `field_name`, and `skip_reason`.
- Analyzed real SROIE results: rule OCR 0.320, Qwen OCR-only 0.710, and Qwen image+OCR 0.761 on 92/100 evaluated rows; identified address extraction as the primary bottleneck with 0.217 normalized accuracy.

## Interview Explanation Version

FlowDoc-VLM is an evaluation-first document VLM project for enterprise field extraction, not a simple OCR wrapper. I built the data adapters, OCR and Qwen2.5-VL baselines, LoRA smoke-training scaffold, same-subset benchmark reporting, and wrong-case analysis loop. On real SROIE receipts, Qwen2.5-VL improved over rule OCR substantially: 0.320 for same-subset rule OCR, 0.710 for Qwen OCR-only, and 0.761 for Qwen image+OCR on 92 evaluated rows. The main finding is honest: LoRA smoke runs did not improve mock results, while real-data evaluation showed the value of VLM baselines and exposed address extraction as the next bottleneck.
