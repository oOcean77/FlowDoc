# FlowDoc-VLM Final Project Summary

## Positioning

FlowDoc-VLM is an enterprise document-image field extraction and multimodal evaluation system. It is not a plain OCR demo, and it does not claim a new VLM architecture. The project focuses on a reproducible evaluation pipeline for document images, OCR text, field-level metrics, and error analysis.

## Technical Path

- Unified schema for document QA and field extraction records.
- Mock-hard data with deliberately difficult OCR cases.
- SROIE real-data adapter for receipt images, OCR text, and entity labels.
- OCR-only rule baseline.
- Qwen2.5-VL zero-shot baselines for OCR-only, image-only, and image+OCR input strategies.
- LoRA smoke training and adapter evaluation scaffolding.
- Benchmark report generation with benchmark-only JSON filtering.
- Skipped-sample tracing from prediction CSV files.
- Normalized wrong-case analysis for raw accuracy, normalized accuracy, and per-field errors.

## Experiment Results

| Dataset | Method | Strategy | Accuracy | Samples / Notes |
| --- | --- | --- | ---: | --- |
| mock-hard | OCR-only rule | OCR text | 0.769 | 39 samples |
| mock-hard | Qwen2.5-VL zero-shot | image-only | 0.692 | 39 samples |
| mock-hard | Qwen2.5-VL zero-shot | OCR-only | 0.744 | 39 samples |
| mock-hard | Qwen2.5-VL zero-shot | image+OCR | 0.718 | 39 samples |
| mock-hard | Qwen2.5-VL LoRA step10 | image+OCR | 0.718 | smoke training only |
| mock-hard | Qwen2.5-VL LoRA step50 | image+OCR | 0.718 | smoke training only |
| SROIE-real | OCR-only rule | OCR text full | 0.298 | 399 QA samples |
| SROIE-real | OCR-only rule | OCR text same-subset | 0.320 | 100 QA samples |
| SROIE-real | Qwen2.5-VL zero-shot | OCR-only | 0.710 | 100 evaluated |
| SROIE-real | Qwen2.5-VL zero-shot | image+OCR | 0.761 | 92 evaluated / 8 skipped |
| SROIE-real | Qwen2.5-VL zero-shot | image+OCR | 0.800 | 50 evaluated |

SROIE image+OCR 100 skipped 8 QA rows because of CUDA OOM on two documents. The score must be read as 92 evaluated out of 100 requested, not a full 100-evaluated run.

## Conclusions

- On mock-hard, VLM and LoRA are not naturally better than the OCR-only rule baseline.
- On SROIE-real, Qwen2.5-VL is substantially stronger than the rule OCR baseline.
- On the SROIE same-subset 100 benchmark, rule OCR is 0.320, Qwen OCR-only is 0.710, and Qwen image+OCR is 0.761 on 92 evaluated rows.
- Image+OCR gives a real-data gain over OCR-only, but the current comparison still needs skipped-row cleanup for a stricter full-100 comparison.
- Address is the main bottleneck: SROIE image+OCR normalized address accuracy is 0.217.
- LoRA step10 and step50 validate the training and adapter-evaluation loop only. They do not demonstrate model improvement and should not be marketed as such.

## Engineering Depth

- Multi-strategy evaluation: OCR-only, image-only, and image+OCR.
- Same-subset comparison to avoid unfair full-vs-subset conclusions.
- Skipped-sample tracing with `sample_id`, `field_name`, and `skip_reason`.
- Raw accuracy versus normalized accuracy for realistic error analysis.
- Prediction CSV names derived from metrics output stems to prevent overwritten runs.
- Benchmark reports automatically filter train logs, environment reports, readiness reports, and other non-benchmark JSON files.
- Real-data adapter for SROIE-style images, OCR text, and entity annotations.
- Reproducible scripts and CPU-safe pytest coverage.

## Current Limitations

- The real-data benchmark currently uses the first 100 documents from the SROIE train subset.
- Qwen image+OCR 100 has 8 skipped QA rows due to CUDA OOM.
- Address evaluation needs a better metric, such as token-level F1 or edit distance.
- There is no formal LoRA SFT result.
- There is no large-scale real-data training set yet.

## Next Steps

- Expand to the full SROIE train/test split.
- Add token-level F1 or edit distance for address fields.
- Run prompt ablations: raw OCR, cleaned OCR, field-aware OCR, and JSON-only output.
- Reduce OOM through image resizing, max-pixel limits, and strict sequential inference.
- Add CORD and FUNSD for broader real-data coverage.
