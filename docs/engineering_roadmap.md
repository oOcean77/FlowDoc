# Engineering Roadmap

FlowDoc-VLM is currently a benchmark and error-analysis system. The next engineering steps should improve reliability, comparability, and operational readiness before any larger training effort.

## v0.2 Benchmark: Current State

Status: complete for the current SROIE 100-document benchmark.

- Unified field-QA schema for mock-hard and SROIE-style data.
- OCR-only rule baseline.
- Qwen2.5-VL OCR-only, image-only, and image+OCR baseline support.
- LoRA smoke-training and adapter-evaluation scaffold.
- Benchmark report filtering for real metrics only.
- Prediction CSV naming based on metrics output stems.
- Skipped-sample summary and normalized wrong-case analysis.

Current benchmark anchor:

- SROIE same-subset rule OCR: `0.320 / 100`
- SROIE Qwen OCR-only: `0.710 / 100`
- SROIE Qwen image+OCR: `0.761 / 92 evaluated / 8 skipped`

## v0.3 OOM Fallback

Goal: make image+OCR inference resilient when image size or memory pressure causes CUDA OOM.

Planned work:

- Add image resize or max-pixel preprocessing controls.
- Add automatic retry with smaller image resolution.
- Add OCR-only fallback when image inference fails.
- Record fallback path in prediction CSV and metrics JSON.
- Separate model failure, image decode failure, and CUDA OOM in skip/fallback reasons.

Success criteria:

- The SROIE image+OCR 100 run evaluates all 100 requested QA rows.
- Fallback rows are clearly marked and can be analyzed separately.

## v0.4 Address Post-processing

Goal: reduce false negatives for long, multi-line address fields without hiding real semantic errors.

Planned work:

- Add address-specific normalization for whitespace, line breaks, punctuation, and common OCR artifacts.
- Add token-level F1 and edit-distance similarity.
- Export address wrong-case clusters.
- Compare strict exact match, normalized exact match, token F1, and edit-distance metrics.

Success criteria:

- Address evaluation becomes more diagnostic than strict exact match alone.
- Reports distinguish formatting errors from wrong-address errors.

## v0.5 Unified Benchmark CLI

Goal: replace multi-command manual runs with a single benchmark entry point.

Planned CLI shape:

```bash
python scripts/run_benchmark.py --dataset sroie --strategy image_ocr --backend qwen2_5_vl --model-name /path/to/qwen --max-docs 100
```

Planned work:

- Chain data preparation, inference, evaluation, skipped tracing, wrong-case export, and report generation.
- Use deterministic output stems for every benchmark run.
- Add `--same-subset` and `--report-only` modes.
- Keep dry-run and dummy modes CPU-safe for CI.

Success criteria:

- A new user can reproduce benchmark artifacts with one command after placing data and model paths locally.

## v0.6 Full SROIE Train/Test Evaluation

Goal: move from the first 100 train documents to a fuller and more defensible benchmark.

Planned work:

- Support full SROIE train and test splits.
- Report train-subset and test-subset metrics separately.
- Add confidence intervals or bootstrap estimates for field-level accuracy.
- Compare OCR-only, Qwen OCR-only, Qwen image+OCR, and fallback-enabled image+OCR on identical samples.

Success criteria:

- Results are no longer framed as a small engineering benchmark only.
- The benchmark can support stronger model and deployment decisions.
