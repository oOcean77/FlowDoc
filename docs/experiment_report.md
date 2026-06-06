# FlowDoc-VLM Experiment Report

This report summarizes the current experimental state honestly. FlowDoc-VLM is currently strongest as an evaluation and error-analysis pipeline, not as a model claiming SOTA document understanding.

## Mock Hard-Case Experiments

The mock hard-case set contains 39 QA samples across receipts, invoices, and access request forms. It deliberately includes multi-value conflicts, ambiguous dates, company-name conflicts, natural-language permission scopes, missing fields, and non-standard labels.

Current mock results:

- OCR-only rule baseline: `0.769`
- Qwen2.5-VL image-only: `0.692`
- Qwen2.5-VL OCR-only: `0.744`
- Qwen2.5-VL image+OCR: `0.718`
- LoRA step10: `0.718`
- LoRA step50: `0.718`

## LoRA Smoke Training Result

LoRA step10 and step50 did not improve over the zero-shot image+OCR baseline on the mock benchmark. The smoke training validated the training and adapter evaluation loop, but it did not demonstrate a model-quality gain.

## Why Stop Training For Now

The mock set is too small and synthetic. Continuing to tune on 39 rows risks overfitting and producing misleading metrics. A higher score on this mock set would not be persuasive without real data.

## Why Move To Real Data

The main project gap is lack of a real public benchmark. SROIE-style receipts provide realistic OCR noise, layout variation, vendor/address/date/total fields, and image-path handling. This is a better next step than more training on mock data.

## SROIE Benchmark

Prepare local SROIE data:

```bash
python scripts/prepare_sroie_data.py --raw-dir data/raw/sroie --output data/processed/sroie_qa.csv --max-docs 100
```

Run OCR-only:

```bash
python scripts/run_field_eval.py --input data/processed/sroie_qa.csv --output outputs/metrics/sroie_ocr_field_eval.json
```

Run Qwen2.5-VL image+OCR:

```bash
python scripts/run_vlm_baseline.py --input data/processed/sroie_qa.csv --strategy image_ocr --backend qwen2_5_vl --model-name /root/autodl-tmp/models/Qwen/Qwen2___5-VL-3B-Instruct --max-samples 50 --output outputs/metrics/sroie_qwen_image_ocr_50.json
```

Generate the benchmark summary:

```bash
python scripts/report_benchmark.py
```

## Current Value

FlowDoc-VLM now provides:

- schema adapters
- baseline execution
- skipped/unavailable handling
- prediction CSVs
- field-level metrics
- error analysis
- benchmark reporting
- training and adapter evaluation scaffolds

The value is the reproducible evaluation loop and error analysis, not a claim of model SOTA.
