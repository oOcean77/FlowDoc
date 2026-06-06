# SROIE Benchmark Report

This report summarizes the current real SROIE-style receipt benchmark results provided from AutoDL. The prepared SROIE train subset has 100 receipt documents and 399 QA samples across `company`, `date`, `address`, and `total_amount`. No training is run in this reporting step.

## Setup

- GPU: RTX 4090 24 GB
- torch: `2.3.0+cu121`
- transformers: `4.51.3`
- model path: `/root/autodl-tmp/models/Qwen/Qwen2___5-VL-3B-Instruct`

## Results

| Dataset | Method | Strategy | Samples | Evaluated | Skipped | Field Accuracy |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| SROIE real | OCR-only rule | OCR text | 399 | 399 | 0 | 0.298 |
| SROIE real | OCR-only rule | OCR text same-subset | 100 | 100 | 0 | 0.320 |
| SROIE real | Qwen2.5-VL zero-shot | OCR-only | 100 | 100 | 0 | 0.710 |
| SROIE real | Qwen2.5-VL zero-shot | image+OCR | 100 | 92 | 8 | 0.761 |
| SROIE real | Qwen2.5-VL zero-shot | image+OCR | 50 | 50 | 0 | 0.800 |

The 399-sample OCR-only full run and the 100-sample Qwen runs are not fully fair horizontal comparisons. The same-subset 100 results are the better comparison point:

| Dataset | Method | Strategy | Requested | Evaluated | Skipped | Accuracy | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| SROIE real | OCR-only rule | OCR text | 100 | 100 | 0 | 0.320 | same-subset rule baseline |
| SROIE real | Qwen2.5-VL zero-shot | OCR-only | 100 | 100 | 0 | 0.710 | same-subset VLM text baseline |
| SROIE real | Qwen2.5-VL zero-shot | image+OCR | 100 | 92 | 8 | 0.761 | 100 requested; 8 QA rows skipped due to CUDA OOM on two documents |

## Per-Field Observation

Address is the weakest field:

- Qwen OCR-only address accuracy: `0.160`
- Qwen image+OCR address accuracy: `0.217`
- Qwen image+OCR wrong-case counts: address `18`, total_amount `2`, company `2`

This suggests the model benefits from visual context overall, but address extraction remains hard because receipts often have long, multi-line address text and OCR/layout ambiguity.

Address matching is intentionally strict in the current field-level metric. It is sensitive to line breaks, punctuation, and token order, so future reporting should add token-level F1 or a field-specific address normalization metric.

## Mock Versus Real Data

The mock hard-case benchmark showed that LoRA smoke training did not improve over the zero-shot image+OCR result:

- Qwen image+OCR zero-shot on mock: `0.718`
- LoRA step10 on mock: `0.718`
- LoRA step50 on mock: `0.718`

The LoRA experiments validate the training and adapter evaluation loop, but they do not demonstrate model improvement. The project should not claim LoRA gains from these smoke runs.

## Current Conclusion

The strongest current result is the real SROIE zero-shot Qwen image+OCR baseline on the 50-sample subset: `0.800`. The 100-sample image+OCR run has `8` skipped samples, so it must be interpreted as `92 evaluated / 100 requested`, not a full-100 score.

The 100-sample SROIE image+OCR error summary is:

- total rows: `100`
- evaluated rows: `92`
- skipped rows: `8`
- skipped reason: CUDA OOM on two documents / 8 QA rows
- raw accuracy: `0.696`
- normalized accuracy: `0.761`
- per-field normalized accuracy: address `0.217`, company `0.913`, date `1.000`, total_amount `0.913`

The project value at this stage is:

- real-data ingestion
- baseline comparison
- skipped-sample accounting
- per-field error analysis
- reproducible reporting

It is not yet a claim of SOTA or a proven LoRA improvement.

## Error Analysis Command

After running a SROIE prediction CSV, export skipped and wrong cases:

```bash
python scripts/analyze_sroie_errors.py --predictions outputs/predictions/sroie_qwen_image_ocr_100_predictions.csv --output-dir outputs/analysis/sroie
```

This writes skipped samples, wrong cases, per-field wrong counts, address wrong cases, and `outputs/analysis/sroie/error_summary.json`.

Generate the aggregate benchmark and skipped-row reports:

```bash
python scripts/report_benchmark.py
```

This writes `outputs/metrics/benchmark_report.md`, `outputs/metrics/skipped_samples_summary.json`, and `outputs/metrics/skipped_samples_summary.md`. If prediction CSV files were overwritten or are missing, the skipped summary explicitly says that skipped rows cannot be traced.

## Current Limitations

- Only the first 100 SROIE train documents are represented in the current benchmark.
- The 100-sample Qwen image+OCR run has 8 skipped rows.
- A larger sample, a held-out test split, and strict same-subset evaluation are still needed.
- Address extraction needs a better metric than strict normalized exact match.
- LoRA mock step10/step50 results did not improve over zero-shot and must not be mixed with SROIE zero-shot conclusions.
