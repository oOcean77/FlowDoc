# SROIE Benchmark Report

This report summarizes the current real SROIE-style receipt benchmark results provided from AutoDL. The prepared SROIE benchmark has 100 receipt documents and 399 QA samples. No training is run in this reporting step.

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

## Per-Field Observation

Address is the weakest field:

- Qwen OCR-only address accuracy: `0.160`
- Qwen image+OCR address accuracy: `0.217`

This suggests the model benefits from visual context overall, but address extraction remains hard because receipts often have long, multi-line address text and OCR/layout ambiguity.

## Mock Versus Real Data

The mock hard-case benchmark showed that LoRA smoke training did not improve over the zero-shot image+OCR result:

- Qwen image+OCR zero-shot on mock: `0.718`
- LoRA step10 on mock: `0.718`
- LoRA step50 on mock: `0.718`

The LoRA experiments validate the training and adapter evaluation loop, but they do not demonstrate model improvement. The project should not claim LoRA gains from these smoke runs.

## Current Conclusion

The strongest current result is the real SROIE zero-shot Qwen image+OCR baseline on the 50-sample subset: `0.800`. The 100-sample image+OCR run has `8` skipped samples, so it must be interpreted as `92 evaluated / 100 requested`, not a full-100 score.

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
python scripts/analyze_sroie_errors.py --predictions outputs/predictions/qwen2_5_vl_image_ocr_predictions.csv --output-dir outputs/error_cases/sroie_qwen_image_ocr
```

This writes skipped samples, wrong cases, per-field wrong counts, and address error examples.
