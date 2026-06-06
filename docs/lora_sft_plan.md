# LoRA SFT Plan

This document reserves the training design for a later phase. The current round only prepares inference smoke tests, instruction data, train/eval splitting, and readiness checks.

## Why Not Train On 39 Mock Samples

The current mock set is useful for validating schema, extraction logic, metrics, and failure analysis. It is not enough for LoRA SFT because:

- 39 QA rows are too few for learning robust document layouts.
- Mock OCR text can make the model memorize synthetic wording.
- The distribution does not represent real receipts, invoices, forms, and noisy screenshots.
- A high training score on this data would be misleading in interviews and reports.

## Required Baseline First

Before LoRA SFT, run Qwen2.5-VL zero-shot or image+OCR inference on a small sample and store real metrics. This establishes whether the model and prompt pipeline work before any training changes the model.

## Minimum Data Requirements

Recommended minimum before training:

- Hundreds to thousands of instruction-answer samples.
- Multiple document types and field types.
- Real or realistic document images with valid `image_path`.
- Stable train/eval split with no duplicate `sample_id`.
- Non-empty answers and clear field labels.
- Baseline metrics for OCR-only and Qwen2.5-VL image+OCR.

## Answer-Only Label Mask

The training target should mask user prompt tokens and compute loss only on the answer text. This reduces the chance that the model learns to reproduce OCR prompts instead of concise field values.

## Train/Eval Split

Use deterministic splitting before training:

```bash
python scripts/split_instruction_data.py --input data/processed/instructions_mock_image_ocr.jsonl --train-output data/processed/train_instructions.jsonl --eval-output data/processed/eval_instructions.jsonl --eval-ratio 0.2 --seed 42
```

Avoid leakage by splitting on document id when real datasets are large enough.

## Evaluation After Training

After LoRA SFT, reuse the same field-level evaluation conventions:

- Generate predictions into CSV with `sample_id`, `field_name`, `gold_answer`, and `pred_answer`.
- Apply only lightweight evaluation normalization.
- Evaluate with `field_metrics.py`.
- Compare against OCR-only and zero-shot Qwen2.5-VL baselines.

## Suggested Training Configuration

Initial candidate:

- model: `Qwen2.5-VL-3B-Instruct`
- LoRA rank: `8`
- learning rate: `1e-4` or `2e-4`
- epochs: `1-3`
- batch size: `1`
- gradient accumulation: `8`
- max_new_tokens: `64`

These are planning defaults, not a training command.

## Risks

- Overfitting to mock data.
- Insufficient GPU memory.
- Incorrect label masking.
- Incorrect image token handling.
- Non-reproducible metrics if skipped runs are mixed with real evaluated runs.
- Training results that are difficult to interpret without strong zero-shot baselines.

## Go/No-Go Rule

Do not start LoRA SFT until `scripts/check_sft_readiness.py` reports `ready=true` and the Qwen2.5-VL smoke baseline has either real evaluated metrics or a documented environment fix plan.
