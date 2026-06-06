# FlowDoc-VLM

FlowDoc-VLM is a document-image field extraction and multimodal evaluation project for enterprise workflow scenarios.

The current week-2 MVP adds real-data adapters, instruction-answer data building, and VLM inference baseline scaffolding on top of the OCR-only evaluation pipeline. It does not run LoRA SFT, DPO, Triton optimization, or any large-model training.

## Naming

This project is named **FlowDoc-VLM**. It does not reproduce the existing DocVLM paper, does not claim a new document VLM architecture, and does not use the name DocVLM-FlowBench to avoid confusion with prior work.

## Why Enterprise Documents

Enterprise workflows often depend on uploaded forms, receipts, invoices, contracts, and access request screenshots. Workflow-Agent can move approval or ticket states, while FlowDoc-VLM extracts structured fields from document images and OCR text so the downstream workflow can validate risk, route human approval, and send notifications.

## Why Not Just OCR

FlowDoc-VLM is not a replacement for OCR. OCR-only is included as a baseline. The project studies where VLM-style document understanding can complement OCR/KIE:

- low-quality scans
- non-standard layouts
- missing or ambiguous field names
- multiple amount fields such as subtotal, tax, and total
- layout understanding after OCR structure is lost
- handwriting, stamps, and dense tables where field ownership is unclear

## Datasets

The week-1 MVP supports local CSV and mock fallback data. The generated mock set includes deliberately hard cases so the OCR-only baseline is not unrealistically perfect:

- subtotal, tax, total, amount due, and weak total labels
- invoice date and due date in the same OCR text
- vendor, company, client, and merchant names in the same document
- natural-language permission scope descriptions
- OCR lines with order different from the visual layout
- missing fields and non-standard labels

Adapters are included for DocVQA-like, FUNSD-like, SROIE-like, and local unified CSV records. Local CSV files can be read from `data/raw/*.csv` and converted into `data/processed/*.csv`. Missing image files are marked with `image_exists=false`; the original `image_path` value is preserved for later environment-specific resolution.

Recommended datasets:

- DocVQA: document visual question answering
- FUNSD: form understanding with entities and relations
- SROIE: scanned receipt OCR and key information extraction
- CORD: optional later expansion

Unified CSV columns are: `sample_id`, `doc_id`, `doc_type`, `image_path`, `question`, `answer`, `field_name`, `field_type`, `ocr_text`, `bbox`, `source_dataset`, and `image_exists`.

## Flow

```mermaid
flowchart LR
  A[Document Image + OCR Text] --> B[Schema Adapter]
  B --> C[OCR-only Baseline]
  B --> D[Image-only Placeholder]
  B --> E[Image+OCR Prompt Builder]
  C --> F[Field Metrics]
  F --> G[Error Analysis]
  C --> H[Workflow Payload]
  H --> I[Workflow-Agent]
```

## Quick Start

```bash
python -m pip install -e ".[dev]"
python scripts/prepare_mock_data.py
python scripts/run_field_eval.py
python scripts/export_error_cases.py
python scripts/build_instruction_data.py --input data/processed/mock_qa.csv --strategy image_ocr --output data/processed/instructions_mock_image_ocr.jsonl
python scripts/check_vlm_env.py
python scripts/run_vlm_baseline.py --input data/processed/mock_qa.csv --strategy image_ocr --backend dummy --output outputs/metrics/vlm_baseline_dummy_image_ocr.json
python scripts/compare_baselines.py
python scripts/demo_workflow_integration.py
python -m pytest -q
```

If `WORKFLOW_AGENT_URL` is set, the workflow demo will POST to that service. Otherwise it saves `outputs/workflow_payloads/payload.json` so the demo works offline.

## Current Capabilities

- unified Pydantic schema for VQA and field extraction samples
- mock document PNG and CSV generation
- local CSV adapter plus DocVQA/FUNSD/SROIE-like converters
- OCR-only rule baseline using `ocr_text`
- image-only placeholder with explicit unsupported status and no fabricated VLM metric
- instruction-answer JSONL construction for `ocr_only`, `image_only`, and `image_ocr`
- Qwen2.5-VL, LLaVA, and dummy VLM runner interfaces
- VLM baseline script with skipped metrics when the model environment is unavailable
- exact match, normalized exact match, regex match, field accuracy, per-field accuracy, missing field rate, and multi-value conflict rate
- error case CSV export plus `outputs/error_cases/analysis_report.md`
- Workflow-Agent payload builder and optional client POST
- CPU-only pytest coverage without model downloads or API keys

`field_level_accuracy` is a baseline validation number on the current mock data, not the final model capability of FlowDoc-VLM. The deliberately hard mock cases are meant to expose common OCR-only failures before image+OCR prompting or VLM-SFT is introduced.

## Week 2 Plan

- prepare real-data-like CSV adapters for FUNSD, SROIE, DocVQA, and local unified records
- build answer-only instruction JSONL for `ocr_only`, `image_only`, and `image_ocr`
- scaffold Qwen2.5-VL and LLaVA inference runners without downloading models in tests
- compare OCR-only and VLM inference baselines, marking unavailable model runs as `skipped=true`
- keep LoRA SFT for week 3 or later

## Input Strategies

- `ocr_only`: question plus OCR text, no image content.
- `image_only`: image path plus question, no OCR text.
- `image_ocr`: image path plus question plus OCR text.

If a real Qwen2.5-VL or LLaVA environment is not installed, `run_vlm_baseline.py` writes `skipped=true` with a clear `skip_reason` and leaves accuracy fields as `null`. The dummy backend is always skipped and is only for pipeline validation.

## Run Qwen2.5-VL Smoke Baseline

Start with an environment check. This does not download a model:

```bash
python scripts/check_vlm_env.py
```

Run the dummy pipeline check:

```bash
python scripts/run_vlm_baseline.py --input data/processed/mock_qa.csv --strategy image_ocr --backend dummy --output outputs/metrics/vlm_baseline_dummy_image_ocr.json
```

Run a Qwen2.5-VL smoke test with the default Hugging Face repo id:

```bash
python scripts/run_vlm_baseline.py --input data/processed/mock_qa.csv --strategy image_ocr --backend qwen2_5_vl --model-name Qwen/Qwen2.5-VL-3B-Instruct --max-samples 3 --smoke-test --output outputs/metrics/vlm_baseline_qwen_image_ocr_smoke.json
```

Run the same smoke test with a local model path:

```bash
python scripts/run_vlm_baseline.py --input data/processed/mock_qa.csv --strategy image_ocr --backend qwen2_5_vl --model-name E:\models\Qwen2.5-VL-3B-Instruct --max-samples 3 --smoke-test --output outputs/metrics/vlm_baseline_qwen_local_image_ocr_smoke.json
```

If dependencies, model files, network access, image files, or GPU resources are unavailable, the result is `skipped=true` with a `skip_reason`. A skipped result is an environment or availability status, not evidence that the model is inaccurate. Skipped runs must not be reported as real VLM metrics. Prediction postprocessing only strips common answer prefixes and lightly extracts amount/date/id values for evaluation normalization; it is not a substitute for model correctness.

## Real Data Commands

```bash
python scripts/prepare_real_data.py --source funsd --input data/raw/funsd_like.csv --output data/processed/funsd_qa.csv
python scripts/prepare_real_data.py --source sroie --input data/raw/sroie_like.csv --output data/processed/sroie_qa.csv
python scripts/prepare_real_data.py --source docvqa --input data/raw/docvqa_like.csv --output data/processed/docvqa_qa.csv
```

## Week 3 Plan

- LoRA rank ablation
- OCR input granularity ablation
- field-type grouped evaluation
- end-to-end demo with Workflow-Agent

## Limitations

The image-only placeholder does not claim real model capability. The OCR baseline is deliberately simple and can fail on dense tables, missing labels, non-standard field names, OCR order changes, natural-language permission scopes, and ambiguous multi-value fields. Qwen2.5-VL and LLaVA runners require local model dependencies; CPU fallback may work but can be very slow. All reported metrics are generated by scripts, not hand-written, and no VLM-SFT score is claimed before that model path exists.
