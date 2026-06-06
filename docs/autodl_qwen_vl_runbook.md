# AutoDL Qwen2.5-VL Runbook

This runbook prepares a reproducible AutoDL environment for FlowDoc-VLM Qwen2.5-VL inference smoke tests. It does not run LoRA SFT, DPO, Triton, or any training job.

## Recommended AutoDL Configuration

- Prefer at least 24 GB GPU memory, such as RTX 4090, A5000, A10, or A100.
- If only 16 GB is available, start with the 3B model, low `max_new_tokens`, and tiny sample counts.
- First goal is smoke-test inference. Do not start training until diagnostics, baseline metrics, and data readiness checks pass.

## Environment Setup

Example commands:

```bash
conda create -n flowdoc-vlm python=3.10 -y
conda activate flowdoc-vlm

# Install a torch build that matches the AutoDL CUDA image.
# Adjust the index URL if your CUDA version differs.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install "transformers>=4.49.0" pillow qwen-vl-utils accelerate peft bitsandbytes
pip install -e ".[dev,vlm]"
```

If the AutoDL image already includes torch, verify it before reinstalling:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Project Upload Or Sync

Preferred:

```bash
git clone <your-flowdoc-vlm-repo-url>
cd FlowDoc
```

Alternative:

- Compress the local project and upload it to AutoDL.
- Exclude `.pytest_cache`, `__pycache__`, `outputs`, local caches, model files, and sensitive real data.
- Keep generated metrics and mock data reproducible by rerunning scripts on AutoDL.

## Model Preparation

Two supported approaches:

1. Use the Hugging Face repo id:

```bash
Qwen/Qwen2.5-VL-3B-Instruct
```

This requires network access, cache space, and model permissions.

2. Prepare a local model directory outside the Git repository:

```bash
/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct
```

Do not put model weights in the Git repository.

## Environment Diagnostics

```bash
python scripts/check_vlm_env.py
```

Outputs:

- `outputs/metrics/vlm_env_report.json`
- `outputs/metrics/vlm_env_report.md`

## Smoke Test Commands

```bash
python scripts/prepare_mock_data.py
python scripts/run_field_eval.py
python scripts/build_instruction_data.py --input data/processed/mock_qa.csv --strategy image_ocr --output data/processed/instructions_mock_image_ocr.jsonl
python scripts/run_vlm_baseline.py --input data/processed/mock_qa.csv --strategy image_ocr --backend qwen2_5_vl --model-name /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct --max-samples 3 --smoke-test --output outputs/metrics/vlm_baseline_qwen_image_ocr_smoke.json
```

If the model or environment is unavailable, the baseline should write `skipped=true` with a clear `skip_reason`. That is expected and should not be reported as model accuracy.

## Expand Baseline After Smoke Test

After the 3-sample smoke test succeeds:

```bash
python scripts/run_vlm_baseline.py --input data/processed/mock_qa.csv --strategy image_ocr --backend qwen2_5_vl --model-name /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct --max-samples 20 --output outputs/metrics/vlm_baseline_qwen_image_ocr_20.json
python scripts/compare_baselines.py
```

## Common Issues

- CUDA unavailable: confirm the AutoDL image has a GPU instance and a compatible torch CUDA build.
- `qwen-vl-utils` missing: install `qwen-vl-utils` in the active conda environment.
- Transformers incompatible: install `transformers>=4.49.0`.
- OOM: use the 3B model, reduce `max_new_tokens`, keep `--max-samples` small, and avoid training.
- Image path not found: rerun `python scripts/prepare_mock_data.py` or verify Linux paths after upload.
- Hugging Face download failure: use a local model directory under `/root/autodl-tmp/models`.
- Windows/Linux path mismatch: replace Windows paths such as `E:\models\...` with Linux paths on AutoDL.

## SFT Gate

Before any LoRA SFT attempt, run:

```bash
python scripts/check_sft_readiness.py --instruction-file data/processed/instructions_mock_image_ocr.jsonl --model-name /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct --baseline-metrics outputs/metrics/vlm_baseline_qwen_image_ocr_smoke.json
```

Only proceed when `ready=true`. For the current MVP, this check is expected to explain what is still missing rather than start training.
