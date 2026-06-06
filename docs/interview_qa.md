# FlowDoc-VLM Interview Q&A

## 1. Why is this not just OCR?

OCR is only one input signal. FlowDoc-VLM evaluates how OCR text, document images, and image+OCR prompting behave under the same schema and metric pipeline. The project includes real-data adapters, field-level evaluation, skipped-sample tracing, and wrong-case analysis, so the core value is a reproducible document-understanding benchmark rather than text extraction alone.

## 2. Why was Qwen image+OCR not better than OCR-only on mock-hard?

The mock-hard set is synthetic and only has 39 QA rows. It was intentionally designed with OCR-friendly labels and controlled expected answers, so rule OCR can do well while a zero-shot VLM may over-explain, choose a nearby field, or return formatting that hurts exact field metrics. That result is useful because it shows the benchmark does not automatically favor VLMs.

## 3. Why did LoRA not improve the result?

The LoRA step10 and step50 runs were smoke training runs on a tiny mock instruction set. They validate the training and adapter evaluation chain, but they are too small and too synthetic to support a model-quality claim. The result stayed at 0.718 on mock image+OCR, so the honest conclusion is that LoRA did not demonstrate improvement.

## 4. Why did Qwen improve clearly on SROIE?

SROIE receipts are real scanned documents with layout, noisy OCR, multi-line content, and field ambiguity. Qwen2.5-VL can use language priors and, in image+OCR mode, visual context. On the same-subset 100 benchmark, rule OCR reached 0.320, Qwen OCR-only reached 0.710, and Qwen image+OCR reached 0.761 on 92 evaluated rows.

## 5. Why is the address field the weakest?

Address answers are long, multi-line, and sensitive to punctuation, line breaks, token order, and OCR noise. The current metric is strict normalized exact match, which is suitable for amounts and dates but harsh for addresses. Address normalized accuracy is 0.217 in the SROIE image+OCR run, so future evaluation should add token-level F1 or edit distance.

## 6. What caused the 8 skipped rows?

The SROIE Qwen image+OCR 100 run requested 100 QA rows but evaluated 92. The 8 skipped QA rows came from CUDA OOM on two documents. The result must be reported as 0.761 on 92 evaluated rows with 8 skipped, not as a full 100-evaluated score.

## 7. How would you optimize this next?

I would first make evaluation stricter and larger: run full SROIE train/test, add token-level F1 for address, and enforce same-subset comparison. Then I would reduce inference OOM with image resizing, max-pixel limits, and sequential inference. After that, I would run prompt ablations for raw OCR, cleaned OCR, field-aware OCR, and JSON output before returning to formal SFT.

## 8. Where is the engineering depth?

The depth is in the end-to-end evaluation system: unified schema, data adapters, multi-strategy baselines, model skip handling, prediction artifact naming, same-subset reporting, skipped-row tracing, raw versus normalized accuracy, per-field wrong cases, and pytest coverage that does not require loading large models.

## 9. How is this different from a simple demo?

A demo usually shows a few examples working. FlowDoc-VLM records what fails, separates mock and real data, avoids fake LoRA gains, tracks skipped samples, compares methods on the same subset, and writes reproducible benchmark reports. That makes it more useful for engineering decisions and interviews.

## 10. What if the interviewer questions the small data size?

I would agree with the concern. The current SROIE benchmark uses 100 train-subset documents and 399 QA samples, so it is not a final model benchmark. The point of this phase is to build a reliable evaluation and error-analysis loop. The next step is to scale to full train/test splits and additional datasets such as CORD and FUNSD.
