# Address Error Plan

The current SROIE image+OCR run has address normalized accuracy of `0.217`. Address is the main bottleneck even though company, date, and total_amount are much stronger.

## Why Address Is Weak

Address fields are harder than amounts and dates for several reasons:

- Long field values: addresses often contain many tokens rather than one compact value.
- Line breaks: receipt addresses may span multiple OCR lines.
- Punctuation: commas, dashes, unit labels, and postal formatting vary across OCR and ground truth.
- OCR order: OCR text can interleave company names, branches, phone numbers, and address lines.
- Strict exact match: a mostly correct address can be marked wrong if one line break, token order, or punctuation mark differs.

## Next-step Plan

### 1. Address Text Normalization

Add field-specific normalization for address values:

- collapse repeated whitespace and line breaks
- normalize punctuation variants
- remove low-value separators
- normalize common OCR artifacts
- preserve important numeric and street tokens

This should be reported separately from strict exact match so the benchmark remains transparent.

### 2. Token-level F1

Add token precision, recall, and F1 for address fields. This captures partial correctness better than exact match and can show whether the model found the right location but formatted it differently.

### 3. Edit-distance Similarity

Add normalized edit-distance similarity for address strings. This helps separate near-miss formatting errors from completely wrong address predictions.

### 4. OCR Candidate Line Stitching

Build a simple candidate generator that stitches adjacent OCR lines likely to form an address:

- nearby street-number tokens
- street or road keywords
- city/state/postcode patterns when available
- line proximity from OCR order or bounding boxes

This can improve rule OCR and also provide better context to VLM prompts.

### 5. Optional Field-aware Prompt

After metric and post-processing improvements, test field-aware prompts for address extraction:

- ask for complete address only
- discourage company, phone, and tax-number leakage
- enforce JSON output with a single address field

Prompt changes should be evaluated on the same subset before considering any new LoRA training.

## Priority

Do post-processing and metrics first. Do not rush into LoRA for address until the benchmark can distinguish formatting errors, partial matches, and genuinely wrong predictions.
