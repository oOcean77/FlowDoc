# Post-processing Design

FlowDoc-VLM keeps raw model predictions for evaluation and audit, then adds a separate cleaned prediction column for downstream workflow or database use. The post-processing layer is intentionally lightweight, rule-based, and field-aware.

## Why Post-processing Is Needed

Relaxed metrics can reveal that a prediction is semantically close, but downstream systems still need clean fields. Examples:

- `Merchant Pine Street Deli` should become `Pine Street Deli` for a company field.
- `$88.50` should become `88.50` for a total amount.
- `Reason: investigate delayed orders` should become `investigate delayed orders`.

The original `pred_answer` is never overwritten. Cleaned output is stored in `cleaned_pred_answer`.

## Rule-based vs LLM-based Schema Enforcer

A rule-based cleaner is preferred for the current stage:

- Low cost: no extra model call.
- Explainable: each output includes `rules_applied`.
- Easy to test: field-level rules can be covered by unit tests.
- Sufficient for current observed errors: prefixes, currency markers, whitespace, punctuation, and line breaks.

An LLM-based schema enforcer can be useful later for low-confidence cases, but it adds cost, latency, and another failure mode.

## Field-level Rules

### Amounts

Fields: `total_amount`, `subtotal`, `tax`.

- Remove currency markers such as `$`, `RM`, `RMB`, `CNY`, `USD`, and common OCR currency artifacts.
- Remove thousands separators.
- Extract the most amount-like numeric candidate from the prediction.
- Do not infer missing digits. `120` stays `120`; it is not guessed into `1200`.

### Dates

Fields: `date`, `deadline`.

- Remove light prefixes such as `Date:`.
- Collapse whitespace.
- Preserve the original date format to avoid incorrect conversion.

### Company-like Fields

Fields: `company`, `merchant`, `vendor`.

- Remove safe prefixes such as `Merchant`, `Company`, `Vendor`, `Store`, and `Shop`.
- Remove obvious trailing noise such as `AP`.
- Keep meaningful legal suffixes such as `Sdn Bhd`, `Ltd`, `LLC`, and `Inc`.
- Preserve original casing style.

### Address

Field: `address`.

- Remove prefixes such as `Address:`, `Addr:`, and `Location:`.
- Merge line breaks into spaces.
- Normalize repeated punctuation and repeated whitespace.
- Keep house numbers, street names, postal codes, and semantic tokens.
- Mark confidence as `low` if the cleaned result is too short or has fewer than three tokens.

### Workflow Fields

Fields: `permission_scope`, `reason`, `deadline`, `security_group`.

- Remove light prefixes such as `Reason:`, `Permission:`, `Scope:`, `Deadline:`, and `Group:`.
- Avoid semantic rewriting.

## Risks

- Rules can accidentally remove valid tokens if prefixes appear as real content.
- Company and address fields must not be over-cleaned.
- Amount/date/ID fields must remain strict and should not use semantic guessing.
- Post-processing can fix errors, but it can also break previously correct predictions; reports must track both.

## Next Steps

- Route low-confidence samples to an optional LLM-based schema enforcer.
- Add address token-F1 and edit-distance reporting to the cleaned evaluation view.
- Track postprocess fixed and broken cases across benchmark runs.
