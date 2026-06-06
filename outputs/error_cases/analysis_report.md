# OCR-only Baseline Error Analysis

This report is generated from the mock evaluation set. The baseline uses OCR text and hand-written rules only; it does not call a VLM or train any large model.

## Summary

- Total evaluated QA samples: 39
- Exported error cases: 9
- Error rate on this mock set: 0.231

## Errors by Field

| Name | Error Count |
| --- | ---: |
| permission_scope | 3 |
| date | 2 |
| company | 2 |
| total_amount | 1 |
| security_group | 1 |

## Errors by Error Type

| Name | Error Count |
| --- | ---: |
| ambiguous_context | 4 |
| missing_field | 1 |
| multi_value_conflict | 1 |
| normalization_error | 1 |
| wrong_candidate | 1 |
| unsupported_field | 1 |

## Main OCR-only Limitations

- Multiple values with similar surface forms are hard to disambiguate, especially subtotal/tax/total and invoice date/due date.
- Company extraction depends on labels and line order, so vendor, merchant, client, and company contexts can be confused.
- Natural-language permission scopes are not stable key-value fields, so rule matching often returns a verbose phrase instead of the canonical scope.
- OCR text order can be different from the visual layout, which makes first-match and last-match rules brittle.
- Missing or non-standard labels expose unsupported fields and weak fallbacks rather than true document understanding.

## Next Improvement Direction

- Image+OCR prompting should use both visual layout and OCR text to resolve field ownership when multiple candidates are present.
- A VLM-SFT stage can learn canonical field answers from instruction-style samples, including natural-language permission descriptions.
- Evaluation should compare OCR-only, image-only placeholder, image+OCR, and later VLM-SFT without fabricating VLM metrics before the model exists.
