# SROIE Data Format

FlowDoc-VLM does not download or ship SROIE data. Put your local SROIE or SROIE-like files under `data/raw/` and run the preparation script.

## Option A: Directory Format

Use this structure:

```text
data/raw/sroie/
  images/
    X00016469612.jpg
  ocr/
    X00016469612.txt
  entities/
    X00016469612.json
```

The entity JSON should contain at least one of:

```json
{
  "company": "ADVANCO COMPANY",
  "date": "2018-03-19",
  "address": "NO 1, ROAD NAME",
  "total": "9.00"
}
```

`total_amount` is also accepted as an alias for `total`.

## Option B: SROIE-like CSV

Use `data/raw/sroie_like.csv` with columns such as:

```text
doc_id,image_path,ocr_text,company,date,address,total
```

`total_amount` can be used instead of `total`.

## Output Schema

Run:

```bash
python scripts/prepare_sroie_data.py --raw-dir data/raw/sroie --output data/processed/sroie_qa.csv --max-docs 100
```

or:

```bash
python scripts/prepare_sroie_data.py --input-csv data/raw/sroie_like.csv --output data/processed/sroie_qa.csv --max-docs 100
```

The output file uses the unified FlowDoc-VLM schema and includes QA rows for:

- `company`
- `date`
- `address`
- `total_amount`

The script checks image paths and reports `missing_image_count`. Missing images are marked with `image_exists=false`; the script does not silently fabricate images or records.

## No Fake Data Rule

If no raw directory or CSV exists, the script fails with a clear error. Mock fallback is only allowed when explicitly passing `--use-mock-fallback`, and it should not be reported as SROIE results.
