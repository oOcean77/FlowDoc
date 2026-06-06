from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.error_analysis import export_error_cases


if __name__ == "__main__":
    df = export_error_cases()
    print(f"Saved {len(df)} error cases to outputs/error_cases/error_cases.csv")
    print("Saved analysis report to outputs/error_cases/analysis_report.md")
