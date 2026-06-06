from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mock_docs import generate_mock_dataset


if __name__ == "__main__":
    df = generate_mock_dataset()
    print(f"Generated {len(df)} QA samples at data/processed/mock_qa.csv")
