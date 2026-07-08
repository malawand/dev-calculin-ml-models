#!/usr/bin/env python3
"""
Export frozen Stage 3 encoder thresholds for live inference.

Run this ONCE after training (or whenever Stage 3 parameters change).
The output must be shipped alongside model.txt in artifacts/.

Usage:
    python3 scripts/export_encoder_artifacts.py
    python3 scripts/export_encoder_artifacts.py --input /path/to/btc_data_15m_labeled.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from constants import DEFAULT_FEATURE_COLS  # noqa: E402
from encoder import fit_encoder_from_history, save_encoder_artifacts  # noqa: E402

DEFAULT_INPUT = (
    PROJECT_ROOT.parent / "2_Build_Labels" / "btc_data_15m_labeled.parquet"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "encoder_artifacts.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export frozen encoder thresholds")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Labeled parquet with indicator columns")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to write encoder_artifacts.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(
            f"Input not found: {input_path}\n"
            "Run Stages 1–2 first, or pass --input pointing at a parquet with the "
            "three indicator columns."
        )

    df = pd.read_parquet(input_path).sort_index()
    missing = [c for c in DEFAULT_FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Input missing columns: {missing}")

    print(f"Loaded {len(df):,} rows from {input_path}")
    enc = fit_encoder_from_history(df[DEFAULT_FEATURE_COLS])

    print("\nLearned thresholds:")
    for feat, thr in enc.thresholds.items():
        print(f"  {feat}: {thr:.5f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_encoder_artifacts(enc, output_path, source=str(input_path), row_count=len(df))
    print(f"\nSaved encoder artifacts -> {output_path}")


if __name__ == "__main__":
    main()
