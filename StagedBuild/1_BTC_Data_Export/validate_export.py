#!/usr/bin/env python3
"""
Validate a Parquet export: detect timestamp gaps, report NaN percentages,
and show per-column statistics (min, max, mean, std).
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd


def validate(path: str, expected_step: int) -> bool:
    """
    Run all checks and print a data-quality report.
    Returns True if the data passes all checks.
    """
    df = pd.read_parquet(path, engine="pyarrow")
    print(f"File   : {path}")
    print(f"Rows   : {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Index  : {df.index.min()} -> {df.index.max()}")
    print()

    passed = True

    # ---- Timestamp gap analysis ----
    print("=" * 60)
    print("TIMESTAMP GAP ANALYSIS")
    print("=" * 60)
    diffs = df.index.to_series().diff().dt.total_seconds().dropna()
    threshold = expected_step * 2
    gaps = diffs[diffs > threshold]
    if len(gaps) == 0:
        print(f"  No gaps > {threshold}s (2x step). OK")
    else:
        passed = False
        print(f"  Found {len(gaps)} gap(s) > {threshold}s:")
        for ts, gap_sec in gaps.items():
            print(f"    {ts}  gap={gap_sec:.0f}s ({gap_sec / 3600:.1f}h)")
        if len(gaps) > 20:
            print(f"    ... and {len(gaps) - 20} more")
    print()

    # ---- NaN report ----
    print("=" * 60)
    print("NaN REPORT")
    print("=" * 60)
    nan_pct = df.isna().mean() * 100
    for col in df.columns:
        pct = nan_pct[col]
        flag = "  WARN" if pct > 5 else ""
        print(f"  {col:50s} {pct:6.2f}%{flag}")
        if pct > 50:
            passed = False
    print()

    # ---- Per-column statistics ----
    print("=" * 60)
    print("COLUMN STATISTICS")
    print("=" * 60)
    stats = df.describe().T[["min", "max", "mean", "std", "count"]]
    stats["count"] = stats["count"].astype(int)
    for col in stats.index:
        row = stats.loc[col]
        print(
            f"  {col:50s}  "
            f"min={row['min']:>14.4f}  "
            f"max={row['max']:>14.4f}  "
            f"mean={row['mean']:>14.4f}  "
            f"std={row['std']:>10.4f}  "
            f"n={row['count']}"
        )
    print()

    # ---- Summary ----
    print("=" * 60)
    if passed:
        print("RESULT: PASS")
    else:
        print("RESULT: ISSUES DETECTED (see above)")
    print("=" * 60)
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Parquet data export")
    parser.add_argument("path", help="Path to Parquet file")
    parser.add_argument("--step", type=int, default=3600, help="Expected step in seconds")
    args = parser.parse_args()

    ok = validate(args.path, args.step)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
