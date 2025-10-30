#!/usr/bin/env python3
"""
Create a clean dataset by excluding July-September 2023 (high missing data)
Keep: April-June 2023 + October 2023 onwards
"""

import pandas as pd
from pathlib import Path

print("="*80)
print("🧹 CREATING CLEAN DATASET")
print("="*80)
print()

# Load the full dataset
data_path = Path('artifacts/historical_data/combined_full_dataset.parquet')
print(f"📥 Loading: {data_path}")
df = pd.read_parquet(data_path)
print(f"   Loaded {len(df):,} samples")
print(f"   Date range: {df.index.min()} → {df.index.max()}")
print()

# Filter out July-September 2023 (bad data months)
print("🔍 Filtering data...")
df_filtered = df[
    ~((df.index >= '2023-07-01') & (df.index < '2023-10-01'))
]

print(f"   ✅ Kept: {len(df_filtered):,} samples")
print(f"   ❌ Removed: {len(df) - len(df_filtered):,} samples (July-Sept 2023)")
print()

# Check data quality
print("📊 DATA QUALITY CHECK")
print("="*80)
missing_pct = df_filtered.isna().sum().sum() / (len(df_filtered) * len(df_filtered.columns)) * 100
print(f"Total missing: {missing_pct:.1f}%")
print()

# Monthly breakdown
print(f"📅 Data quality by month:")
monthly_groups = df_filtered.groupby(pd.Grouper(freq='ME'))
for date, group in monthly_groups:
    if len(group) > 0:
        missing_pct = group.isna().sum().sum() / (len(group) * len(group.columns)) * 100
        print(f"  {date.strftime('%Y-%m')}: {len(group):>6,} samples | Missing: {missing_pct:>5.1f}%")

# Save
output_path = Path('artifacts/historical_data/combined_full_dataset.parquet')
print(f"\n💾 Saving clean dataset: {output_path}")
df_filtered.to_parquet(output_path)
print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

print("\n" + "="*80)
print("✅ CLEAN DATASET CREATED!")
print("="*80)
print(f"\nFinal dataset:")
print(f"  • Samples: {len(df_filtered):,}")
print(f"  • Date range: {df_filtered.index.min().date()} → {df_filtered.index.max().date()}")
print(f"  • Duration: {(df_filtered.index.max() - df_filtered.index.min()).days} days")
print(f"  • Years: {(df_filtered.index.max() - df_filtered.index.min()).days / 365.25:.2f}")
print(f"  • Missing data: {df_filtered.isna().sum().sum() / (len(df_filtered) * len(df_filtered.columns)) * 100:.1f}%")



