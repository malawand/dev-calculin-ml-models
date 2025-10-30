#!/usr/bin/env python3
"""
Compute missing derivative and derivative prime metrics for early 2023 data.
Replicates Prometheus deriv() function behavior.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

def compute_derivative(series: pd.Series, window: str) -> pd.Series:
    """
    Compute derivative (rate of change) over a rolling window.
    Replicates Prometheus deriv() function using linear regression.
    
    Args:
        series: Time series data
        window: Window size (e.g., '5m', '1h', '7d')
    
    Returns:
        Series of derivatives
    """
    # Convert window to number of 15-minute periods
    window_map = {
        '5m': 1, '10m': 1, '15m': 1, '30m': 2, '45m': 3,
        '1h': 4, '2h': 8, '4h': 16, '8h': 32, '12h': 48, '16h': 64,
        '24h': 96, '48h': 192,
        '3d': 288, '4d': 384, '5d': 480, '6d': 576, '7d': 672,
        '14d': 1344, '30d': 2880
    }
    
    periods = window_map.get(window, 4)  # Default to 1h
    
    def rolling_slope(x):
        """Calculate linear regression slope"""
        if len(x) < 2:
            return np.nan
        # Remove NaN values
        valid_idx = ~np.isnan(x)
        if valid_idx.sum() < 2:
            return np.nan
        x_valid = x[valid_idx]
        y_valid = np.arange(len(x))[valid_idx]
        try:
            slope, _, _, _, _ = stats.linregress(y_valid, x_valid)
            return slope
        except Exception:
            return np.nan
    
    min_periods = min(2, periods)  # Can't have min_periods > window
    return series.rolling(window=periods, min_periods=min_periods).apply(rolling_slope, raw=True)


def compute_derivative_prime(deriv_series: pd.Series, window: str) -> pd.Series:
    """
    Compute second derivative (derivative of derivative).
    
    Args:
        deriv_series: First derivative series
        window: Window size for second derivative
    
    Returns:
        Series of second derivatives (acceleration)
    """
    return compute_derivative(deriv_series, window)


print("="*80)
print("🔧 COMPUTING MISSING DERIVATIVES FOR FULL DATASET")
print("="*80)
print()

# Load the full dataset
data_path = Path('artifacts/historical_data/combined_full_dataset.parquet')
print(f"📥 Loading: {data_path}")
df = pd.read_parquet(data_path)
print(f"   Loaded {len(df):,} samples")
print()

# Ensure we have the price column
if 'crypto_last_price' not in df.columns:
    raise ValueError("crypto_last_price column not found!")

price = df['crypto_last_price']

# Define all derivative metrics we need
derivative_specs = [
    ('5m', 'job:crypto_last_price:deriv5m'),
    ('10m', 'job:crypto_last_price:deriv10m'),
    ('15m', 'job:crypto_last_price:deriv15m'),
    ('30m', 'job:crypto_last_price:deriv30m'),
    ('45m', 'job:crypto_last_price:deriv45m'),
    ('1h', 'job:crypto_last_price:deriv1h'),
    ('2h', 'job:crypto_last_price:deriv2h'),
    ('4h', 'job:crypto_last_price:deriv4h'),
    ('8h', 'job:crypto_last_price:deriv8h'),
    ('12h', 'job:crypto_last_price:deriv12h'),
    ('16h', 'job:crypto_last_price:deriv16h'),
    ('24h', 'job:crypto_last_price:deriv24h'),
    ('48h', 'job:crypto_last_price:deriv48h'),
    ('3d', 'job:crypto_last_price:deriv3d'),
    ('4d', 'job:crypto_last_price:deriv4d'),
    ('5d', 'job:crypto_last_price:deriv5d'),
    ('6d', 'job:crypto_last_price:deriv6d'),
    ('7d', 'job:crypto_last_price:deriv7d'),
    ('14d', 'job:crypto_last_price:deriv14d'),
    ('30d', 'job:crypto_last_price:deriv30d'),
]

# Compute first derivatives
print("📊 Computing First Derivatives...")
computed_count = 0
for window, col_name in derivative_specs:
    if col_name in df.columns:
        # Fill missing values
        missing_before = df[col_name].isna().sum()
        if missing_before > 0:
            print(f"   {col_name:45} - Filling {missing_before:>6,} missing values")
            computed = compute_derivative(price, window)
            df[col_name] = df[col_name].fillna(computed)
            computed_count += 1
    else:
        # Create new column
        print(f"   {col_name:45} - Creating new column")
        df[col_name] = compute_derivative(price, window)
        computed_count += 1

print(f"   ✅ Computed/filled {computed_count} first derivatives")
print()

# Define derivative prime (second derivative) specs
# Format: (base_deriv_window, prime_window, column_name)
derivative_prime_specs = [
    ('1h', '1h', 'job:crypto_last_price:deriv1h_prime1h'),
    ('1h', '2h', 'job:crypto_last_price:deriv1h_prime2h'),
    ('2h', '1h', 'job:crypto_last_price:deriv2h_prime1h'),
    ('2h', '2h', 'job:crypto_last_price:deriv2h_prime2h'),
    ('4h', '30m', 'job:crypto_last_price:deriv4h_prime30m'),
    ('4h', '1h', 'job:crypto_last_price:deriv4h_prime1h'),
    ('4h', '2h', 'job:crypto_last_price:deriv4h_prime2h'),
    ('8h', '30m', 'job:crypto_last_price:deriv8h_prime30m'),
    ('8h', '1h', 'job:crypto_last_price:deriv8h_prime1h'),
    ('8h', '2h', 'job:crypto_last_price:deriv8h_prime2h'),
    ('8h', '4h', 'job:crypto_last_price:deriv8h_prime4h'),
    ('12h', '30m', 'job:crypto_last_price:deriv12h_prime30m'),
    ('12h', '1h', 'job:crypto_last_price:deriv12h_prime1h'),
    ('12h', '2h', 'job:crypto_last_price:deriv12h_prime2h'),
    ('12h', '4h', 'job:crypto_last_price:deriv12h_prime4h'),
    ('12h', '8h', 'job:crypto_last_price:deriv12h_prime8h'),
    ('12h', '16h', 'job:crypto_last_price:deriv12h_prime16h'),
    ('16h', '30m', 'job:crypto_last_price:deriv16h_prime30m'),
    ('16h', '1h', 'job:crypto_last_price:deriv16h_prime1h'),
    ('16h', '2h', 'job:crypto_last_price:deriv16h_prime2h'),
    ('16h', '4h', 'job:crypto_last_price:deriv16h_prime4h'),
    ('16h', '8h', 'job:crypto_last_price:deriv16h_prime8h'),
    ('16h', '12h', 'job:crypto_last_price:deriv16h_prime12h'),
    ('16h', '16h', 'job:crypto_last_price:deriv16h_prime16h'),
    ('16h', '24h', 'job:crypto_last_price:deriv16h_prime24h'),
    ('24h', '30m', 'job:crypto_last_price:deriv24h_prime30m'),
    ('24h', '1h', 'job:crypto_last_price:deriv24h_prime1h'),
    ('24h', '2h', 'job:crypto_last_price:deriv24h_prime2h'),
    ('24h', '4h', 'job:crypto_last_price:deriv24h_prime4h'),
    ('24h', '8h', 'job:crypto_last_price:deriv24h_prime8h'),
    ('24h', '12h', 'job:crypto_last_price:deriv24h_prime12h'),
    ('24h', '16h', 'job:crypto_last_price:deriv24h_prime16h'),
    ('24h', '24h', 'job:crypto_last_price:deriv24h_prime24h'),
    ('24h', '48h', 'job:crypto_last_price:deriv24h_prime48h'),
    ('48h', '30m', 'job:crypto_last_price:deriv48h_prime30m'),
    ('48h', '1h', 'job:crypto_last_price:deriv48h_prime1h'),
    ('48h', '2h', 'job:crypto_last_price:deriv48h_prime2h'),
    ('48h', '4h', 'job:crypto_last_price:deriv48h_prime4h'),
    ('48h', '8h', 'job:crypto_last_price:deriv48h_prime8h'),
    ('48h', '12h', 'job:crypto_last_price:deriv48h_prime12h'),
    ('48h', '16h', 'job:crypto_last_price:deriv48h_prime16h'),
    ('48h', '24h', 'job:crypto_last_price:deriv48h_prime24h'),
    ('48h', '48h', 'job:crypto_last_price:deriv48h_prime48h'),
    ('3d', '30m', 'job:crypto_last_price:deriv3d_prime30m'),
    ('3d', '1h', 'job:crypto_last_price:deriv3d_prime1h'),
    ('3d', '2h', 'job:crypto_last_price:deriv3d_prime2h'),
    ('3d', '4h', 'job:crypto_last_price:deriv3d_prime4h'),
    ('3d', '8h', 'job:crypto_last_price:deriv3d_prime8h'),
    ('3d', '12h', 'job:crypto_last_price:deriv3d_prime12h'),
    ('3d', '16h', 'job:crypto_last_price:deriv3d_prime16h'),
    ('3d', '24h', 'job:crypto_last_price:deriv3d_prime24h'),
    ('3d', '48h', 'job:crypto_last_price:deriv3d_prime48h'),
    ('3d', '3d', 'job:crypto_last_price:deriv3d_prime3d'),
    ('3d', '4d', 'job:crypto_last_price:deriv3d_prime4d'),
    ('4d', '4d', 'job:crypto_last_price:deriv4d_prime4d'),
    ('5d', '5d', 'job:crypto_last_price:deriv5d_prime5d'),
    ('6d', '6d', 'job:crypto_last_price:deriv6d_prime6d'),
    ('7d', '7d', 'job:crypto_last_price:deriv7d_prime7d'),
    ('14d', '14d', 'job:crypto_last_price:deriv14d_prime14d'),
    ('30d', '30d', 'job:crypto_last_price:deriv30d_prime30d'),
]

# Compute second derivatives
print("📊 Computing Second Derivatives (Derivative Primes)...")
computed_prime_count = 0
for base_window, prime_window, col_name in derivative_prime_specs:
    # Get the base derivative column
    base_col_name = f'job:crypto_last_price:deriv{base_window}'
    
    if base_col_name not in df.columns:
        print(f"   ⚠️  Skipping {col_name} - base derivative {base_col_name} not found")
        continue
    
    base_deriv = df[base_col_name]
    
    if col_name in df.columns:
        # Fill missing values
        missing_before = df[col_name].isna().sum()
        if missing_before > 0:
            print(f"   {col_name:50} - Filling {missing_before:>6,} missing values")
            computed = compute_derivative_prime(base_deriv, prime_window)
            df[col_name] = df[col_name].fillna(computed)
            computed_prime_count += 1
    else:
        # Create new column
        print(f"   {col_name:50} - Creating new column")
        df[col_name] = compute_derivative_prime(base_deriv, prime_window)
        computed_prime_count += 1

print(f"   ✅ Computed/filled {computed_prime_count} second derivatives")
print()

# Check final data quality
print("="*80)
print("📊 FINAL DATA QUALITY CHECK")
print("="*80)
missing_per_col = df.isna().sum()
high_missing = missing_per_col[missing_per_col > len(df) * 0.5]

print(f"\nTotal samples: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"Total missing: {df.isna().sum().sum():,} ({df.isna().sum().sum() / (len(df) * len(df.columns)) * 100:.1f}%)")

if len(high_missing) > 0:
    print(f"\n⚠️  Columns still with >50% missing:")
    for col, count in high_missing.items():
        print(f"   {col}: {count / len(df) * 100:.1f}%")
else:
    print(f"\n✅ No columns with >50% missing data!")

# Monthly breakdown
print(f"\n📅 Data quality by month:")
monthly_groups = df.groupby(pd.Grouper(freq='ME'))
for date, group in monthly_groups:
    if len(group) > 0:
        missing_pct = group.isna().sum().sum() / (len(group) * len(group.columns)) * 100
        print(f"  {date.strftime('%Y-%m')}: {len(group):>6,} samples | Missing: {missing_pct:>5.1f}%")

# Save the complete dataset
output_path = Path('artifacts/historical_data/combined_full_dataset.parquet')
print(f"\n💾 Saving complete dataset: {output_path}")
df.to_parquet(output_path)
print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

print("\n" + "="*80)
print("✅ DERIVATIVE COMPUTATION COMPLETE!")
print("="*80)
print("\nNext: Re-run incremental training experiments on the complete dataset")

