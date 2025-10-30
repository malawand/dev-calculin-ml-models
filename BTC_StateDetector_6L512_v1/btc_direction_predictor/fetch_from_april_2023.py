#!/usr/bin/env python3
"""
Fetch full 2.5 years of Bitcoin data from Cortex (April 2023 → October 2025)
Using the DataFrame-based PrometheusClient
"""

import pandas as pd
from datetime import datetime, timedelta
from src.data.prom import PrometheusClient
import time
from pathlib import Path

print("="*80)
print("📊 FETCHING 2.5 YEARS OF DATA FROM CORTEX")
print("="*80)
print()

# Date range: April 2023 → October 2025
start_date = datetime(2023, 4, 1, 0, 0, 0)
end_date = datetime(2025, 10, 17, 0, 0, 0)
step = '15m'
symbol = 'BTCUSDT'

# Cortex connection
prom = PrometheusClient(
    base_url="http://10.1.20.60:9009",
    read_api="/prometheus/api/v1/query_range"
)

# Core metrics to fetch
metrics = [
    'crypto_last_price',
    # Averages
    'job:crypto_last_price:avg10m', 'job:crypto_last_price:avg15m', 'job:crypto_last_price:avg30m',
    'job:crypto_last_price:avg45m', 'job:crypto_last_price:avg1h', 'job:crypto_last_price:avg2h',
    'job:crypto_last_price:avg4h', 'job:crypto_last_price:avg8h', 'job:crypto_last_price:avg12h',
    'job:crypto_last_price:avg24h', 'job:crypto_last_price:avg48h', 'job:crypto_last_price:avg3d',
    'job:crypto_last_price:avg4d', 'job:crypto_last_price:avg5d', 'job:crypto_last_price:avg6d',
    'job:crypto_last_price:avg7d', 'job:crypto_last_price:avg14d',
    # Derivatives
    'job:crypto_last_price:deriv5m', 'job:crypto_last_price:deriv10m', 'job:crypto_last_price:deriv15m',
    'job:crypto_last_price:deriv30m', 'job:crypto_last_price:deriv45m', 'job:crypto_last_price:deriv1h',
    'job:crypto_last_price:deriv2h', 'job:crypto_last_price:deriv4h', 'job:crypto_last_price:deriv8h',
    'job:crypto_last_price:deriv12h', 'job:crypto_last_price:deriv16h', 'job:crypto_last_price:deriv24h',
    'job:crypto_last_price:deriv48h', 'job:crypto_last_price:deriv3d', 'job:crypto_last_price:deriv4d',
    'job:crypto_last_price:deriv5d', 'job:crypto_last_price:deriv6d', 'job:crypto_last_price:deriv7d',
    'job:crypto_last_price:deriv14d', 'job:crypto_last_price:deriv30d',
    # Key derivative primes
    'job:crypto_last_price:deriv7d_prime7d', 'job:crypto_last_price:deriv4d_prime4d',
    'job:crypto_last_price:deriv24h_prime24h', 'job:crypto_last_price:deriv12h_prime12h',
]

print(f"📅 Date range: {start_date.date()} → {end_date.date()}")
print(f"   Duration: {(end_date - start_date).days} days (~2.5 years)")
print(f"📊 Metrics to fetch: {len(metrics)}")
print(f"⏱️  Step: {step}")
print(f"🎯 Symbol: {symbol}")
print()

# Fetch in 3-month chunks to avoid timeout
chunk_size_days = 90
all_dfs = []

current_start = start_date
chunk_num = 1

while current_start < end_date:
    current_end = min(current_start + timedelta(days=chunk_size_days), end_date)
    
    print(f"📦 Chunk {chunk_num}: {current_start.date()} → {current_end.date()}")
    
    chunk_dfs = []
    
    for i, metric in enumerate(metrics, 1):
        try:
            query = f'{metric}{{symbol="{symbol}"}}'
            
            result_df = prom.query_range(
                query=query,
                start=current_start,
                end=current_end,
                step=step
            )
            
            # Check if we got data
            if result_df is not None and not result_df.empty:
                # Rename column to metric name
                result_df.columns = [metric]
                chunk_dfs.append(result_df)
                
                if i % 10 == 0:
                    print(f"      Progress: {i}/{len(metrics)} metrics ({len(chunk_dfs)} successful)")
                    
        except Exception as e:
            if i % 10 == 0:
                print(f"      Progress: {i}/{len(metrics)} metrics ({len(chunk_dfs)} successful)")
            continue
    
    if chunk_dfs:
        # Combine all metrics for this chunk by timestamp
        df_chunk = pd.concat(chunk_dfs, axis=1)
        all_dfs.append(df_chunk)
        print(f"   ✅ Fetched {len(df_chunk)} samples with {len(df_chunk.columns)} metrics")
    else:
        print(f"   ⚠️  No data in this chunk")
    
    current_start = current_end
    chunk_num += 1
    time.sleep(0.5)  # Rate limiting
    print()

# Combine all chunks
print("="*80)
print("📊 COMBINING ALL CHUNKS")
print("="*80)

if all_dfs:
    df_combined = pd.concat(all_dfs)
    df_combined = df_combined.sort_index()
    
    # Remove duplicates
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
    
    # Fill forward missing values (interpolate gaps)
    df_combined = df_combined.ffill(limit=4)  # Fill forward up to 1 hour gaps
    
    print(f"\n✅ Total samples: {len(df_combined):,}")
    print(f"   Date range: {df_combined.index.min()} → {df_combined.index.max()}")
    print(f"   Duration: {(df_combined.index.max() - df_combined.index.min()).days} days")
    print(f"   Years: {(df_combined.index.max() - df_combined.index.min()).days / 365.25:.2f}")
    print(f"   Columns: {len(df_combined.columns)}")
    print(f"   Missing data: {df_combined.isna().sum().sum():,} NaN values ({df_combined.isna().sum().sum() / (len(df_combined) * len(df_combined.columns)) * 100:.1f}%)")
    
    # Save
    output_path = Path('artifacts/historical_data/combined_full_dataset.parquet')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_combined.to_parquet(output_path)
    print(f"\n💾 Saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Monthly breakdown
    print(f"\n📅 Monthly breakdown:")
    monthly = df_combined.resample('ME').size()
    for date, count in monthly.items():
        if count > 0:
            print(f"  {date.strftime('%Y-%m')}: {count:>6,} samples")
else:
    print("❌ No data fetched!")
    
print("\n" + "="*80)
print("✅ DATA FETCH COMPLETE!")
print("="*80)



