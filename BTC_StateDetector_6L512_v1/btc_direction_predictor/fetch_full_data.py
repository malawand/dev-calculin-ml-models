#!/usr/bin/env python3
"""
Fetch 2.5 years of Bitcoin data from Cortex (April 2023 → October 2025)
"""

import pandas as pd
from datetime import datetime, timedelta
from src.data.prom import PrometheusClient
import time
from pathlib import Path
import yaml

print("="*80)
print("📊 FETCHING 2.5 YEARS OF DATA FROM CORTEX")
print("="*80)
print()

# Configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Date range: April 2023 → October 2025
start_date = datetime(2023, 4, 1, 0, 0, 0)
end_date = datetime(2025, 10, 17, 0, 0, 0)
step = '15m'

# Cortex connection
prom = PrometheusClient(
    base_url=config['cortex']['base_url'],
    read_api=config['cortex']['read_api']
)

# All metrics to fetch (use the same list from btc_minimal_start experiments)
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
    # Key derivative primes (subset - most useful ones)
    'job:crypto_last_price:deriv7d_prime7d', 'job:crypto_last_price:deriv4d_prime4d',
    'job:crypto_last_price:deriv24h_prime24h', 'job:crypto_last_price:deriv12h_prime12h',
]

symbol = config['cortex']['symbol']

print(f"📅 Date range: {start_date.date()} → {end_date.date()}")
print(f"   Duration: {(end_date - start_date).days} days (~2.5 years)")
print(f"📊 Metrics to fetch: {len(metrics)}")
print(f"⏱️  Step: {step}")
print(f"🎯 Symbol: {symbol}")
print()

# Fetch in 3-month chunks to avoid timeout
chunk_size_days = 90
all_chunks = []

current_start = start_date
chunk_num = 1

while current_start < end_date:
    current_end = min(current_start + timedelta(days=chunk_size_days), end_date)
    
    print(f"📦 Chunk {chunk_num}: {current_start.date()} → {current_end.date()}")
    print(f"   Fetching {len(metrics)} metrics...")
    
    chunk_data = {}
    
    for i, metric in enumerate(metrics, 1):
        try:
            query = f'{metric}{{symbol="{symbol}"}}'
            
            result = prom.query_range(
                query=query,
                start=current_start,
                end=current_end,
                step=step
            )
            
            if result and len(result) > 0 and 'values' in result[0]:
                timestamps = [datetime.fromtimestamp(v[0]) for v in result[0]['values']]
                values = [float(v[1]) for v in result[0]['values']]
                
                for ts, val in zip(timestamps, values):
                    if ts not in chunk_data:
                        chunk_data[ts] = {}
                    chunk_data[ts][metric] = val
            
            if i % 10 == 0:
                print(f"      Progress: {i}/{len(metrics)} metrics")
                
        except Exception as e:
            print(f"      ⚠️  Error fetching {metric}: {e}")
            continue
    
    if chunk_data:
        df_chunk = pd.DataFrame.from_dict(chunk_data, orient='index')
        df_chunk.index.name = 'timestamp'
        all_chunks.append(df_chunk)
        print(f"   ✅ Fetched {len(df_chunk)} samples")
    else:
        print(f"   ⚠️  No data in this chunk")
    
    current_start = current_end
    chunk_num += 1
    time.sleep(1)  # Rate limiting
    print()

# Combine all chunks
print("="*80)
print("📊 COMBINING ALL CHUNKS")
print("="*80)

if all_chunks:
    df_combined = pd.concat(all_chunks)
    df_combined = df_combined.sort_index()
    
    # Remove duplicates
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
    
    print(f"✅ Total samples: {len(df_combined):,}")
    print(f"   Date range: {df_combined.index.min()} → {df_combined.index.max()}")
    print(f"   Columns: {len(df_combined.columns)}")
    print(f"   Missing data: {df_combined.isna().sum().sum():,} NaN values")
    
    # Save
    output_path = Path('artifacts/historical_data/combined_2.5year_dataset.parquet')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_combined.to_parquet(output_path)
    print(f"\n💾 Saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
else:
    print("❌ No data fetched!")
    
print("\n" + "="*80)
print("✅ DATA FETCH COMPLETE!")
print("="*80)

