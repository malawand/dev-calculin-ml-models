#!/usr/bin/env python3
"""
Fetch ALL 2.5 YEARS of 1-MINUTE data from Cortex for scalping
April 2023 → October 2025 = ~1.3 million samples
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cortex configuration
CORTEX_BASE_URL = "http://10.1.20.60:9009"
CORTEX_API = "/prometheus/api/v1/query_range"
SYMBOL = "BTCUSDT"

# Essential metrics for scalping
METRICS = [
    'crypto_last_price',
    'job:crypto_last_price:deriv5m',
    'job:crypto_last_price:deriv10m',
    'job:crypto_last_price:deriv15m',
    'job:crypto_last_price:deriv30m',
    'job:crypto_last_price:deriv1h',
    'job:crypto_last_price:avg5m',
    'job:crypto_last_price:avg10m',
    'job:crypto_last_price:avg15m',
    'job:crypto_last_price:avg30m',
    'job:crypto_last_price:avg1h',
    'crypto_volume',
    'job:crypto_volume:deriv5m',
    'job:crypto_volume:deriv15m',
    'job:crypto_volume:avg5m',
    'job:crypto_volume:avg15m',
]

def fetch_cortex_data(metric, start_time, end_time, step='1m'):
    """Fetch data from Cortex."""
    url = f"{CORTEX_BASE_URL}{CORTEX_API}"
    query = f'{metric}{{symbol="{SYMBOL}"}}'
    
    params = {
        'query': query,
        'start': int(start_time.timestamp()),
        'end': int(end_time.timestamp()),
        'step': step
    }
    
    try:
        response = requests.get(url, params=params, timeout=180)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'success':
            results = data.get('data', {}).get('result', [])
            if results:
                values = results[0].get('values', [])
                if values:
                    df = pd.DataFrame(values, columns=['timestamp', 'value'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    df = df.set_index('timestamp')
                    df.columns = [metric]
                    return df
        return None
    except Exception as e:
        logger.error(f"Error fetching {metric}: {e}")
        return None

def fetch_and_save_chunk(chunk_start, chunk_end, chunk_idx, total_chunks, output_dir):
    """Fetch a chunk and save it immediately."""
    logger.info(f"\n{'='*80}")
    logger.info(f"CHUNK {chunk_idx}/{total_chunks}: {chunk_start.date()} → {chunk_end.date()}")
    logger.info(f"{'='*80}")
    
    chunk_data = {}
    
    for i, metric in enumerate(METRICS, 1):
        df = fetch_cortex_data(metric, chunk_start, chunk_end, step='1m')
        if df is not None and len(df) > 0:
            chunk_data[metric] = df
            logger.info(f"   [{i}/{len(METRICS)}] {metric}: ✅ {len(df):,} samples")
        else:
            logger.warning(f"   [{i}/{len(METRICS)}] {metric}: ❌ No data")
        
        time.sleep(0.2)  # Rate limiting
    
    if chunk_data:
        # Combine and save chunk
        df_chunk = pd.concat(chunk_data.values(), axis=1)
        df_chunk.columns = chunk_data.keys()
        
        # Save chunk
        chunk_file = output_dir / f'chunk_{chunk_idx:03d}.parquet'
        df_chunk.to_parquet(chunk_file)
        
        logger.info(f"   ✅ Saved chunk: {chunk_file.name} ({df_chunk.shape[0]:,} × {df_chunk.shape[1]})")
        return True
    return False

def main():
    logger.info("="*80)
    logger.info("🚀 FETCHING ALL 2.5 YEARS OF 1-MINUTE DATA")
    logger.info("="*80)
    logger.info(f"Symbol: {SYMBOL}")
    logger.info(f"Granularity: 1 minute")
    logger.info(f"Metrics: {len(METRICS)}")
    logger.info("")
    
    # Date range: April 2023 → October 2025
    start_time = datetime(2023, 4, 1, tzinfo=None)
    end_time = datetime(2025, 10, 19, tzinfo=None)
    
    duration_days = (end_time - start_time).days
    
    logger.info(f"📅 Full Period:")
    logger.info(f"   Start: {start_time}")
    logger.info(f"   End:   {end_time}")
    logger.info(f"   Duration: {duration_days} days ({duration_days/365:.1f} years)")
    logger.info(f"   Expected samples: ~{duration_days * 1440:,} per metric")
    logger.info(f"   Total data points: ~{duration_days * 1440 * len(METRICS):,}")
    logger.info("")
    
    # Create chunks directory
    output_dir = Path(__file__).parent / 'artifacts' / 'historical_data' / 'chunks_1min'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split into 7-day chunks
    chunk_days = 7
    chunks = []
    current_start = start_time
    while current_start < end_time:
        current_end = min(current_start + timedelta(days=chunk_days), end_time)
        chunks.append((current_start, current_end))
        current_start = current_end
    
    logger.info(f"📦 Strategy:")
    logger.info(f"   Chunk size: {chunk_days} days")
    logger.info(f"   Total chunks: {len(chunks)}")
    logger.info(f"   Estimated time: {len(chunks) * len(METRICS) * 0.5 / 60:.1f} minutes")
    logger.info("")
    
    input(f"⚠️  This will fetch ~{duration_days * 1440 * len(METRICS):,} data points. Press Enter to continue...")
    
    logger.info("\n🚀 Starting fetch...\n")
    
    successful_chunks = 0
    failed_chunks = []
    
    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks, 1):
        try:
            success = fetch_and_save_chunk(chunk_start, chunk_end, chunk_idx, len(chunks), output_dir)
            if success:
                successful_chunks += 1
            else:
                failed_chunks.append(chunk_idx)
        except Exception as e:
            logger.error(f"   ❌ Chunk {chunk_idx} failed: {e}")
            failed_chunks.append(chunk_idx)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 FETCH SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"   Successful: {successful_chunks}/{len(chunks)}")
    logger.info(f"   Failed: {len(failed_chunks)}")
    if failed_chunks:
        logger.warning(f"   Failed chunks: {failed_chunks}")
    
    if successful_chunks > 0:
        logger.info(f"\n🔗 Combining all chunks...")
        
        # Load all chunks
        chunk_files = sorted(output_dir.glob('chunk_*.parquet'))
        logger.info(f"   Found {len(chunk_files)} chunk files")
        
        all_chunks = []
        for chunk_file in chunk_files:
            df = pd.read_parquet(chunk_file)
            all_chunks.append(df)
            logger.info(f"   Loaded {chunk_file.name}: {df.shape}")
        
        # Combine
        df_combined = pd.concat(all_chunks)
        df_combined = df_combined.sort_index()
        df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
        
        logger.info(f"\n   Combined shape: {df_combined.shape}")
        logger.info(f"   Date range: {df_combined.index.min()} → {df_combined.index.max()}")
        logger.info(f"   Duration: {(df_combined.index.max() - df_combined.index.min()).days} days")
        
        # Clean
        logger.info(f"\n🧹 Cleaning...")
        before = len(df_combined)
        df_combined = df_combined.dropna(thresh=int(len(df_combined.columns) * 0.5))
        df_combined = df_combined.ffill().bfill()
        df_combined = df_combined.replace([float('inf'), float('-inf')], float('nan'))
        df_combined = df_combined.fillna(0)
        after = len(df_combined)
        logger.info(f"   Removed {before - after:,} rows")
        logger.info(f"   Clean samples: {after:,}")
        
        # Save final
        final_path = Path(__file__).parent / 'artifacts' / 'historical_data' / 'scalping_1min_2.5years.parquet'
        df_combined.to_parquet(final_path)
        
        logger.info(f"\n💾 Final file saved:")
        logger.info(f"   Path: {final_path}")
        logger.info(f"   Size: {final_path.stat().st_size / 1024 / 1024:.2f} MB")
        logger.info(f"   Samples: {len(df_combined):,}")
        logger.info(f"   Columns: {len(df_combined.columns)}")
        
        # Verify granularity
        time_diffs = df_combined.index.to_series().diff()
        time_diffs_seconds = time_diffs.dt.total_seconds()
        most_common = time_diffs_seconds.mode().values[0] if len(time_diffs_seconds.mode()) > 0 else None
        
        logger.info(f"\n⏱️  Granularity:")
        logger.info(f"   Interval: {most_common:.0f} seconds ({most_common/60:.1f} minutes)")
        logger.info(f"   Samples/day: {len(df_combined) / ((df_combined.index.max() - df_combined.index.min()).days):.0f}")
        
        logger.info(f"\n✅ ALL DATA FETCHED!")
        logger.info(f"\n🎯 Ready to train scalping model on 2.5 years of 1-minute data")
        
        # Clean up chunk files
        logger.info(f"\n🧹 Cleaning up chunk files...")
        for chunk_file in chunk_files:
            chunk_file.unlink()
        output_dir.rmdir()
        logger.info(f"   ✅ Chunk files removed")
    else:
        logger.error(f"\n❌ No successful chunks - fetch failed")

if __name__ == "__main__":
    main()



