#!/usr/bin/env python3
"""
Fetch 2.5 years of 1-minute data with rate limiting and retry logic
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cortex configuration
CORTEX_URL = "http://10.1.20.60:9009"
CORTEX_API_PATH = "/prometheus/api/v1/query_range"
SYMBOL = "BTCUSDT"

# Rate limiting
REQUEST_DELAY = 0.5  # 500ms between requests
CHUNK_DELAY = 2.0  # 2 seconds between chunks
MAX_RETRIES = 3
RETRY_DELAY = 5  # 5 seconds before retry

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

def fetch_cortex_data(metric, start_time, end_time, step, retry_count=0):
    """Fetches data for a single metric with retry logic."""
    query = f'{metric}{{symbol="{SYMBOL}"}}'
    params = {
        'query': query,
        'start': int(start_time.timestamp()),
        'end': int(end_time.timestamp()),
        'step': step
    }
    url = f"{CORTEX_URL}{CORTEX_API_PATH}"
    
    try:
        time.sleep(REQUEST_DELAY)  # Rate limiting
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] == 'success' and data['data']['result']:
            values = data['data']['result'][0]['values']
            df = pd.DataFrame(values, columns=['timestamp', metric])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('timestamp')
            df[metric] = pd.to_numeric(df[metric])
            return df
        else:
            logger.warning(f"No data for {metric}")
            return None
    
    except requests.exceptions.RequestException as e:
        if retry_count < MAX_RETRIES:
            logger.warning(f"Request failed for {metric}, retry {retry_count+1}/{MAX_RETRIES}: {e}")
            time.sleep(RETRY_DELAY)
            return fetch_cortex_data(metric, start_time, end_time, step, retry_count+1)
        else:
            logger.error(f"Max retries exceeded for {metric}: {e}")
            return None
    
    except Exception as e:
        logger.error(f"Error processing {metric}: {e}")
        return None

def main():
    logger.info("="*80)
    logger.info("🚀 FETCHING 2.5 YEARS OF 1-MINUTE DATA (WITH RATE LIMITING)")
    logger.info("="*80)
    logger.info(f"Symbol: {SYMBOL}")
    logger.info(f"Step: 1 minute")
    logger.info(f"Metrics: {len(METRICS)}")
    logger.info(f"Rate Limiting: {REQUEST_DELAY}s/request, {CHUNK_DELAY}s/chunk")
    logger.info(f"Retries: {MAX_RETRIES} per request")
    logger.info("")
    
    # Date range: April 2023 → Today
    end_time = datetime.now()
    start_time = datetime(2023, 4, 1)
    
    logger.info(f"📅 Full Range:")
    logger.info(f"   Start: {start_time}")
    logger.info(f"   End:   {end_time}")
    logger.info(f"   Duration: {(end_time - start_time).days} days")
    logger.info("")
    
    # Split into 1-week chunks (smaller to avoid overload)
    chunk_duration = timedelta(days=7)
    total_chunks = int((end_time - start_time) / chunk_duration) + 1
    
    logger.info(f"📦 Fetching in {total_chunks} chunks (7-day each)")
    logger.info(f"   Expected time: ~{total_chunks * len(METRICS) * REQUEST_DELAY / 60:.0f} minutes")
    logger.info("")
    
    output_dir = Path(__file__).parent / 'artifacts' / 'historical_data' / '1min_chunks'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    successful_chunks = 0
    failed_chunks = []
    
    for chunk_idx in range(total_chunks):
        chunk_start = start_time + (chunk_idx * chunk_duration)
        chunk_end = min(chunk_start + chunk_duration, end_time)
        
        logger.info("="*80)
        logger.info(f"CHUNK {chunk_idx+1}/{total_chunks}: {chunk_start.date()} → {chunk_end.date()}")
        logger.info("="*80)
        
        chunk_data = {}
        success_count = 0
        
        for metric_idx, metric in enumerate(METRICS, 1):
            df_metric = fetch_cortex_data(metric, chunk_start, chunk_end, '1m')
            
            if df_metric is not None and not df_metric.empty:
                chunk_data[metric] = df_metric
                success_count += 1
                logger.info(f"   [{metric_idx}/{len(METRICS)}] {metric}... ✅ {len(df_metric):,} samples")
            else:
                logger.info(f"   [{metric_idx}/{len(METRICS)}] {metric}... ❌")
        
        # Save chunk if we got data
        if chunk_data:
            df_combined = pd.concat(chunk_data.values(), axis=1)
            chunk_file = output_dir / f"chunk_{chunk_idx:03d}_{chunk_start.strftime('%Y%m%d')}_{chunk_end.strftime('%Y%m%d')}.parquet"
            df_combined.to_parquet(chunk_file)
            
            logger.info(f"\n✅ Saved chunk {chunk_idx+1}: {len(df_combined):,} samples, {success_count}/{len(METRICS)} metrics")
            logger.info(f"   {chunk_file}")
            successful_chunks += 1
        else:
            logger.warning(f"\n❌ Failed to fetch chunk {chunk_idx+1}")
            failed_chunks.append(chunk_idx+1)
        
        # Rate limit between chunks
        if chunk_idx < total_chunks - 1:
            time.sleep(CHUNK_DELAY)
    
    logger.info("\n" + "="*80)
    logger.info("📊 FETCH SUMMARY")
    logger.info("="*80)
    logger.info(f"Successful: {successful_chunks}/{total_chunks} chunks")
    if failed_chunks:
        logger.info(f"Failed: {failed_chunks}")
    
    # Combine all chunks
    if successful_chunks > 0:
        logger.info("\n🔄 Combining chunks...")
        all_chunks = []
        for chunk_file in sorted(output_dir.glob("chunk_*.parquet")):
            df_chunk = pd.read_parquet(chunk_file)
            all_chunks.append(df_chunk)
            logger.info(f"   ✅ {chunk_file.name}: {len(df_chunk):,} samples")
        
        df_full = pd.concat(all_chunks, axis=0)
        df_full = df_full.sort_index()
        df_full = df_full[~df_full.index.duplicated(keep='first')]
        
        # Clean data
        logger.info(f"\n🧹 Cleaning data...")
        logger.info(f"   Before: {len(df_full):,} samples")
        
        # Drop rows with too many NaNs
        threshold = len(df_full.columns) * 0.5
        df_full = df_full.dropna(thresh=threshold)
        
        # Fill remaining NaNs
        df_full = df_full.ffill().bfill()
        df_full = df_full.replace([float('inf'), float('-inf')], float('nan'))
        df_full = df_full.fillna(0)
        
        logger.info(f"   After: {len(df_full):,} samples")
        
        # Save
        output_file = Path(__file__).parent / 'artifacts' / 'historical_data' / 'scalping_1min_full.parquet'
        df_full.to_parquet(output_file)
        
        logger.info(f"\n💾 Final dataset saved:")
        logger.info(f"   {output_file}")
        logger.info(f"   Samples: {len(df_full):,}")
        logger.info(f"   Features: {len(df_full.columns)}")
        logger.info(f"   Date range: {df_full.index.min()} → {df_full.index.max()}")
        logger.info(f"   Duration: {(df_full.index.max() - df_full.index.min()).days} days")
        
        logger.info("\n✅ READY TO TRAIN!")
    else:
        logger.error("\n❌ NO DATA FETCHED - Check Cortex connection")
    
    logger.info("\n" + "="*80)

if __name__ == "__main__":
    main()

