#!/usr/bin/env python3
"""
Fetch 1-MINUTE granularity data from Cortex for scalping
This gives us 1,440 samples per day (vs 96 with 15-min data)
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cortex configuration
CORTEX_BASE_URL = "http://10.1.20.60:9009"
CORTEX_API = "/prometheus/api/v1/query_range"
SYMBOL = "BTCUSDT"

# Essential metrics for scalping
METRICS = [
    # Spot price - CRITICAL
    'crypto_last_price',
    
    # Short-term derivatives (velocity indicators)
    'job:crypto_last_price:deriv5m',
    'job:crypto_last_price:deriv10m',
    'job:crypto_last_price:deriv15m',
    'job:crypto_last_price:deriv30m',
    'job:crypto_last_price:deriv1h',
    
    # Short-term averages (trend indicators)
    'job:crypto_last_price:avg5m',
    'job:crypto_last_price:avg10m',
    'job:crypto_last_price:avg15m',
    'job:crypto_last_price:avg30m',
    'job:crypto_last_price:avg1h',
    
    # Volume (liquidity indicators - CRITICAL for scalping)
    'crypto_volume',
    'job:crypto_volume:deriv5m',
    'job:crypto_volume:deriv15m',
    'job:crypto_volume:avg5m',
    'job:crypto_volume:avg15m',
]

def fetch_cortex_data(metric, start_time, end_time, step='1m'):
    """Fetch data from Cortex with specified step size."""
    url = f"{CORTEX_BASE_URL}{CORTEX_API}"
    
    query = f'{metric}{{symbol="{SYMBOL}"}}'
    
    params = {
        'query': query,
        'start': int(start_time.timestamp()),
        'end': int(end_time.timestamp()),
        'step': step
    }
    
    try:
        response = requests.get(url, params=params, timeout=120)
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

def fetch_chunk(start, end, metrics):
    """Fetch all metrics for a time chunk."""
    chunk_data = {}
    
    for i, metric in enumerate(metrics, 1):
        df = fetch_cortex_data(metric, start, end, step='1m')
        if df is not None and len(df) > 0:
            chunk_data[metric] = df
            logger.info(f"      [{i}/{len(metrics)}] {metric}: ✅ {len(df):,} samples")
        else:
            logger.info(f"      [{i}/{len(metrics)}] {metric}: ❌ No data")
        
        time.sleep(0.3)  # Rate limiting
    
    return chunk_data

def main():
    logger.info("="*80)
    logger.info("🚀 FETCHING 1-MINUTE GRANULARITY DATA FOR SCALPING")
    logger.info("="*80)
    logger.info(f"Symbol: {SYMBOL}")
    logger.info(f"Step: 1 minute")
    logger.info(f"Metrics: {len(METRICS)}")
    logger.info("")
    
    # Fetch last 30 days of 1-minute data
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    logger.info(f"📅 Fetching Period:")
    logger.info(f"   Start: {start_time}")
    logger.info(f"   End:   {end_time}")
    logger.info(f"   Duration: 30 days")
    logger.info(f"   Expected samples: ~{30 * 24 * 60:,} per metric (1-min intervals)")
    logger.info("")
    
    # Split into 7-day chunks to avoid timeout
    chunks = []
    current_start = start_time
    while current_start < end_time:
        current_end = min(current_start + timedelta(days=7), end_time)
        chunks.append((current_start, current_end))
        current_start = current_end
    
    logger.info(f"📦 Split into {len(chunks)} chunks (7 days each)")
    logger.info("")
    
    all_chunk_data = []
    
    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks, 1):
        logger.info(f"   Chunk {chunk_idx}/{len(chunks)}: {chunk_start.date()} → {chunk_end.date()}")
        
        chunk_data = fetch_chunk(chunk_start, chunk_end, METRICS)
        
        if chunk_data:
            # Combine metrics for this chunk
            df_chunk = pd.concat(chunk_data.values(), axis=1)
            df_chunk.columns = chunk_data.keys()
            all_chunk_data.append(df_chunk)
            logger.info(f"      ✅ Chunk complete: {df_chunk.shape}")
        else:
            logger.warning(f"      ⚠️  No data for this chunk")
        
        logger.info("")
    
    if all_chunk_data:
        # Combine all chunks
        logger.info(f"🔗 Combining {len(all_chunk_data)} chunks...")
        df_combined = pd.concat(all_chunk_data)
        df_combined = df_combined.sort_index()
        
        # Remove duplicates
        df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
        
        logger.info(f"   Combined shape: {df_combined.shape}")
        logger.info(f"   Date range: {df_combined.index.min()} → {df_combined.index.max()}")
        logger.info(f"   Duration: {(df_combined.index.max() - df_combined.index.min()).days} days")
        
        # Clean data
        logger.info(f"\n🧹 Cleaning data...")
        before_clean = len(df_combined)
        
        # Remove rows with too many NaNs
        nan_threshold = 0.5
        df_combined = df_combined.dropna(thresh=int(len(df_combined.columns) * nan_threshold))
        
        # Fill remaining NaNs
        df_combined = df_combined.ffill().bfill()
        
        # Remove inf values
        df_combined = df_combined.replace([float('inf'), float('-inf')], float('nan'))
        df_combined = df_combined.fillna(0)
        
        after_clean = len(df_combined)
        logger.info(f"   Removed {before_clean - after_clean:,} rows")
        logger.info(f"   Clean samples: {after_clean:,}")
        
        # Verify granularity
        time_diffs = df_combined.index.to_series().diff()
        time_diffs_seconds = time_diffs.dt.total_seconds()
        most_common = time_diffs_seconds.mode().values[0] if len(time_diffs_seconds.mode()) > 0 else None
        
        logger.info(f"\n⏱️  Granularity Check:")
        logger.info(f"   Most common interval: {most_common:.0f} seconds ({most_common/60:.1f} minutes)")
        logger.info(f"   Min interval: {time_diffs_seconds.min():.0f} seconds")
        logger.info(f"   Max interval: {time_diffs_seconds.max():.0f} seconds")
        
        # Save to parquet
        output_dir = Path(__file__).parent / 'artifacts' / 'historical_data'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / 'scalping_1min_30days.parquet'
        df_combined.to_parquet(output_path)
        
        logger.info(f"\n💾 Saved to: {output_path}")
        logger.info(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Statistics
        logger.info(f"\n📊 Data Statistics:")
        logger.info(f"   Total samples: {len(df_combined):,}")
        logger.info(f"   Samples per day: {len(df_combined) / ((df_combined.index.max() - df_combined.index.min()).days):.0f}")
        logger.info(f"   Columns: {len(df_combined.columns)}")
        logger.info(f"   Completeness: {(1 - df_combined.isna().sum().sum() / (df_combined.shape[0] * df_combined.shape[1])):.1%}")
        
        logger.info(f"\n✅ FETCH COMPLETE!")
        logger.info(f"\n📋 Summary:")
        logger.info(f"   • 1-minute granularity ✅")
        logger.info(f"   • 30 days of data ✅")
        logger.info(f"   • {len(df_combined):,} samples ✅")
        logger.info(f"   • {len(df_combined.columns)} features ✅")
        logger.info(f"\n🎯 Next: Train scalping model on 1-minute data")
    else:
        logger.error(f"❌ No data fetched")

if __name__ == "__main__":
    main()

