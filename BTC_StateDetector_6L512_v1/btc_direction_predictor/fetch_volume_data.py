#!/usr/bin/env python3
"""
Fetch volume metrics from Cortex and merge with existing price dataset.
Uses symbol="BTCUSDT" filter for all volume queries.
"""
import pandas as pd
import requests
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_metric_chunk(base_url, metric, symbol, start, end, step='15m'):
    """Fetch a single metric for a given time range."""
    try:
        query = f'{metric}{{symbol="{symbol}"}}'
        params = {
            'query': query,
            'start': start.timestamp(),
            'end': end.timestamp(),
            'step': step
        }
        
        response = requests.get(base_url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] != 'success':
            logger.warning(f"Query failed for {metric}: {data}")
            return None
        
        results = data['data']['result']
        if not results:
            logger.warning(f"No data for {metric}")
            return None
        
        # Convert to DataFrame
        values = results[0]['values']  # Assuming single series
        df = pd.DataFrame(values, columns=['timestamp', metric])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df[metric] = pd.to_numeric(df[metric], errors='coerce')
        
        return df
    
    except Exception as e:
        logger.warning(f"Error fetching {metric}: {e}")
        return None

def main():
    logger.info("=" * 80)
    logger.info("📊 FETCHING VOLUME DATA FROM CORTEX")
    logger.info("=" * 80)
    
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    cortex_url = config['cortex']['base_url']
    cortex_path = config['cortex']['read_api']
    symbol = config['cortex']['symbol']
    
    base_url = f"{cortex_url}{cortex_path}"
    
    logger.info(f"Cortex URL: {base_url}")
    logger.info(f"Symbol: {symbol}")
    
    # Load existing price dataset to get date range
    price_data_path = Path(__file__).parent / 'artifacts/historical_data/combined_full_dataset.parquet'
    
    if not price_data_path.exists():
        logger.error(f"❌ Price dataset not found: {price_data_path}")
        logger.info("   Please run fetch_full_data.py first to get price data.")
        return
    
    logger.info(f"\n📥 Loading existing price dataset...")
    df_price = pd.read_parquet(price_data_path)
    
    if 'timestamp' in df_price.columns:
        df_price = df_price.set_index('timestamp')
    df_price.index = pd.to_datetime(df_price.index, utc=True)
    df_price = df_price.sort_index()
    
    start_date = df_price.index.min()
    end_date = df_price.index.max()
    
    logger.info(f"   Price data: {len(df_price):,} samples")
    logger.info(f"   Date range: {start_date} → {end_date}")
    logger.info(f"   Duration: {(end_date - start_date).days} days")
    
    # Load volume metrics list
    volume_metrics_file = Path(__file__).parent / 'volume_metrics_list.txt'
    with open(volume_metrics_file, 'r') as f:
        all_volume_metrics = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    logger.info(f"\n📊 Will fetch {len(all_volume_metrics)} volume metrics")
    
    # Fetch data in 3-month chunks (same as price data was fetched)
    chunk_months = 3
    chunk_delta = timedelta(days=chunk_months * 30)
    
    all_volume_dfs = []
    chunk_start = start_date
    chunk_num = 1
    
    while chunk_start < end_date:
        chunk_end = min(chunk_start + chunk_delta, end_date)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📦 CHUNK {chunk_num}: {chunk_start.date()} → {chunk_end.date()}")
        logger.info(f"{'='*80}")
        
        chunk_dfs = []
        
        for i, metric in enumerate(all_volume_metrics, 1):
            df_metric = fetch_metric_chunk(base_url, metric, symbol, chunk_start, chunk_end, step='15m')
            
            if df_metric is not None:
                logger.info(f"  [{i}/{len(all_volume_metrics)}] {metric:45s} ✅ {len(df_metric):,} samples")
                chunk_dfs.append(df_metric.set_index('timestamp'))
            else:
                logger.info(f"  [{i}/{len(all_volume_metrics)}] {metric:45s} ❌ No data")
            
            # Rate limit
            time.sleep(0.5)
        
        if chunk_dfs:
            # Merge all metrics for this chunk
            logger.info(f"\n  🔗 Merging {len(chunk_dfs)} metrics for chunk {chunk_num}...")
            chunk_combined = pd.concat(chunk_dfs, axis=1)
            all_volume_dfs.append(chunk_combined)
            logger.info(f"  ✅ Chunk {chunk_num} complete: {len(chunk_combined):,} samples, {len(chunk_combined.columns)} metrics")
        
        chunk_start = chunk_end
        chunk_num += 1
    
    if not all_volume_dfs:
        logger.error("\n❌ No volume data fetched!")
        return
    
    # Combine all chunks
    logger.info(f"\n{'='*80}")
    logger.info(f"🔗 COMBINING ALL CHUNKS...")
    logger.info(f"{'='*80}")
    
    df_volume_full = pd.concat(all_volume_dfs, axis=0)
    df_volume_full = df_volume_full.sort_index()
    
    # Remove duplicates (in case of overlapping chunks)
    df_volume_full = df_volume_full[~df_volume_full.index.duplicated(keep='first')]
    
    logger.info(f"\n📊 Volume data fetched:")
    logger.info(f"   Samples: {len(df_volume_full):,}")
    logger.info(f"   Metrics: {len(df_volume_full.columns)}")
    logger.info(f"   Date range: {df_volume_full.index.min()} → {df_volume_full.index.max()}")
    
    # Merge with price data
    logger.info(f"\n🔗 Merging volume data with price data...")
    logger.info(f"   Price data: {len(df_price):,} samples, {len(df_price.columns)} columns")
    logger.info(f"   Volume data: {len(df_volume_full):,} samples, {len(df_volume_full.columns)} columns")
    
    # Merge on timestamp index
    df_combined = df_price.join(df_volume_full, how='inner')
    
    logger.info(f"\n✅ Merged dataset:")
    logger.info(f"   Samples: {len(df_combined):,}")
    logger.info(f"   Total features: {len(df_combined.columns)}")
    logger.info(f"   Price features: {len(df_price.columns)}")
    logger.info(f"   Volume features: {len(df_volume_full.columns)}")
    logger.info(f"   Date range: {df_combined.index.min()} → {df_combined.index.max()}")
    
    # Check for missing data
    logger.info(f"\n📊 DATA QUALITY:")
    missing_pct = df_combined.isnull().sum().sum() / (df_combined.shape[0] * df_combined.shape[1])
    logger.info(f"   Missing data: {missing_pct:.1%}")
    
    # Show which columns have the most missing data
    missing_by_col = df_combined.isnull().sum().sort_values(ascending=False)
    high_missing = missing_by_col[missing_by_col > len(df_combined) * 0.1]  # >10% missing
    
    if len(high_missing) > 0:
        logger.info(f"\n⚠️  Columns with >10% missing data:")
        for col, count in high_missing.head(10).items():
            pct = count / len(df_combined) * 100
            logger.info(f"   {col:50s} {pct:5.1f}% missing")
    else:
        logger.info(f"   ✅ All columns have <10% missing data")
    
    # Save combined dataset
    output_path = Path(__file__).parent / 'artifacts/historical_data/combined_with_volume.parquet'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n💾 Saving combined dataset to: {output_path}")
    df_combined.reset_index().to_parquet(output_path, index=False)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"   File size: {file_size_mb:.1f} MB")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ VOLUME DATA FETCH COMPLETE!")
    logger.info(f"{'='*80}")
    logger.info(f"\nNext steps:")
    logger.info(f"1. Update feature engineering to include volume features")
    logger.info(f"2. Retrain model with: python btc_minimal_start/incremental_simple.py")
    logger.info(f"3. Compare accuracy with/without volume data")
    
    logger.info(f"\nDataset summary:")
    logger.info(f"  • {len(df_combined):,} samples ({(df_combined.index.max() - df_combined.index.min()).days} days)")
    logger.info(f"  • {len(df_price.columns)} price features")
    logger.info(f"  • {len(df_volume_full.columns)} volume features")
    logger.info(f"  • {len(df_combined.columns)} total features")
    logger.info(f"  • {missing_pct:.1%} missing data")

if __name__ == "__main__":
    main()

