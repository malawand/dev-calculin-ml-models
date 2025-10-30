#!/usr/bin/env python3
"""
Fetch 30-second granularity data from Cortex
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

# Metrics to fetch
METRICS = [
    # Spot price
    'crypto_last_price',
    
    # Key derivatives for scalping (short-term)
    'job:crypto_last_price:deriv5m',
    'job:crypto_last_price:deriv10m',
    'job:crypto_last_price:deriv15m',
    'job:crypto_last_price:deriv30m',
    'job:crypto_last_price:deriv1h',
    'job:crypto_last_price:deriv2h',
    'job:crypto_last_price:deriv4h',
    
    # Key averages for scalping
    'job:crypto_last_price:avg5m',
    'job:crypto_last_price:avg10m',
    'job:crypto_last_price:avg15m',
    'job:crypto_last_price:avg30m',
    'job:crypto_last_price:avg1h',
    'job:crypto_last_price:avg2h',
    
    # Volume (important for scalping)
    'crypto_volume',
    'job:crypto_volume:deriv5m',
    'job:crypto_volume:deriv15m',
    'job:crypto_volume:deriv30m',
    'job:crypto_volume:avg5m',
    'job:crypto_volume:avg15m',
    'job:crypto_volume:avg30m',
]

def fetch_cortex_data(metric, start_time, end_time, step='30s'):
    """Fetch data from Cortex with specified step size."""
    url = f"{CORTEX_BASE_URL}{CORTEX_API}"
    
    query = f'{metric}{{symbol="{SYMBOL}"}}'
    
    params = {
        'query': query,
        'start': int(start_time.timestamp()),
        'end': int(end_time.timestamp()),
        'step': step  # 30 seconds
    }
    
    try:
        response = requests.get(url, params=params, timeout=60)
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

def main():
    logger.info("="*80)
    logger.info("🚀 FETCHING 30-SECOND GRANULARITY DATA FROM CORTEX")
    logger.info("="*80)
    logger.info(f"Symbol: {SYMBOL}")
    logger.info(f"Step: 30 seconds")
    logger.info(f"Metrics: {len(METRICS)}")
    logger.info("")
    
    # Test with recent data first (last 7 days)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    logger.info(f"📅 Test Fetch:")
    logger.info(f"   Start: {start_time}")
    logger.info(f"   End:   {end_time}")
    logger.info(f"   Expected samples: ~{7 * 24 * 60 * 2:,} (7 days at 30s intervals)")
    logger.info("")
    
    # Test with first metric
    test_metric = 'crypto_last_price'
    logger.info(f"🧪 Testing with {test_metric}...")
    
    df_test = fetch_cortex_data(test_metric, start_time, end_time, step='30s')
    
    if df_test is not None and len(df_test) > 0:
        logger.info(f"   ✅ SUCCESS!")
        logger.info(f"   Samples received: {len(df_test):,}")
        logger.info(f"   Date range: {df_test.index.min()} → {df_test.index.max()}")
        
        # Check actual granularity
        time_diffs = df_test.index.to_series().diff()
        time_diffs_seconds = time_diffs.dt.total_seconds()
        most_common_interval = time_diffs_seconds.mode().values[0] if len(time_diffs_seconds.mode()) > 0 else None
        
        logger.info(f"   Actual interval: {most_common_interval:.0f} seconds")
        
        if most_common_interval == 30:
            logger.info(f"   ✅ Confirmed 30-second granularity!")
        else:
            logger.warning(f"   ⚠️  Interval is {most_common_interval:.0f}s, not 30s")
            logger.info(f"   Cortex may be aggregating to {most_common_interval:.0f}s minimum")
        
        # Now fetch all metrics
        logger.info(f"\n📥 Fetching all {len(METRICS)} metrics...")
        
        all_data = {}
        for i, metric in enumerate(METRICS, 1):
            logger.info(f"   [{i}/{len(METRICS)}] Fetching {metric}...")
            df = fetch_cortex_data(metric, start_time, end_time, step='30s')
            if df is not None and len(df) > 0:
                all_data[metric] = df
                logger.info(f"      ✅ {len(df):,} samples")
            else:
                logger.warning(f"      ❌ No data")
            time.sleep(0.5)  # Rate limiting
        
        if all_data:
            # Combine all metrics
            logger.info(f"\n🔗 Combining {len(all_data)} metrics...")
            df_combined = pd.concat(all_data.values(), axis=1)
            df_combined.columns = all_data.keys()
            
            logger.info(f"   Combined shape: {df_combined.shape}")
            logger.info(f"   Columns: {len(df_combined.columns)}")
            logger.info(f"   Samples: {len(df_combined):,}")
            
            # Save to parquet
            output_dir = Path(__file__).parent / 'artifacts' / 'historical_data'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / 'scalping_30s_data.parquet'
            df_combined.to_parquet(output_path)
            
            logger.info(f"\n💾 Saved to: {output_path}")
            logger.info(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
            
            # Show statistics
            logger.info(f"\n📊 Data Statistics:")
            logger.info(f"   Samples: {len(df_combined):,}")
            logger.info(f"   Duration: {(df_combined.index.max() - df_combined.index.min()).total_seconds() / 3600:.1f} hours")
            logger.info(f"   Completeness: {(1 - df_combined.isna().sum().sum() / (df_combined.shape[0] * df_combined.shape[1])):.1%}")
            
            logger.info(f"\n✅ FETCH COMPLETE!")
            logger.info(f"\nNext steps:")
            logger.info(f"   1. Verify data quality")
            logger.info(f"   2. Fetch more historical data (30 days, 90 days, etc.)")
            logger.info(f"   3. Train scalping model on 30s data")
        else:
            logger.error(f"❌ No data fetched for any metric")
    else:
        logger.error(f"❌ Test fetch failed - 30s data may not be available")
        logger.info(f"\nTrying with 1-minute step instead...")
        
        df_test_1m = fetch_cortex_data(test_metric, start_time, end_time, step='1m')
        if df_test_1m is not None and len(df_test_1m) > 0:
            logger.info(f"   ✅ 1-minute data available!")
            logger.info(f"   Samples: {len(df_test_1m):,}")
            logger.info(f"\nRecommendation: Use 1-minute intervals for scalping")
        else:
            logger.error(f"   ❌ Even 1-minute data not available")
            logger.info(f"\nRecommendation: Stick with 15-minute data or query exchange directly")

if __name__ == "__main__":
    main()



