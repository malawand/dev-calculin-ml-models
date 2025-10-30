#!/usr/bin/env python3
"""
Discover all volume metrics available in Cortex.
Query for job:crypto_volume:* metrics.
"""
import requests
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    cortex_url = config['cortex']['base_url']
    cortex_path = config['cortex']['read_api']
    symbol = config['cortex']['symbol']
    
    base_url = f"{cortex_url}{cortex_path}"
    
    logger.info("=" * 80)
    logger.info("🔍 DISCOVERING VOLUME METRICS IN CORTEX")
    logger.info("=" * 80)
    logger.info(f"Cortex URL: {base_url}")
    logger.info(f"Symbol: {symbol}")
    
    # Query for all metrics matching job:crypto_volume:*
    try:
        # Use label_values API to discover metrics
        label_url = f"{cortex_url}/prometheus/api/v1/label/__name__/values"
        logger.info(f"\n📊 Querying: {label_url}")
        
        response = requests.get(label_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] != 'success':
            logger.error(f"❌ Query failed: {data}")
            return
        
        all_metrics = data['data']
        
        # Filter for volume metrics
        volume_metrics = [m for m in all_metrics if 'crypto_volume' in m]
        
        logger.info(f"\n✅ Found {len(volume_metrics)} volume metrics:")
        logger.info("=" * 80)
        
        for metric in sorted(volume_metrics):
            logger.info(f"   - {metric}")
        
        # Now test if these metrics have data for BTCUSDT
        logger.info(f"\n📊 Checking which metrics have data for {symbol}...")
        logger.info("=" * 80)
        
        available_metrics = []
        
        for metric in volume_metrics:
            try:
                # Query last 1 hour to check if data exists
                query = f'{metric}{{symbol="{symbol}"}}'
                params = {
                    'query': query,
                    'time': 'now'
                }
                
                query_url = f"{cortex_url}/prometheus/api/v1/query"
                resp = requests.get(query_url, params=params, timeout=10)
                resp.raise_for_status()
                result = resp.json()
                
                if result['status'] == 'success' and result['data']['result']:
                    value = result['data']['result'][0]['value'][1]
                    available_metrics.append(metric)
                    logger.info(f"   ✅ {metric:50s} → {value}")
                else:
                    logger.info(f"   ❌ {metric:50s} → No data")
                    
            except Exception as e:
                logger.info(f"   ⚠️  {metric:50s} → Error: {e}")
        
        logger.info("\n" + "=" * 80)
        logger.info(f"📊 SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total volume metrics found: {len(volume_metrics)}")
        logger.info(f"Available for {symbol}: {len(available_metrics)}")
        
        if available_metrics:
            logger.info(f"\n✅ Available volume metrics for {symbol}:")
            for metric in sorted(available_metrics):
                logger.info(f"   - {metric}")
            
            # Save to file
            output_file = Path(__file__).parent / 'artifacts' / 'available_volume_metrics.txt'
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                f.write(f"# Available volume metrics for {symbol}\n")
                f.write(f"# Discovered: {len(available_metrics)} metrics\n\n")
                for metric in sorted(available_metrics):
                    f.write(f"{metric}\n")
            
            logger.info(f"\n💾 Saved to: {output_file}")
        else:
            logger.warning(f"\n⚠️  No volume metrics found for {symbol}")
            logger.info("   This might mean:")
            logger.info("   1. Volume metrics use a different naming pattern")
            logger.info("   2. Volume data is not being collected")
            logger.info("   3. The symbol name is different in volume metrics")
            
            logger.info(f"\n🔍 Let's check what symbols ARE available in volume metrics...")
            
            # Query for all symbols in crypto_volume metrics
            for metric in volume_metrics[:3]:  # Check first 3 metrics
                try:
                    label_url = f"{cortex_url}/prometheus/api/v1/label/symbol/values?match[]={metric}"
                    resp = requests.get(label_url, timeout=10)
                    resp.raise_for_status()
                    result = resp.json()
                    
                    if result['status'] == 'success':
                        symbols = result['data']
                        logger.info(f"\n   Symbols in {metric}:")
                        for sym in symbols[:10]:  # Show first 10
                            logger.info(f"      - {sym}")
                        if len(symbols) > 10:
                            logger.info(f"      ... and {len(symbols) - 10} more")
                except Exception as e:
                    logger.info(f"   Could not fetch symbols for {metric}: {e}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ DISCOVERY COMPLETE")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Error discovering metrics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

