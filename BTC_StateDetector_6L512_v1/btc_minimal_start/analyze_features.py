#!/usr/bin/env python3
"""
Analyze what features are actually available in the training data.
"""
import sys
sys.path.insert(0, '/Users/mazenlawand/Documents/Caculin ML/btc_direction_predictor')

import pandas as pd
import numpy as np
from pathlib import Path
from src.features.engineering import FeatureEngineer
from src.labels.labels import LabelCreator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    horizon = '24h'
    
    logger.info("=" * 80)
    logger.info("📊 ANALYZING ALL AVAILABLE FEATURES")
    logger.info("=" * 80)
    
    # Load raw data
    data_path = Path('/Users/mazenlawand/Documents/Caculin ML/btc_direction_predictor/artifacts/historical_data/combined_full_dataset.parquet')
    logger.info(f"📥 Loading: {data_path}")
    df_raw = pd.read_parquet(data_path)
    
    if 'timestamp' in df_raw.columns:
        df_raw = df_raw.set_index('timestamp')
    df_raw.index = pd.to_datetime(df_raw.index)
    df_raw = df_raw.sort_index()
    
    if 'crypto_last_price' in df_raw.columns:
        df_raw = df_raw.rename(columns={'crypto_last_price': 'price'})
    
    logger.info(f"   Loaded {len(df_raw):,} samples with {len(df_raw.columns)} raw columns")
    
    # Engineer features
    logger.info("🔧 Engineering features...")
    config = {
        'features': {
            'price_lags': [1, 3, 6, 12],
            'deriv_lags': [1, 3, 6],
            'rolling_windows': [12, 24, 72]
        },
        'prometheus': {
            'metrics': {
                'spot': ['crypto_last_price'],
                'averages': [],
                'derivatives': [],
                'derivative_primes': []
            }
        },
        'price_col': 'price',
        'target_horizons': [horizon]
    }
    
    feature_engineer = FeatureEngineer(config)
    df_engineered = feature_engineer.engineer(df_raw.copy())
    
    logger.info(f"   Engineered {len(df_engineered.columns)} features")
    
    # Create labels
    logger.info("🎯 Creating labels...")
    config['labels'] = {
        'horizons': [horizon],
        'threshold_pct': 0.0
    }
    label_creator = LabelCreator(config)
    df_labeled = label_creator.create_labels(df_engineered)
    
    # Get all features
    leakage_patterns = ['return_24', 'return_72', 'return_96', 'lag1_24h', 'lag3_24h', 'lag6_24h', 'lag12_24h']
    
    all_features = []
    for col in df_labeled.columns:
        if col.startswith('label_') or col == 'crypto_last_price' or col == 'timestamp' or col == 'price':
            continue
        is_leakage = any(pattern in col for pattern in leakage_patterns)
        if not is_leakage:
            all_features.append(col)
    
    logger.info(f"   Total features: {len(all_features)}")
    
    # Categorize features
    avg_features = sorted([f for f in all_features if 'avg' in f.lower()])
    deriv_features = sorted([f for f in all_features if 'deriv' in f.lower() and 'prime' not in f.lower()])
    prime_features = sorted([f for f in all_features if 'prime' in f.lower()])
    roc_features = sorted([f for f in all_features if '_roc' in f])
    lag_features = sorted([f for f in all_features if '_lag' in f])
    spread_features = sorted([f for f in all_features if 'spread' in f])
    zscore_features = sorted([f for f in all_features if 'zscore' in f])
    volatility_features = sorted([f for f in all_features if 'volatility' in f])
    return_features = sorted([f for f in all_features if 'return' in f])
    momentum_features = sorted([f for f in all_features if 'momentum' in f.lower()])
    velocity_features = sorted([f for f in all_features if 'velocity' in f.lower() or 'accel' in f.lower()])
    divergence_features = sorted([f for f in all_features if 'diverg' in f.lower()])
    coherence_features = sorted([f for f in all_features if 'coherence' in f.lower()])
    sign_features = sorted([f for f in all_features if 'sign' in f.lower()])
    
    # Get remaining features
    categorized = (avg_features + deriv_features + prime_features + roc_features + 
                  lag_features + spread_features + zscore_features + volatility_features + 
                  return_features + momentum_features + velocity_features + divergence_features +
                  coherence_features + sign_features)
    other_features = sorted([f for f in all_features if f not in categorized])
    
    logger.info("\n" + "=" * 80)
    logger.info("📊 FEATURE BREAKDOWN")
    logger.info("=" * 80)
    
    print(f"\n✅ Moving Average features: {len(avg_features)}")
    for f in avg_features[:10]:
        print(f"   - {f}")
    if len(avg_features) > 10:
        print(f"   ... and {len(avg_features) - 10} more")
    
    print(f"\n✅ Derivative features: {len(deriv_features)}")
    for f in deriv_features[:10]:
        print(f"   - {f}")
    if len(deriv_features) > 10:
        print(f"   ... and {len(deriv_features) - 10} more")
    
    print(f"\n✅ Derivative Prime features: {len(prime_features)}")
    for f in prime_features[:10]:
        print(f"   - {f}")
    if len(prime_features) > 10:
        print(f"   ... and {len(prime_features) - 10} more")
    
    print(f"\n✅ ROC features: {len(roc_features)}")
    for f in roc_features[:10]:
        print(f"   - {f}")
    if len(roc_features) > 10:
        print(f"   ... and {len(roc_features) - 10} more")
    
    print(f"\n✅ Spread features: {len(spread_features)}")
    for f in spread_features[:5]:
        print(f"   - {f}")
    if len(spread_features) > 5:
        print(f"   ... and {len(spread_features) - 5} more")
    
    print(f"\n✅ Z-score features: {len(zscore_features)}")
    print(f"\n✅ Volatility features: {len(volatility_features)}")
    print(f"\n✅ Return features: {len(return_features)}")
    print(f"\n✅ Lag features: {len(lag_features)}")
    print(f"\n✅ Momentum features: {len(momentum_features)}")
    print(f"\n✅ Velocity/Accel features: {len(velocity_features)}")
    print(f"\n✅ Divergence features: {len(divergence_features)}")
    print(f"\n✅ Coherence features: {len(coherence_features)}")
    print(f"\n✅ Sign features: {len(sign_features)}")
    print(f"\n✅ Other features: {len(other_features)}")
    if other_features:
        for f in other_features[:20]:
            print(f"   - {f}")
        if len(other_features) > 20:
            print(f"   ... and {len(other_features) - 20} more")
    
    print("\n" + "=" * 80)
    print(f"TOTAL: {len(all_features)} features")
    print("=" * 80)
    
    # Now compute correlations with label
    logger.info("\n🔍 Computing correlations with label...")
    df_test = df_labeled[all_features + [f'label_{horizon}']].dropna()
    correlations = df_test[all_features].corrwith(df_test[f'label_{horizon}']).abs().sort_values(ascending=False)
    
    print("\n" + "=" * 80)
    print("📊 TOP 50 FEATURES BY CORRELATION WITH LABEL")
    print("=" * 80)
    
    for i, (feature, corr) in enumerate(correlations.head(50).items(), 1):
        # Categorize feature
        if 'avg' in feature.lower():
            cat = "AVG"
        elif 'prime' in feature.lower():
            cat = "PRIME"
        elif 'deriv' in feature.lower():
            cat = "DERIV"
        elif 'roc' in feature:
            cat = "ROC"
        elif 'volatility' in feature:
            cat = "VOL"
        elif 'momentum' in feature.lower():
            cat = "MOM"
        elif 'spread' in feature:
            cat = "SPREAD"
        else:
            cat = "OTHER"
        
        print(f"  {i:2d}. [{cat:6s}] {feature:50s} {corr:.6f}")
    
    print("\n" + "=" * 80)
    print("🎯 ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()



