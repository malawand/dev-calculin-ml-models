#!/usr/bin/env python3
"""
Quick training script - minimal data for testing
Fetches 6 hours of data and trains a single model
"""
import sys
import yaml
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.prom import PrometheusClient
from src.model.trees import LightGBMClassifier
from src.eval.metrics import MetricsCalculator

def quick_train():
    """Quick training with minimal data"""
    
    print("=" * 80)
    print("QUICK TRAINING - Bitcoin Direction Predictor")
    print("=" * 80)
    
    # Load config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Step 1: Fetch minimal data (6 hours)
    print("\n[1/5] Fetching 6 hours of data...")
    client = PrometheusClient(config['cortex']['base_url'])
    
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=6)
    
    # Fetch just spot price and a few key metrics
    metrics_to_fetch = [
        'crypto_last_price',
        'job:crypto_last_price:avg1h',
        'job:crypto_last_price:avg4h',
        'job:crypto_last_price:deriv1h',
        'job:crypto_last_price:deriv4h'
    ]
    
    dfs = []
    for metric in metrics_to_fetch:
        try:
            query = client.build_metric_query(metric, 'BTCUSDT')
            df = client.query_range(query, start, end, '5m')
            if not df.empty:
                df.columns = [metric.replace('job:crypto_last_price:', '').replace('crypto_last_price', 'price')]
                dfs.append(df)
                print(f"  ✓ {metric}: {len(df)} points")
        except Exception as e:
            print(f"  ✗ {metric}: {e}")
    
    if not dfs:
        print("❌ No data fetched. Check Prometheus endpoint.")
        return
    
    df = dfs[0]
    for d in dfs[1:]:
        df = df.join(d, how='outer')
    
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    print(f"\n✓ Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Step 2: Simple features
    print("\n[2/5] Creating features...")
    
    # Returns
    df['return_1'] = df['price'].pct_change()
    df['return_12'] = df['price'].pct_change(12)
    
    # Rolling stats
    df['volatility_12'] = df['return_1'].rolling(12).std()
    df['zscore_12'] = (df['price'] - df['price'].rolling(12).mean()) / df['price'].rolling(12).std()
    
    # Lags
    if 'deriv1h' in df.columns:
        df['deriv1h_lag1'] = df['deriv1h'].shift(1)
    
    df.dropna(inplace=True)
    
    print(f"✓ Features: {df.shape[1]} columns, {df.shape[0]} rows after cleanup")
    
    # Step 3: Create labels
    print("\n[3/5] Creating labels...")
    
    # 1h ahead label (12 steps for 5m bars)
    df['return_1h_future'] = df['price'].pct_change(12).shift(-12)
    df['label'] = (df['return_1h_future'] > 0.0).astype(int)
    
    df.dropna(inplace=True)
    
    up_pct = df['label'].mean()
    print(f"✓ Labels created: {len(df)} samples")
    print(f"  UP: {up_pct:.1%}, DOWN: {1-up_pct:.1%}")
    
    if len(df) < 20:
        print(f"❌ Not enough samples ({len(df)}). Need at least 20.")
        print("   Try: Increase hours or check data availability.")
        return
    
    # Step 4: Train
    print("\n[4/5] Training LightGBM...")
    
    # Prepare data
    feature_cols = [c for c in df.columns if c not in ['label', 'return_1h_future']]
    X = df[feature_cols].values
    y = df['label'].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Simple config for quick training
    quick_config = {
        'modeling': {
            'random_search_iters': 0,  # No search for speed
            'lightgbm': {}
        }
    }
    
    # Train
    model = LightGBMClassifier(quick_config)
    model.train(X_train_scaled, y_train)
    
    print("✓ Training complete")
    
    # Step 5: Evaluate
    print("\n[5/5] Evaluating...")
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    
    calc = MetricsCalculator()
    metrics = calc.calculate_classification_metrics(y_test, y_pred, y_proba)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Accuracy:       {metrics['accuracy']:.2%}")
    print(f"Precision (UP): {metrics['precision_up']:.2%}")
    print(f"Recall (UP):    {metrics['recall_up']:.2%}")
    print(f"F1 (UP):        {metrics['f1_up']:.3f}")
    print(f"MCC:            {metrics['mcc']:.3f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC:        {metrics['roc_auc']:.3f}")
    print("=" * 80)
    
    # Feature importance
    print("\nTop Features:")
    importance = model.get_feature_importance(feature_cols)
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (feat, imp) in enumerate(sorted_imp, 1):
        print(f"  {i}. {feat}: {imp:.4f}")
    
    # Save model
    Path('artifacts/models').mkdir(parents=True, exist_ok=True)
    model.save('artifacts/models/quick_test.pkl')
    print(f"\n✓ Model saved to: artifacts/models/quick_test.pkl")
    
    print("\n" + "=" * 80)
    print("✓ QUICK TRAINING COMPLETE")
    print("=" * 80)
    print("\nInterpretation:")
    if metrics['accuracy'] > 0.55:
        print("  ✓ Good! Above random baseline (50%)")
    elif metrics['accuracy'] > 0.52:
        print("  → OK. Slight edge over random.")
    else:
        print("  ⚠ Near random. Need more data for meaningful results.")
    
    print("\nNext steps:")
    print("  1. Run full training: python -m src.pipeline.train")
    print("  2. Use more data (weeks/months) for better performance")
    print("  3. Check QUICKSTART.md for guidance")

if __name__ == '__main__':
    try:
        quick_train()
    except Exception as e:
        logger.error(f"Quick training failed: {e}", exc_info=True)
        sys.exit(1)
