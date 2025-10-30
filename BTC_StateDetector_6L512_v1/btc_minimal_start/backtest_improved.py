#!/usr/bin/env python3
"""
Backtest Improved Model
Compatible with 44-feature advanced model
"""
import sys
sys.path.insert(0, str(__file__).replace('backtest_improved.py', 'train_improved_model.py'))

from train_improved_model import fetch_data_with_volume, compute_advanced_features
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score

print("="*80)
print("🔬 BACKTESTING IMPROVED MODEL (Multiple Windows)")
print("="*80)
print()

# Load model
models_dir = Path(__file__).parent / 'models'

with open(models_dir / 'scalping_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(models_dir / 'scalping_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open(models_dir / 'scalping_config.json', 'r') as f:
    config = json.load(f)

with open(models_dir / 'feature_names.json', 'r') as f:
    feature_names = json.load(f)

print(f"📥 Model: {config['horizon']} @ ±{config['threshold']}%")
print(f"   Features: {len(feature_names)}")
print()

# Parse horizon
horizon_str = config['horizon']
threshold = config['threshold']

if 'min' in horizon_str:
    horizon_minutes = int(horizon_str.replace('min', ''))
else:
    horizon_minutes = 15

# Backtest windows
windows = [('4h', 4), ('8h', 8), ('12h', 12), ('24h', 24), ('48h', 48)]

for window_name, window_hours in windows:
    print(f"\n{'='*80}")
    print(f"📊 BACKTEST: {window_name} ({window_hours} hours)")
    print(f"{'='*80}")
    
    # Fetch data
    df = fetch_data_with_volume(hours=window_hours + 24)  # Extra for lookback
    
    if df is None or len(df) < 500:
        print(f"❌ Not enough data for {window_name}")
        continue
    
    # Test on recent window
    end_time = df.index.max()
    start_time = end_time - timedelta(hours=window_hours)
    df_window = df[df.index >= start_time]
    
    print(f"   Data: {len(df_window)} samples")
    print(f"   Range: {start_time.strftime('%H:%M')} → {end_time.strftime('%H:%M')}")
    
    # Backtest
    results = []
    
    for i in range(240, len(df) - horizon_minutes, 15):  # Test every 15 min
        if df.index[i] < start_time:
            continue
        
        features_dict = compute_advanced_features(df, i)
        if features_dict is None:
            continue
        
        # Convert to array in correct order
        features = np.array([[features_dict[fn] for fn in feature_names]])
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Actual outcome
        current_price = df.iloc[i]['price']
        future_price = df.iloc[i + horizon_minutes]['price']
        change_pct = (future_price - current_price) / current_price * 100
        
        if change_pct > threshold:
            actual = 2  # UP
        elif change_pct < -threshold:
            actual = 0  # DOWN
        else:
            actual = 1  # SIDEWAYS
        
        results.append({
            'predicted': prediction,
            'actual': actual,
            'correct': prediction == actual,
            'confidence': probabilities[prediction],
            'change_pct': change_pct
        })
    
    if not results:
        print(f"   ❌ No predictions made")
        continue
    
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    overall_acc = results_df['correct'].mean()
    
    # Per-class
    pred_counts = results_df['predicted'].value_counts()
    up_pred = results_df[results_df['predicted'] == 2]
    down_pred = results_df[results_df['predicted'] == 0]
    sideways_pred = results_df[results_df['predicted'] == 1]
    
    up_acc = up_pred['correct'].mean() if len(up_pred) > 0 else 0
    down_acc = down_pred['correct'].mean() if len(down_pred) > 0 else 0
    sideways_acc = sideways_pred['correct'].mean() if len(sideways_pred) > 0 else 0
    
    # Directional
    directional = results_df[results_df['predicted'] != 1]
    dir_acc = directional['correct'].mean() if len(directional) > 0 else 0
    
    # Trading signals
    trading = results_df[(results_df['predicted'] != 1) & (results_df['confidence'] > 0.70)]
    trade_acc = trading['correct'].mean() if len(trading) > 0 else 0
    
    # High confidence
    high_conf = results_df[results_df['confidence'] > 0.80]
    high_conf_acc = high_conf['correct'].mean() if len(high_conf) > 0 else 0
    
    print()
    print(f"   📊 Results ({len(results)} predictions):")
    print(f"      Overall Accuracy:     {overall_acc:.1%}")
    print(f"      Directional Accuracy: {dir_acc:.1%}")
    print(f"      Trading Accuracy:     {trade_acc:.1%} ({len(trading)} signals)")
    print(f"      High-Conf Accuracy:   {high_conf_acc:.1%} ({len(high_conf)} signals)")
    print()
    print(f"   📈 Predictions:")
    print(f"      UP:       {pred_counts.get(2, 0):3d} ({up_acc:.1%} correct)")
    print(f"      DOWN:     {pred_counts.get(0, 0):3d} ({down_acc:.1%} correct)")
    print(f"      SIDEWAYS: {pred_counts.get(1, 0):3d} ({sideways_acc:.1%} correct)")
    
    # Grade
    if trade_acc >= 0.70:
        grade = "✅ EXCELLENT - Trade with confidence!"
    elif trade_acc >= 0.60:
        grade = "✅ GOOD - Trade with normal size"
    elif trade_acc >= 0.50:
        grade = "⚠️  OK - Trade with reduced size"
    else:
        grade = "❌ POOR - Don't trade"
    
    print(f"\n   {grade}")

print()
print("="*80)
print("✅ BACKTEST COMPLETE!")
print("="*80)



