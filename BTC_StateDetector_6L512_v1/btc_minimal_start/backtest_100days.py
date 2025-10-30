#!/usr/bin/env python3
"""100-Day Backtest"""
import sys
sys.path.insert(0, str(__file__).replace('backtest_100days.py', 'train_improved_model.py'))
from train_improved_model import fetch_data_with_volume, compute_advanced_features
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔬 100-DAY BACKTEST")
print("="*80)

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

print(f"Model: {config['horizon']} @ ±{config['threshold']}%")
print(f"Features: {len(feature_names)}")

# Parse config
horizon_str = config['horizon']
threshold = config['threshold']
if 'min' in horizon_str:
    horizon_minutes = int(horizon_str.replace('min', ''))
else:
    horizon_minutes = 15

# Fetch 100 days + extra for lookback
print(f"\nFetching 110 days of data...")
df = fetch_data_with_volume(hours=110*24)

if df is None or len(df) < 1000:
    print("❌ Not enough data")
    sys.exit(1)

print(f"   ✅ {len(df)} samples")
print(f"   Range: {df.index.min()} → {df.index.max()}")

# Test on last 100 days
end_time = df.index.max()
start_time = end_time - timedelta(days=100)
df_test = df[df.index >= start_time]

print(f"\nBacktesting...")
print(f"   Period: {start_time.strftime('%Y-%m-%d')} → {end_time.strftime('%Y-%m-%d')}")

results = []
for i in range(240, len(df) - horizon_minutes, 60):  # Every hour
    if df.index[i] < start_time:
        continue
    
    features_dict = compute_advanced_features(df, i)
    if features_dict is None:
        continue
    
    features = np.array([[features_dict[fn] for fn in feature_names]])
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
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
        'timestamp': df.index[i],
        'predicted': prediction,
        'actual': actual,
        'correct': prediction == actual,
        'confidence': probabilities[prediction],
        'change_pct': change_pct
    })

if not results:
    print("❌ No predictions made")
    sys.exit(1)

results_df = pd.DataFrame(results)

# Calculate metrics
overall_acc = results_df['correct'].mean()

pred_counts = results_df['predicted'].value_counts()
up_pred = results_df[results_df['predicted'] == 2]
down_pred = results_df[results_df['predicted'] == 0]
sideways_pred = results_df[results_df['predicted'] == 1]

up_acc = up_pred['correct'].mean() if len(up_pred) > 0 else 0
down_acc = down_pred['correct'].mean() if len(down_pred) > 0 else 0
sideways_acc = sideways_pred['correct'].mean() if len(sideways_pred) > 0 else 0

directional = results_df[results_df['predicted'] != 1]
dir_acc = directional['correct'].mean() if len(directional) > 0 else 0

trading = results_df[(results_df['predicted'] != 1) & (results_df['confidence'] > 0.70)]
trade_acc = trading['correct'].mean() if len(trading) > 0 else 0

high_conf = results_df[results_df['confidence'] > 0.80]
high_conf_acc = high_conf['correct'].mean() if len(high_conf) > 0 else 0

print()
print("="*80)
print("📊 RESULTS")
print("="*80)
print(f"\nTotal Predictions: {len(results)}")
print(f"\nOverall Accuracy:     {overall_acc:.1%}")
print(f"Directional Accuracy: {dir_acc:.1%}")
print(f"Trading Accuracy:     {trade_acc:.1%} ({len(trading)} signals)")
print(f"High-Conf Accuracy:   {high_conf_acc:.1%} ({len(high_conf)} signals)")

print(f"\nPredictions:")
print(f"   UP:       {pred_counts.get(2, 0):4d} ({up_acc:.1%} correct)")
print(f"   DOWN:     {pred_counts.get(0, 0):4d} ({down_acc:.1%} correct)")
print(f"   SIDEWAYS: {pred_counts.get(1, 0):4d} ({sideways_acc:.1%} correct)")

# Grade
if trade_acc >= 0.70:
    grade = "✅ EXCELLENT"
elif trade_acc >= 0.60:
    grade = "✅ GOOD"
elif trade_acc >= 0.50:
    grade = "⚠️  OK"
else:
    grade = "❌ POOR"

print(f"\n{grade}")

# Save
output_file = Path(__file__).parent / f'backtest_100days_{datetime.now().strftime("%Y%m%d")}.csv'
results_df.to_csv(output_file, index=False)
print(f"\n💾 Saved: {output_file.name}")
print("="*80)



