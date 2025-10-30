#!/usr/bin/env python3
"""
Backtest Momentum Detection on 2.5 Years

Test the hand-crafted momentum calculator's accuracy:
- When it says momentum is BUILDING, does price actually move?
- When it says direction is UP, does price go up?
- How accurate is the strength measurement?
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add paths
sys.path.append(str(Path(__file__).parent))

from momentum_calculator import MomentumCalculator

print("="*80)
print("📊 MOMENTUM DETECTION - 2.5 YEAR BACKTEST")
print("="*80)
print("Testing hand-crafted momentum calculator accuracy on historical data")
print()

# Load historical data
print("📥 Loading 2.5 years of historical data...")
data_paths = [
    Path(__file__).parent.parent / 'btc_direction_predictor/artifacts/historical_data/combined_with_volume.parquet',
    Path(__file__).parent.parent / 'btc_direction_predictor/artifacts/historical_data/cleaned_2023_2025.parquet',
]

df = None
for data_path in data_paths:
    if data_path.exists():
        print(f"   Found: {data_path.name}")
        df = pd.read_parquet(data_path)
        break

if df is None:
    print("❌ No historical data found!")
    sys.exit(1)

# Prepare data
if 'timestamp' in df.columns:
    df = df.set_index('timestamp')
df.index = pd.to_datetime(df.index)
df = df.sort_index()

if 'crypto_last_price' in df.columns:
    df = df.rename(columns={'crypto_last_price': 'price'})

if 'crypto_volume' in df.columns and 'volume' not in df.columns:
    df['volume'] = df['crypto_volume']
elif 'volume' not in df.columns:
    df['volume'] = 0.0

print(f"   ✅ Loaded {len(df):,} samples")
print(f"   📅 Date range: {df.index.min().strftime('%Y-%m-%d')} → {df.index.max().strftime('%Y-%m-%d')}")
print()

# Test momentum detection
print("🔄 Testing momentum detection accuracy...")
print()

momentum_calc = MomentumCalculator()

lookback = 300  # 5 hours of history needed
sample_every = 30  # Test every 30 minutes
forward_periods = [15, 30, 60, 120, 240]  # Look forward 15min to 4h

results = []

print("   Analyzing historical momentum signals...")
for i in range(lookback, len(df) - max(forward_periods), sample_every):
    if i % 5000 == 0:
        pct = (i / len(df)) * 100
        print(f"   Progress: {pct:.1f}%")
    
    # Get window
    df_window = df.iloc[i-lookback:i+1].copy()
    
    if len(df_window) < 240:
        continue
    
    # Calculate momentum
    try:
        momentum_report = momentum_calc.get_momentum_report(df_window)
    except:
        continue
    
    current_price = df['price'].iloc[i]
    current_time = df.index[i]
    
    # Look forward to see what actually happened
    for forward in forward_periods:
        future_idx = i + forward
        if future_idx >= len(df):
            break
        
        future_price = df['price'].iloc[future_idx]
        price_change_pct = ((future_price - current_price) / current_price) * 100
        
        # Determine actual direction
        if price_change_pct > 0.2:
            actual_direction = 'UP'
        elif price_change_pct < -0.2:
            actual_direction = 'DOWN'
        else:
            actual_direction = 'NONE'
        
        # Determine actual momentum strength based on move size
        actual_strength = min(abs(price_change_pct) * 30, 100)  # Scale to 0-100
        
        # Record result
        result = {
            'timestamp': current_time,
            'forward_mins': forward,
            'predicted_direction': momentum_report['direction'],
            'actual_direction': actual_direction,
            'predicted_strength': momentum_report['strength'],
            'actual_strength': actual_strength,
            'predicted_phase': momentum_report['phase'],
            'price_change_pct': price_change_pct,
            'confidence': momentum_report['confidence']
        }
        
        results.append(result)
        break  # Only take first forward period

print(f"\n   ✅ Analyzed {len(results):,} momentum predictions")
print()

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Calculate accuracy metrics
print("="*80)
print("📊 MOMENTUM DETECTION ACCURACY")
print("="*80)
print()

# Overall direction accuracy
correct_direction = (results_df['predicted_direction'] == results_df['actual_direction']).sum()
total = len(results_df)
direction_accuracy = (correct_direction / total) * 100

print(f"DIRECTION ACCURACY:")
print(f"   Overall:       {direction_accuracy:.1f}%")
print(f"   Correct:       {correct_direction:,} / {total:,}")
print()

# Direction accuracy by prediction
for direction in ['UP', 'DOWN', 'NONE']:
    mask = results_df['predicted_direction'] == direction
    if mask.sum() > 0:
        correct = ((results_df[mask]['predicted_direction'] == results_df[mask]['actual_direction']).sum())
        total_dir = mask.sum()
        acc = (correct / total_dir) * 100
        print(f"   When predicted {direction:5s}: {acc:.1f}% ({correct:,}/{total_dir:,})")

print()

# Strength accuracy (MAE)
strength_error = np.abs(results_df['predicted_strength'] - results_df['actual_strength'])
mae_strength = strength_error.mean()

print(f"STRENGTH ACCURACY:")
print(f"   MAE:           {mae_strength:.1f} points")
print(f"   Avg Predicted: {results_df['predicted_strength'].mean():.1f}")
print(f"   Avg Actual:    {results_df['actual_strength'].mean():.1f}")
print()

# Phase analysis
print(f"PHASE DISTRIBUTION:")
phase_counts = results_df['predicted_phase'].value_counts()
for phase, count in phase_counts.items():
    pct = (count / total) * 100
    avg_move = results_df[results_df['predicted_phase'] == phase]['price_change_pct'].mean()
    print(f"   {phase:10s}: {pct:5.1f}% ({count:4,} times) → Avg move: {avg_move:+.2f}%")
print()

# Confidence vs Accuracy
print(f"CONFIDENCE CALIBRATION:")
high_conf = results_df[results_df['confidence'] > 0.6]
med_conf = results_df[(results_df['confidence'] > 0.4) & (results_df['confidence'] <= 0.6)]
low_conf = results_df[results_df['confidence'] <= 0.4]

for name, subset in [('High (>60%)', high_conf), ('Medium (40-60%)', med_conf), ('Low (<40%)', low_conf)]:
    if len(subset) > 0:
        acc = ((subset['predicted_direction'] == subset['actual_direction']).sum() / len(subset)) * 100
        print(f"   {name:15s}: {acc:.1f}% accuracy ({len(subset):,} samples)")
print()

# Time-based analysis
print(f"ACCURACY BY TIME HORIZON:")
for forward in sorted(results_df['forward_mins'].unique()):
    subset = results_df[results_df['forward_mins'] == forward]
    acc = ((subset['predicted_direction'] == subset['actual_direction']).sum() / len(subset)) * 100
    avg_move = subset['price_change_pct'].mean()
    print(f"   {forward:3d} minutes: {acc:.1f}% accuracy → Avg move: {avg_move:+.2f}%")
print()

# Trading implications
print("="*80)
print("💡 TRADING IMPLICATIONS")
print("="*80)
print()

# Filter by confidence
high_conf_mask = results_df['confidence'] > 0.5
high_conf_results = results_df[high_conf_mask]

print(f"FILTERING BY CONFIDENCE >50%:")
print(f"   Signals:       {len(high_conf_results):,} ({len(high_conf_results)/len(results_df)*100:.1f}%)")

direction_acc_filtered = ((high_conf_results['predicted_direction'] == high_conf_results['actual_direction']).sum() / len(high_conf_results)) * 100
print(f"   Accuracy:      {direction_acc_filtered:.1f}%")

# When momentum is BUILDING
building_mask = (results_df['predicted_phase'] == 'BUILDING') & (results_df['confidence'] > 0.5)
building_results = results_df[building_mask]

if len(building_results) > 0:
    print()
    print(f"WHEN MOMENTUM IS BUILDING (confidence >50%):")
    print(f"   Occurrences:   {len(building_results):,}")
    
    building_direction_acc = ((building_results['predicted_direction'] == building_results['actual_direction']).sum() / len(building_results)) * 100
    print(f"   Direction Acc: {building_direction_acc:.1f}%")
    
    avg_move = building_results['price_change_pct'].mean()
    print(f"   Avg Move:      {avg_move:+.2f}%")
    
    # Win rate if trading
    up_correct = ((building_results['predicted_direction'] == 'UP') & (building_results['actual_direction'] == 'UP')).sum()
    down_correct = ((building_results['predicted_direction'] == 'DOWN') & (building_results['actual_direction'] == 'DOWN')).sum()
    tradable = ((building_results['predicted_direction'] == 'UP') | (building_results['predicted_direction'] == 'DOWN')).sum()
    
    if tradable > 0:
        win_rate = ((up_correct + down_correct) / tradable) * 100
        print(f"   Trading Win:   {win_rate:.1f}% ({up_correct + down_correct}/{tradable})")

# When momentum is STRONG
strong_mask = (results_df['predicted_phase'] == 'STRONG') & (results_df['confidence'] > 0.5)
strong_results = results_df[strong_mask]

if len(strong_results) > 0:
    print()
    print(f"WHEN MOMENTUM IS STRONG (confidence >50%):")
    print(f"   Occurrences:   {len(strong_results):,}")
    
    strong_direction_acc = ((strong_results['predicted_direction'] == strong_results['actual_direction']).sum() / len(strong_results)) * 100
    print(f"   Direction Acc: {strong_direction_acc:.1f}%")
    
    avg_move = strong_results['price_change_pct'].mean()
    print(f"   Avg Move:      {avg_move:+.2f}%")

print()
print("="*80)
print("✅ BACKTEST COMPLETE")
print("="*80)
print()
print(f"Summary:")
print(f"   Direction Accuracy:   {direction_accuracy:.1f}%")
print(f"   Strength MAE:         {mae_strength:.1f} points")
print(f"   High Confidence:      {direction_acc_filtered:.1f}% accuracy")
print(f"   Trading Signals:      {len(building_results):,} BUILDING + {len(strong_results):,} STRONG")
print()


