#!/usr/bin/env python3
"""
Train ML-Enhanced Momentum Calculator on 2.5 years of data

This trains ML to accurately measure momentum by learning from
what actually happened (not hand-crafted formulas).
"""
import sys
from pathlib import Path
import pandas as pd

# Add paths
sys.path.append(str(Path(__file__).parent))

from ml_momentum_calculator import MLMomentumCalculator

print("="*80)
print("🤖 TRAINING ML MOMENTUM CALCULATOR")
print("="*80)
print("Learning to measure momentum from 2.5 years of actual outcomes")
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

# Train ML momentum calculator
ml_momentum = MLMomentumCalculator()
results = ml_momentum.train_on_historical_data(df)

# Save models
print("💾 Saving trained models...")
models_dir = Path(__file__).parent / 'ml_models'
ml_momentum.save(models_dir)
print(f"   ✅ Saved to: {models_dir}/")
print()

# Test on current market
print("="*80)
print("🧪 TESTING ON CURRENT MARKET")
print("="*80)
print()

# Use last 300 bars for test
test_df = df.iloc[-300:].copy()
ml_result = ml_momentum.calculate_momentum(test_df)

print("ML-CALCULATED MOMENTUM:")
print("─" * 80)
print(f"   Strength:      {ml_result['strength']:.0f}/100")
print(f"   Direction:     {ml_result['direction']}")
print(f"   Phase:         {ml_result['phase']}")
print(f"   Confidence:    {ml_result['confidence']*100:.0f}%")
print()
print(f"   Direction Conf: {ml_result['direction_confidence']*100:.0f}%")
print(f"   Phase Conf:     {ml_result['phase_confidence']*100:.0f}%")
print()

# Compare with hand-crafted (if available)
try:
    from momentum_calculator import MomentumCalculator
    
    traditional_calc = MomentumCalculator()
    traditional_result = traditional_calc.get_momentum_report(test_df)
    
    print("COMPARISON:")
    print("─" * 80)
    print(f"                   ML Model      Hand-Crafted")
    print(f"   Strength:       {ml_result['strength']:.0f}/100        {traditional_result['strength']:.0f}/100")
    print(f"   Direction:      {ml_result['direction']:10s}  {traditional_result['direction']:10s}")
    print(f"   Phase:          {ml_result['phase']:10s}  {traditional_result['phase']:10s}")
    print(f"   Confidence:     {ml_result['confidence']*100:.0f}%           {traditional_result['confidence']*100:.0f}%")
    print()
    
    diff = abs(ml_result['strength'] - traditional_result['strength'])
    if diff > 20:
        print(f"⚠️  Large difference detected: {diff:.0f} points")
        print(f"   ML learned different patterns from actual outcomes")
    else:
        print(f"✅ Similar results (diff: {diff:.0f} points)")
except:
    pass

print()
print("="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print()
print("📊 Model Performance:")
print(f"   Strength:  R² = {results['strength']['test_r2']:.3f}, MAE = {results['strength']['mae']:.1f} points")
print(f"   Direction: Accuracy = {results['direction']['test_acc']*100:.1f}%")
print(f"   Phase:     Accuracy = {results['phase']['test_acc']*100:.1f}%")
print()
print("🚀 ML Momentum Calculator is ready!")
print()
print("To use:")
print("   from ml_momentum_calculator import MLMomentumCalculator")
print("   ml_momentum = MLMomentumCalculator()")
print("   ml_momentum.load('ml_models/')")
print("   result = ml_momentum.calculate_momentum(df)")
print()


