#!/usr/bin/env python3
"""Quick prediction with 44-feature model"""
import sys
sys.path.insert(0, str(__file__).replace('predict_now.py', 'train_improved_model.py'))
from train_improved_model import fetch_data_with_volume, compute_advanced_features
import pickle
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

print("="*80)
print("🎯 BTC PREDICTION - 44 FEATURES")
print("="*80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

# Fetch data (need 4h for lookback)
print("📡 Fetching 6h of data from Cortex...")
df = fetch_data_with_volume(hours=6)

if df is None or len(df) < 300:
    print("❌ Not enough data")
    sys.exit(1)

print(f"   ✅ {len(df)} samples")
print()

# Compute features for latest point
features_dict = compute_advanced_features(df, len(df) - 1)

if features_dict is None:
    print("❌ Could not compute features")
    sys.exit(1)

# Convert to array
features = np.array([[features_dict[fn] for fn in feature_names]])
features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

# Scale and predict
features_scaled = scaler.transform(features)
prediction = model.predict(features_scaled)[0]
probabilities = model.predict_proba(features_scaled)[0]

labels = ['DOWN', 'SIDEWAYS', 'UP']
predicted_label = labels[prediction]
confidence = probabilities[prediction]

# Display
print("="*80)
print("📊 PREDICTION")
print("="*80)
print()
print(f"   Current Price:  ${features_dict['price']:,.2f}")
print(f"   Direction:      {predicted_label}")
print(f"   Confidence:     {confidence:.1%}")
print()
print("   Probabilities:")
print(f"      DOWN:     {probabilities[0]:.1%}")
print(f"      SIDEWAYS: {probabilities[1]:.1%}")
print(f"      UP:       {probabilities[2]:.1%}")
print()

# Show key indicators
print("   Market Indicators:")
print(f"      RSI-14:           {features_dict.get('rsi_14', 0):.1f}")
print(f"      MACD:             {features_dict.get('macd', 0):.4f}")
print(f"      Bollinger:        {features_dict.get('bollinger_position', 0):.1%}")
print(f"      Volatility (15m): {features_dict.get('volatility_15m', 0):.3f}%")
print()

# Regime
regime_trending_up = features_dict.get('regime_trending_up', 0)
regime_trending_down = features_dict.get('regime_trending_down', 0)
regime_sideways = features_dict.get('regime_sideways', 0)
regime_high_vol = features_dict.get('regime_high_vol', 0)

if regime_trending_up:
    regime = "📈 TRENDING UP"
elif regime_trending_down:
    regime = "📉 TRENDING DOWN"
elif regime_high_vol:
    regime = "⚡ HIGH VOLATILITY"
else:
    regime = "↔️  SIDEWAYS"

print(f"   Market Regime: {regime}")
print()

# Trading signal
print("="*80)
print("🚨 TRADING SIGNAL")
print("="*80)
print()

if prediction == 1:
    print("   ⏸️  NO TRADE - Market sideways")
elif confidence < 0.70:
    print(f"   🔕 LOW CONFIDENCE ({confidence:.1%}) - Skip")
elif confidence > 0.85:
    action = "SELL/SHORT" if prediction == 0 else "BUY/LONG"
    print(f"   🔥 HIGH CONFIDENCE: {action}")
    print(f"   Position: 1-2%")
    print(f"   Stop-loss: {config['threshold'] * 1.5:.2f}%")
    print(f"   Take-profit: {config['threshold'] * 2:.2f}%")
elif confidence > 0.70:
    action = "SELL/SHORT" if prediction == 0 else "BUY/LONG"
    print(f"   ⚠️  MEDIUM CONFIDENCE: {action}")
    print(f"   Position: 0.5-1%")
    print(f"   Stop-loss: {config['threshold'] * 1.5:.2f}%")
    print(f"   Take-profit: {config['threshold'] * 2:.2f}%")

print()
print("="*80)



