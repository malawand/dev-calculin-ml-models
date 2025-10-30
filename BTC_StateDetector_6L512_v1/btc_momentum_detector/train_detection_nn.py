#!/usr/bin/env python3
"""
Neural Network for Current State Detection (Not Prediction!)

Train NN to detect:
- Current direction (UP/DOWN/NONE)
- Current momentum strength
- Current phase

Goal: Improve from 62% → 70%+ by learning better feature combinations
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier, MLPRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🤖 NEURAL NETWORK FOR STATE DETECTION")
print("="*80)
print("Training NN to detect current momentum/trend state")
print()

# Load data
data_paths = [
    Path(__file__).parent.parent / 'btc_direction_predictor/artifacts/historical_data/combined_with_volume.parquet',
]

df = None
for data_path in data_paths:
    if data_path.exists():
        df = pd.read_parquet(data_path)
        break

if df is None:
    print("❌ No data found!")
    sys.exit(1)

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
print()

# Extract features and labels
print("📊 Extracting features and current state labels...")

lookback = 300  # 5 hours in minutes (short-term features only)
test_window = 60  # Measure current state over last 60 min
sample_every = 30

samples = []

for i in range(lookback, len(df) - test_window, sample_every):
    if i % 10000 == 0:
        pct = (i / len(df)) * 100
        print(f"   Progress: {pct:.1f}%")
    
    # Extract features from current moment
    window = df.iloc[i-lookback:i+1].copy()
    
    if len(window) < 100:
        continue
    
    # Features
    prices = window['price'].values
    volumes = window['volume'].values if 'volume' in window.columns else np.zeros(len(prices))
    
    features = {}
    
    # Price features (multiple timeframes)
    for period in [5, 15, 30, 60, 120, 240]:
        if len(prices) >= period:
            # ROC
            roc = (prices[-1] - prices[-period]) / prices[-period]
            features[f'roc_{period}'] = roc
            
            # Volatility
            returns = np.diff(prices[-period:]) / prices[-period:-1]
            features[f'vol_{period}'] = np.std(returns)
            
            # Trend
            x = np.arange(period)
            y = prices[-period:]
            if len(x) == len(y):
                corr = np.corrcoef(x, y)[0, 1]
                features[f'trend_{period}'] = corr
    
    # Volume features
    for period in [15, 30, 60, 120]:
        if len(volumes) >= period and volumes.sum() > 0:
            recent_vol = np.mean(volumes[-period//2:])
            avg_vol = np.mean(volumes[-period:])
            if avg_vol > 0:
                features[f'vol_ratio_{period}'] = recent_vol / avg_vol
    
    # Acceleration
    if len(prices) >= 60:
        roc_recent = (prices[-1] - prices[-30]) / prices[-30]
        roc_earlier = (prices[-30] - prices[-60]) / prices[-60]
        features['acceleration'] = roc_recent - roc_earlier
    
    # Calculate TRUE current state (ground truth)
    prices_recent = prices[-60:]
    
    # True strength (volatility)
    returns_recent = np.diff(prices_recent) / prices_recent[:-1]
    true_volatility = np.std(returns_recent) * 100
    true_strength = min(true_volatility * 50, 100)
    
    # True direction
    price_change = (prices_recent[-1] - prices_recent[0]) / prices_recent[0] * 100
    if price_change > 0.3:
        true_direction = 1  # UP
    elif price_change < -0.3:
        true_direction = -1  # DOWN
    else:
        true_direction = 0  # NONE
    
    samples.append({
        'features': features,
        'true_strength': true_strength,
        'true_direction': true_direction
    })

print(f"\n   ✅ Extracted {len(samples):,} training samples")
print()

# Convert to arrays
feature_names = sorted(list(samples[0]['features'].keys()))
X = []
y_strength = []
y_direction = []

for sample in samples:
    X.append([sample['features'].get(fn, 0) for fn in feature_names])
    y_strength.append(sample['true_strength'])
    y_direction.append(sample['true_direction'])

X = np.array(X)
y_strength = np.array(y_strength)
y_direction = np.array(y_direction)

# Handle NaN/inf
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
y_strength = np.nan_to_num(y_strength, nan=0.0, posinf=0.0, neginf=0.0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_str_train, y_str_test, y_dir_train, y_dir_test = train_test_split(
    X_scaled, y_strength, y_direction, test_size=0.2, shuffle=False
)

print(f"   Training set: {len(X_train):,}")
print(f"   Test set:     {len(X_test):,}")
print(f"   Features:     {len(feature_names)}")
print()

# Train
print("🤖 Training Neural Network...")
print()

# Train strength model (regression)
print("   Training Strength Model...")
strength_model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    max_iter=100,
    learning_rate_init=0.001,
    early_stopping=True,
    random_state=42,
    verbose=False
)
strength_model.fit(X_train, y_str_train)

# Train direction model (classification)
print("   Training Direction Model...")
direction_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    max_iter=100,
    learning_rate_init=0.001,
    early_stopping=True,
    random_state=42,
    verbose=False
)
direction_model.fit(X_train, y_dir_train)

print()
print("✅ Training complete!")
print()

# Test
strength_pred_np = strength_model.predict(X_test)
direction_pred_original = direction_model.predict(X_test)

direction_accuracy = accuracy_score(y_dir_test, direction_pred_original)

# Strength correlation
strength_corr = np.corrcoef(strength_pred_np, y_str_test)[0, 1]
strength_mae = np.mean(np.abs(strength_pred_np - y_str_test))

print("="*80)
print("📊 NEURAL NETWORK RESULTS")
print("="*80)
print()

print(f"DIRECTION DETECTION:")
print(f"   NN Accuracy:        {direction_accuracy*100:.1f}%")
print(f"   Baseline (hand):    55.6%")
print(f"   Improvement:        {(direction_accuracy*100 - 55.6):+.1f} pp")
print()

# Per-class accuracy
for true_dir, name in [(-1, 'DOWN'), (0, 'NONE'), (1, 'UP')]:
    mask = y_dir_test == true_dir
    if mask.sum() > 0:
        acc = (direction_pred_original[mask] == true_dir).sum() / mask.sum()
        print(f"   When {name:5s}: {acc*100:.1f}%")
print()

print(f"STRENGTH MEASUREMENT:")
print(f"   NN Correlation:     {strength_corr:.3f}")
print(f"   Baseline (hand):    0.556")
print(f"   Improvement:        {(strength_corr - 0.556):+.3f}")
print(f"   MAE:                {strength_mae:.1f} points")
print()

print("="*80)
print("🎯 VERDICT")
print("="*80)
print()

if direction_accuracy > 0.62 or strength_corr > 0.60:
    print("✅ NEURAL NETWORK IMPROVES DETECTION!")
    print()
    if direction_accuracy > 0.62:
        print(f"   • Direction: {direction_accuracy*100:.1f}% (up from 62%)")
    if strength_corr > 0.60:
        print(f"   • Strength: {strength_corr:.3f} correlation (up from 0.556)")
    print()
    print("   NN learned better feature combinations for state detection.")
    print("   Recommended: Use NN instead of hand-crafted formulas.")
elif direction_accuracy > 0.55 or strength_corr > 0.55:
    print("⚠️  NEURAL NETWORK COMPARABLE TO HAND-CRAFTED")
    print()
    print(f"   • Direction: {direction_accuracy*100:.1f}% (baseline 62%)")
    print(f"   • Strength: {strength_corr:.3f} (baseline 0.556)")
    print()
    print("   NN is similar to hand-crafted. Either approach works.")
else:
    print("❌ NEURAL NETWORK DOESN'T IMPROVE")
    print()
    print(f"   • Direction: {direction_accuracy*100:.1f}% (baseline 62%)")
    print(f"   • Strength: {strength_corr:.3f} (baseline 0.556)")
    print()
    print("   Hand-crafted formulas work better. Keep using those.")

print()
print("="*80)

# Save models
import pickle

models_dir = Path(__file__).parent
print()
print("💾 Saving models...")

with open(models_dir / 'direction_model.pkl', 'wb') as f:
    pickle.dump(direction_model, f)
print(f"   ✅ Saved direction_model.pkl")

with open(models_dir / 'strength_model.pkl', 'wb') as f:
    pickle.dump(strength_model, f)
print(f"   ✅ Saved strength_model.pkl")

with open(models_dir / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"   ✅ Saved scaler.pkl")

# Save feature names
import json
with open(models_dir / 'feature_names.json', 'w') as f:
    json.dump(feature_names, f)
print(f"   ✅ Saved feature_names.json")

print()
print(f"📁 Models saved to: {models_dir}")
print()
print("🎯 Ready to use! Run:")
print(f"   python monitor_state_realtime.py")
print()
print("="*80)

