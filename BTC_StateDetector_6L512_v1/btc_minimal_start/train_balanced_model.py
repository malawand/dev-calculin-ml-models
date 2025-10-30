#!/usr/bin/env python3
"""Train a BALANCED model - not too conservative, not too aggressive"""
import sys
sys.path.insert(0, str(__file__).replace('train_balanced_model.py', 'train_improved_model.py'))
from train_improved_model import fetch_data_with_volume
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
import pickle
import json
from pathlib import Path
from datetime import datetime

print("="*80)
print("🎯 TRAINING BALANCED MODEL (GOLDILOCKS APPROACH)")
print("="*80)
print("Finding the sweet spot between conservative and aggressive...")
print()

# Fetch data
print("📥 Fetching 96h of data...")
df = fetch_data_with_volume(hours=96)

if df is None or len(df) < 300:
    print("❌ Not enough data")
    sys.exit(1)

print(f"   ✅ {len(df)} samples")
print()

def compute_features_simple(df, idx):
    """Simplified feature computation focusing on most important features"""
    if idx < 240:
        return None
    
    window = df.iloc[idx-240:idx+1]
    prices = window['price'].values
    volumes = window['volume'].values if 'volume' in window.columns else np.zeros(len(window))
    weighted_derivs = window['weighted_deriv'].values if 'weighted_deriv' in window.columns else np.zeros(len(window))
    
    if len(prices) < 241 or np.any(np.isnan(prices)):
        return None
    
    features = {}
    
    # Top features from previous analysis
    # 1. Volatilities (most important)
    features['volatility_5m'] = np.std(prices[-5:]) if len(prices) >= 5 else 0
    features['volatility_15m'] = np.std(prices[-15:]) if len(prices) >= 15 else 0
    features['volatility_30m'] = np.std(prices[-30:]) if len(prices) >= 30 else 0
    features['volatility_60m'] = np.std(prices[-60:]) if len(prices) >= 60 else 0
    features['volatility_120m'] = np.std(prices[-120:]) if len(prices) >= 120 else 0
    features['volatility_240m'] = np.std(prices[-240:]) if len(prices) >= 240 else 0
    
    # 2. Weighted derivative (rank #3 feature)
    features['weighted_deriv'] = weighted_derivs[-1]
    features['weighted_deriv_change'] = weighted_derivs[-1] - weighted_derivs[-30] if len(weighted_derivs) >= 30 else 0
    
    # 3. Price derivatives
    for window_size in [15, 30, 60, 120, 240]:
        if len(prices) >= window_size:
            deriv = (prices[-1] - prices[-window_size]) / (prices[-window_size] + 1e-10)
            features[f'deriv_{window_size}m'] = deriv
    
    # 4. Moving averages
    for window_size in [15, 30, 60, 120, 240]:
        if len(prices) >= window_size:
            ma = np.mean(prices[-window_size:])
            features[f'ma_{window_size}m'] = ma
            features[f'price_vs_ma_{window_size}m'] = (prices[-1] - ma) / (ma + 1e-10)
    
    # 5. Volume features
    if np.any(volumes > 0):
        features['volume'] = volumes[-1]
        features['volume_ratio'] = volumes[-1] / (np.mean(volumes[-60:]) + 1e-10) if len(volumes) >= 60 else 1.0
        
        for window_size in [15, 30]:
            if len(volumes) >= window_size and volumes[-window_size] > 0:
                features[f'volume_deriv_{window_size}m'] = (volumes[-1] - volumes[-window_size]) / (volumes[-window_size] + 1e-10)
            else:
                features[f'volume_deriv_{window_size}m'] = 0
    else:
        features['volume'] = 0
        features['volume_ratio'] = 1.0
        features['volume_deriv_15m'] = 0
        features['volume_deriv_30m'] = 0
    
    # 6. Momentum
    for window_size in [15, 30, 60]:
        if len(prices) >= window_size + 1:
            recent = (prices[-1] - prices[-window_size]) / (prices[-window_size] + 1e-10)
            earlier = (prices[-window_size] - prices[-window_size*2]) / (prices[-window_size*2] + 1e-10) if len(prices) >= window_size*2 else 0
            features[f'momentum_accel_{window_size}m'] = recent - earlier
    
    return features

def train_balanced_config(df, horizon_minutes, threshold_pct, class_weights):
    """Train a single configuration"""
    print(f"\n🔧 Training {horizon_minutes}min @ ±{threshold_pct}% (weights: {class_weights})...")
    
    # Prepare data
    X_list = []
    y_list = []
    feature_names = None
    
    for i in range(240, len(df) - horizon_minutes):
        features_dict = compute_features_simple(df, i)
        if features_dict is None:
            continue
        
        if feature_names is None:
            feature_names = list(features_dict.keys())
        
        current_price = df.iloc[i]['price']
        future_price = df.iloc[i + horizon_minutes]['price']
        change_pct = (future_price - current_price) / current_price * 100
        
        # Create label
        if change_pct > threshold_pct:
            label = 2  # UP
        elif change_pct < -threshold_pct:
            label = 0  # DOWN
        else:
            label = 1  # SIDEWAYS
        
        X_list.append([features_dict[fn] for fn in feature_names])
        y_list.append(label)
    
    if len(X_list) == 0:
        return None
    
    X = np.array(X_list)
    y = np.array(y_list)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Train/test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    model = LGBMClassifier(
        random_state=42,
        n_estimators=400,
        learning_rate=0.02,
        num_leaves=35,
        max_depth=7,
        class_weight=class_weights,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # Per-class
    up_mask = y_test == 2
    down_mask = y_test == 0
    sideways_mask = y_test == 1
    
    up_acc = accuracy_score(y_test[up_mask], y_pred[up_mask]) if up_mask.sum() > 0 else 0
    down_acc = accuracy_score(y_test[down_mask], y_pred[down_mask]) if down_mask.sum() > 0 else 0
    sideways_acc = accuracy_score(y_test[sideways_mask], y_pred[sideways_mask]) if sideways_mask.sum() > 0 else 0
    
    # Directional (only when model predicts UP/DOWN)
    directional_mask = (y_test != 1) & (y_pred != 1)
    dir_acc = accuracy_score(y_test[directional_mask], y_pred[directional_mask]) if directional_mask.sum() > 0 else 0
    
    # Signal distribution
    pred_up_pct = (y_pred == 2).sum() / len(y_pred) * 100
    pred_down_pct = (y_pred == 0).sum() / len(y_pred) * 100
    pred_sideways_pct = (y_pred == 1).sum() / len(y_pred) * 100
    
    # High-confidence accuracy (>70%)
    high_conf_mask = np.max(y_proba, axis=1) > 0.70
    high_conf_acc = accuracy_score(y_test[high_conf_mask], y_pred[high_conf_mask]) if high_conf_mask.sum() > 0 else 0
    high_conf_count = high_conf_mask.sum()
    
    print(f"   Samples: {len(X_train)} train, {len(X_test)} test, {len(feature_names)} features")
    print(f"   ✅ Overall: {accuracy:.1%}")
    print(f"      UP: {up_acc:.1%}, DOWN: {down_acc:.1%}, SIDEWAYS: {sideways_acc:.1%}")
    print(f"      Directional: {dir_acc:.1%}")
    print(f"      Signals: UP {pred_up_pct:.1f}%, DOWN {pred_down_pct:.1f}%, SIDEWAYS {pred_sideways_pct:.1f}%")
    print(f"      High-conf (>70%): {high_conf_acc:.1%} ({high_conf_count} samples)")
    
    # Balanced score: weighs all factors
    # - Overall accuracy: 20%
    # - Directional accuracy: 30%
    # - High-confidence accuracy: 25%
    # - Signal balance: 25% (prefer 20-40% directional, 60-80% sideways)
    directional_signal_pct = pred_up_pct + pred_down_pct
    balance_score = 1.0 - abs(directional_signal_pct - 30) / 100  # Optimal: 30% directional signals
    
    score = (
        accuracy * 0.20 +
        dir_acc * 0.30 +
        high_conf_acc * 0.25 +
        balance_score * 0.25
    )
    
    print(f"      Score: {score:.3f} (target: >0.600)")
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'config': {'horizon': f'{horizon_minutes}min', 'threshold': threshold_pct, 'class_weights': class_weights},
        'metrics': {
            'accuracy': accuracy,
            'up_acc': up_acc,
            'down_acc': down_acc,
            'sideways_acc': sideways_acc,
            'directional_acc': dir_acc,
            'high_conf_acc': high_conf_acc,
            'high_conf_count': high_conf_count,
            'pred_up_pct': pred_up_pct,
            'pred_down_pct': pred_down_pct,
            'pred_sideways_pct': pred_sideways_pct,
            'score': score
        }
    }

# Test multiple balanced configurations
print("="*80)
print("🎯 TESTING BALANCED CONFIGURATIONS")
print("="*80)

configs = [
    # Moderate thresholds with balanced weights
    (30, 0.10, {0: 1.5, 1: 0.7, 2: 1.5}),  # Slightly favor directional
    (30, 0.12, {0: 1.5, 1: 0.7, 2: 1.5}),
    (30, 0.15, {0: 1.5, 1: 0.7, 2: 1.5}),
    (60, 0.12, {0: 1.5, 1: 0.7, 2: 1.5}),
    (60, 0.15, {0: 1.5, 1: 0.7, 2: 1.5}),
    (60, 0.18, {0: 1.5, 1: 0.7, 2: 1.5}),
    # More conservative (higher thresholds)
    (30, 0.15, {0: 1.2, 1: 0.9, 2: 1.2}),
    (60, 0.20, {0: 1.2, 1: 0.9, 2: 1.2}),
]

results = []
for horizon, threshold, weights in configs:
    result = train_balanced_config(df, horizon, threshold, weights)
    if result is not None:
        results.append(result)

# Sort by score
results.sort(key=lambda x: x['metrics']['score'], reverse=True)

print()
print("="*80)
print("🏆 TOP 5 CONFIGURATIONS")
print("="*80)

for i, result in enumerate(results[:5], 1):
    cfg = result['config']
    m = result['metrics']
    print(f"\n{i}. {cfg['horizon']} @ ±{cfg['threshold']}%")
    print(f"   Score: {m['score']:.3f}")
    print(f"   Overall: {m['accuracy']:.1%}, Directional: {m['directional_acc']:.1%}, High-conf: {m['high_conf_acc']:.1%}")
    print(f"   Signals: UP {m['pred_up_pct']:.1f}%, DOWN {m['pred_down_pct']:.1f}%, SIDEWAYS {m['pred_sideways_pct']:.1f}%")

# Save best model
best = results[0]
print()
print("="*80)
print(f"🥇 BEST MODEL: {best['config']['horizon']} @ ±{best['config']['threshold']}%")
print("="*80)
print(f"Score: {best['metrics']['score']:.3f}")
print(f"Overall Accuracy: {best['metrics']['accuracy']:.1%}")
print(f"Directional Accuracy: {best['metrics']['directional_acc']:.1%}")
print(f"High-Confidence Accuracy: {best['metrics']['high_conf_acc']:.1%} ({best['metrics']['high_conf_count']} samples)")
print(f"Signal Distribution:")
print(f"   UP:       {best['metrics']['pred_up_pct']:.1f}%")
print(f"   DOWN:     {best['metrics']['pred_down_pct']:.1f}%")
print(f"   SIDEWAYS: {best['metrics']['pred_sideways_pct']:.1f}%")
print()

# Save
models_dir = Path(__file__).parent / 'models'
models_dir.mkdir(exist_ok=True)

with open(models_dir / 'scalping_model.pkl', 'wb') as f:
    pickle.dump(best['model'], f)

with open(models_dir / 'scalping_scaler.pkl', 'wb') as f:
    pickle.dump(best['scaler'], f)

with open(models_dir / 'scalping_config.json', 'w') as f:
    json.dump(best['config'], f, indent=2)

with open(models_dir / 'feature_names.json', 'w') as f:
    json.dump(best['feature_names'], f, indent=2)

print("💾 Saved:")
print("   - scalping_model.pkl")
print("   - scalping_scaler.pkl")
print("   - scalping_config.json")
print("   - feature_names.json")
print()
print("="*80)
print("✅ BALANCED MODEL TRAINED!")
print("="*80)
print()
print("This model aims for:")
print("   ✅ 60-70% overall accuracy")
print("   ✅ 60-70% directional accuracy")
print("   ✅ 70-80% high-confidence accuracy")
print("   ✅ 20-40% directional signals (balanced)")
print()
print("Test it: python test_weighted_deriv.py")
print()


