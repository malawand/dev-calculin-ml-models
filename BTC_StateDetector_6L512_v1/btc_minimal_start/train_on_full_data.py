#!/usr/bin/env python3
"""Train on FULL 2.5 year dataset for better learning"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
import pickle
import json

print("="*80)
print("🚀 TRAINING ON FULL 2.5 YEAR DATASET")
print("="*80)
print("Using comprehensive historical data for better predictions...")
print()

# Load full dataset
print("📥 Loading full dataset...")
data_path = Path(__file__).parent.parent / 'btc_direction_predictor/artifacts/historical_data/combined_with_volume.parquet'
df = pd.read_parquet(data_path)

if 'timestamp' in df.columns:
    df = df.set_index('timestamp')
df.index = pd.to_datetime(df.index)
df = df.sort_index()

if 'crypto_last_price' in df.columns:
    df = df.rename(columns={'crypto_last_price': 'price'})

print(f"   ✅ Loaded {len(df):,} samples")
print(f"   Date range: {df.index.min()} → {df.index.max()}")
print(f"   Duration: {(df.index.max() - df.index.min()).days} days")
print()

# Simple feature engineering - just use what we have
print("🔧 Engineering features...")

# Returns
df['return_15m'] = df['price'].pct_change(periods=15)
df['return_30m'] = df['price'].pct_change(periods=30)
df['return_60m'] = df['price'].pct_change(periods=60)
df['return_4h'] = df['price'].pct_change(periods=240)

# Volatility
df['volatility_30m'] = df['price'].rolling(30).std()
df['volatility_1h'] = df['price'].rolling(60).std()
df['volatility_4h'] = df['price'].rolling(240).std()

# Moving averages
df['ma_15m'] = df['price'].rolling(15).mean()
df['ma_30m'] = df['price'].rolling(30).mean()
df['ma_1h'] = df['price'].rolling(60).mean()
df['ma_4h'] = df['price'].rolling(240).mean()

# Price vs MA
df['price_vs_ma_30m'] = (df['price'] - df['ma_30m']) / (df['ma_30m'] + 1e-10)
df['price_vs_ma_1h'] = (df['price'] - df['ma_1h']) / (df['ma_1h'] + 1e-10)
df['price_vs_ma_4h'] = (df['price'] - df['ma_4h']) / (df['ma_4h'] + 1e-10)

# Use available derivatives if they exist
deriv_cols = [col for col in df.columns if 'deriv' in col.lower() and 'crypto_last_price' in col]
print(f"   Found {len(deriv_cols)} derivative columns")

# Volume features
if 'crypto_volume' in df.columns:
    df['volume'] = df['crypto_volume']
    df['volume_ma_1h'] = df['volume'].rolling(60).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma_1h'] + 1e-10)
else:
    df['volume'] = 0
    df['volume_ma_1h'] = 0
    df['volume_ratio'] = 1.0

print(f"   ✅ Total features: {len(df.columns)}")
print()

# Define features to use
feature_cols = [
    'return_15m', 'return_30m', 'return_60m', 'return_4h',
    'volatility_30m', 'volatility_1h', 'volatility_4h',
    'price_vs_ma_30m', 'price_vs_ma_1h', 'price_vs_ma_4h',
    'volume', 'volume_ratio'
]

# Add some derivative columns if available
if len(deriv_cols) > 0:
    # Use up to 10 most relevant derivative columns
    important_derivs = [
        'job:crypto_last_price:deriv30m',
        'job:crypto_last_price:deriv1h',
        'job:crypto_last_price:deriv4h',
        'job:crypto_last_price:deriv24h',
        'job:crypto_last_price:weighted_deriv:24h:48h:7d',  # Our new metric!
    ]
    for deriv in important_derivs:
        if deriv in df.columns:
            feature_cols.append(deriv)
            print(f"   ✅ Using {deriv}")

print(f"\n📊 Final feature count: {len(feature_cols)}")
print()

# Train multiple configurations
def train_config(df, horizon_minutes, threshold_pct, test_days=30):
    """Train a configuration"""
    print(f"\n🔧 Training {horizon_minutes}min @ ±{threshold_pct}%...")
    
    # Create labels
    df_copy = df.copy()
    df_copy['future_price'] = df_copy['price'].shift(-horizon_minutes)
    df_copy['return'] = (df_copy['future_price'] - df_copy['price']) / df_copy['price'] * 100
    
    df_copy['label'] = 1  # SIDEWAYS
    df_copy.loc[df_copy['return'] > threshold_pct, 'label'] = 2  # UP
    df_copy.loc[df_copy['return'] < -threshold_pct, 'label'] = 0  # DOWN
    
    # Drop NaN
    df_model = df_copy[feature_cols + ['label']].dropna()
    
    if len(df_model) < 1000:
        print(f"   ⚠️  Only {len(df_model)} samples, skipping")
        return None
    
    print(f"   Samples: {len(df_model):,}")
    
    # Split: use last test_days for testing
    test_samples = test_days * 24 * 60  # days * hours * minutes
    split_idx = len(df_model) - test_samples
    
    train_df = df_model.iloc[:split_idx]
    test_df = df_model.iloc[split_idx:]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    # Handle inf/nan
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"   Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train with balanced approach
    model = LGBMClassifier(
        random_state=42,
        n_estimators=500,
        learning_rate=0.01,
        num_leaves=31,
        max_depth=8,
        class_weight={0: 1.3, 1: 0.8, 2: 1.3},  # Slightly favor directional
        min_child_samples=50,
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
    up_acc = accuracy_score(y_test[y_test == 2], y_pred[y_test == 2]) if (y_test == 2).sum() > 0 else 0
    down_acc = accuracy_score(y_test[y_test == 0], y_pred[y_test == 0]) if (y_test == 0).sum() > 0 else 0
    sideways_acc = accuracy_score(y_test[y_test == 1], y_pred[y_test == 1]) if (y_test == 1).sum() > 0 else 0
    
    # Directional
    dir_mask = (y_test != 1) & (y_pred != 1)
    dir_acc = accuracy_score(y_test[dir_mask], y_pred[dir_mask]) if dir_mask.sum() > 0 else 0
    
    # Signal distribution
    pred_up = (y_pred == 2).sum() / len(y_pred) * 100
    pred_down = (y_pred == 0).sum() / len(y_pred) * 100
    pred_sideways = (y_pred == 1).sum() / len(y_pred) * 100
    
    # High confidence
    high_conf = (np.max(y_proba, axis=1) > 0.75)
    high_conf_acc = accuracy_score(y_test[high_conf], y_pred[high_conf]) if high_conf.sum() > 0 else 0
    
    print(f"   ✅ Overall: {accuracy:.1%}")
    print(f"      UP: {up_acc:.1%}, DOWN: {down_acc:.1%}, SIDEWAYS: {sideways_acc:.1%}")
    print(f"      Directional: {dir_acc:.1%}")
    print(f"      Signals: UP {pred_up:.1f}%, DOWN {pred_down:.1f}%, SIDEWAYS {pred_sideways:.1f}%")
    print(f"      High-conf (>75%): {high_conf_acc:.1%} ({high_conf.sum()} samples)")
    
    # Balanced score
    score = accuracy * 0.4 + dir_acc * 0.4 + high_conf_acc * 0.2
    print(f"      Score: {score:.3f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_cols,
        'config': {'horizon': f'{horizon_minutes}min', 'threshold': threshold_pct},
        'metrics': {
            'accuracy': accuracy,
            'directional_acc': dir_acc,
            'high_conf_acc': high_conf_acc,
            'pred_up': pred_up,
            'pred_down': pred_down,
            'pred_sideways': pred_sideways,
            'score': score
        }
    }

# Test configurations
print("="*80)
print("🎯 TESTING CONFIGURATIONS ON FULL DATASET")
print("="*80)

configs = [
    (15, 0.08),
    (15, 0.10),
    (15, 0.12),
    (30, 0.10),
    (30, 0.12),
    (30, 0.15),
    (60, 0.15),
    (60, 0.18),
    (60, 0.20),
]

results = []
for horizon, threshold in configs:
    result = train_config(df, horizon, threshold)
    if result:
        results.append(result)

# Sort by score
results.sort(key=lambda x: x['metrics']['score'], reverse=True)

print()
print("="*80)
print("🏆 TOP 5 BEST MODELS")
print("="*80)

for i, r in enumerate(results[:5], 1):
    cfg = r['config']
    m = r['metrics']
    print(f"\n{i}. {cfg['horizon']} @ ±{cfg['threshold']}%")
    print(f"   Score: {m['score']:.3f}")
    print(f"   Overall: {m['accuracy']:.1%}, Directional: {m['directional_acc']:.1%}, High-conf: {m['high_conf_acc']:.1%}")
    print(f"   Signals: UP {m['pred_up']:.1f}%, DOWN {m['pred_down']:.1f}%, SIDEWAYS {m['pred_sideways']:.1f}%")

# Save best
best = results[0]
print()
print("="*80)
print(f"🥇 SAVING BEST MODEL: {best['config']['horizon']} @ ±{best['config']['threshold']}%")
print("="*80)

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

print("\n💾 Saved model files")
print("\n✅ Training complete on 2.5 years of data!")
print("Test with: python test_weighted_deriv.py")
print()


