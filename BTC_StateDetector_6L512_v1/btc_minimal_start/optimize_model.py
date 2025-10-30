#!/usr/bin/env python3
"""
Comprehensive Model Optimization
Tests multiple configurations and finds the optimal model
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
import json
import pickle
from datetime import datetime, timedelta
import requests
from itertools import product

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'btc_direction_predictor'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'btc_lstm_ensemble'))

from src.features.engineering import FeatureEngineer
from src.labels.labels import LabelCreator

# Cortex configuration
CORTEX_URL = "http://10.2.20.60:9009"
CORTEX_API_RANGE = "/prometheus/api/v1/query_range"
SYMBOL = "BTCUSDT"

print("="*80)
print("🚀 COMPREHENSIVE MODEL OPTIMIZATION")
print("="*80)
print()

# Configuration space to explore
CONFIGS = [
    {'horizon': '15min', 'threshold': 0.10},
    {'horizon': '15min', 'threshold': 0.15},
    {'horizon': '15min', 'threshold': 0.20},
    {'horizon': '30min', 'threshold': 0.10},
    {'horizon': '30min', 'threshold': 0.15},
    {'horizon': '30min', 'threshold': 0.20},
    {'horizon': '1h', 'threshold': 0.15},
    {'horizon': '1h', 'threshold': 0.20},
    {'horizon': '1h', 'threshold': 0.25},
]

BACKTEST_WINDOWS = ['4h', '8h', '12h', '24h', '48h']

def fetch_recent_data(hours=48):
    """Fetch recent data for training and backtesting."""
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    print(f"📥 Fetching last {hours} hours of data...")
    
    query = f'crypto_last_price{{symbol="{SYMBOL}"}}'
    params = {
        'query': query,
        'start': int(start_time.timestamp()),
        'end': int(end_time.timestamp()),
        'step': '60s'
    }
    
    try:
        response = requests.get(f"{CORTEX_URL}{CORTEX_API_RANGE}", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] == 'success' and data['data']['result']:
            values = data['data']['result'][0]['values']
            df = pd.DataFrame(values, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df = df.set_index('timestamp')
            df = df.dropna()
            
            print(f"   ✅ Fetched {len(df)} data points")
            return df
        else:
            print(f"❌ No data returned")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def compute_features(df, idx):
    """Compute features for a given index."""
    if idx < 60:
        return None
    
    window = df.iloc[idx-60:idx+1]
    prices = window['price'].values
    
    if len(prices) < 61 or np.any(np.isnan(prices)):
        return None
    
    features = []
    features.append(prices[-1])  # Current price
    
    # Derivatives
    for window_size in [5, 10, 15, 30]:
        deriv = (prices[-1] - prices[-window_size]) / prices[-window_size] if len(prices) >= window_size else 0
        features.append(deriv)
    
    # Moving averages
    for window_size in [5, 10, 15]:
        ma = np.mean(prices[-window_size:]) if len(prices) >= window_size else prices[-1]
        features.append(ma)
    
    # Volume placeholders
    features.extend([0.0, 0.0, 0.0])
    
    return np.array(features).reshape(1, -1)

def train_model(df, config):
    """Train a model with given configuration."""
    horizon_str = config['horizon']
    threshold = config['threshold']
    
    # Convert horizon to minutes
    if 'min' in horizon_str:
        horizon_minutes = int(horizon_str.replace('min', ''))
    elif 'h' in horizon_str:
        horizon_minutes = int(horizon_str.replace('h', '')) * 60
    else:
        horizon_minutes = 15
    
    # Prepare training data
    X_list = []
    y_list = []
    
    for i in range(60, len(df) - horizon_minutes):
        features = compute_features(df, i)
        if features is None:
            continue
        
        current_price = df.iloc[i]['price']
        future_price = df.iloc[i + horizon_minutes]['price']
        change_pct = (future_price - current_price) / current_price * 100
        
        # Create label
        if change_pct > threshold:
            label = 2  # UP
        elif change_pct < -threshold:
            label = 0  # DOWN
        else:
            label = 1  # SIDEWAYS
        
        X_list.append(features[0])
        y_list.append(label)
    
    if len(X_list) == 0:
        return None, None, None
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Train/test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train with class balancing
    model = LGBMClassifier(
        random_state=42,
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=30,
        max_depth=6,
        class_weight='balanced',
        verbose=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Per-class metrics
    up_mask = y_test == 2
    down_mask = y_test == 0
    sideways_mask = y_test == 1
    
    up_acc = accuracy_score(y_test[up_mask], y_pred[up_mask]) if up_mask.sum() > 0 else 0
    down_acc = accuracy_score(y_test[down_mask], y_pred[down_mask]) if down_mask.sum() > 0 else 0
    sideways_acc = accuracy_score(y_test[sideways_mask], y_pred[sideways_mask]) if sideways_mask.sum() > 0 else 0
    
    return model, scaler, {
        'accuracy': accuracy,
        'up_acc': up_acc,
        'down_acc': down_acc,
        'sideways_acc': sideways_acc,
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }

def backtest_model(model, scaler, df, config, window_hours):
    """Backtest model on a specific time window."""
    horizon_str = config['horizon']
    threshold = config['threshold']
    
    if 'min' in horizon_str:
        horizon_minutes = int(horizon_str.replace('min', ''))
    elif 'h' in horizon_str:
        horizon_minutes = int(horizon_str.replace('h', '')) * 60
    else:
        horizon_minutes = 15
    
    # Get data for the window
    end_time = df.index.max()
    start_time = end_time - timedelta(hours=window_hours)
    df_window = df[df.index >= start_time]
    
    if len(df_window) < 100:
        return None
    
    # Backtest
    results = []
    for i in range(60, len(df_window) - horizon_minutes, 15):  # Test every 15 min
        features = compute_features(df_window, i)
        if features is None:
            continue
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        current_price = df_window.iloc[i]['price']
        future_price = df_window.iloc[i + horizon_minutes]['price']
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
            'correct': prediction == actual
        })
    
    if len(results) == 0:
        return None
    
    results_df = pd.DataFrame(results)
    
    accuracy = results_df['correct'].mean()
    
    # Directional accuracy (UP/DOWN only)
    directional = results_df[results_df['predicted'] != 1]
    dir_accuracy = directional['correct'].mean() if len(directional) > 0 else 0
    
    # Trading signals accuracy
    trading = results_df[results_df['predicted'] != 1]
    trade_accuracy = trading['correct'].mean() if len(trading) > 0 else 0
    
    # Prediction distribution
    pred_dist = results_df['predicted'].value_counts()
    
    return {
        'accuracy': accuracy,
        'directional_accuracy': dir_accuracy,
        'trading_accuracy': trade_accuracy,
        'predictions': len(results),
        'up_predictions': pred_dist.get(2, 0),
        'down_predictions': pred_dist.get(0, 0),
        'sideways_predictions': pred_dist.get(1, 0),
    }

print("📊 Step 1: Fetching data...")
df_full = fetch_recent_data(hours=72)  # Get 72h for training

if df_full is None or len(df_full) < 1000:
    print("❌ Not enough data")
    sys.exit(1)

print()
print("="*80)
print("🔧 Step 2: Training models with different configurations")
print("="*80)
print()

all_results = []

for idx, config in enumerate(CONFIGS, 1):
    print(f"\n{'='*80}")
    print(f"🧪 Configuration {idx}/{len(CONFIGS)}: {config['horizon']} @ ±{config['threshold']}%")
    print(f"{'='*80}")
    
    # Train
    print("   Training...")
    model, scaler, train_metrics = train_model(df_full, config)
    
    if model is None:
        print("   ❌ Training failed")
        continue
    
    print(f"   ✅ Training complete:")
    print(f"      Accuracy: {train_metrics['accuracy']:.1%}")
    print(f"      UP: {train_metrics['up_acc']:.1%}, DOWN: {train_metrics['down_acc']:.1%}, SIDEWAYS: {train_metrics['sideways_acc']:.1%}")
    
    # Backtest on different windows
    print("   Backtesting...")
    backtest_results = {}
    
    for window in BACKTEST_WINDOWS:
        window_hours = int(window.replace('h', ''))
        bt_result = backtest_model(model, scaler, df_full, config, window_hours)
        
        if bt_result:
            backtest_results[window] = bt_result
            print(f"      {window:4s}: {bt_result['accuracy']:.1%} (Dir: {bt_result['directional_accuracy']:.1%}, Trade: {bt_result['trading_accuracy']:.1%})")
    
    # Calculate overall score
    avg_accuracy = np.mean([bt_result['accuracy'] for bt_result in backtest_results.values()])
    avg_directional = np.mean([bt_result['directional_accuracy'] for bt_result in backtest_results.values()])
    avg_trading = np.mean([bt_result['trading_accuracy'] for bt_result in backtest_results.values()])
    
    # Composite score (weighted)
    composite_score = (avg_accuracy * 0.3) + (avg_directional * 0.3) + (avg_trading * 0.4)
    
    result = {
        'config': config,
        'train_metrics': train_metrics,
        'backtest_results': backtest_results,
        'avg_accuracy': avg_accuracy,
        'avg_directional': avg_directional,
        'avg_trading': avg_trading,
        'composite_score': composite_score,
        'model': model,
        'scaler': scaler
    }
    
    all_results.append(result)
    
    print(f"\n   📊 Overall Score: {composite_score:.1%}")
    print(f"      Avg Accuracy: {avg_accuracy:.1%}")
    print(f"      Avg Directional: {avg_directional:.1%}")
    print(f"      Avg Trading: {avg_trading:.1%}")

print()
print("="*80)
print("🏆 Step 3: Ranking models")
print("="*80)
print()

# Sort by composite score
all_results.sort(key=lambda x: x['composite_score'], reverse=True)

print("TOP 5 MODELS:")
print()

for rank, result in enumerate(all_results[:5], 1):
    config = result['config']
    print(f"{rank}. {config['horizon']} @ ±{config['threshold']}%")
    print(f"   Composite Score: {result['composite_score']:.1%}")
    print(f"   Avg Accuracy:    {result['avg_accuracy']:.1%}")
    print(f"   Avg Directional: {result['avg_directional']:.1%}")
    print(f"   Avg Trading:     {result['avg_trading']:.1%}")
    print()

# Save best model
best_result = all_results[0]
best_config = best_result['config']
best_model = best_result['model']
best_scaler = best_result['scaler']

models_dir = Path(__file__).parent / 'models'
models_dir.mkdir(exist_ok=True)

with open(models_dir / 'scalping_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open(models_dir / 'scalping_scaler.pkl', 'wb') as f:
    pickle.dump(best_scaler, f)

with open(models_dir / 'scalping_config.json', 'w') as f:
    json.dump(best_config, f, indent=2)

print("="*80)
print("✅ OPTIMIZATION COMPLETE!")
print("="*80)
print()
print(f"🏆 Best Model: {best_config['horizon']} @ ±{best_config['threshold']}%")
print(f"   Composite Score: {best_result['composite_score']:.1%}")
print()
print("💾 Saved to models/")
print("   - scalping_model.pkl")
print("   - scalping_scaler.pkl")
print("   - scalping_config.json")
print()

# Save full results
results_file = Path(__file__).parent / f'optimization_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
results_data = []
for r in all_results:
    results_data.append({
        'config': r['config'],
        'train_metrics': r['train_metrics'],
        'backtest_results': r['backtest_results'],
        'avg_accuracy': r['avg_accuracy'],
        'avg_directional': r['avg_directional'],
        'avg_trading': r['avg_trading'],
        'composite_score': r['composite_score']
    })

with open(results_file, 'w') as f:
    json.dump(results_data, f, indent=2)

print(f"📊 Full results saved to: {results_file.name}")
print()
print("="*80)

