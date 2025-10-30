#!/usr/bin/env python3
"""
Improved Model Training with:
1. Actual volume data
2. Market regime detection
3. Extended lookback (4 hours)
4. Technical indicators (RSI, MACD, Bollinger Bands)
5. Ensemble approach
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
import json
import pickle
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

# Cortex configuration
CORTEX_URL = "http://10.2.20.60:9009"
CORTEX_API_RANGE = "/prometheus/api/v1/query_range"
SYMBOL = "BTCUSDT"

def fetch_data_with_volume(hours=96):
    """Fetch price, volume, and weighted derivative data."""
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    print(f"📥 Fetching {hours}h of price and volume data...")
    
    # Fetch price
    query = f'crypto_last_price{{symbol="{SYMBOL}"}}'
    params = {
        'query': query,
        'start': int(start_time.timestamp()),
        'end': int(end_time.timestamp()),
        'step': '60s'
    }
    
    try:
        response = requests.get(f"{CORTEX_URL}{CORTEX_API_RANGE}", params=params, timeout=30)
        data = response.json()
        
        if data['status'] == 'success' and data['data']['result']:
            values = data['data']['result'][0]['values']
            df = pd.DataFrame(values, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df = df.set_index('timestamp')
            df = df.dropna()
            
            print(f"   ✅ Price: {len(df)} points")
            
            # Fetch volume
            query_vol = f'crypto_volume{{symbol="{SYMBOL}"}}'
            params_vol = {
                'query': query_vol,
                'start': int(start_time.timestamp()),
                'end': int(end_time.timestamp()),
                'step': '60s'
            }
            
            response_vol = requests.get(f"{CORTEX_URL}{CORTEX_API_RANGE}", params=params_vol, timeout=30)
            data_vol = response_vol.json()
            
            if data_vol['status'] == 'success' and data_vol['data']['result']:
                values_vol = data_vol['data']['result'][0]['values']
                df_vol = pd.DataFrame(values_vol, columns=['timestamp', 'volume'])
                df_vol['timestamp'] = pd.to_datetime(df_vol['timestamp'], unit='s')
                df_vol['volume'] = pd.to_numeric(df_vol['volume'], errors='coerce')
                df_vol = df_vol.set_index('timestamp')
                
                # Merge
                df = df.join(df_vol, how='left')
                df['volume'] = df['volume'].fillna(0)
                
                print(f"   ✅ Volume: {(df['volume'] > 0).sum()} points")
            else:
                print(f"   ⚠️  No volume data, using zeros")
                df['volume'] = 0
            
            # Fetch weighted derivative
            query_wderiv = f'job:crypto_last_price:weighted_deriv:24h:48h:7d{{symbol="{SYMBOL}"}}'
            params_wderiv = {
                'query': query_wderiv,
                'start': int(start_time.timestamp()),
                'end': int(end_time.timestamp()),
                'step': '60s'
            }
            
            response_wderiv = requests.get(f"{CORTEX_URL}{CORTEX_API_RANGE}", params=params_wderiv, timeout=30)
            data_wderiv = response_wderiv.json()
            
            if data_wderiv['status'] == 'success' and data_wderiv['data']['result']:
                values_wderiv = data_wderiv['data']['result'][0]['values']
                df_wderiv = pd.DataFrame(values_wderiv, columns=['timestamp', 'weighted_deriv'])
                df_wderiv['timestamp'] = pd.to_datetime(df_wderiv['timestamp'], unit='s')
                df_wderiv['weighted_deriv'] = pd.to_numeric(df_wderiv['weighted_deriv'], errors='coerce')
                df_wderiv = df_wderiv.set_index('timestamp')
                
                # Merge
                df = df.join(df_wderiv, how='left')
                df['weighted_deriv'] = df['weighted_deriv'].fillna(0)
                
                print(f"   ✅ Weighted deriv: {(df['weighted_deriv'] != 0).sum()} points")
            else:
                print(f"   ⚠️  No weighted deriv data, using zeros")
                df['weighted_deriv'] = 0
            
            print(f"   📊 Total: {len(df)} samples")
            return df
        else:
            print(f"❌ No data")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def detect_market_regime(prices, window=60):
    """Detect current market regime."""
    if len(prices) < window:
        return 'unknown'
    
    recent = prices[-window:]
    
    # Calculate trend strength
    returns = np.diff(recent) / recent[:-1]
    avg_return = np.mean(returns)
    volatility = np.std(returns)
    
    # Trend detection
    if avg_return > 0.0005 and volatility < 0.002:
        return 'trending_up'
    elif avg_return < -0.0005 and volatility < 0.002:
        return 'trending_down'
    elif volatility > 0.003:
        return 'high_volatility'
    else:
        return 'sideways'

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    if len(prices) < period + 1:
        return 50.0
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(prices):
    """Calculate MACD indicator."""
    if len(prices) < 26:
        return 0.0
    
    ema12 = pd.Series(prices).ewm(span=12, adjust=False).mean().iloc[-1]
    ema26 = pd.Series(prices).ewm(span=26, adjust=False).mean().iloc[-1]
    
    macd = ema12 - ema26
    return macd

def calculate_bollinger_bands(prices, period=20):
    """Calculate Bollinger Bands position."""
    if len(prices) < period:
        return 0.5
    
    recent = prices[-period:]
    ma = np.mean(recent)
    std = np.std(recent)
    
    current_price = prices[-1]
    upper = ma + (2 * std)
    lower = ma - (2 * std)
    
    if upper == lower:
        return 0.5
    
    # Position within bands (0 = lower, 1 = upper)
    position = (current_price - lower) / (upper - lower)
    return np.clip(position, 0, 1)

def compute_advanced_features(df, idx):
    """Compute advanced features with 4-hour lookback."""
    if idx < 240:  # Need 4 hours (240 minutes)
        return None
    
    window = df.iloc[idx-240:idx+1]
    prices = window['price'].values
    volumes = window['volume'].values if 'volume' in window.columns else np.zeros(len(window))
    weighted_derivs = window['weighted_deriv'].values if 'weighted_deriv' in window.columns else np.zeros(len(window))
    
    if len(prices) < 241 or np.any(np.isnan(prices)):
        return None
    
    features = {}
    
    # Basic price features
    features['price'] = prices[-1]
    
    # Weighted derivative feature (from Prometheus)
    features['weighted_deriv'] = weighted_derivs[-1]
    features['weighted_deriv_norm'] = weighted_derivs[-1] / (prices[-1] + 1e-10)
    
    # Multi-timeframe derivatives
    for window_size in [5, 10, 15, 30, 60, 120, 240]:
        if len(prices) >= window_size:
            deriv = (prices[-1] - prices[-window_size]) / prices[-window_size]
            features[f'deriv_{window_size}m'] = deriv
    
    # Multi-timeframe moving averages
    for window_size in [5, 10, 15, 30, 60, 120, 240]:
        if len(prices) >= window_size:
            ma = np.mean(prices[-window_size:])
            features[f'ma_{window_size}m'] = ma
            # Price position vs MA
            features[f'price_vs_ma_{window_size}m'] = (prices[-1] - ma) / ma
    
    # Volatility features
    for window_size in [15, 30, 60, 120]:
        if len(prices) >= window_size:
            vol = np.std(prices[-window_size:]) / np.mean(prices[-window_size:])
            features[f'volatility_{window_size}m'] = vol
    
    # Momentum features
    for window_size in [5, 15, 30, 60]:
        if len(prices) >= window_size + 1:
            recent_momentum = (prices[-1] - prices[-window_size]) / prices[-window_size]
            earlier_momentum = (prices[-window_size] - prices[-window_size*2]) / prices[-window_size*2] if len(prices) >= window_size*2 else 0
            features[f'momentum_accel_{window_size}m'] = recent_momentum - earlier_momentum
    
    # Technical indicators
    features['rsi_14'] = calculate_rsi(prices, 14)
    features['rsi_28'] = calculate_rsi(prices, 28)
    features['macd'] = calculate_macd(prices)
    features['bollinger_position'] = calculate_bollinger_bands(prices, 20)
    
    # Market regime
    regime = detect_market_regime(prices, 60)
    features['regime_trending_up'] = 1 if regime == 'trending_up' else 0
    features['regime_trending_down'] = 1 if regime == 'trending_down' else 0
    features['regime_sideways'] = 1 if regime == 'sideways' else 0
    features['regime_high_vol'] = 1 if regime == 'high_volatility' else 0
    
    # Volume features (if available)
    if np.any(volumes > 0):
        features['volume'] = volumes[-1]
        features['volume_ma_15'] = np.mean(volumes[-15:]) if len(volumes) >= 15 else volumes[-1]
        features['volume_ratio'] = volumes[-1] / np.mean(volumes[-30:]) if len(volumes) >= 30 and np.mean(volumes[-30:]) > 0 else 1.0
    else:
        features['volume'] = 0
        features['volume_ma_15'] = 0
        features['volume_ratio'] = 1.0
    
    # Volume derivatives (always create these features)
    for window_size in [5, 15, 30]:
        if np.any(volumes > 0) and len(volumes) >= window_size and volumes[-window_size] > 0:
            vol_deriv = (volumes[-1] - volumes[-window_size]) / volumes[-window_size]
            features[f'volume_deriv_{window_size}m'] = vol_deriv
        else:
            features[f'volume_deriv_{window_size}m'] = 0
    
    return features

def train_model(df, config):
    """Train model with configuration."""
    horizon_str = config['horizon']
    threshold = config['threshold']
    
    if 'min' in horizon_str:
        horizon_minutes = int(horizon_str.replace('min', ''))
    elif 'h' in horizon_str:
        horizon_minutes = int(horizon_str.replace('h', '')) * 60
    else:
        horizon_minutes = 15
    
    print(f"\n🔧 Training {horizon_str} @ ±{threshold}%...")
    
    # Prepare data
    X_list = []
    y_list = []
    feature_names = None
    
    for i in range(240, len(df) - horizon_minutes):
        features_dict = compute_advanced_features(df, i)
        if features_dict is None:
            continue
        
        if feature_names is None:
            feature_names = list(features_dict.keys())
        
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
        
        X_list.append([features_dict[fn] for fn in feature_names])
        y_list.append(label)
    
    if len(X_list) == 0:
        return None, None, None, None
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Replace inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Train/test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {len(feature_names)}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Custom aggressive class weights: heavily favor UP/DOWN over SIDEWAYS
    # DOWN (0): 3.0, SIDEWAYS (1): 0.2, UP (2): 3.0
    aggressive_weights = {0: 3.0, 1: 0.2, 2: 3.0}
    
    # Train
    model = LGBMClassifier(
        random_state=42,
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=40,
        max_depth=8,
        class_weight=aggressive_weights,  # Aggressive: penalize SIDEWAYS heavily
        verbose=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Per-class
    up_mask = y_test == 2
    down_mask = y_test == 0
    sideways_mask = y_test == 1
    
    up_acc = accuracy_score(y_test[up_mask], y_pred[up_mask]) if up_mask.sum() > 0 else 0
    down_acc = accuracy_score(y_test[down_mask], y_pred[down_mask]) if down_mask.sum() > 0 else 0
    sideways_acc = accuracy_score(y_test[sideways_mask], y_pred[sideways_mask]) if sideways_mask.sum() > 0 else 0
    
    # Directional
    directional_mask = (y_test != 1) & (y_pred != 1)
    dir_acc = accuracy_score(y_test[directional_mask], y_pred[directional_mask]) if directional_mask.sum() > 0 else 0
    
    # Count signal distribution
    pred_up_pct = (y_pred == 2).sum() / len(y_pred) * 100
    pred_down_pct = (y_pred == 0).sum() / len(y_pred) * 100
    pred_sideways_pct = (y_pred == 1).sum() / len(y_pred) * 100
    
    print(f"   ✅ Accuracy: {accuracy:.1%}")
    print(f"      UP: {up_acc:.1%}, DOWN: {down_acc:.1%}, SIDEWAYS: {sideways_acc:.1%}")
    print(f"      Directional: {dir_acc:.1%}")
    print(f"      Signal distribution: UP {pred_up_pct:.1f}%, DOWN {pred_down_pct:.1f}%, SIDEWAYS {pred_sideways_pct:.1f}%")
    
    return model, scaler, feature_names, {
        'accuracy': accuracy,
        'pred_up_pct': pred_up_pct,
        'pred_down_pct': pred_down_pct,
        'pred_sideways_pct': pred_sideways_pct,
        'up_acc': up_acc,
        'down_acc': down_acc,
        'sideways_acc': sideways_acc,
        'directional_acc': dir_acc
    }

if __name__ == "__main__":
    print("="*80)
    print("🚀 TRAINING IMPROVED MODEL WITH ADVANCED FEATURES")
    print("="*80)
    print()
    
    # Fetch data
    df = fetch_data_with_volume(hours=96)

    if df is None or len(df) < 1000:
        print("❌ Not enough data")
        sys.exit(1)

    print()

    # Train AGGRESSIVE configurations (lower thresholds = more signals)
    configs = [
        {'horizon': '15min', 'threshold': 0.03},  # Very aggressive
        {'horizon': '15min', 'threshold': 0.05},  # Aggressive
        {'horizon': '15min', 'threshold': 0.08},  # Moderate
        {'horizon': '30min', 'threshold': 0.05},  # Aggressive
        {'horizon': '30min', 'threshold': 0.08},  # Moderate
        {'horizon': '1h', 'threshold': 0.10},     # Moderate
    ]

    best_model = None
    best_score = 0
    best_config = None
    best_scaler = None
    best_feature_names = None

    for config in configs:
        model, scaler, feature_names, metrics = train_model(df, config)
        
        if model is not None:
            # AGGRESSIVE scoring: favor models with more UP/DOWN signals
            # Penalize high SIDEWAYS percentage
            directional_signal_pct = metrics['pred_up_pct'] + metrics['pred_down_pct']
            sideways_penalty = max(0, (metrics['pred_sideways_pct'] - 50) / 100)  # Penalize if >50% sideways
            
            # Composite score: favor directional accuracy + signal generation
            score = (
                metrics['directional_acc'] * 0.5 +  # Directional accuracy is key
                (directional_signal_pct / 100) * 0.3 +  # Reward more signals
                metrics['accuracy'] * 0.2 -  # Overall accuracy
                sideways_penalty * 0.3  # Penalize excessive sideways
            )
            
            print(f"      Directional signals: {directional_signal_pct:.1f}%, Score: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_model = model
                best_config = config
                best_scaler = scaler
                best_feature_names = feature_names

    print()
    print("="*80)
    print(f"🏆 BEST MODEL: {best_config['horizon']} @ ±{best_config['threshold']}%")
    print(f"   Score: {best_score:.1%}")
    print("="*80)
    print()

    # Save
    models_dir = Path(__file__).parent / 'models'
    models_dir.mkdir(exist_ok=True)

    with open(models_dir / 'scalping_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    with open(models_dir / 'scalping_scaler.pkl', 'wb') as f:
        pickle.dump(best_scaler, f)

    with open(models_dir / 'scalping_config.json', 'w') as f:
        json.dump(best_config, f, indent=2)

    with open(models_dir / 'feature_names.json', 'w') as f:
        json.dump(best_feature_names, f, indent=2)

    print("💾 Saved:")
    print("   - scalping_model.pkl")
    print("   - scalping_scaler.pkl")
    print("   - scalping_config.json")
    print("   - feature_names.json")
    print()
    print("="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)

