#!/usr/bin/env python3
"""
Live Bitcoin Price Direction Prediction (WITH 1-HOUR CONTEXT)
Uses the last 1 hour of data to better assess market momentum and trends
"""
import pickle
import numpy as np
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path

# Cortex configuration
CORTEX_URL = "http://10.2.20.60:9009"
CORTEX_API_INSTANT = "/prometheus/api/v1/query"
CORTEX_API_RANGE = "/prometheus/api/v1/query_range"
SYMBOL = "BTCUSDT"

# Core metrics to fetch (1 hour of history)
CORE_METRICS = [
    'crypto_last_price',
    'crypto_volume',
    'job:crypto_last_price:deriv5m',
    'job:crypto_last_price:deriv10m',
    'job:crypto_last_price:deriv15m',
    'job:crypto_last_price:avg5m',
    'job:crypto_last_price:avg10m',
    'job:crypto_last_price:avg15m',
]

def fetch_1hour_data():
    """Fetch last 1 hour of data for better trend analysis."""
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    fetch_time = end_time
    
    print(f"📡 Cortex Query Details (1-Hour Context):")
    print(f"   Endpoint: {CORTEX_URL}{CORTEX_API_RANGE}")
    print(f"   Symbol: {SYMBOL}")
    print(f"   Query time: {fetch_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"   Time range: {start_time.strftime('%H:%M:%S')} → {end_time.strftime('%H:%M:%S')}")
    print(f"   Duration: 1 hour (60 data points at 1min intervals)")
    print()
    
    all_series = {}
    
    for idx, metric in enumerate(CORE_METRICS, 1):
        query = f'{metric}{{symbol="{SYMBOL}"}}'
        params = {
            'query': query,
            'start': int(start_time.timestamp()),
            'end': int(end_time.timestamp()),
            'step': '60s'  # 1 minute intervals
        }
        
        full_url = f"{CORTEX_URL}{CORTEX_API_RANGE}?query={query}&start={int(start_time.timestamp())}&end={int(end_time.timestamp())}&step=60s"
        
        try:
            response = requests.get(f"{CORTEX_URL}{CORTEX_API_RANGE}", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'success' and data['data']['result']:
                values = data['data']['result'][0]['values']
                df = pd.DataFrame(values, columns=['timestamp', metric])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df[metric] = pd.to_numeric(df[metric], errors='coerce')
                all_series[metric] = df
                
                if idx == 1:
                    print(f"   Sample query: {full_url}")
                    print(f"   Fetched {len(df)} data points")
                    print(f"   Latest timestamp: {df['timestamp'].iloc[-1]}")
                    print(f"   Data age: {(fetch_time - df['timestamp'].iloc[-1]).total_seconds():.1f}s ago")
                    print()
            else:
                print(f"⚠️  Warning: No data for {metric}")
                all_series[metric] = None
        except Exception as e:
            print(f"❌ Error fetching {metric}: {e}")
            all_series[metric] = None
    
    return all_series, fetch_time

def compute_trend_features(all_series):
    """Compute enhanced trend features from 1-hour data."""
    features = {}
    
    # Get price series
    price_series = all_series.get('crypto_last_price')
    if price_series is not None and not price_series.empty:
        prices = price_series['crypto_last_price'].values
        
        # Current price
        features['current_price'] = prices[-1]
        
        # Price momentum over different windows
        if len(prices) >= 60:
            features['momentum_5min'] = (prices[-1] - prices[-5]) / prices[-5] * 100  # Last 5 min
            features['momentum_15min'] = (prices[-1] - prices[-15]) / prices[-15] * 100  # Last 15 min
            features['momentum_30min'] = (prices[-1] - prices[-30]) / prices[-30] * 100  # Last 30 min
            features['momentum_60min'] = (prices[-1] - prices[0]) / prices[0] * 100  # Full hour
            
            # Trend acceleration (is momentum increasing or decreasing?)
            recent_momentum = (prices[-1] - prices[-5]) / prices[-5]
            earlier_momentum = (prices[-5] - prices[-10]) / prices[-10]
            features['momentum_acceleration'] = recent_momentum - earlier_momentum
            
            # Volatility (how much price is bouncing around)
            features['volatility_15min'] = np.std(prices[-15:]) / np.mean(prices[-15:]) * 100
            features['volatility_60min'] = np.std(prices) / np.mean(prices) * 100
            
            # Trend consistency (how many of last 15 bars moved in same direction)
            price_changes = np.diff(prices[-16:])
            features['trend_consistency'] = np.sum(price_changes > 0) / len(price_changes) * 100
            
            # Price position vs moving averages
            ma_5 = np.mean(prices[-5:])
            ma_15 = np.mean(prices[-15:])
            ma_30 = np.mean(prices[-30:])
            features['price_vs_ma5'] = (prices[-1] - ma_5) / ma_5 * 100
            features['price_vs_ma15'] = (prices[-1] - ma_15) / ma_15 * 100
            features['price_vs_ma30'] = (prices[-1] - ma_30) / ma_30 * 100
        else:
            # Not enough data, use defaults
            for key in ['momentum_5min', 'momentum_15min', 'momentum_30min', 'momentum_60min',
                       'momentum_acceleration', 'volatility_15min', 'volatility_60min',
                       'trend_consistency', 'price_vs_ma5', 'price_vs_ma15', 'price_vs_ma30']:
                features[key] = 0.0
    
    # Volume features
    volume_series = all_series.get('crypto_volume')
    if volume_series is not None and not volume_series.empty:
        volumes = volume_series['crypto_volume'].values
        features['current_volume'] = volumes[-1]
        
        if len(volumes) >= 60:
            # Volume trend
            features['volume_trend'] = (volumes[-1] - np.mean(volumes[-15:])) / np.mean(volumes[-15:]) * 100
            features['volume_spike'] = volumes[-1] / np.mean(volumes[:-1]) if len(volumes) > 1 else 1.0
        else:
            features['volume_trend'] = 0.0
            features['volume_spike'] = 1.0
    else:
        features['current_volume'] = 0.0
        features['volume_trend'] = 0.0
        features['volume_spike'] = 1.0
    
    # Derivative features (already computed by Prometheus)
    for metric in ['job:crypto_last_price:deriv5m', 'job:crypto_last_price:deriv10m', 'job:crypto_last_price:deriv15m']:
        series = all_series.get(metric)
        if series is not None and not series.empty:
            values = series[metric].values
            features[metric] = values[-1]
        else:
            features[metric] = 0.0
    
    # Average features
    for metric in ['job:crypto_last_price:avg5m', 'job:crypto_last_price:avg10m', 'job:crypto_last_price:avg15m']:
        series = all_series.get(metric)
        if series is not None and not series.empty:
            values = series[metric].values
            features[metric] = values[-1]
        else:
            features[metric] = 0.0
    
    return features

def load_model():
    """Load the trained model, scaler, and config."""
    model_dir = Path(__file__).parent / 'models'
    
    with open(model_dir / 'scalping_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open(model_dir / 'scalping_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open(model_dir / 'scalping_config.json', 'r') as f:
        config = json.load(f)
    
    return model, scaler, config

def predict():
    """Make a live prediction with 1-hour context."""
    print("="*80)
    print("🎯 BTC SCALPING SIGNAL - WITH 1-HOUR CONTEXT")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load model
    print("📥 Loading model...")
    model, scaler, config = load_model()
    print(f"   Model: {config['horizon']} horizon, ±{config['threshold']}% threshold")
    print()
    
    # Fetch 1-hour data
    print("📡 Fetching 1-hour historical data from Cortex...")
    print()
    all_series, fetch_time = fetch_1hour_data()
    print(f"   ✅ Fetched data for {len([s for s in all_series.values() if s is not None])} metrics")
    print()
    
    # Compute enhanced features
    print("🔧 Computing trend features from 1-hour context...")
    trend_features = compute_trend_features(all_series)
    print(f"   ✅ Computed {len(trend_features)} trend features")
    print()
    
    # Show key trend features
    print("📋 1-Hour Trend Analysis:")
    print(f"   Current Price:        ${trend_features.get('current_price', 0):,.2f}")
    print(f"   5-min momentum:       {trend_features.get('momentum_5min', 0):+.3f}%")
    print(f"   15-min momentum:      {trend_features.get('momentum_15min', 0):+.3f}%")
    print(f"   30-min momentum:      {trend_features.get('momentum_30min', 0):+.3f}%")
    print(f"   60-min momentum:      {trend_features.get('momentum_60min', 0):+.3f}%")
    print(f"   Momentum acceleration: {trend_features.get('momentum_acceleration', 0):+.6f}")
    print(f"   Trend consistency:    {trend_features.get('trend_consistency', 0):.1f}% bullish")
    print(f"   Volatility (15min):   {trend_features.get('volatility_15min', 0):.3f}%")
    print(f"   Price vs MA(5):       {trend_features.get('price_vs_ma5', 0):+.3f}%")
    print(f"   Price vs MA(15):      {trend_features.get('price_vs_ma15', 0):+.3f}%")
    print(f"   Volume trend:         {trend_features.get('volume_trend', 0):+.1f}%")
    print()
    
    # Map trend features to model's expected features
    # NOTE: The model expects the original 11 features, so we need to map our enhanced features
    # For now, use the core features the model was trained on
    model_features = np.array([
        trend_features.get('current_price', 0),
        trend_features.get('job:crypto_last_price:deriv5m', 0),
        trend_features.get('job:crypto_last_price:deriv10m', 0),
        trend_features.get('job:crypto_last_price:deriv15m', 0),
        trend_features.get('momentum_30min', 0) * 10,  # Use 30min momentum as proxy for 30m deriv
        trend_features.get('job:crypto_last_price:avg5m', 0),
        trend_features.get('job:crypto_last_price:avg10m', 0),
        trend_features.get('job:crypto_last_price:avg15m', 0),
        trend_features.get('current_volume', 0),
        trend_features.get('volume_trend', 0),
        trend_features.get('current_volume', 0) * 0.99,  # Approximate volume avg
    ]).reshape(1, -1)
    
    # Replace NaN/inf with 0
    model_features = np.nan_to_num(model_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale features
    features_scaled = scaler.transform(model_features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    labels = ['DOWN', 'SIDEWAYS', 'UP']
    predicted_label = labels[prediction]
    confidence = probabilities[prediction]
    
    # Display results
    print("="*80)
    print("📊 PREDICTION RESULT (WITH 1-HOUR CONTEXT)")
    print("="*80)
    print()
    
    # Parse horizon to minutes
    horizon_str = config['horizon']
    if 'min' in horizon_str:
        horizon_minutes = int(horizon_str.replace('min', ''))
    elif 'h' in horizon_str:
        horizon_minutes = int(horizon_str.replace('h', '')) * 60
    else:
        horizon_minutes = 15
    
    prediction_target_time = fetch_time + timedelta(minutes=horizon_minutes)
    
    print(f"   Prediction for: {prediction_target_time.strftime('%Y-%m-%d %H:%M:%S')} ({horizon_str} from now)")
    print(f"   Direction:      {predicted_label}")
    print(f"   Confidence:     {confidence:.1%}")
    print()
    
    # Enhanced context interpretation
    print("   Trend Context:")
    momentum_60 = trend_features.get('momentum_60min', 0)
    momentum_15 = trend_features.get('momentum_15min', 0)
    acceleration = trend_features.get('momentum_acceleration', 0)
    
    if momentum_60 > 0.1:
        print(f"     • Strong uptrend over last hour (+{momentum_60:.2f}%)")
    elif momentum_60 < -0.1:
        print(f"     • Strong downtrend over last hour ({momentum_60:.2f}%)")
    else:
        print(f"     • Sideways/choppy last hour ({momentum_60:+.2f}%)")
    
    if acceleration > 0.001:
        print(f"     • Momentum ACCELERATING (gaining strength)")
    elif acceleration < -0.001:
        print(f"     • Momentum DECELERATING (losing strength)")
    else:
        print(f"     • Momentum stable")
    
    print()
    print("   All Probabilities:")
    print(f"     DOWN:     {probabilities[0]:.1%}")
    print(f"     SIDEWAYS: {probabilities[1]:.1%}")
    print(f"     UP:       {probabilities[2]:.1%}")
    print()
    
    # Trading signal
    print("="*80)
    print("🚨 TRADING SIGNAL")
    print("="*80)
    
    if prediction == 1:  # SIDEWAYS
        print("   ⏸️  NO TRADE - Market is sideways")
        print("   Action: Stay out, wait for clearer signal")
    elif confidence > 0.80:
        action = "SELL" if prediction == 0 else "BUY"
        print(f"   🔥 HIGH CONFIDENCE SIGNAL: {action}")
        print(f"   Action: Execute {action} with 1-2% position")
        print(f"   Entry: Current price (${trend_features.get('current_price', 0):,.2f})")
        print(f"   Stop-loss: {config['threshold'] * 1.5:.2f}%")
        print(f"   Take-profit: {config['threshold'] * 2:.2f}%")
    elif confidence > 0.70:
        action = "SELL" if prediction == 0 else "BUY"
        print(f"   ⚠️  MEDIUM CONFIDENCE: {action}")
        print(f"   Action: Execute {action} with 0.5-1% position")
        print(f"   Entry: Current price (${trend_features.get('current_price', 0):,.2f})")
        print(f"   Stop-loss: {config['threshold'] * 1.5:.2f}%")
        print(f"   Take-profit: {config['threshold'] * 2:.2f}%")
    else:
        action = "SELL" if prediction == 0 else "BUY"
        print(f"   🔕 LOW CONFIDENCE: {action}")
        print(f"   Action: SKIP - Confidence too low ({confidence:.1%})")
        print("   Wait for better setup")
    
    print()
    print("="*80)
    
    # Return structured data
    return {
        'timestamp': datetime.now().isoformat(),
        'prediction': predicted_label,
        'prediction_code': int(prediction),
        'confidence': float(confidence),
        'probabilities': {
            'down': float(probabilities[0]),
            'sideways': float(probabilities[1]),
            'up': float(probabilities[2])
        },
        'should_trade': bool(prediction != 1 and confidence > 0.70),
        'action': 'SELL' if prediction == 0 else ('BUY' if prediction == 2 else 'NONE'),
        'trend_features': {k: float(v) if isinstance(v, (int, float)) else v for k, v in trend_features.items()}
    }

if __name__ == "__main__":
    try:
        result = predict()
        
        # Save to log
        log_file = Path(__file__).parent / 'predictions_with_context.jsonl'
        with open(log_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
        
        print(f"💾 Prediction logged to: {log_file.name}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

