#!/usr/bin/env python3
"""
Live Bitcoin Price Direction Prediction
Uses the trained aggressive scalping model to predict 15-min direction
"""
import pickle
import numpy as np
import requests
import json
from datetime import datetime
from pathlib import Path

# Cortex configuration
CORTEX_URL = "http://10.2.20.60:9009"
CORTEX_API = "/prometheus/api/v1/query"
SYMBOL = "BTCUSDT"

# Features the model needs (in order!)
REQUIRED_FEATURES = [
    'crypto_last_price',
    'job:crypto_last_price:deriv5m',
    'job:crypto_last_price:deriv10m',
    'job:crypto_last_price:deriv15m',
    'job:crypto_last_price:deriv30m',
    'job:crypto_last_price:avg5m',
    'job:crypto_last_price:avg10m',
    'job:crypto_last_price:avg15m',
    'crypto_volume',
    'job:crypto_volume:deriv5m',
    'job:crypto_volume:avg5m',
]

def fetch_current_data():
    """Fetch current values for all required features from Cortex."""
    features = []
    fetch_time = datetime.now()
    
    print(f"📡 Cortex Query Details:")
    print(f"   Endpoint: {CORTEX_URL}{CORTEX_API}")
    print(f"   Symbol: {SYMBOL}")
    print(f"   Query time: {fetch_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()
    
    for idx, metric in enumerate(REQUIRED_FEATURES, 1):
        query = f'{metric}{{symbol="{SYMBOL}"}}'
        params = {'query': query}
        
        # Build full URL for transparency
        full_url = f"{CORTEX_URL}{CORTEX_API}?query={query}"
        
        try:
            response = requests.get(f"{CORTEX_URL}{CORTEX_API}", params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'success' and data['data']['result']:
                value = float(data['data']['result'][0]['value'][1])
                timestamp = float(data['data']['result'][0]['value'][0])
                data_time = datetime.fromtimestamp(timestamp)
                
                features.append(value)
                
                # Show first and last metric for transparency
                if idx == 1:
                    print(f"   Sample query: {full_url}")
                    print(f"   Data timestamp: {data_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"   Data age: {(fetch_time - data_time).total_seconds():.1f}s ago")
                    print()
            else:
                print(f"⚠️  Warning: No data for {metric}, using 0")
                features.append(0.0)
        except Exception as e:
            print(f"❌ Error fetching {metric}: {e}")
            features.append(0.0)
    
    return np.array(features).reshape(1, -1), fetch_time

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
    """Make a live prediction."""
    print("="*80)
    print("🎯 BTC SCALPING SIGNAL - LIVE PREDICTION")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load model
    print("📥 Loading model...")
    model, scaler, config = load_model()
    print(f"   Model: {config['horizon']} horizon, ±{config['threshold']}% threshold")
    print()
    
    # Fetch data
    print("📡 Fetching current data from Cortex...")
    print()
    features, fetch_time = fetch_current_data()
    print(f"   ✅ Fetched {len(features[0])} features successfully")
    print()
    
    # Show feature values for transparency
    print("📋 Current Feature Values:")
    print(f"   BTC Price:           ${features[0][0]:,.2f}")
    print(f"   5m derivative:       {features[0][1]:,.4f}")
    print(f"   10m derivative:      {features[0][2]:,.4f}")
    print(f"   15m derivative:      {features[0][3]:,.4f}")
    print(f"   30m derivative:      {features[0][4]:,.4f}")
    print(f"   5m average:          ${features[0][5]:,.2f}")
    print(f"   10m average:         ${features[0][6]:,.2f}")
    print(f"   15m average:         ${features[0][7]:,.2f}")
    print(f"   Volume:              {features[0][8]:,.2f}")
    print(f"   Volume 5m deriv:     {features[0][9]:,.4f}")
    print(f"   Volume 5m avg:       {features[0][10]:,.2f}")
    print()
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    labels = ['DOWN', 'SIDEWAYS', 'UP']
    predicted_label = labels[prediction]
    confidence = probabilities[prediction]
    
    # Display results
    print("="*80)
    print("📊 PREDICTION RESULT")
    print("="*80)
    print()
    
    # Parse horizon to minutes
    horizon_str = config['horizon']
    if 'min' in horizon_str:
        horizon_minutes = int(horizon_str.replace('min', ''))
    elif 'h' in horizon_str:
        horizon_minutes = int(horizon_str.replace('h', '')) * 60
    else:
        horizon_minutes = 15  # default
    
    from datetime import timedelta
    prediction_target_time = fetch_time + timedelta(minutes=horizon_minutes)
    
    print(f"   Prediction for: {prediction_target_time.strftime('%Y-%m-%d %H:%M:%S')} ({horizon_str} from now)")
    print(f"   Direction:      {predicted_label}")
    print(f"   Confidence:     {confidence:.1%}")
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
        print(f"   Entry: Current price")
        print(f"   Stop-loss: {config['threshold'] * 1.5:.2f}%")
        print(f"   Take-profit: {config['threshold'] * 2:.2f}%")
    elif confidence > 0.70:
        action = "SELL" if prediction == 0 else "BUY"
        print(f"   ⚠️  MEDIUM CONFIDENCE: {action}")
        print(f"   Action: Execute {action} with 0.5-1% position")
        print(f"   Entry: Current price")
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
        'features': [float(f) for f in features[0]]
    }

if __name__ == "__main__":
    try:
        result = predict()
        
        # Optionally save to file for logging
        log_file = Path(__file__).parent / 'predictions.jsonl'
        with open(log_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
