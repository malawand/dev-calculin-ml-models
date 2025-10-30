#!/usr/bin/env python3
"""
Backtest Recent Performance
Tests the model on the last 4 hours of data to see how it would have performed
"""
import pickle
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
import json

# Cortex configuration
CORTEX_URL = "http://10.2.20.60:9009"
CORTEX_API_RANGE = "/prometheus/api/v1/query_range"
SYMBOL = "BTCUSDT"

def fetch_recent_data(hours=4):
    """Fetch recent data for backtesting."""
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    print(f"📥 Fetching last {hours} hours of data...")
    print(f"   From: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   To:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Fetch price data at 1-minute intervals
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
            
            print(f"   ✅ Fetched {len(df)} data points")
            print(f"   Price range: ${df['price'].min():,.2f} - ${df['price'].max():,.2f}")
            print()
            
            return df
        else:
            print(f"❌ No data returned")
            return None
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

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

def compute_features(df, idx):
    """Compute features for a given index using historical data."""
    # Need at least 60 previous points for 1-hour context
    if idx < 60:
        return None
    
    # Get historical window
    window = df.iloc[idx-60:idx+1]
    prices = window['price'].values
    
    if len(prices) < 61 or np.any(np.isnan(prices)):
        return None
    
    # Compute features (matching the model's expected features)
    features = []
    
    # Current price
    features.append(prices[-1])
    
    # Derivatives (rate of change)
    deriv_5m = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
    deriv_10m = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
    deriv_15m = (prices[-1] - prices[-15]) / prices[-15] if len(prices) >= 15 else 0
    deriv_30m = (prices[-1] - prices[-30]) / prices[-30] if len(prices) >= 30 else 0
    
    features.extend([deriv_5m, deriv_10m, deriv_15m, deriv_30m])
    
    # Moving averages
    avg_5m = np.mean(prices[-5:]) if len(prices) >= 5 else prices[-1]
    avg_10m = np.mean(prices[-10:]) if len(prices) >= 10 else prices[-1]
    avg_15m = np.mean(prices[-15:]) if len(prices) >= 15 else prices[-1]
    
    features.extend([avg_5m, avg_10m, avg_15m])
    
    # Volume placeholders (we don't have volume in simple fetch)
    features.extend([0.0, 0.0, 0.0])
    
    return np.array(features).reshape(1, -1)

def backtest_recent(hours=4):
    """Backtest the model on recent data."""
    print("="*80)
    print("🔬 BACKTESTING LAST 4 HOURS")
    print("="*80)
    print()
    
    # Load model
    print("📥 Loading model...")
    model, scaler, config = load_model()
    print(f"   Model: {config['horizon']} horizon, ±{config['threshold']}% threshold")
    print()
    
    # Fetch data
    df = fetch_recent_data(hours)
    if df is None or len(df) < 100:
        print("❌ Not enough data for backtesting")
        return
    
    # Parse horizon
    horizon_str = config['horizon']
    if 'min' in horizon_str:
        horizon_minutes = int(horizon_str.replace('min', ''))
    else:
        horizon_minutes = 15
    
    threshold = config['threshold']
    
    print(f"🧪 Running backtest...")
    print(f"   Testing every 15 minutes")
    print(f"   Prediction horizon: {horizon_minutes} minutes")
    print(f"   Threshold: ±{threshold}%")
    print()
    
    # Test at 15-minute intervals
    test_intervals = list(range(60, len(df) - horizon_minutes, 15))
    
    results = []
    predictions_made = 0
    
    for i in test_intervals:
        # Compute features at this point
        features = compute_features(df, i)
        if features is None:
            continue
        
        # Make prediction
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        labels = ['DOWN', 'SIDEWAYS', 'UP']
        predicted_label = labels[prediction]
        confidence = probabilities[prediction]
        
        # Get actual outcome
        current_price = df.iloc[i]['price']
        future_idx = min(i + horizon_minutes, len(df) - 1)
        future_price = df.iloc[future_idx]['price']
        
        actual_change_pct = (future_price - current_price) / current_price * 100
        
        # Determine actual label
        if actual_change_pct > threshold:
            actual_label = 'UP'
        elif actual_change_pct < -threshold:
            actual_label = 'DOWN'
        else:
            actual_label = 'SIDEWAYS'
        
        # Record result
        result = {
            'timestamp': df.index[i],
            'price': current_price,
            'predicted': predicted_label,
            'actual': actual_label,
            'confidence': confidence,
            'actual_change_pct': actual_change_pct,
            'correct': predicted_label == actual_label,
            'should_trade': prediction != 1 and confidence > 0.70
        }
        
        results.append(result)
        predictions_made += 1
    
    if not results:
        print("❌ No predictions could be made")
        return
    
    # Calculate metrics
    results_df = pd.DataFrame(results)
    
    total_predictions = len(results_df)
    correct_predictions = results_df['correct'].sum()
    overall_accuracy = correct_predictions / total_predictions * 100
    
    # Accuracy by class
    up_pred = results_df[results_df['predicted'] == 'UP']
    down_pred = results_df[results_df['predicted'] == 'DOWN']
    sideways_pred = results_df[results_df['predicted'] == 'SIDEWAYS']
    
    up_correct = up_pred['correct'].sum() if len(up_pred) > 0 else 0
    down_correct = down_pred['correct'].sum() if len(down_pred) > 0 else 0
    sideways_correct = sideways_pred['correct'].sum() if len(sideways_pred) > 0 else 0
    
    up_accuracy = (up_correct / len(up_pred) * 100) if len(up_pred) > 0 else 0
    down_accuracy = (down_correct / len(down_pred) * 100) if len(down_pred) > 0 else 0
    sideways_accuracy = (sideways_correct / len(sideways_pred) * 100) if len(sideways_pred) > 0 else 0
    
    # Directional accuracy (ignoring SIDEWAYS)
    directional = results_df[results_df['predicted'] != 'SIDEWAYS']
    directional_accuracy = (directional['correct'].sum() / len(directional) * 100) if len(directional) > 0 else 0
    
    # High-confidence accuracy
    high_conf = results_df[results_df['confidence'] > 0.80]
    high_conf_accuracy = (high_conf['correct'].sum() / len(high_conf) * 100) if len(high_conf) > 0 else 0
    
    # Trading signals
    trading_signals = results_df[results_df['should_trade']]
    trading_accuracy = (trading_signals['correct'].sum() / len(trading_signals) * 100) if len(trading_signals) > 0 else 0
    
    # Display results
    print("="*80)
    print("📊 BACKTEST RESULTS (LAST 4 HOURS)")
    print("="*80)
    print()
    
    print(f"⏱️  Test Period:")
    print(f"   From: {results_df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   To:   {results_df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Duration: {(results_df['timestamp'].max() - results_df['timestamp'].min()).total_seconds() / 3600:.1f} hours")
    print()
    
    print(f"📈 Predictions Made: {total_predictions}")
    print(f"   UP:       {len(up_pred):3d} ({len(up_pred)/total_predictions*100:5.1f}%)")
    print(f"   DOWN:     {len(down_pred):3d} ({len(down_pred)/total_predictions*100:5.1f}%)")
    print(f"   SIDEWAYS: {len(sideways_pred):3d} ({len(sideways_pred)/total_predictions*100:5.1f}%)")
    print()
    
    print(f"🎯 Overall Accuracy: {overall_accuracy:.1f}% ({correct_predictions}/{total_predictions})")
    print()
    
    print(f"📊 Per-Class Accuracy:")
    print(f"   UP:       {up_accuracy:5.1f}% ({up_correct}/{len(up_pred)})")
    print(f"   DOWN:     {down_accuracy:5.1f}% ({down_correct}/{len(down_pred)})")
    print(f"   SIDEWAYS: {sideways_accuracy:5.1f}% ({sideways_correct}/{len(sideways_pred)})")
    print()
    
    print(f"🎲 Directional Accuracy: {directional_accuracy:.1f}% (UP/DOWN only)")
    print()
    
    print(f"🔥 High-Confidence (>80%): {high_conf_accuracy:.1f}% ({len(high_conf)} predictions)")
    print()
    
    print(f"💰 Trading Signals (>70% conf, not SIDEWAYS):")
    print(f"   Signals: {len(trading_signals)}")
    print(f"   Accuracy: {trading_accuracy:.1f}%")
    print(f"   Expected win rate for trading: {trading_accuracy:.1f}%")
    print()
    
    # Recent predictions
    print("="*80)
    print("📜 LAST 10 PREDICTIONS:")
    print("="*80)
    print()
    
    for _, row in results_df.tail(10).iterrows():
        status = "✅" if row['correct'] else "❌"
        conf_icon = "🔥" if row['confidence'] > 0.80 else ("⚠️" if row['confidence'] > 0.70 else "🔕")
        
        print(f"{status} {row['timestamp'].strftime('%H:%M:%S')} | "
              f"Pred: {row['predicted']:8s} ({row['confidence']:5.1%}) {conf_icon} | "
              f"Actual: {row['actual']:8s} ({row['actual_change_pct']:+6.3f}%) | "
              f"Price: ${row['price']:,.2f}")
    
    print()
    print("="*80)
    
    # Save results
    output_file = Path(__file__).parent / f'backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"💾 Results saved to: {output_file.name}")
    print()
    
    # Summary
    print("="*80)
    print("📊 SUMMARY")
    print("="*80)
    print()
    
    if overall_accuracy >= 60:
        print("✅ EXCELLENT - Model performing well!")
    elif overall_accuracy >= 50:
        print("⚠️  GOOD - Model performing okay")
    else:
        print("❌ POOR - Model struggling in current conditions")
    
    print()
    print(f"   Overall: {overall_accuracy:.1f}%")
    print(f"   Directional: {directional_accuracy:.1f}%")
    print(f"   Trading signals: {trading_accuracy:.1f}%")
    print()
    
    if trading_accuracy >= 70:
        print("💰 TRADE - High confidence signals look good!")
    elif trading_accuracy >= 60:
        print("⚠️  CAUTION - Trade with smaller size")
    else:
        print("🛑 WAIT - Model not performing well, skip trading")
    
    print()
    print("="*80)

if __name__ == "__main__":
    try:
        backtest_recent(hours=4)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

