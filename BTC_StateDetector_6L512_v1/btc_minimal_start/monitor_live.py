#!/usr/bin/env python3
"""Continuous monitoring with 44-feature model"""
import sys
sys.path.insert(0, str(__file__).replace('monitor_live.py', 'train_improved_model.py'))
from train_improved_model import fetch_data_with_volume, compute_advanced_features
import pickle
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import time

print("="*80)
print("🤖 BTC LIVE MONITOR - CONTINUOUS")
print("="*80)
print("Checking every 60 seconds...")
print("Press Ctrl+C to stop")
print("="*80)
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

check_count = 0
signal_count = 0
last_action = None

while True:
    try:
        check_count += 1
        
        # Fetch data
        df = fetch_data_with_volume(hours=6)
        
        if df is None or len(df) < 300:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Data fetch failed")
            time.sleep(60)
            continue
        
        # Compute features
        features_dict = compute_advanced_features(df, len(df) - 1)
        
        if features_dict is None:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Feature compute failed")
            time.sleep(60)
            continue
        
        # Predict
        features = np.array([[features_dict[fn] for fn in feature_names]])
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        labels = ['DOWN', 'SIDEWAYS', 'UP']
        predicted_label = labels[prediction]
        confidence = probabilities[prediction]
        
        # Check for trading signal
        should_trade = (prediction != 1 and confidence > 0.70)
        
        # Format probabilities
        prob_down = probabilities[0]
        prob_sideways = probabilities[1]
        prob_up = probabilities[2]
        
        if should_trade:
            signal_count += 1
            action = 'SELL' if prediction == 0 else 'BUY'
            
            # Alert
            print("\n" + "🔔"*40)
            print(f"🚨 TRADING SIGNAL #{signal_count}")
            print(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
            print(f"   Price: ${features_dict['price']:,.2f}")
            print(f"   Action: {action}")
            print()
            print("   Probabilities:")
            print(f"      DOWN:     {prob_down:6.2%}")
            print(f"      SIDEWAYS: {prob_sideways:6.2%}")
            print(f"      UP:       {prob_up:6.2%}")
            print(f"   → Predicted: {predicted_label} ({confidence:.2%})")
            print()
            
            if confidence > 0.85:
                print(f"   🔥 HIGH CONFIDENCE - 1-2% position")
            else:
                print(f"   ⚠️  MEDIUM CONFIDENCE - 0.5-1% position")
            
            print(f"   Stop-loss: {config['threshold'] * 1.5:.2f}%")
            print(f"   Take-profit: {config['threshold'] * 2:.2f}%")
            print("🔔"*40 + "\n")
            
            last_action = action
        else:
            # No trade - show table format with highlighted prediction
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Price: ${features_dict['price']:,.2f}")
            print(f"  ┌──────────┬────────────┬────────────┐")
            print(f"  │ Direction│ Confidence │ Predicted? │")
            print(f"  ├──────────┼────────────┼────────────┤")
            
            down_marker = "  ← ✓" if prediction == 0 else ""
            sideways_marker = "  ← ✓" if prediction == 1 else ""
            up_marker = "  ← ✓" if prediction == 2 else ""
            
            print(f"  │ DOWN     │   {prob_down:6.2%}   │{down_marker:12s}│")
            print(f"  │ SIDEWAYS │   {prob_sideways:6.2%}   │{sideways_marker:12s}│")
            print(f"  │ UP       │   {prob_up:6.2%}   │{up_marker:12s}│")
            print(f"  └──────────┴────────────┴────────────┘")
            print(f"  Prediction: {predicted_label} (Confidence: {confidence:.2%}) | Checks:{check_count} Signals:{signal_count}")
        
        # Wait 60 seconds
        time.sleep(60)
        
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("📊 SESSION SUMMARY")
        print("="*80)
        print(f"Total checks: {check_count}")
        print(f"Trading signals: {signal_count}")
        if check_count > 0:
            print(f"Signal rate: {signal_count/check_count:.1%}")
        print("\n✅ Monitor stopped")
        break
    except Exception as e:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ❌ Error: {e}")
        print("Retrying in 60 seconds...")
        time.sleep(60)

