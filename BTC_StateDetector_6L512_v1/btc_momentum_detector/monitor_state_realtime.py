#!/usr/bin/env python3
"""
Real-Time State Detection Monitor

Continuously detects market state (UP/DOWN/NONE) every 60 seconds
Displays current state, strength, confidence, and historical trend
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime
from collections import deque

sys.path.append(str(Path(__file__).parent.parent / 'btc_minimal_start'))
from train_improved_model import fetch_data_with_volume
from feature_extractor import extract_features

# Configuration
UPDATE_INTERVAL = 60  # seconds
HISTORY_SIZE = 20  # Keep last 20 readings
LOG_FILE = Path(__file__).parent / 'state_detection_log.csv'
HOURS_TO_FETCH = 6  # 6 hours for short-term features (original model)

# Load models
print("🔄 Loading state detection models...")
try:
    with open(Path(__file__).parent / 'direction_model.pkl', 'rb') as f:
        direction_model = pickle.load(f)
    with open(Path(__file__).parent / 'strength_model.pkl', 'rb') as f:
        strength_model = pickle.load(f)
    with open(Path(__file__).parent / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Models loaded successfully")
except FileNotFoundError:
    print("❌ Models not found! Run train_detection_nn.py first")
    sys.exit(1)

# History tracking
history = deque(maxlen=HISTORY_SIZE)

def detect_state(df):
    """Detect current market state"""
    try:
        # Extract features
        features = extract_features(df)
        feature_names = sorted(features.keys())
        X = np.array([[features[fn] for fn in feature_names]])
        
        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale
        X_scaled = scaler.transform(X)
        
        # Predict
        direction = direction_model.predict(X_scaled)[0]
        strength = strength_model.predict(X_scaled)[0]
        direction_proba = direction_model.predict_proba(X_scaled)[0]
        confidence = direction_proba[direction + 1]
        
        # Map to labels
        direction_label = {-1: 'DOWN', 0: 'NONE', 1: 'UP'}[direction]
        
        return {
            'timestamp': datetime.now(),
            'price': df['price'].iloc[-1],
            'direction': direction_label,
            'strength': strength,
            'confidence': confidence,
            'success': True
        }
    except Exception as e:
        return {
            'timestamp': datetime.now(),
            'error': str(e),
            'success': False
        }

def display_state(result):
    """Display current state with formatting"""
    print("\033[2J\033[H")  # Clear screen
    
    print("="*80)
    print("📊 REAL-TIME STATE DETECTION")
    print("="*80)
    print(f"Updated: {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if not result['success']:
        print(f"❌ Error: {result.get('error', 'Unknown')}")
        return
    
    # Current state
    print("CURRENT STATE:")
    print()
    
    direction_emoji = {'UP': '📈', 'DOWN': '📉', 'NONE': '↔️'}
    emoji = direction_emoji.get(result['direction'], '❓')
    
    print(f"   Price:      ${result['price']:,.2f}")
    print(f"   Direction:  {emoji} {result['direction']}")
    print(f"   Strength:   {result['strength']:.1f}/100")
    print(f"   Confidence: {result['confidence']*100:.1f}%")
    print()
    
    # Strength bar
    strength_bar_len = int(result['strength'] / 2)  # 0-100 -> 0-50 chars
    strength_bar = '█' * strength_bar_len + '░' * (50 - strength_bar_len)
    print(f"   [{strength_bar}]")
    print()
    
    # Confidence bar
    conf_bar_len = int(result['confidence'] * 50)  # 0-1 -> 0-50 chars
    conf_bar = '█' * conf_bar_len + '░' * (50 - conf_bar_len)
    print(f"   [{conf_bar}]")
    print()
    
    # Assessment
    if result['confidence'] > 0.9 and result['strength'] > 60:
        assessment = f"🟢 STRONG {result['direction']} - High confidence trade"
    elif result['confidence'] > 0.85 and result['strength'] > 50:
        assessment = f"🟡 MODERATE {result['direction']} - Good signal"
    elif result['confidence'] > 0.7:
        assessment = f"🟠 WEAK {result['direction']} - Low confidence"
    else:
        assessment = "🔴 UNCERTAIN - Wait for better signal"
    
    print(f"   Assessment: {assessment}")
    print()
    
    # Historical trend
    if len(history) > 1:
        print("="*80)
        print("📈 RECENT HISTORY (Last 20 readings)")
        print("="*80)
        print()
        
        # Direction changes
        directions = [h['direction'] for h in history if h['success']]
        if len(directions) > 0:
            up_count = directions.count('UP')
            down_count = directions.count('DOWN')
            none_count = directions.count('NONE')
            
            print(f"   UP:   {up_count:2d} ({up_count/len(directions)*100:4.1f}%)  {'█' * (up_count * 2)}")
            print(f"   DOWN: {down_count:2d} ({down_count/len(directions)*100:4.1f}%)  {'█' * (down_count * 2)}")
            print(f"   NONE: {none_count:2d} ({none_count/len(directions)*100:4.1f}%)  {'█' * (none_count * 2)}")
            print()
        
        # Trend over time
        print("   Last 10 states:")
        print("   ", end="")
        for h in list(history)[-10:]:
            if h['success']:
                emoji = direction_emoji.get(h['direction'], '❓')
                print(f"{emoji} ", end="")
        print()
        print()
        
        # Average strength and confidence
        strengths = [h['strength'] for h in history if h['success']]
        confidences = [h['confidence'] for h in history if h['success']]
        
        if len(strengths) > 0:
            avg_strength = np.mean(strengths)
            avg_confidence = np.mean(confidences)
            
            print(f"   Avg Strength:   {avg_strength:.1f}/100")
            print(f"   Avg Confidence: {avg_confidence*100:.1f}%")
            print()
    
    print("="*80)
    print(f"Next update in {UPDATE_INTERVAL} seconds... (Ctrl+C to stop)")
    print("="*80)

def save_to_log(result):
    """Save result to CSV log"""
    if not result['success']:
        return
    
    log_data = {
        'timestamp': [result['timestamp']],
        'price': [result['price']],
        'direction': [result['direction']],
        'strength': [result['strength']],
        'confidence': [result['confidence']]
    }
    
    df_log = pd.DataFrame(log_data)
    
    # Append to existing log or create new
    if LOG_FILE.exists():
        df_existing = pd.read_csv(LOG_FILE)
        df_log = pd.concat([df_existing, df_log], ignore_index=True)
    
    df_log.to_csv(LOG_FILE, index=False)

def main():
    """Main monitoring loop"""
    print("="*80)
    print("🚀 STARTING REAL-TIME STATE DETECTION")
    print("="*80)
    print()
    print(f"Update interval: {UPDATE_INTERVAL} seconds")
    print(f"History size: {HISTORY_SIZE} readings")
    print(f"Log file: {LOG_FILE}")
    print()
    print("Press Ctrl+C to stop")
    print()
    time.sleep(2)
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            
            # Fetch data
            try:
                df = fetch_data_with_volume(hours=HOURS_TO_FETCH)
                
                if df is None or len(df) < 300:  # Need ~6 hours for short-term features
                    print(f"❌ Data fetch failed or insufficient (got {len(df) if df is not None else 0} samples, need 300+)")
                    time.sleep(UPDATE_INTERVAL)
                    continue
                
                # Detect state
                result = detect_state(df)
                
                # Add to history
                history.append(result)
                
                # Display
                display_state(result)
                
                # Save to log
                save_to_log(result)
                
            except Exception as e:
                print(f"❌ Error in iteration {iteration}: {e}")
            
            # Wait for next update
            time.sleep(UPDATE_INTERVAL)
            
    except KeyboardInterrupt:
        print()
        print("="*80)
        print("🛑 STOPPING REAL-TIME MONITORING")
        print("="*80)
        print()
        print(f"Total iterations: {iteration}")
        print(f"Log saved to: {LOG_FILE}")
        print()
        
        if len(history) > 0:
            successful = [h for h in history if h['success']]
            print(f"Successful readings: {len(successful)}/{len(history)}")
            
            if len(successful) > 0:
                directions = [h['direction'] for h in successful]
                print()
                print("Final Summary:")
                print(f"   UP:   {directions.count('UP')} times")
                print(f"   DOWN: {directions.count('DOWN')} times")
                print(f"   NONE: {directions.count('NONE')} times")
        
        print()
        print("✅ Monitoring stopped")

if __name__ == "__main__":
    main()

