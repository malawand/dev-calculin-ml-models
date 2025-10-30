#!/usr/bin/env python3
"""
Combined Real-Time Monitor

Displays both State Detection AND Fair Value Analysis together
Updates every 60 seconds with trading signals
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime
from collections import deque

# Add paths
sys.path.append(str(Path(__file__).parent / 'btc_momentum_detector'))
sys.path.append(str(Path(__file__).parent / 'btc_fair_value'))
sys.path.append(str(Path(__file__).parent / 'btc_minimal_start'))

from fair_value_calculator import FairValueCalculator
from train_improved_model import fetch_data_with_volume
from feature_extractor import extract_features
from write_prometheus_metrics import write_metrics

# Configuration
UPDATE_INTERVAL = 30  # seconds
HISTORY_SIZE = 20  # Keep last 20 readings
LOG_FILE = Path(__file__).parent / 'combined_signals_log.csv'
HOURS_TO_FETCH = 6  # 6 hours for short-term features (original model)

# Load state detection models - USING ULTRA-DEEP (95.3% accurate!)
print("🔄 Loading ULTRA-DEEP state detection models...")
print("   (95.3% accuracy, 0.951 strength correlation)")
try:
    models_dir = Path(__file__).parent / 'btc_ultra_deep_detector'
    with open(models_dir / 'direction_model.pkl', 'rb') as f:
        direction_model = pickle.load(f)
    with open(models_dir / 'strength_model.pkl', 'rb') as f:
        strength_model = pickle.load(f)
    with open(models_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Ultra-Deep state detection models loaded (BEST of 3 models tested)")
except FileNotFoundError:
    print("❌ Ultra-Deep models not found!")
    print("   Run: cd btc_ultra_deep_detector && python train_ultra_deep.py")
    sys.exit(1)

# Initialize fair value calculator
calculator = FairValueCalculator()
print("✅ Fair value calculator ready")

# History tracking
history = deque(maxlen=HISTORY_SIZE)

def detect_state(df):
    """Detect current market state"""
    try:
        features = extract_features(df)
        feature_names = sorted(features.keys())
        X = np.array([[features[fn] for fn in feature_names]])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = scaler.transform(X)
        
        direction = direction_model.predict(X_scaled)[0]
        strength = strength_model.predict(X_scaled)[0]
        direction_proba = direction_model.predict_proba(X_scaled)[0]
        confidence = direction_proba[direction + 1]
        
        direction_label = {-1: 'DOWN', 0: 'NONE', 1: 'UP'}[direction]
        
        # Probability distribution: [DOWN, NONE, UP]
        prob_down = direction_proba[0] * 100
        prob_none = direction_proba[1] * 100
        prob_up = direction_proba[2] * 100
        
        # Determine lean when SIDEWAYS
        lean = None
        if direction_label == 'NONE':
            if prob_up > prob_down:
                lean = 'UP'
            elif prob_down > prob_up:
                lean = 'DOWN'
            else:
                lean = 'NEUTRAL'
        
        return {
            'direction': direction_label,
            'strength': strength,
            'confidence': confidence,
            'prob_down': prob_down,
            'prob_none': prob_none,
            'prob_up': prob_up,
            'lean': lean,
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def calculate_fair_value(df):
    """Calculate fair value"""
    try:
        result = calculator.calculate_comprehensive_fair_value(df)
        if result is None:
            return {'success': False}
        result['success'] = True
        return result
    except Exception as e:
        return {'success': False, 'error': str(e)}

def generate_trading_signal(state, fair):
    """Generate trading signal from combined analysis"""
    if not state['success'] or not fair['success']:
        return {
            'action': 'ERROR',
            'position_size': 0.0,
            'reasoning': ['System error']
        }
    
    action = None
    position_size = 0.0
    reasoning = []
    
    # STRONG BUY (convergence)
    if (state['direction'] == 'UP' and 
        state['confidence'] > 0.9 and 
        fair['assessment'] in ['UNDERVALUED', 'SLIGHTLY_UNDERVALUED']):
        
        action = 'STRONG_BUY'
        position_size = 2.0
        reasoning = [
            f"✅ Strong UP momentum ({state['confidence']*100:.0f}% conf)",
            f"✅ Price below fair (${fair['current_price']:,.0f} < ${fair['fair_value']:,.0f})",
            "→ Both systems agree - high probability"
        ]
    
    # STRONG SELL (convergence)
    elif (state['direction'] == 'DOWN' and 
          state['confidence'] > 0.9 and 
          fair['assessment'] in ['OVERVALUED', 'SLIGHTLY_OVERVALUED']):
        
        action = 'STRONG_SELL'
        position_size = 2.0
        reasoning = [
            f"✅ Strong DOWN momentum ({state['confidence']*100:.0f}% conf)",
            f"✅ Price above fair (${fair['current_price']:,.0f} > ${fair['fair_value']:,.0f})",
            "→ Both systems agree - high probability"
        ]
    
    # Mean reversion SELL (at upper bound)
    elif fair['position_in_range_pct'] > 85 and state['direction'] != 'UP':
        action = 'SELL'
        position_size = 1.5
        reasoning = [
            f"⚠️  Near upper bound ({fair['position_in_range_pct']:.0f}%)",
            f"✅ No strong UP momentum",
            "→ Mean reversion opportunity"
        ]
    
    # Mean reversion BUY (at lower bound)
    elif fair['position_in_range_pct'] < 15 and state['direction'] != 'DOWN':
        action = 'BUY'
        position_size = 1.5
        reasoning = [
            f"⚠️  Near lower bound ({fair['position_in_range_pct']:.0f}%)",
            f"✅ No strong DOWN momentum",
            "→ Mean reversion opportunity"
        ]
    
    # Follow momentum at fair value
    elif (fair['assessment'] == 'FAIR' and 
          state['confidence'] > 0.85 and 
          state['strength'] > 60):
        
        if state['direction'] == 'UP':
            action = 'BUY'
            position_size = 1.0
            reasoning = [
                "✅ Price at fair value",
                f"✅ Strong UP momentum ({state['strength']:.0f}/100)",
                "→ Follow trend"
            ]
        elif state['direction'] == 'DOWN':
            action = 'SELL'
            position_size = 1.0
            reasoning = [
                "✅ Price at fair value",
                f"✅ Strong DOWN momentum ({state['strength']:.0f}/100)",
                "→ Follow trend"
            ]
    
    # WAIT (conflicting or uncertain)
    else:
        action = 'WAIT'
        position_size = 0.0
        
        if state['confidence'] < 0.7:
            reasoning.append(f"⚠️  Low confidence ({state['confidence']*100:.0f}%)")
        if fair['assessment'] == 'FAIR' and state['direction'] == 'NONE':
            reasoning.append("⚠️  Sideways at fair - no edge")
        if state['direction'] == 'UP' and fair['assessment'] == 'OVERVALUED':
            reasoning.append("⚠️  UP momentum but overvalued")
        if state['direction'] == 'DOWN' and fair['assessment'] == 'UNDERVALUED':
            reasoning.append("⚠️  DOWN momentum but undervalued")
        
        if not reasoning:
            reasoning.append("⚠️  No clear signal")
        
        reasoning.append("→ Wait for better setup")
    
    return {
        'action': action,
        'position_size': position_size,
        'reasoning': reasoning
    }

def display_combined(state, fair, signal, timestamp):
    """Display combined analysis"""
    print("\033[2J\033[H")  # Clear screen
    
    print("="*80)
    print("🤖 COMBINED REAL-TIME TRADING SYSTEM")
    print("="*80)
    print(f"Updated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using: Ultra-Deep Model (95.3% accurate, 0.951 strength correlation)")
    print()
    
    # Left column: State Detection
    print("┌" + "─"*38 + "┬" + "─"*39 + "┐")
    print("│ 📊 STATE DETECTION (Ultra-Deep)" + " "*5 + "│ 💰 FAIR VALUE ANALYSIS" + " "*15 + "│")
    print("├" + "─"*38 + "┼" + "─"*39 + "┤")
    
    if not state['success']:
        print("│ ❌ Error" + " "*29 + "│", end="")
    else:
        direction_emoji = {'UP': '📈', 'DOWN': '📉', 'NONE': '↔️'}
        emoji = direction_emoji.get(state['direction'], '❓')
        
        print(f"│ Direction: {emoji} {state['direction']:<20s}│", end="")
    
    if not fair['success']:
        print(" ❌ Error" + " "*30 + "│")
    else:
        print(f" Current: ${fair['current_price']:>10,.2f}     │")
    
    if state['success']:
        print(f"│ Strength:  {state['strength']:5.1f}/100" + " "*16 + "│", end="")
    else:
        print("│" + " "*38 + "│", end="")
    
    if fair['success']:
        print(f" Fair:    ${fair['fair_value']:>10,.2f}     │")
    else:
        print(" " + " "*39 + "│")
    
    if state['success']:
        print(f"│ Confidence: {state['confidence']*100:4.1f}%" + " "*19 + "│", end="")
    else:
        print("│" + " "*38 + "│", end="")
    
    if fair['success']:
        deviation = fair['deviation_pct']
        print(f" Deviation: {deviation:>+6.2f}%" + " "*13 + "│")
    else:
        print(" " + " "*39 + "│")
    
    # Show probability distribution
    if state['success']:
        print(f"│ Probabilities:" + " "*23 + "│", end="")
    else:
        print("│" + " "*38 + "│", end="")
    
    print(" " + " "*39 + "│")
    
    if state['success']:
        print(f"│   DOWN: {state['prob_down']:4.1f}%  NONE: {state['prob_none']:4.1f}%  UP: {state['prob_up']:4.1f}%  │", end="")
    else:
        print("│" + " "*38 + "│", end="")
    
    # Fair value position
    if fair['success']:
        assessment_emoji = {
            'OVERVALUED': '🔴', 'SLIGHTLY_OVERVALUED': '🟠',
            'FAIR': '🟢', 'SLIGHTLY_UNDERVALUED': '🔵',
            'UNDERVALUED': '🟣'
        }
        emoji = assessment_emoji.get(fair['assessment'], '⚪')
        print(f" {emoji} {fair['assessment']:<26s}│")
    else:
        print(" " + " "*39 + "│")
    
    # Show lean when SIDEWAYS
    if state['success'] and state['lean']:
        lean_emoji = {'UP': '↗️', 'DOWN': '↘️', 'NEUTRAL': '↔️'}
        emoji = lean_emoji.get(state['lean'], '❓')
        print(f"│ Leaning: {emoji} {state['lean']:<22s}│", end="")
    else:
        print("│" + " "*38 + "│", end="")
    
    print(" " + " "*39 + "│")
    
    print("│" + " "*38 + "│" + " "*39 + "│")
    
    # State bar
    if state['success']:
        strength_bar = int(state['strength'] / 100 * 30)
        bar = '█' * strength_bar + '░' * (30 - strength_bar)
        print(f"│ [{bar}] │", end="")
    else:
        print("│" + " "*38 + "│", end="")
    
    # Fair value bar - just empty space for alignment
    print(" " + " "*39 + "│")
    
    print("└" + "─"*38 + "┴" + "─"*39 + "┘")
    print()
    
    # Trading Signal
    print("="*80)
    print("🎯 TRADING SIGNAL")
    print("="*80)
    print()
    
    signal_emoji = {
        'STRONG_BUY': '🟢🟢', 'BUY': '🟢',
        'STRONG_SELL': '🔴🔴', 'SELL': '🔴',
        'WAIT': '⏸️', 'ERROR': '❌'
    }
    
    emoji = signal_emoji.get(signal['action'], '⚪')
    print(f"   Action: {emoji} {signal['action']}")
    print(f"   Position Size: {signal['position_size']:.1f}%")
    print()
    
    print("   Reasoning:")
    for r in signal['reasoning']:
        print(f"      {r}")
    print()
    
    # Risk management
    if signal['action'] in ['STRONG_BUY', 'BUY'] and fair['success']:
        target = fair['fair_value']
        stop = fair['empirical_min'] * 0.99
        current = fair['current_price']
        
        print("   Trade Setup:")
        print(f"      Entry:  ${current:,.2f}")
        print(f"      Target: ${target:,.2f} (+{((target/current)-1)*100:.2f}%)")
        print(f"      Stop:   ${stop:,.2f} ({((stop/current)-1)*100:.2f}%)")
        rr = abs((target-current)/(current-stop))
        print(f"      R:R:    {rr:.2f}")
    
    elif signal['action'] in ['STRONG_SELL', 'SELL'] and fair['success']:
        target = fair['fair_value']
        stop = fair['empirical_max'] * 1.01
        current = fair['current_price']
        
        print("   Trade Setup:")
        print(f"      Entry:  ${current:,.2f}")
        print(f"      Target: ${target:,.2f} ({((target/current)-1)*100:.2f}%)")
        print(f"      Stop:   ${stop:,.2f} (+{((stop/current)-1)*100:.2f}%)")
        rr = abs((current-target)/(stop-current))
        print(f"      R:R:    {rr:.2f}")
    
    print()
    
    # Historical performance
    if len(history) > 1:
        print("="*80)
        print("📈 RECENT HISTORY")
        print("="*80)
        print()
        
        actions = [h['signal']['action'] for h in history if h.get('signal')]
        if len(actions) > 0:
            signal_counts = {}
            for a in actions:
                signal_counts[a] = signal_counts.get(a, 0) + 1
            
            print("   Signal Distribution (last 20):")
            for sig, count in sorted(signal_counts.items(), key=lambda x: -x[1]):
                emoji = signal_emoji.get(sig, '⚪')
                pct = count / len(actions) * 100
                print(f"      {emoji} {sig:12s}: {count:2d} ({pct:4.1f}%)")
            print()
    
    print("="*80)

def countdown(seconds):
    """Display countdown timer"""
    for remaining in range(seconds, 0, -1):
        print(f"\r⏱️  Next update in {remaining} seconds... (Ctrl+C to stop)    ", end='', flush=True)
        time.sleep(1)
    print("\r" + " "*60 + "\r", end='', flush=True)  # Clear the line

def save_to_log(state, fair, signal, timestamp):
    """Save combined result to CSV log"""
    if not state.get('success') or not fair.get('success'):
        return
    
    log_data = {
        'timestamp': [timestamp],
        'price': [fair['current_price']],
        'direction': [state['direction']],
        'strength': [state['strength']],
        'confidence': [state['confidence']],
        'fair_value': [fair['fair_value']],
        'deviation_pct': [fair['deviation_pct']],
        'assessment': [fair['assessment']],
        'signal': [signal['action']],
        'position_size': [signal['position_size']]
    }
    
    df_log = pd.DataFrame(log_data)
    
    if LOG_FILE.exists():
        df_existing = pd.read_csv(LOG_FILE)
        df_log = pd.concat([df_existing, df_log], ignore_index=True)
    
    df_log.to_csv(LOG_FILE, index=False)

def main():
    """Main monitoring loop"""
    print("="*80)
    print("🚀 STARTING COMBINED REAL-TIME MONITORING")
    print("="*80)
    print()
    print(f"Update interval: {UPDATE_INTERVAL} seconds")
    print(f"History size: {HISTORY_SIZE} readings")
    print(f"Log file: {LOG_FILE}")
    print()
    print("Combining:")
    print("   • State Detection (90.6% accurate)")
    print("   • Fair Value Analysis (4 methods)")
    print()
    print("Press Ctrl+C to stop")
    print()
    time.sleep(3)
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            timestamp = datetime.now()
            
            try:
                # Fetch data
                df = fetch_data_with_volume(hours=HOURS_TO_FETCH)
                
                if df is None or len(df) < 300:  # Need ~6 hours for short-term features
                    print(f"❌ Data fetch failed or insufficient (got {len(df) if df is not None else 0} samples, need 300+)")
                    countdown(UPDATE_INTERVAL)
                    continue
                
                # Detect state
                state = detect_state(df)
                
                # Calculate fair value
                fair = calculate_fair_value(df)
                
                # Generate trading signal
                signal = generate_trading_signal(state, fair)
                
                # Store in history
                history.append({
                    'timestamp': timestamp,
                    'state': state,
                    'fair': fair,
                    'signal': signal
                })
                
                # Display
                display_combined(state, fair, signal, timestamp)
                
                # Save to log
                save_to_log(state, fair, signal, timestamp)
                
                # Write Prometheus metrics
                write_metrics(state, fair, signal, fair.get('current_price', 0))
                
            except Exception as e:
                print(f"❌ Error in iteration {iteration}: {e}")
            
            # Wait for next update with countdown
            countdown(UPDATE_INTERVAL)
            
    except KeyboardInterrupt:
        print()
        print("="*80)
        print("🛑 STOPPING COMBINED MONITORING")
        print("="*80)
        print()
        print(f"Total iterations: {iteration}")
        print(f"Log saved to: {LOG_FILE}")
        print()
        print("✅ Monitoring stopped")

if __name__ == "__main__":
    main()

