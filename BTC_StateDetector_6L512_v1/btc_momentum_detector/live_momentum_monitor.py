#!/usr/bin/env python3
"""
Live Momentum Monitor
Continuously displays momentum strength, direction, and phase
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'btc_minimal_start'))

from momentum_calculator import MomentumCalculator
from momentum_signals import MomentumSignals
from train_improved_model import fetch_data_with_volume
import time
from datetime import datetime

def get_momentum_emoji(phase):
    """Get emoji for momentum phase"""
    return {
        'BUILDING': '🚀',
        'STRONG': '💪',
        'FADING': '📉',
        'ABSENT': '😴'
    }.get(phase, '❓')

def get_direction_emoji(direction):
    """Get emoji for direction"""
    return {
        'UP': '📈',
        'DOWN': '📉',
        'NONE': '➡️'
    }.get(direction, '❓')

def get_strength_bar(strength):
    """Get visual strength bar"""
    bars = int(strength / 10)
    filled = '█' * bars
    empty = '░' * (10 - bars)
    return filled + empty

def get_strength_description(strength):
    """Get text description of strength"""
    if strength < 20:
        return "VERY WEAK"
    elif strength < 40:
        return "WEAK"
    elif strength < 60:
        return "MODERATE"
    elif strength < 80:
        return "STRONG"
    else:
        return "VERY STRONG"

print("="*80)
print("🎯 MOMENTUM DETECTOR - LIVE MONITOR")
print("="*80)
print("Monitoring momentum every 60 seconds...")
print("Press Ctrl+C to stop")
print("="*80)
print()

calculator = MomentumCalculator()
signal_gen = MomentumSignals(
    min_strength_entry=35,
    min_confidence_entry=0.5
)

signal_count = 0
last_signal = None

while True:
    try:
        # Fetch latest data
        df = fetch_data_with_volume(hours=48)
        
        if df is None or len(df) < 300:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Data fetch failed")
            time.sleep(60)
            continue
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get momentum report
        report = calculator.get_momentum_report(df)
        
        # Get trading signal
        signal = signal_gen.generate_entry_signal(df)
        
        # Display
        print(f"\n{'='*80}")
        print(f"⏰ {timestamp}")
        print(f"{'='*80}")
        print(f"💰 Current Price: ${report['current_price']:,.2f}")
        print()
        
        # Momentum metrics
        phase_emoji = get_momentum_emoji(report['phase'])
        dir_emoji = get_direction_emoji(report['direction'])
        strength_bar = get_strength_bar(report['strength'])
        strength_desc = get_strength_description(report['strength'])
        
        print(f"{'='*80}")
        print(f"📊 MOMENTUM ANALYSIS")
        print(f"{'='*80}")
        print(f"Phase:        {phase_emoji} {report['phase']}")
        print(f"Direction:    {dir_emoji} {report['direction']}")
        print(f"Strength:     {strength_bar} {report['strength']:.0f}/100 ({strength_desc})")
        print(f"Confidence:   {'█' * int(report['confidence'] * 10)}{'░' * (10 - int(report['confidence'] * 10))} {report['confidence']*100:.0f}%")
        print()
        
        # Rate of Change details
        print(f"📈 RATE OF CHANGE:")
        print(f"   15m:  {report['roc_15m']*100:+.3f}%")
        print(f"   1h:   {report['roc_1h']*100:+.3f}%")
        print(f"   4h:   {report['roc_4h']*100:+.3f}%")
        print(f"   Accel: {report['acceleration']*100:+.4f}% (momentum change)")
        print()
        
        # Trading signal
        if signal['action'] != 'NONE':
            signal_count += 1
            last_signal = {
                'time': timestamp,
                'action': signal['action'],
                'price': signal['entry_price']
            }
            
            action_emoji = "🟢 BUY" if signal['action'] == 'BUY' else "🔴 SELL"
            
            print(f"{'='*80}")
            print(f"🚨 TRADING SIGNAL #{signal_count}")
            print(f"{'='*80}")
            print(f"Action:        {action_emoji}")
            print(f"Confidence:    {signal['confidence']*100:.1f}%")
            print(f"Entry Price:   ${signal['entry_price']:,.2f}")
            print(f"Target Price:  ${signal['target_price']:,.2f} ({signal['expected_return_pct']:+.2f}%)")
            print(f"Stop Loss:     ${signal['stop_loss']:,.2f} ({-signal['risk_pct']:.2f}%)")
            print(f"Risk/Reward:   1:{signal['reward_risk_ratio']:.2f}")
            print(f"Position Size: {signal['position_multiplier']*100:.0f}% of normal")
            print()
            print(f"Reason: {signal['reason']}")
            print()
            
            if signal['confidence'] >= 0.7:
                print(f"💡 RECOMMENDATION: ✅ STRONG SIGNAL - Execute trade")
            elif signal['confidence'] >= 0.5:
                print(f"💡 RECOMMENDATION: ⚠️  MODERATE SIGNAL - Use caution")
            else:
                print(f"💡 RECOMMENDATION: ❌ WEAK SIGNAL - Skip")
            
            print(f"{'='*80}")
        
        elif report['phase'] == 'ABSENT':
            print(f"😴 NO MOMENTUM - Market is choppy/consolidating")
            print(f"   Strength: {report['strength']:.0f}/100 (need >35 for signal)")
            print(f"   Wait for momentum to build...")
        
        elif report['phase'] == 'FADING':
            print(f"📉 MOMENTUM FADING - Don't enter, consider exits")
            print(f"   Strength: {report['strength']:.0f}/100")
            print(f"   Direction: {report['direction']}")
        
        elif report['direction'] == 'NONE':
            print(f"➡️  NO CLEAR DIRECTION - Waiting for directional momentum")
            print(f"   Strength: {report['strength']:.0f}/100")
            print(f"   Phase: {report['phase']}")
        
        else:
            print(f"⏸️  NO SIGNAL")
            print(f"   Phase: {report['phase']}, Direction: {report['direction']}")
            print(f"   Strength: {report['strength']:.0f}/100 (need >{signal_gen.min_strength_entry})")
            print(f"   Confidence: {report['confidence']*100:.0f}% (need >{signal_gen.min_confidence_entry*100:.0f}%)")
        
        # Summary
        print()
        if last_signal:
            print(f"📊 Session: {signal_count} signals | Last: {last_signal['action']} at ${last_signal['price']:,.2f} ({last_signal['time']})")
        else:
            print(f"📊 Session: {signal_count} signals generated")
        
        print(f"\n⏳ Next update in 60 seconds...")
        time.sleep(60)
        
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("🛑 MONITOR STOPPED")
        print("="*80)
        print(f"Total signals generated: {signal_count}")
        if last_signal:
            print(f"Last signal: {last_signal['action']} at ${last_signal['price']:,.2f}")
        print("="*80)
        break
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("Retrying in 60 seconds...")
        time.sleep(60)


