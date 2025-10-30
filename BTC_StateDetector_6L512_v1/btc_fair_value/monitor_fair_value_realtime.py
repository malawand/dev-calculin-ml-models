#!/usr/bin/env python3
"""
Real-Time Fair Value Monitor

Continuously calculates Bitcoin fair value, max, min every 60 seconds
Displays current valuation and tracks historical trends
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time
from datetime import datetime
from collections import deque

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'btc_minimal_start'))

from fair_value_calculator import FairValueCalculator
from train_improved_model import fetch_data_with_volume

# Configuration
UPDATE_INTERVAL = 60  # seconds
HISTORY_SIZE = 20  # Keep last 20 readings
LOG_FILE = Path(__file__).parent / 'fair_value_log.csv'

# Initialize calculator
calculator = FairValueCalculator()

# History tracking
history = deque(maxlen=HISTORY_SIZE)

def calculate_fair_value(df):
    """Calculate fair value"""
    try:
        result = calculator.calculate_comprehensive_fair_value(df)
        
        if result is None:
            return {
                'timestamp': datetime.now(),
                'success': False,
                'error': 'Calculation failed'
            }
        
        result['timestamp'] = datetime.now()
        result['success'] = True
        return result
        
    except Exception as e:
        return {
            'timestamp': datetime.now(),
            'success': False,
            'error': str(e)
        }

def display_fair_value(result):
    """Display fair value with formatting"""
    print("\033[2J\033[H")  # Clear screen
    
    print("="*80)
    print("💰 REAL-TIME FAIR VALUE ANALYSIS")
    print("="*80)
    print(f"Updated: {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if not result['success']:
        print(f"❌ Error: {result.get('error', 'Unknown')}")
        return
    
    # Current values
    print("CURRENT VALUATION:")
    print()
    
    current = result['current_price']
    fair = result['fair_value']
    max_price = result['empirical_max']
    min_price = result['empirical_min']
    
    print(f"   Current Price:  ${current:,.2f}")
    print(f"   Fair Value:     ${fair:,.2f}")
    print(f"   Empirical Max:  ${max_price:,.2f}")
    print(f"   Empirical Min:  ${min_price:,.2f}")
    print()
    
    print(f"   Deviation:      {result['deviation_pct']:+.2f}%")
    print(f"   Position:       {result['position_in_range_pct']:.1f}% of range")
    print()
    
    # Visual representation
    position_pct = result['position_in_range_pct']
    bar_width = 50
    position_bar = int(position_pct / 100 * bar_width)
    position_bar = max(0, min(bar_width - 1, position_bar))
    
    bar = ['─'] * bar_width
    bar[position_bar] = '█'
    
    print("PRICE POSITION:")
    print()
    print(f"   Min ${min_price:,.0f}")
    print(f"   {''.join(bar)}")
    print(f"   Max ${max_price:,.0f}")
    print()
    
    # Assessment with emoji
    assessment_emoji = {
        'OVERVALUED': '🔴',
        'SLIGHTLY_OVERVALUED': '🟠',
        'FAIR': '🟢',
        'SLIGHTLY_UNDERVALUED': '🔵',
        'UNDERVALUED': '🟣',
        'EXTREMELY_OVERVALUED': '🔴🔴',
        'EXTREMELY_UNDERVALUED': '🟣🟣'
    }
    
    emoji = assessment_emoji.get(result['assessment'], '⚪')
    
    print(f"   Assessment: {emoji} {result['assessment']}")
    print(f"   Expected:   {result['expected_move']}")
    print()
    
    # Trading implication
    if position_pct > 85:
        print("   ⚠️  NEAR UPPER BOUND - High reversal risk")
        print("   💡 Consider: Selling or taking profits")
    elif position_pct < 15:
        print("   ⚠️  NEAR LOWER BOUND - High bounce probability")
        print("   💡 Consider: Buying opportunity")
    elif 45 <= position_pct <= 55:
        print("   ℹ️  MIDDLE OF RANGE - No clear edge")
        print("   💡 Consider: Wait for extremes")
    else:
        if position_pct > 55:
            print("   📊 Above fair value")
            print("   💡 Consider: Reduce longs, watch for reversal")
        else:
            print("   📊 Below fair value")
            print("   💡 Consider: Scale into longs")
    
    print()
    
    # Method breakdown
    methods = result['methods']
    print("METHODS:")
    print()
    if methods['vwap_4h']:
        print(f"   VWAP (4h):      ${methods['vwap_4h']:,.2f}")
    if methods['statistical_mean']:
        print(f"   Statistical:    ${methods['statistical_mean']:,.2f}")
    if methods['derivative_fair']:
        print(f"   Derivative:     ${methods['derivative_fair']:,.2f}")
    if methods['zscore'] is not None:
        print(f"   Z-Score:        {methods['zscore']:.2f} ({methods['zscore_interpretation']})")
    print()
    
    # Historical trend
    if len(history) > 1:
        print("="*80)
        print("📈 RECENT HISTORY (Last 20 readings)")
        print("="*80)
        print()
        
        # Assessment distribution
        assessments = [h['assessment'] for h in history if h['success']]
        if len(assessments) > 0:
            assessment_counts = {}
            for a in assessments:
                assessment_counts[a] = assessment_counts.get(a, 0) + 1
            
            print("   Assessment Distribution:")
            for assess, count in sorted(assessment_counts.items(), key=lambda x: -x[1])[:5]:
                emoji = assessment_emoji.get(assess, '⚪')
                pct = count / len(assessments) * 100
                bar = '█' * int(pct / 5)
                print(f"   {emoji} {assess:20s}: {count:2d} ({pct:4.1f}%) {bar}")
            print()
        
        # Price trend
        prices = [h['current_price'] for h in history if h['success']]
        if len(prices) > 1:
            price_change = prices[-1] - prices[0]
            price_change_pct = (price_change / prices[0]) * 100
            
            print(f"   Price Change (last {len(prices)} readings):")
            print(f"   ${prices[0]:,.2f} → ${prices[-1]:,.2f}")
            print(f"   {price_change:+,.2f} ({price_change_pct:+.2f}%)")
            print()
        
        # Average deviation
        deviations = [h['deviation_pct'] for h in history if h['success']]
        if len(deviations) > 0:
            avg_dev = np.mean(deviations)
            print(f"   Average Deviation: {avg_dev:+.2f}%")
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
        'current_price': [result['current_price']],
        'fair_value': [result['fair_value']],
        'empirical_max': [result['empirical_max']],
        'empirical_min': [result['empirical_min']],
        'deviation_pct': [result['deviation_pct']],
        'position_in_range_pct': [result['position_in_range_pct']],
        'assessment': [result['assessment']],
        'expected_move': [result['expected_move']]
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
    print("🚀 STARTING REAL-TIME FAIR VALUE MONITORING")
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
                df = fetch_data_with_volume(hours=48)
                
                if df is None or len(df) < 300:
                    print(f"❌ Data fetch failed (iteration {iteration})")
                    time.sleep(UPDATE_INTERVAL)
                    continue
                
                # Calculate fair value
                result = calculate_fair_value(df)
                
                # Add to history
                history.append(result)
                
                # Display
                display_fair_value(result)
                
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
                assessments = [h['assessment'] for h in successful]
                print()
                print("Final Summary:")
                for assess in set(assessments):
                    count = assessments.count(assess)
                    print(f"   {assess}: {count} times")
        
        print()
        print("✅ Monitoring stopped")

if __name__ == "__main__":
    main()


