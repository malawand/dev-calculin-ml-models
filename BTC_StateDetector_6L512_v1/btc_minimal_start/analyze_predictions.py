#!/usr/bin/env python3
"""
Analyze prediction log to track real-world model performance
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from pathlib import Path

def parse_log_file(log_file):
    """Parse predictions.log into DataFrame"""
    data = []
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                parts = line.strip().split(' | ')
                timestamp = parts[0]
                price = float(parts[1].split('$')[1].replace(',', ''))
                prob = float(parts[2].split(': ')[1])
                direction = parts[3].split(': ')[1]
                signal = parts[4].split(': ')[1]
                
                data.append({
                    'timestamp': pd.to_datetime(timestamp),
                    'price': price,
                    'prob_up': prob,
                    'direction': direction,
                    'signal': signal
                })
            except Exception as e:
                continue
    
    return pd.DataFrame(data)


def fetch_actual_prices(start_time, end_time, cortex_url='10.1.20.60', cortex_port=9009):
    """Fetch actual BTC prices for verification"""
    url = f"http://{cortex_url}:{cortex_port}/prometheus/api/v1/query_range"
    
    params = {
        'query': 'crypto_last_price{symbol="BTCUSDT"}',
        'start': start_time.isoformat() + 'Z',
        'end': end_time.isoformat() + 'Z',
        'step': '15m'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] == 'success' and data['data']['result']:
            values = data['data']['result'][0]['values']
            df = pd.DataFrame(values, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            df['price'] = df['price'].astype(float)
            return df
    except Exception as e:
        print(f"⚠️ Could not fetch actual prices: {e}")
        return None


def calculate_accuracy(df):
    """Calculate prediction accuracy"""
    # For each prediction, check if price moved in predicted direction 24h later
    results = []
    
    for i in range(len(df) - 96):  # 96 = 24 hours in 15-min intervals
        pred = df.iloc[i]
        future = df.iloc[i + 96]
        
        price_change = future['price'] - pred['price']
        actual_direction = "UP" if price_change > 0 else "DOWN"
        
        correct = (pred['direction'] == actual_direction)
        results.append({
            'timestamp': pred['timestamp'],
            'predicted': pred['direction'],
            'actual': actual_direction,
            'correct': correct,
            'prob_up': pred['prob_up'],
            'signal': pred['signal'],
            'price_change_pct': (price_change / pred['price']) * 100
        })
    
    return pd.DataFrame(results)


def analyze_performance(results_df):
    """Analyze model performance"""
    print("\n" + "=" * 70)
    print("                    📊 MODEL PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    # Overall accuracy
    overall_accuracy = results_df['correct'].mean()
    print(f"\n📈 OVERALL ACCURACY: {overall_accuracy:.2%}")
    print(f"   Total predictions: {len(results_df)}")
    print(f"   Correct: {results_df['correct'].sum()}")
    print(f"   Incorrect: {(~results_df['correct']).sum()}")
    
    # Accuracy by signal strength
    print(f"\n🎯 ACCURACY BY SIGNAL STRENGTH:")
    signal_order = ['STRONG_BUY', 'BUY', 'WEAK_BUY', 'HOLD', 'WEAK_SELL', 'SELL', 'STRONG_SELL']
    
    for signal in signal_order:
        signal_df = results_df[results_df['signal'] == signal]
        if len(signal_df) > 0:
            accuracy = signal_df['correct'].mean()
            count = len(signal_df)
            avg_gain = signal_df['price_change_pct'].mean()
            print(f"   {signal:15} | Accuracy: {accuracy:6.2%} | Count: {count:4} | Avg Gain: {avg_gain:+6.2f}%")
    
    # Accuracy by confidence level
    print(f"\n💪 ACCURACY BY CONFIDENCE LEVEL:")
    bins = [0.0, 0.4, 0.6, 0.7, 0.8, 1.0]
    labels = ['LOW (0.4-0.6)', 'MEDIUM (0.6-0.7)', 'HIGH (0.7-0.8)', 'VERY HIGH (0.8+)', 'VERY HIGH (0.0-0.2)']
    
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        if i == 0:
            mask = (results_df['prob_up'] >= low) & (results_df['prob_up'] <= high)
        else:
            mask = (results_df['prob_up'] > low) & (results_df['prob_up'] <= high)
        
        confidence_df = results_df[mask]
        if len(confidence_df) > 0:
            accuracy = confidence_df['correct'].mean()
            count = len(confidence_df)
            print(f"   {labels[i]:20} | Accuracy: {accuracy:6.2%} | Count: {count:4}")
    
    # Also check very low (SELL signals)
    very_low_mask = results_df['prob_up'] < 0.2
    very_low_df = results_df[very_low_mask]
    if len(very_low_df) > 0:
        accuracy = very_low_df['correct'].mean()
        count = len(very_low_df)
        print(f"   {'VERY HIGH (0.0-0.2)':20} | Accuracy: {accuracy:6.2%} | Count: {count:4}")
    
    # Direction bias
    print(f"\n📊 DIRECTION BIAS:")
    up_count = (results_df['predicted'] == 'UP').sum()
    down_count = (results_df['predicted'] == 'DOWN').sum()
    print(f"   UP predictions:   {up_count} ({up_count/len(results_df):.1%})")
    print(f"   DOWN predictions: {down_count} ({down_count/len(results_df):.1%})")
    
    # ROI simulation
    print(f"\n💰 SIMULATED ROI (if trading all signals with 1% risk):")
    total_roi = 0.0
    win_count = 0
    loss_count = 0
    
    for _, row in results_df.iterrows():
        if row['correct']:
            # Win: assume 2% gain (2:1 risk/reward)
            total_roi += 2.0
            win_count += 1
        else:
            # Loss: -1%
            total_roi -= 1.0
            loss_count += 1
    
    print(f"   Total ROI: {total_roi:+.2f}%")
    print(f"   Win rate: {win_count}/{len(results_df)} ({win_count/len(results_df):.2%})")
    print(f"   Avg gain per trade: {total_roi/len(results_df):+.3f}%")
    
    # High confidence only
    high_conf_mask = (results_df['prob_up'] >= 0.75) | (results_df['prob_up'] <= 0.25)
    high_conf_df = results_df[high_conf_mask]
    
    if len(high_conf_df) > 0:
        print(f"\n🌟 HIGH CONFIDENCE ONLY (prob ≥ 0.75 or ≤ 0.25):")
        accuracy = high_conf_df['correct'].mean()
        count = len(high_conf_df)
        
        high_conf_roi = 0.0
        for _, row in high_conf_df.iterrows():
            if row['correct']:
                high_conf_roi += 2.0
            else:
                high_conf_roi -= 1.0
        
        print(f"   Accuracy: {accuracy:.2%}")
        print(f"   Predictions: {count}")
        print(f"   Total ROI: {high_conf_roi:+.2f}%")
        print(f"   Avg gain per trade: {high_conf_roi/len(high_conf_df):+.3f}%")
    
    # Recent trend (last 24 hours)
    recent_mask = results_df['timestamp'] >= (results_df['timestamp'].max() - timedelta(hours=24))
    recent_df = results_df[recent_mask]
    
    if len(recent_df) > 0:
        print(f"\n🕐 LAST 24 HOURS:")
        recent_accuracy = recent_df['correct'].mean()
        print(f"   Accuracy: {recent_accuracy:.2%}")
        print(f"   Predictions: {len(recent_df)}")
        
        if recent_accuracy < overall_accuracy - 0.10:
            print(f"   ⚠️ WARNING: Recent accuracy significantly lower than overall!")
            print(f"   Consider checking data quality or market conditions.")
    
    print("=" * 70 + "\n")
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    if overall_accuracy >= 0.75:
        print("   ✅ Model performing excellently! Keep current settings.")
    elif overall_accuracy >= 0.65:
        print("   ⚠️ Model performing adequately. Consider monitoring closely.")
    else:
        print("   ❌ Model underperforming. Consider:")
        print("      - Checking data quality")
        print("      - Retraining with recent data")
        print("      - Adjusting thresholds or reducing position sizes")
    
    if high_conf_mask.sum() > 0:
        high_conf_acc = results_df[high_conf_mask]['correct'].mean()
        if high_conf_acc >= 0.80:
            print("   ✅ High confidence signals very reliable. Consider:")
            print("      - Trading only high confidence signals (≥0.75 or ≤0.25)")
            print("      - Larger position sizes on these signals")
    
    print()


def main():
    parser = argparse.ArgumentParser(description='Analyze prediction log')
    parser.add_argument('--log', default='predictions.log', help='Path to predictions log file')
    parser.add_argument('--days', type=int, default=7, help='Number of days to analyze')
    parser.add_argument('--export', help='Export results to CSV file')
    
    args = parser.parse_args()
    
    if not Path(args.log).exists():
        print(f"❌ Log file not found: {args.log}")
        print("   Make sure predict_live.py has been running and logging predictions.")
        return
    
    print("📊 Loading predictions...")
    df = parse_log_file(args.log)
    
    if df.empty:
        print("❌ No predictions found in log file.")
        return
    
    print(f"✅ Loaded {len(df)} predictions")
    
    # Filter by date range
    cutoff_date = datetime.now() - timedelta(days=args.days)
    df = df[df['timestamp'] >= cutoff_date]
    print(f"📅 Analyzing last {args.days} days ({len(df)} predictions)")
    
    if len(df) < 96:  # Less than 24 hours of data
        print("⚠️ Not enough data to calculate accuracy (need at least 24 hours)")
        print("   Continue running predict_live.py to collect more data.")
        return
    
    # Calculate accuracy
    print("🔍 Calculating accuracy (checking 24h outcomes)...")
    results = calculate_accuracy(df)
    
    if results.empty:
        print("❌ Not enough data to verify predictions yet.")
        print("   Need at least 24 hours of future data to verify each prediction.")
        return
    
    # Analyze performance
    analyze_performance(results)
    
    # Export if requested
    if args.export:
        results.to_csv(args.export, index=False)
        print(f"💾 Results exported to: {args.export}")


if __name__ == "__main__":
    main()



