#!/usr/bin/env python3
"""
Write ML Model Predictions to Prometheus Textfile

Writes metrics to a .prom file that Prometheus textfile_collector can scrape.
"""
import time
from pathlib import Path

# Where to write the metrics file
METRICS_FILE = Path("./metrics/btc_ml_metrics.prom")  # Change path if needed

def write_metrics(state, fair, signal, price):
    """
    Write metrics to .prom file for Prometheus textfile_collector
    
    Args:
        state: State detection result dict
        fair: Fair value result dict
        signal: Trading signal dict
        price: Current BTC price
    """
    if not state.get('success') or not fair.get('success'):
        return
    
    # Map direction to numeric value
    direction_map = {'UP': 1, 'NONE': 0, 'DOWN': -1}
    direction_value = direction_map.get(state['direction'], 0)
    
    # Map assessment to numeric value
    assessment_map = {
        'UNDERVALUED': -2,
        'SLIGHTLY_UNDERVALUED': -1,
        'FAIR': 0,
        'SLIGHTLY_OVERVALUED': 1,
        'OVERVALUED': 2
    }
    assessment_value = assessment_map.get(fair['assessment'], 0)
    
    # Map signal to numeric value
    signal_map = {
        'STRONG_BUY': 2,
        'BUY': 1,
        'WAIT': 0,
        'SELL': -1,
        'STRONG_SELL': -2
    }
    signal_value = signal_map.get(signal['action'], 0)
    
    # Write atomically (write to temp file, then rename)
    temp_file = METRICS_FILE.with_suffix('.prom.tmp')
    
    with open(temp_file, 'w') as f:
        # Direction prediction
        f.write('# HELP btc_ml_direction_prediction Predicted direction: -1=DOWN, 0=NONE, 1=UP\n')
        f.write('# TYPE btc_ml_direction_prediction gauge\n')
        f.write(f'btc_ml_direction_prediction{{model="ultra_deep",symbol="BTCUSDT"}} {direction_value}\n')
        f.write('\n')
        
        # Momentum strength
        f.write('# HELP btc_ml_momentum_strength Predicted momentum strength (0-100)\n')
        f.write('# TYPE btc_ml_momentum_strength gauge\n')
        f.write(f'btc_ml_momentum_strength{{model="ultra_deep",symbol="BTCUSDT"}} {state["strength"]}\n')
        f.write('\n')
        
        # Confidence
        f.write('# HELP btc_ml_confidence Model confidence in prediction (0-100%)\n')
        f.write('# TYPE btc_ml_confidence gauge\n')
        f.write(f'btc_ml_confidence{{model="ultra_deep",symbol="BTCUSDT"}} {state["confidence"] * 100}\n')
        f.write('\n')
        
        # Probabilities
        f.write('# HELP btc_ml_probability_down Probability of DOWN direction (0-100%)\n')
        f.write('# TYPE btc_ml_probability_down gauge\n')
        f.write(f'btc_ml_probability_down{{model="ultra_deep",symbol="BTCUSDT"}} {state["prob_down"]}\n')
        f.write('\n')
        
        f.write('# HELP btc_ml_probability_none Probability of SIDEWAYS direction (0-100%)\n')
        f.write('# TYPE btc_ml_probability_none gauge\n')
        f.write(f'btc_ml_probability_none{{model="ultra_deep",symbol="BTCUSDT"}} {state["prob_none"]}\n')
        f.write('\n')
        
        f.write('# HELP btc_ml_probability_up Probability of UP direction (0-100%)\n')
        f.write('# TYPE btc_ml_probability_up gauge\n')
        f.write(f'btc_ml_probability_up{{model="ultra_deep",symbol="BTCUSDT"}} {state["prob_up"]}\n')
        f.write('\n')
        
        # Fair value
        f.write('# HELP btc_ml_fair_value Calculated fair value\n')
        f.write('# TYPE btc_ml_fair_value gauge\n')
        f.write(f'btc_ml_fair_value{{symbol="BTCUSDT"}} {fair["fair_value"]}\n')
        f.write('\n')
        
        # Deviation from fair value
        f.write('# HELP btc_ml_deviation_percent Deviation from fair value (%)\n')
        f.write('# TYPE btc_ml_deviation_percent gauge\n')
        f.write(f'btc_ml_deviation_percent{{symbol="BTCUSDT"}} {fair["deviation_pct"]}\n')
        f.write('\n')
        
        # Assessment
        f.write('# HELP btc_ml_assessment Fair value assessment: -2=UNDERVALUED, -1=SLIGHTLY_UNDER, 0=FAIR, 1=SLIGHTLY_OVER, 2=OVERVALUED\n')
        f.write('# TYPE btc_ml_assessment gauge\n')
        f.write(f'btc_ml_assessment{{symbol="BTCUSDT"}} {assessment_value}\n')
        f.write('\n')
        
        # Trading signal
        f.write('# HELP btc_ml_trading_signal Trading signal: -2=STRONG_SELL, -1=SELL, 0=WAIT, 1=BUY, 2=STRONG_BUY\n')
        f.write('# TYPE btc_ml_trading_signal gauge\n')
        f.write(f'btc_ml_trading_signal{{symbol="BTCUSDT"}} {signal_value}\n')
        f.write('\n')
        
        # Position size
        f.write('# HELP btc_ml_position_size Recommended position size (%)\n')
        f.write('# TYPE btc_ml_position_size gauge\n')
        f.write(f'btc_ml_position_size{{symbol="BTCUSDT"}} {signal["position_size"]}\n')
        f.write('\n')
        
        # Current price (for reference)
        f.write('# HELP btc_ml_current_price Current BTC price (for reference)\n')
        f.write('# TYPE btc_ml_current_price gauge\n')
        f.write(f'btc_ml_current_price{{symbol="BTCUSDT"}} {price}\n')
        f.write('\n')
        
        # Last update timestamp
        f.write('# HELP btc_ml_last_update_timestamp Unix timestamp of last update\n')
        f.write('# TYPE btc_ml_last_update_timestamp gauge\n')
        f.write(f'btc_ml_last_update_timestamp{{model="ultra_deep"}} {time.time()}\n')
    
    # Atomic rename
    temp_file.rename(METRICS_FILE)

def main():
    """Test the write function"""
    print("Testing .prom file writing...")
    print(f"Metrics file: {METRICS_FILE}")
    print()
    
    # Example data
    state = {
        'success': True,
        'direction': 'UP',
        'strength': 75.3,
        'confidence': 0.892,
        'prob_down': 5.3,
        'prob_none': 5.5,
        'prob_up': 89.2
    }
    
    fair = {
        'success': True,
        'fair_value': 108500.0,
        'deviation_pct': 0.35,
        'assessment': 'FAIR',
        'current_price': 108850.0
    }
    
    signal = {
        'action': 'BUY',
        'position_size': 1.5
    }
    
    price = 108850.0
    
    write_metrics(state, fair, signal, price)
    
    print(f"✅ Wrote metrics to {METRICS_FILE}")
    print()
    print("Contents:")
    print("-" * 60)
    with open(METRICS_FILE) as f:
        print(f.read())
    print("-" * 60)

if __name__ == '__main__':
    main()

