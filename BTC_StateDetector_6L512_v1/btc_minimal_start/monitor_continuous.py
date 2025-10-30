#!/usr/bin/env python3
"""
Continuous BTC Scalping Monitor
Checks every minute and alerts on trading signals
"""
import time
from datetime import datetime
from predict_live import predict

def main():
    print("="*80)
    print("🤖 BTC SCALPING MONITOR - CONTINUOUS MODE")
    print("="*80)
    print("Checking every 60 seconds for trading signals...")
    print("Press Ctrl+C to stop")
    print("="*80)
    print()
    
    last_action = None
    signal_count = 0
    trade_count = 0
    
    while True:
        try:
            # Get prediction
            result = predict()
            
            # Track stats
            signal_count += 1
            
            # Check if new trading signal
            current_action = result['action']
            
            if result['should_trade']:
                trade_count += 1
                
                # Alert
                print("\n" + "🔔"*40)
                print(f"🚨 TRADING SIGNAL #{trade_count}")
                print(f"   Action: {current_action}")
                print(f"   Confidence: {result['confidence']:.1%}")
                print(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
                print("🔔"*40 + "\n")
                
                last_action = current_action
            else:
                # No trade
                status = "⏸️  No Trade" if result['prediction'] == 'SIDEWAYS' else f"🔕 Low Conf ({result['confidence']:.0%})"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {status} | Checked: {signal_count} | Traded: {trade_count}")
            
            # Wait 60 seconds
            time.sleep(60)
            
        except KeyboardInterrupt:
            print("\n\n" + "="*80)
            print("📊 SESSION SUMMARY")
            print("="*80)
            print(f"Total checks: {signal_count}")
            print(f"Trading signals: {trade_count}")
            print(f"Signal rate: {trade_count/signal_count:.1%}")
            print("\n✅ Monitor stopped")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Retrying in 60 seconds...")
            time.sleep(60)

if __name__ == "__main__":
    main()



