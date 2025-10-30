#!/usr/bin/env python3
"""
Monitor Incremental Training Progress

Watch the incremental training in real-time
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime


def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')


def monitor():
    """Monitor training progress"""
    progress_file = Path('results/incremental_progress.json')
    
    print("="*80)
    print("📊 INCREMENTAL TRAINING MONITOR")
    print("="*80)
    print("\nWatching: results/incremental_progress.json")
    print("Press Ctrl+C to stop monitoring\n")
    
    last_iteration = 0
    
    try:
        while True:
            if not progress_file.exists():
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for training to start...")
                time.sleep(5)
                continue
            
            # Load progress
            with open(progress_file, 'r') as f:
                data = json.load(f)
            
            # Check if new iteration
            current_iteration = len(data['history'])
            if current_iteration > last_iteration:
                last_iteration = current_iteration
                
                # Clear and redisplay
                clear_screen()
                
                print("="*80)
                print("📊 INCREMENTAL TRAINING PROGRESS")
                print("="*80)
                print(f"\nLast Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Iteration: {current_iteration}")
                print(f"\n{'='*80}\n")
                
                # Current stats
                print("📈 Current Model:")
                print(f"   Features: {len(data['current_features'])}")
                print(f"   Accuracy: {data['current_accuracy']:.4f}")
                print(f"   ROC-AUC: {data['current_roc_auc']:.4f}")
                
                print(f"\n🏆 Best Model So Far:")
                print(f"   Features: {len(data['best_features'])}")
                print(f"   Accuracy: {data['best_accuracy']:.4f}")
                
                # Show features
                print(f"\n🎯 Current Features ({len(data['current_features'])}):")
                for i, feat in enumerate(data['current_features'], 1):
                    print(f"   {i:2d}. {feat}")
                
                # Recent history
                print(f"\n📜 Recent Actions (last 5):")
                for hist in data['history'][-5:]:
                    action = hist['action']
                    iter_num = hist['iteration']
                    acc = hist['accuracy']
                    
                    if action == 'add':
                        feat = hist.get('feature', '?')
                        imp = hist.get('improvement', 0)
                        print(f"   Iter {iter_num:2d}: ✅ Added {feat} (+{imp:.4f}) → {acc:.4f}")
                    elif action == 'skip':
                        print(f"   Iter {iter_num:2d}: ❌ No improvement → {acc:.4f}")
                    elif action == 'init':
                        print(f"   Iter {iter_num:2d}: 🎯 Initial baseline → {acc:.4f}")
                
                print(f"\n{'='*80}")
                print(f"Refreshing every 10 seconds... (Ctrl+C to stop)")
            
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\n✋ Monitoring stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    monitor()



