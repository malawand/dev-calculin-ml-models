#!/usr/bin/env python3
"""
Compare 30-day baseline vs 2.5-year full dataset results
"""
import json
from pathlib import Path
import sys

def load_results(path):
    with open(path) as f:
        return json.load(f)

def main():
    # Load baseline (30-day)
    baseline_path = Path(__file__).parent / 'checkpoints/30day_baseline/scalping_model_results.json'
    
    # Load current (2.5-year)
    current_path = Path(__file__).parent / 'results/scalping_model_results.json'
    
    if not baseline_path.exists():
        print("❌ Baseline results not found!")
        sys.exit(1)
    
    if not current_path.exists():
        print("❌ Current results not found - training may not be complete")
        sys.exit(1)
    
    baseline = load_results(baseline_path)
    current = load_results(current_path)
    
    print("="*80)
    print("📊 COMPARISON: 30-Day Baseline vs 2.5-Year Full Dataset")
    print("="*80)
    print()
    
    # Dataset info
    print("📁 Dataset Information:")
    print(f"   Baseline: {baseline['data_samples']:,} samples")
    print(f"   Current:  {current['data_samples']:,} samples")
    print(f"   Increase: {current['data_samples'] / baseline['data_samples']:.1f}x more data")
    print()
    
    # Best models
    baseline_best = baseline['best_config']
    current_best = current['best_config']
    
    print("🥇 BEST CONFIGURATIONS:")
    print()
    print("Baseline (30 days):")
    print(f"   Config: {baseline_best['name']}")
    print(f"   Overall Accuracy:     {baseline_best['overall_accuracy']:.2%}")
    print(f"   Directional Accuracy: {baseline_best['directional_accuracy']:.2%}")
    print(f"   High-Conf Accuracy:   {baseline_best['high_conf_accuracy']:.2%}")
    print(f"   Trading Signals:      {baseline_best['directional_signals_pct']:.1%} of time")
    print()
    
    print("Current (2.5 years):")
    print(f"   Config: {current_best['name']}")
    print(f"   Overall Accuracy:     {current_best['overall_accuracy']:.2%}")
    print(f"   Directional Accuracy: {current_best['directional_accuracy']:.2%}")
    print(f"   High-Conf Accuracy:   {current_best['high_conf_accuracy']:.2%}")
    print(f"   Trading Signals:      {current_best['directional_signals_pct']:.1%} of time")
    print()
    
    # Changes
    print("📈 CHANGES (Current - Baseline):")
    overall_diff = current_best['overall_accuracy'] - baseline_best['overall_accuracy']
    dir_diff = current_best['directional_accuracy'] - baseline_best['directional_accuracy']
    hc_diff = current_best['high_conf_accuracy'] - baseline_best['high_conf_accuracy']
    
    print(f"   Overall Accuracy:     {overall_diff:+.2%} {'✅' if overall_diff > 0 else '⚠️'}")
    print(f"   Directional Accuracy: {dir_diff:+.2%} {'✅' if dir_diff > 0 else '⚠️'}")
    print(f"   High-Conf Accuracy:   {hc_diff:+.2%} {'✅' if hc_diff > 0 else '⚠️'}")
    print()
    
    # Per-class comparison
    print("🎯 PER-CLASS PERFORMANCE:")
    print()
    print("DOWN Detection:")
    print(f"   Baseline: {baseline_best.get('down_acc', 0):.2%}")
    print(f"   Current:  {current_best.get('down_acc', 0):.2%}")
    print(f"   Change:   {current_best.get('down_acc', 0) - baseline_best.get('down_acc', 0):+.2%}")
    print()
    
    print("SIDEWAYS Detection:")
    print(f"   Baseline: {baseline_best.get('sideways_acc', 0):.2%}")
    print(f"   Current:  {current_best.get('sideways_acc', 0):.2%}")
    print(f"   Change:   {current_best.get('sideways_acc', 0) - baseline_best.get('sideways_acc', 0):+.2%}")
    print()
    
    print("UP Detection:")
    print(f"   Baseline: {baseline_best.get('up_acc', 0):.2%}")
    print(f"   Current:  {current_best.get('up_acc', 0):.2%}")
    print(f"   Change:   {current_best.get('up_acc', 0) - baseline_best.get('up_acc', 0):+.2%}")
    print()
    
    # Recommendation
    print("="*80)
    print("💡 RECOMMENDATION:")
    print("="*80)
    
    if dir_diff > 0.05:  # 5%+ improvement in directional accuracy
        print("✅ Use 2.5-year model - Significantly better directional accuracy!")
        print("   The model learned more robust patterns across market cycles.")
    elif dir_diff > 0:
        print("✅ Use 2.5-year model - Improved directional accuracy.")
        print("   More data provides better generalization.")
    elif dir_diff > -0.05:
        print("⚠️  Similar performance - Consider ensemble approach")
        print("   Use 30-day for recent trends, 2.5-year for robustness.")
    else:
        print("⚠️  30-day model performed better on this metric")
        print("   But 2.5-year model is likely more robust long-term.")
    
    print()
    print("="*80)

if __name__ == "__main__":
    main()



