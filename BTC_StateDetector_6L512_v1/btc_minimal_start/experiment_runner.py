#!/usr/bin/env python3
"""
Run Multiple Incremental Training Experiments

Tests different starting feature combinations to find the best approach.
"""

import subprocess
import json
import time
from pathlib import Path

# Define different starting feature sets to experiment with
EXPERIMENTS = [
    {
        'name': 'exp1_trend_focus',
        'features': ['deriv30d_roc', 'deriv7d_roc', 'deriv3d_roc'],
        'description': 'Long/medium/short term trends'
    },
    {
        'name': 'exp2_volatility_focus',
        'features': ['volatility_72', 'volatility_24', 'volatility_12'],
        'description': 'Multi-timeframe volatility'
    },
    {
        'name': 'exp3_derivative_primes',
        'features': ['deriv7d_prime7d', 'deriv24h_prime24h', 'deriv3d_prime3d'],
        'description': 'Acceleration indicators (2nd derivatives)'
    },
    {
        'name': 'exp4_momentum_mix',
        'features': ['persistence_7d', 'momentum_vol_4h', 'align_strength_7d_7d'],
        'description': 'Advanced momentum indicators'
    },
    {
        'name': 'exp5_mixed_best',
        'features': ['deriv7d_prime7d', 'volatility_24', 'deriv4d_roc'],
        'description': 'Top features from LightGBM baseline'
    },
    {
        'name': 'exp6_lagged_derivs',
        'features': ['deriv7d_lag6', 'deriv4d_lag6', 'deriv24h_lag6'],
        'description': 'Lagged derivative signals'
    }
]


def run_experiment(exp_name, features, description):
    """Run one incremental training experiment"""
    
    print(f"\n{'='*80}")
    print(f"🔬 EXPERIMENT: {exp_name}")
    print(f"   {description}")
    print(f"   Starting features: {features}")
    print(f"{'='*80}\n")
    
    # Create modified training script for this experiment
    script_content = f'''#!/usr/bin/env python3
import sys
sys.path.insert(0, "/Users/mazenlawand/Documents/Caculin ML")

# Import the incremental training main
exec(open("/Users/mazenlawand/Documents/Caculin ML/btc_minimal_start/incremental_simple.py").read())

# Override starting features
MINIMAL_FEATURES = {features}

if __name__ == "__main__":
    main()
'''
    
    # Save temporary script
    temp_script = f'/tmp/{exp_name}.py'
    with open(temp_script, 'w') as f:
        f.write(script_content)
    
    # Run training
    log_file = f'logs/{exp_name}.log'
    result_file = f'results/{exp_name}_results.json'
    
    start_time = time.time()
    
    try:
        subprocess.run(
            f'cd "/Users/mazenlawand/Documents/Caculin ML/btc_minimal_start" && '
            f'source venv/bin/activate && '
            f'python3 {temp_script} > {log_file} 2>&1',
            shell=True,
            check=True,
            executable='/bin/zsh'
        )
        
        elapsed = time.time() - start_time
        
        # Load results
        if Path(f'results/incremental_final.json').exists():
            with open('results/incremental_final.json', 'r') as f:
                results = json.load(f)
            
            # Save with experiment name
            with open(result_file, 'w') as f:
                results['experiment'] = exp_name
                results['description'] = description
                results['elapsed_time'] = elapsed
                json.dump(results, f, indent=2)
            
            print(f"\n✅ {exp_name} complete!")
            print(f"   Final accuracy: {results['best_accuracy']:.4f}")
            print(f"   Features used: {len(results['best_features'])}")
            print(f"   Time: {elapsed:.1f}s")
            
            return results
        else:
            print(f"❌ {exp_name} failed - no results file")
            return None
            
    except Exception as e:
        print(f"❌ {exp_name} failed: {e}")
        return None


def main():
    """Run all experiments and compare"""
    
    print("="*80)
    print("🧪 RUNNING MULTIPLE INCREMENTAL TRAINING EXPERIMENTS")
    print("="*80)
    print(f"\nTotal experiments: {len(EXPERIMENTS)}")
    print(f"Expected duration: ~{len(EXPERIMENTS) * 5} minutes\n")
    
    Path('logs').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    all_results = []
    
    for exp in EXPERIMENTS:
        result = run_experiment(
            exp['name'],
            exp['features'],
            exp['description']
        )
        
        if result:
            all_results.append(result)
        
        time.sleep(2)  # Brief pause between experiments
    
    # Compare results
    print("\n" + "="*80)
    print("📊 FINAL COMPARISON")
    print("="*80)
    
    if not all_results:
        print("❌ No successful experiments")
        return
    
    # Sort by accuracy
    all_results.sort(key=lambda x: x['best_accuracy'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Experiment':<25} {'Features':<10} {'Accuracy':<12} {'Description'}")
    print("-"*80)
    
    for i, result in enumerate(all_results, 1):
        exp_name = result.get('experiment', 'unknown')
        desc = result.get('description', '')
        n_features = len(result['best_features'])
        accuracy = result['best_accuracy']
        
        print(f"{i:<6} {exp_name:<25} {n_features:<10} {accuracy:.4f} ({accuracy*100:.2f}%)  {desc[:30]}")
    
    # Best result
    best = all_results[0]
    print("\n" + "="*80)
    print("🏆 BEST RESULT")
    print("="*80)
    print(f"\nExperiment: {best.get('experiment')}")
    print(f"Description: {best.get('description')}")
    print(f"Final accuracy: {best['best_accuracy']:.4f} ({best['best_accuracy']*100:.2f}%)")
    print(f"Number of features: {len(best['best_features'])}")
    print(f"\nBest features:")
    for i, feat in enumerate(best['best_features'], 1):
        print(f"  {i}. {feat}")
    
    # Save summary
    with open('results/experiment_summary.json', 'w') as f:
        json.dump({
            'experiments': all_results,
            'best_experiment': best
        }, f, indent=2)
    
    print(f"\n💾 Summary saved: results/experiment_summary.json")
    
    # Comparison to baseline
    baseline_acc = 0.5836
    best_acc = best['best_accuracy']
    
    print(f"\n📊 vs Baseline:")
    print(f"   Baseline: 58.36% (9 features)")
    print(f"   Best:     {best_acc*100:.2f}% ({len(best['best_features'])} features)")
    print(f"   Gap:      {(baseline_acc - best_acc)*100:.2f}%")
    
    if best_acc > baseline_acc:
        print(f"\n🎉 WE BEAT THE BASELINE! +{(best_acc - baseline_acc)*100:.2f}%")
    elif best_acc > 0.57:
        print(f"\n✅ Very close to baseline! Just {(baseline_acc - best_acc)*100:.2f}% away")
    else:
        print(f"\n⚠️  Still {(baseline_acc - best_acc)*100:.2f}% below baseline")


if __name__ == "__main__":
    main()



