import json

print("="*80)
print("🏆 INCREMENTAL TRAINING EXPERIMENTS - FINAL RESULTS")
print("="*80)
print()

experiments = [
    ('Experiment 1: Multi-Timeframe Derivatives', 'logs/exp1_derivatives.log', 'deriv30d_roc, deriv7d_roc, deriv3d_roc'),
    ('Experiment 2: Multi-Timeframe Volatility', 'logs/exp2_volatility.log', 'volatility_72, volatility_24, volatility_12'),
    ('Experiment 3: Advanced Features (Top from Baseline)', 'logs/exp3_advanced.log', 'deriv7d_prime7d, deriv4d_roc, volatility_24'),
]

results = []

for name, log_file, starting in experiments:
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Extract results
        if 'TRAINING COMPLETE' in content:
            lines = content.split('\n')
            accuracy = None
            n_features = None
            features = []
            
            for i, line in enumerate(lines):
                if 'Best Model:' in line:
                    # Get accuracy from next few lines
                    for j in range(i, min(i+10, len(lines))):
                        if 'Accuracy:' in lines[j]:
                            accuracy = float(lines[j].split('Accuracy:')[1].strip())
                        if 'Features:' in lines[j] and 'Best Features:' not in lines[j]:
                            n_features = int(lines[j].split('Features:')[1].strip())
                if 'Best Features:' in line:
                    # Get feature list
                    for j in range(i+1, min(i+20, len(lines))):
                        if lines[j].strip() and lines[j].strip()[0].isdigit():
                            feat_name = lines[j].split('. ', 1)[1].strip()
                            features.append(feat_name)
                        elif lines[j].strip().startswith('💾'):
                            break
            
            results.append({
                'name': name,
                'starting': starting,
                'accuracy': accuracy,
                'n_features': n_features,
                'features': features,
                'status': 'COMPLETE'
            })
        else:
            results.append({
                'name': name,
                'starting': starting,
                'status': 'INCOMPLETE'
            })
    except Exception as e:
        print(f"Error reading {log_file}: {e}")

# Sort by accuracy (descending)
results.sort(key=lambda x: x.get('accuracy', 0), reverse=True)

# Display results
print(f"{'Rank':<6} {'Experiment':<45} {'Features':<10} {'Accuracy'}")
print("-"*80)

for i, r in enumerate(results, 1):
    if r['status'] == 'COMPLETE':
        print(f"{i:<6} {r['name']:<45} {r['n_features']:<10} {r['accuracy']:.4f} ({r['accuracy']*100:.2f}%)")
    else:
        print(f"{i:<6} {r['name']:<45} {'N/A':<10} {r['status']}")

print("\n" + "="*80)
print("🥇 WINNER")
print("="*80)

if results and results[0]['status'] == 'COMPLETE':
    best = results[0]
    print(f"\n{best['name']}")
    print(f"\nStarting features: {best['starting']}")
    print(f"\n✅ Final Accuracy: {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")
    print(f"✅ Total Features: {best['n_features']}")
    print(f"\n📋 Complete Feature List:")
    for i, feat in enumerate(best['features'], 1):
        print(f"   {i}. {feat}")
    
    # Compare to baseline
    baseline = 0.5836
    first_run = 0.5483
    
    print(f"\n📊 Performance Comparison:")
    print(f"   Baseline (LightGBM, 9 features):    58.36%")
    print(f"   First Run (Original 3):             54.83%")
    print(f"   Best Experiment:                    {best['accuracy']*100:.2f}%")
    
    if best['accuracy'] > baseline:
        print(f"\n🎉 WE BEAT THE BASELINE! +{(best['accuracy'] - baseline)*100:.2f}%")
    elif best['accuracy'] > first_run:
        print(f"\n✅ Improved over first run: +{(best['accuracy'] - first_run)*100:.2f}%")
    
    print(f"\n💡 Key Insights:")
    print(f"   - Used {best['n_features']} features (vs baseline's 9)")
    print(f"   - Starting point: {best['starting']}")
    print(f"   - Achieved {best['accuracy']*100:.2f}% accuracy")
    
    # Check for common patterns
    avg_features = [f for f in best['features'] if f.startswith('avg')]
    deriv_features = [f for f in best['features'] if 'deriv' in f]
    vol_features = [f for f in best['features'] if 'volatility' in f]
    
    print(f"\n📈 Feature Breakdown:")
    print(f"   - Moving averages: {len(avg_features)}")
    print(f"   - Derivatives: {len(deriv_features)}")
    print(f"   - Volatility: {len(vol_features)}")
    print(f"   - Other: {best['n_features'] - len(avg_features) - len(deriv_features) - len(vol_features)}")

print("\n" + "="*80)
