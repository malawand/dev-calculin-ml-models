print("="*80)
print("🏆 OPTION B: INCREMENTAL TRAINING ON ~2 YEARS - FINAL RESULTS")
print("="*80)
print()

experiments = [
    ('Experiment 1: Derivatives', 'logs/2year_exp1_derivatives.log'),
    ('Experiment 2: Volatility', 'logs/2year_exp2_volatility.log'),
    ('Experiment 3: Advanced (Top Features)', 'logs/2year_exp3_advanced.log'),
]

results = []

for name, log_file in experiments:
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        accuracy = None
        n_features = None
        features = []
        
        for i, line in enumerate(lines):
            if 'Best Model:' in line:
                for j in range(i, min(i+10, len(lines))):
                    if 'Accuracy:' in lines[j]:
                        accuracy = float(lines[j].split('Accuracy:')[1].strip())
                    if 'Features:' in lines[j] and 'Best Features:' not in lines[j]:
                        n_features = int(lines[j].split('Features:')[1].strip())
            if 'Best Features:' in line:
                for j in range(i+1, min(i+20, len(lines))):
                    if lines[j].strip() and lines[j].strip()[0].isdigit():
                        feat_name = lines[j].split('. ', 1)[1].strip()
                        features.append(feat_name)
                    elif '💾' in lines[j] or '=' in lines[j]:
                        break
        
        results.append({
            'name': name,
            'accuracy': accuracy,
            'n_features': n_features,
            'features': features
        })
    except Exception as e:
        print(f"Error reading {log_file}: {e}")

# Sort by accuracy
results.sort(key=lambda x: x.get('accuracy', 0), reverse=True)

# Display results
print(f"{'Rank':<6} {'Experiment':<40} {'Features':<10} {'Accuracy'}")
print("-"*80)

for i, r in enumerate(results, 1):
    print(f"{i:<6} {r['name']:<40} {r['n_features']:<10} {r['accuracy']:.4f} ({r['accuracy']*100:.2f}%)")

print("\n" + "="*80)
print("🥇 WINNER: " + results[0]['name'])
print("="*80)

best = results[0]
print(f"\n✅ Final Accuracy: {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")
print(f"✅ Total Features: {best['n_features']}")
print(f"\n📋 Complete Feature List:")
for i, feat in enumerate(best['features'], 1):
    print(f"   {i}. {feat}")

# Comparisons
print(f"\n{'='*80}")
print("📊 COMPARISON: 1 YEAR vs 2 YEARS")
print(f"{'='*80}\n")

one_year = {
    'Exp 1': 0.6037,
    'Exp 2': 0.6189,
    'Exp 3': 0.7336
}

two_year = {
    'Exp 1': 0.6088,
    'Exp 2': 0.6335,
    'Exp 3': 0.7646
}

print(f"{'Experiment':<25} {'1 Year':<12} {'2 Years':<12} {'Change'}")
print("-"*65)
for exp in ['Exp 1', 'Exp 2', 'Exp 3']:
    change = two_year[exp] - one_year[exp]
    emoji = '✅' if change > 0 else '⚠️'
    print(f"{exp:<25} {one_year[exp]*100:>6.2f}%     {two_year[exp]*100:>6.2f}%     {change*100:>+6.2f}% {emoji}")

print(f"\n{'='*80}")
print("💡 KEY INSIGHTS")
print(f"{'='*80}\n")

print("✅ Experiment 3 (Advanced Features) WINS AGAIN!")
print(f"   - 1 Year:  73.36%")
print(f"   - 2 Years: 76.46% (+3.10%)")
print(f"   → The optimal starting features are VALIDATED!")
print()
print("✅ ALL experiments improved with more data!")
print(f"   - Exp 1: +0.51%")
print(f"   - Exp 2: +1.46%")
print(f"   - Exp 3: +3.10%")
print(f"   → More data = Better generalization")
print()
print("✅ Consistent pattern:")
print(f"   - Starting with proven top features works best")
print(f"   - Multi-timeframe derivatives work well")
print(f"   - Volatility signals are useful")
print()

# vs fixed 8 features
print(f"🔍 Why did Exp 3 beat the fixed 8 features?")
print(f"   - Fixed 8 features (Option A): 54.50%")
print(f"   - Incremental discovery (Option B): 76.46%")
print(f"   - Difference: +21.96%!")
print(f"   → Incremental approach discovered BETTER features for 2 years")
print()

print(f"{'='*80}")
