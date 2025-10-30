# 🚀 Getting Started with Minimal Start

## What You Have Now

```
✅ Runner-Ups/baseline_58pct/btc_lstm_ensemble/
   └── Your working 58.36% baseline model (SAFE!)

✅ btc_minimal_start/
   └── New incremental training system (START HERE!)
```

---

## 🎯 The Plan

### Phase 1: Train Minimal Model (Right Now!)

**Start with ONLY 3 features**:
1. `deriv30d_roc` - Long-term trend
2. `volatility_24` - Market volatility
3. `avg14d_spread` - Mean reversion signal

**Expected Result**: ~52-54% accuracy (barely better than random)

**Why only 3?**  
We want to establish the absolute baseline. This tells us: "Can we predict Bitcoin direction with just trend + volatility + mean reversion?"

---

### Phase 2: Add Features One-by-One (Tomorrow)

Once we have the 3-feature baseline, we'll **gradually add features** using the incremental trainer:

```python
# Pseudocode of what will happen:
current_features = [deriv30d_roc, volatility_24, avg14d_spread]
current_accuracy = 0.523

for iteration in range(50):
    # 1. Rank remaining features by importance
    candidates = rank_features_by_correlation(remaining_features)
    
    # 2. Try adding top 3 candidates
    for candidate in candidates[:3]:
        test_features = current_features + [candidate]
        new_accuracy = train_and_test(test_features)
        
        if new_accuracy > current_accuracy + 0.001:
            # Keep it!
            current_features.append(candidate)
            current_accuracy = new_accuracy
            break
    
    # 3. Stop if no improvement for 5 iterations
    if no_improvement_count >= 5:
        break

# Result: Optimal feature set (hopefully 5-10 features)
```

---

## 📊 Expected Results

| Phase | Features | Expected Accuracy | Training Time |
|-------|----------|-------------------|---------------|
| **Minimal (3 feat)** | 3 | 52-54% | ~2 minutes |
| **After 5 iterations** | 8 | 54-56% | ~10 minutes |
| **After 10 iterations** | 12 | 56-58% | ~20 minutes |
| **Final (optimal)** | 5-15 | >58% | ~30 minutes |
| **Baseline** | 9 | 58.36% | N/A |

**Goal**: Beat 58.36% with fewer features!

---

## 🚀 Quick Start Commands

### Step 1: Train Minimal Model (3 features)

```bash
cd "/Users/mazenlawand/Documents/Caculin ML/btc_minimal_start"

# Create virtual environment (if needed)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch pandas numpy scikit-learn pyyaml tqdm

# Train minimal model
python train_minimal.py --horizon 24h
```

**This will**:
- Load 1 year of data from `btc_direction_predictor/artifacts`
- Use only 3 features
- Train a small LSTM (32x32 units)
- Save results to `results/minimal_3feat_results.json`
- Take ~2 minutes

**Expected output**:
```
🎯 MINIMAL MODEL - 3 Features Only
================================================================================

Starting features:
  1. deriv30d_roc
  2. volatility_24
  3. avg14d_spread

================================================================================

📥 Loading cached data...
   Total features available: 419
   Using only 3 features

🔧 Preparing data...
   Valid samples: 16899
   Features shape: (16899, 3)
   Class balance: UP=9448/16899 (55.9%)

📊 Splitting data...
   Train: 13519 samples
   Test: 3380 samples

...

📊 FINAL EVALUATION
================================================================================

Minimal Model (3 features):
  Accuracy:  0.5234 (52.34%)
  ROC-AUC:   0.5512
  Features:  3

================================================================================
```

---

### Step 2: View Results

```bash
# See what you got
cat results/minimal_3feat_results.json

# Example output:
{
  "horizon": "24h",
  "features": [
    "deriv30d_roc",
    "volatility_24",
    "avg14d_spread"
  ],
  "num_features": 3,
  "accuracy": 0.5234,
  "roc_auc": 0.5512,
  "train_samples": 13495,
  "test_samples": 3356
}
```

---

### Step 3: Run Incremental Training (Next!)

**Coming soon**: `python incremental_train.py`

This will:
- Start with your 3-feature model
- Add features one at a time
- Track progress in `results/incremental_progress.json`
- Stop when no improvement for 5 iterations
- Save final model to `models/incremental_best.pt`

---

## 📈 What to Look For

### Good Signs ✅
- Minimal model gets >50% accuracy (better than random)
- Accuracy improves as you add features
- Model converges (not oscillating wildly)
- Features make intuitive sense

### Warning Signs ⚠️
- Minimal model <50% accuracy (something's wrong with data)
- Train accuracy 95%+ but validation 52% (overfitting)
- Adding features makes accuracy worse (wrong features)
- Model doesn't converge after 30 epochs (learning rate too high)

---

## 🎯 Success Metrics

### Minimal Success
- ✅ 3-feature model works (>50%)
- ✅ Can run full training pipeline
- ✅ Results saved correctly

### Good Success
- ✅ Incremental training improves accuracy
- ✅ Find feature set that matches baseline (58%)
- ✅ Use fewer features than baseline (9)

### Exceptional Success
- ✅ Beat baseline (>58.36%) with <7 features
- ✅ Understand why each feature helps
- ✅ Simple, interpretable model that works

---

## 🔬 Experiments to Try

After you get the minimal model working:

### Experiment 1: Different Starting Features

Try different 3-feature combinations:

```python
# Trend-focused
['deriv30d_roc', 'deriv7d_roc', 'deriv3d_roc']

# Volatility-focused
['volatility_24', 'volatility_12', 'volatility_72']

# Derivative-focused
['deriv24h_prime24h', 'deriv7d_prime7d', 'deriv4h_prime1h']
```

### Experiment 2: Different Model Sizes

```yaml
# Smaller model
lstm_hidden: [16, 16]  # Even simpler

# Bigger model
lstm_hidden: [64, 64]  # Same as baseline
```

### Experiment 3: Different Incremental Strategies

```yaml
# Aggressive: Add features faster
candidates_per_iter: 5
min_improvement: 0.0005

# Conservative: Only add if big improvement
candidates_per_iter: 1
min_improvement: 0.005
```

---

## 🐛 Troubleshooting

### "Missing features" Error
- Check that `deriv30d_roc`, `volatility_24`, `avg14d_spread` exist in data
- Run: `python -c "import pandas as pd; df = pd.read_parquet('../btc_direction_predictor/artifacts/historical_data/combined_full_dataset.parquet'); print(df.columns.tolist())"`

### "Not enough data" Error
- Make sure you have at least 1 year of data fetched
- Check: `ls -lh ../btc_direction_predictor/artifacts/historical_data/`

### "Model not converging" Error
- Try lower learning rate: `learning_rate: 0.0005`
- Try more epochs: `epochs: 50`
- Try smaller model: `lstm_hidden: [16, 16]`

### "Accuracy stuck at 50%" Error
- This is actually expected for minimal model!
- 50% = random guessing
- 52-54% = slight signal (good for 3 features)
- 56-58% = strong signal (what we want after adding features)

---

## 📚 What's Next

### Today
1. ✅ Train minimal 3-feature model
2. Review results
3. Understand baseline

### Tomorrow
1. Run incremental training
2. Track which features get added
3. Compare to baseline

### This Week
1. Experiment with different starting sets
2. Find optimal feature set
3. Beat the baseline!

---

## 🎓 Learning Goals

By the end of this, you'll know:
1. **Minimum viable features**: What's the smallest set that works?
2. **Feature importance**: Which features actually matter?
3. **Diminishing returns**: When does adding features stop helping?
4. **Model complexity**: What's the sweet spot?

---

**Ready?**

```bash
cd "/Users/mazenlawand/Documents/Caculin ML/btc_minimal_start"
python train_minimal.py --horizon 24h
```

**Let's start minimal and build up! 🚀**



