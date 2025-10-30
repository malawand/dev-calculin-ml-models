# Bitcoin Minimal Start - Incremental Feature Training

## 🎯 Philosophy

**Start Small, Add Gradually**

Instead of throwing 419 features at a model and hoping for the best, we:
1. Start with **3 minimal features**
2. Train and measure accuracy
3. Add **1 feature at a time**
4. Keep it **only if accuracy improves**
5. Stop when no improvement for 5 iterations

---

## 📊 Starting Features (3 Total)

We begin with the **absolute minimum** to capture different market aspects:

1. **`deriv30d_roc`** - Long-term trend (30-day derivative rate of change)
2. **`volatility_24`** - Market volatility (24-period rolling std)
3. **`avg14d_spread`** - Mean reversion signal (spread from 14-day average)

**Why these 3?**
- ✅ Diverse: Trend + Volatility + Mean Reversion
- ✅ Simple: Easy to understand and interpret
- ✅ Proven: These were in the top 9 of the baseline model
- ✅ Not correlated: Independent signals

---

## 🚀 How It Works

### Phase 1: Train Minimal Model (3 features)
```bash
python train_minimal.py --horizon 24h
```

**Expected**: ~52-54% accuracy (barely better than random)

### Phase 2: Add Features One-by-One
```bash
python incremental_train.py --start-features 3 --max-features 20
```

**Process**:
1. Rank remaining 416 features by correlation with target
2. Try top 3 candidates
3. Add best one if accuracy improves
4. Repeat until no improvement

### Phase 3: Final Model
- Save best feature set
- Retrain with optimal features
- Compare vs baseline (58.36%)

**Goal**: Beat baseline with **fewer features** = simpler, more robust model

---

## 📁 Directory Structure

```
btc_minimal_start/
├── README.md                      # This file
├── config.yaml                    # Configuration
├── train_minimal.py               # Train with 3 features
├── incremental_train.py           # Add features gradually
├── models/
│   ├── minimal_3feat.pt           # 3-feature model
│   ├── incremental_best.pt        # Best incremental model
│   └── feature_history.json       # Track which features added
├── results/
│   ├── minimal_results.json       # 3-feature results
│   └── incremental_progress.json  # Iteration history
└── logs/
    └── training.log               # Training logs
```

---

## 🎯 Strategy

### The Incremental Approach

```
Iteration 1: [deriv30d_roc, volatility_24, avg14d_spread]
             Accuracy: 52.3%
             
Iteration 2: Try adding deriv3d_roc
             Accuracy: 54.1% ✅ KEEP (improved!)
             
Iteration 3: Try adding align_strength_7d_7d  
             Accuracy: 55.8% ✅ KEEP (improved!)
             
Iteration 4: Try adding deriv7d_prime7d_jerk
             Accuracy: 57.2% ✅ KEEP (improved!)
             
Iteration 5: Try adding deriv24h_prime24h
             Accuracy: 57.1% ❌ SKIP (no improvement)
             
Iteration 6: Try adding momentum_state_7d_24h_vol_adj
             Accuracy: 58.5% ✅ KEEP (improved!)
             
Iteration 7: Try adding medium_magnitude_std
             Accuracy: 58.3% ❌ SKIP (no improvement)
             
... continue until 5 consecutive no-improvements

Final: 6 features, 58.5% accuracy
```

---

## 💡 Key Insights

### Why This Works Better

1. **Avoid Overfitting**: Fewer features = less chance to memorize noise
2. **Find True Signal**: Only keep features that actually help
3. **Simpler Model**: Easier to understand and debug
4. **Faster Training**: 6 features trains faster than 419!
5. **More Robust**: Generalizes better to new data

### Comparison to Baseline

| Approach | Features | Accuracy | Training Time |
|----------|----------|----------|---------------|
| **Baseline** | 9 (LightGBM selected) | 58.36% | ~30 min |
| **Incremental** | TBD (start with 3) | TBD | ~10 min per iteration |
| **Goal** | 5-10 | >58% | Faster |

---

## 🔬 Experiments to Run

### Experiment 1: Different Starting Sets

Try different 3-feature combinations:

**Set A (Trend-focused)**:
- deriv30d_roc, deriv7d_roc, deriv3d_roc

**Set B (Volatility-focused)**:
- volatility_24, volatility_12, volatility_72

**Set C (Balanced - RECOMMENDED)**:
- deriv30d_roc, volatility_24, avg14d_spread

### Experiment 2: Different Addition Strategies

**Greedy (current)**:
- Add best feature each iteration

**Batch**:
- Try adding 2-3 features at once

**Stepwise Regression**:
- Remove features that become redundant

---

## 📊 Expected Timeline

### Day 1: Minimal Model
- Train 3-feature model: 5 minutes
- Result: ~52-54% accuracy
- Baseline established

### Day 2-3: Incremental Training
- Run 10-15 iterations: ~2-3 hours
- Find optimal feature set: 5-10 features
- Result: ~56-59% accuracy

### Day 4: Comparison & Analysis
- Compare vs baseline
- Understand which features matter most
- Document findings

---

## 🎯 Success Criteria

### Minimal Success
- ✅ 3-feature model works (>50% accuracy)
- ✅ Incremental process completes
- ✅ Find feature set better than random

### Good Success
- ✅ Match baseline (58%) with fewer features
- ✅ Clear feature ranking
- ✅ Understand why each feature helps

### Exceptional Success
- ✅ Beat baseline (>58%) with <7 features
- ✅ Faster training
- ✅ More robust to new data

---

## 🚀 Quick Start

```bash
# 1. Train minimal 3-feature model
cd btc_minimal_start
python train_minimal.py --horizon 24h

# 2. Run incremental training
python incremental_train.py --start-features 3 --max-features 20

# 3. View results
cat results/incremental_progress.json

# 4. Compare to baseline
python compare_to_baseline.py
```

---

## 📝 Notes

- Uses same data as baseline (1 year from btc_direction_predictor/artifacts)
- Same LSTM architecture (2-layer, 64 units each)
- Same validation split (80/20)
- Same evaluation metrics (Accuracy, ROC-AUC, MCC)

**The ONLY difference**: Feature selection strategy

---

## 🎓 What We'll Learn

1. **Minimum Viable Features**: What's the smallest set that works?
2. **Feature Interactions**: Do some features only work together?
3. **Diminishing Returns**: At what point does adding features stop helping?
4. **Optimal Complexity**: What's the sweet spot between simple and complex?

---

**Created**: 2025-10-17  
**Status**: 🆕 Ready to start  
**Goal**: Beat 58.36% baseline with <10 features



