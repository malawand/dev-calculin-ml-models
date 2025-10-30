# 🧪 Option B: 3 Experiments on ~2 Years of Data - RUNNING

**Dataset:** 2023-12-30 → 2025-10-17 (~1.8 years, 40,109 samples)  
**Status:** All 3 experiments running in parallel  
**Expected Completion:** 30-40 minutes

---

## 📊 Experiments Running

### Experiment 1: Multi-Timeframe Derivatives
- **Starting Features:** `deriv30d_roc`, `deriv7d_roc`, `deriv3d_roc`
- **Hypothesis:** Multi-timeframe trend signals
- **1-Year Result:** 60.37% (4 features)
- **Log:** `logs/2year_exp1_derivatives.log`

### Experiment 2: Multi-Timeframe Volatility
- **Starting Features:** `volatility_72`, `volatility_24`, `volatility_12`
- **Hypothesis:** Volatility regime detection
- **1-Year Result:** 61.89% (7 features)
- **Log:** `logs/2year_exp2_volatility.log`

### Experiment 3: Advanced Features (Top from Baseline) ⭐
- **Starting Features:** `deriv7d_prime7d`, `deriv4d_roc`, `volatility_24`
- **Hypothesis:** Proven top features should work
- **1-Year Result:** **73.36%** (8 features) ← WINNER
- **Log:** `logs/2year_exp3_advanced.log`

---

## 🎯 What We're Looking For

### Success Criteria:
1. **Test Accuracy > 58%** - Better than baseline
2. **Train/Test Gap < 15%** - Not overfitting
3. **Consistent features** - Same features emerge as important
4. **Robust across time** - Works on different market periods

### Key Questions:
- ❓ Do the same starting features win on 2 years?
- ❓ Do we discover the same 8 features?
- ❓ Is the 73% result reproducible or was it lucky?
- ❓ What's the true generalizable accuracy?

---

## 📈 Comparison Targets

| Experiment | 1 Year Data | 2 Year Data (Expected) |
|-----------|------------|----------------------|
| **Exp 1 (Derivatives)** | 60.37% | ? |
| **Exp 2 (Volatility)** | 61.89% | ? |
| **Exp 3 (Advanced)** | **73.36%** | **? ⭐** |
| Fixed 8 Features | 73.36% | **54.50%** (already tested) |

---

## 🔍 Already Known

From Option A (training fixed 8 features on 2 years):
- **Result:** 54.50% accuracy
- **Issue:** Large train/test gap (21.65%)
- **Conclusion:** The exact 8 features from 1-year are overfit

**This means:** We need to let incremental training discover better features for the 2-year period!

---

## 🎯 Expected Outcomes

### Best Case:
- Experiment 3 finds different 8 features
- Achieves 65-70% accuracy (higher than 54.50%)
- Small train/test gap (<15%)
- **Conclusion:** Incremental approach works! Use these new features.

### Good Case:
- One experiment achieves 60-65% accuracy
- Moderate train/test gap (15-20%)
- Features make intuitive sense
- **Conclusion:** Model is useful but needs periodic retraining.

### Concerning Case:
- All experiments < 58% accuracy
- Large train/test gaps (>20%)
- Unstable feature selection
- **Conclusion:** Bitcoin is too non-stationary for this approach.

---

## ⏱️ Timeline

```
[Started]     All 3 experiments launched
              ↓
[~5-10 min]   Feature engineering complete
              ↓
[~10-20 min]  Incremental training in progress
              ↓
[~30-40 min]  All experiments complete ✅
              ↓
[Analysis]    Compare results, identify winner
```

---

## 🚀 Next Steps After Completion

1. **Compare all results** - Which starting point won?
2. **Analyze features** - Are they different from 1-year?
3. **Check overfitting** - Train/test gaps acceptable?
4. **Make decision:**
   - If results are good (>60%): Use for production
   - If results are mixed (55-60%): Implement periodic retraining
   - If results are poor (<55%): Consider other approaches (ensemble, regime-switching)

---

**Monitor progress:** Run `./check_progress.sh` or `tail -f logs/2year_exp*.log`



