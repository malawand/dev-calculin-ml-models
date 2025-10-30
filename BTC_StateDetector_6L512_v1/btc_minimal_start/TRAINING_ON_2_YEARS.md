# 🚀 Training on 2 Years of Data - Status

**Goal:** Validate the winning 8-feature model on 2 years of historical data

---

## Phase 1: Option A - Retrain Optimal Features (IN PROGRESS)

### Step 1: Fetch 2 Years of Data ⏳
- **Date Range:** 2023-10-17 → 2025-10-17
- **Expected Samples:** ~70,000 (15-minute bars)
- **Duration:** 5-10 minutes
- **Status:** FETCHING NOW...

### Step 2: Train with Optimal 8 Features
- **Features to Use:**
  1. `deriv7d_prime7d` - 7-day acceleration
  2. `deriv4d_roc` - 4-day trend
  3. `volatility_24` - Volatility regime
  4. `avg30m` - 30-minute average
  5. `avg45m` - 45-minute average
  6. `avg1h` - 1-hour average
  7. `avg15m` - 15-minute average
  8. `avg10m` - 10-minute average

- **Expected Outcome:**
  - If accuracy stays ~73%: Features are robust across time
  - If accuracy improves: More data helps generalization
  - If accuracy drops: Possible regime shift or 1-year overfitting

### Step 3: Analysis
- Compare to 1-year result (73.36%)
- Check for overfitting (train vs test gap)
- Validate feature importance remains consistent

---

## Phase 2: Option B - Re-run All Experiments (WAITING)

After Option A completes, we'll run all 3 experiments on 2 years:

1. **Experiment 1:** Derivatives focus
2. **Experiment 2:** Volatility focus
3. **Experiment 3:** Advanced features (top from baseline)

This will validate:
- Do the same starting features win?
- Does the incremental algorithm discover the same 8 features?
- Is the winning pattern consistent across different time periods?

---

## Why 2 Years Matters

### Benefits:
- ✅ **More diverse market conditions** - Bull markets, bear markets, sideways
- ✅ **Better generalization** - Less likely to overfit to specific period
- ✅ **Robustness validation** - Does 73% hold up or was it lucky?
- ✅ **Regime testing** - See performance across different volatility regimes

### Risks:
- ⚠️ **Regime shifts** - Market dynamics may have changed
- ⚠️ **Non-stationarity** - Crypto markets evolve rapidly
- ⚠️ **Training time** - More data = longer training

---

## What We're Looking For

### Best Case Scenario:
- Test accuracy stays >70% on 2 years
- Similar train/test gap (not overfitting)
- Same 8 features remain most important
- **Conclusion:** Model is robust and production-ready!

### Good Scenario:
- Test accuracy 65-70% on 2 years
- Slightly wider train/test gap but acceptable
- Most of the 8 features still important
- **Conclusion:** Model works but may need periodic retraining

### Concerning Scenario:
- Test accuracy <60% on 2 years
- Large train/test gap (overfitting)
- Different features become important
- **Conclusion:** 1-year result was overfit, need more work

---

## Current Status

```
[=====>                    ] 20% - Fetching 2-year data from Cortex
```

**Next:** Once data is ready, run `train_2year_optimal.py` to see results!



