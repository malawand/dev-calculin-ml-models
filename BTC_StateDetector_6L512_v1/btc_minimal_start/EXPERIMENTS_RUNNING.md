# 🧪 Incremental Training Experiments - RUNNING

**Status:** 3 experiments running in parallel  
**Started:** Just now  
**Expected completion:** ~30-40 minutes  

---

## Experiments

### Experiment 1: Multi-Timeframe Derivatives
- **Starting Features:**
  - `deriv30d_roc` (30-day trend)
  - `deriv7d_roc` (7-day trend)
  - `deriv3d_roc` (3-day trend)
- **Hypothesis:** Different timeframe trends capture different market regimes
- **Log:** `logs/exp1_derivatives.log`

### Experiment 2: Multi-Timeframe Volatility
- **Starting Features:**
  - `volatility_72` (72-period volatility)
  - `volatility_24` (24-period volatility)
  - `volatility_12` (12-period volatility)
- **Hypothesis:** Volatility regime changes predict price direction
- **Log:** `logs/exp2_volatility.log`

### Experiment 3: Advanced Features (Top from Baseline)
- **Starting Features:**
  - `deriv7d_prime7d` (7-day acceleration)
  - `deriv4d_roc` (4-day trend)
  - `volatility_24` (24-period volatility)
- **Hypothesis:** Starting with proven top features will converge faster to optimal
- **Log:** `logs/exp3_advanced.log`

---

## Comparison to Previous Results

### Baseline Results (from first run):
- **Starting:** `deriv30d_roc`, `volatility_24`, `avg14d_spread`
- **Final Accuracy:** 54.83%
- **Features Used:** 6 features
- **Features Added:** `avg15m`, `avg10m`, `avg1h`

### Target:
- **Baseline Model:** 58.36% (9 features)
- **Gap to close:** 3.53%

---

## What We're Testing

Each experiment starts with 3 strategically chosen features and incrementally adds the best-performing features until:
1. No improvement for 5 consecutive iterations, OR
2. Maximum of 20 features reached

The system will automatically:
- Rank remaining features by correlation with target
- Test top 3 candidates
- Add the best one if it improves accuracy by ≥0.1%
- Repeat until convergence

---

## Expected Insights

We'll discover:
1. **Which starting point converges faster**
2. **Which features naturally cluster together**
3. **What's the minimum feature set for 55%+ accuracy**
4. **Can we reach 58%+ with fewer than 9 features**

---

## Next Steps

After experiments complete:
1. Compare final accuracy across all experiments
2. Analyze which features were discovered by each approach
3. Identify common features across successful experiments
4. Recommend optimal starting strategy for production

---

**Note:** Each experiment is capped at 10 minutes. If it doesn't converge by then, it will be stopped and results will be analyzed from the last completed iteration.



