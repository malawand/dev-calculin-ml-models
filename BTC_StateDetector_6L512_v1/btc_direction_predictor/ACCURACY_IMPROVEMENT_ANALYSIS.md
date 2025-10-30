# Accuracy Improvement Analysis

## Current Status

### Performance Progression (15-minute horizon)

| Version | Features | Accuracy | ROC-AUC | MCC | F1 (UP) | Key Changes |
|---------|----------|----------|---------|-----|---------|-------------|
| **Baseline** | 220 | 52.45% | 0.5290 | 0.0490 | 0.5211 | Basic derivatives, averages, ROC |
| **+ Advanced** | 326 | 52.39% | 0.5325 | 0.0460 | 0.5199 | Momentum regimes, divergences, jerk |
| **+ Ultra-Advanced** | 424 | **52.14%** | **0.5302** | **0.0422** | **0.5059** | Feature interactions, microstructure, stat arb |

### Key Observations

1. **Diminishing Returns**: Adding 200+ sophisticated features didn't improve accuracy - it got slightly worse!
2. **Feature Importance Shift**: The new advanced features ARE being used (5 of top 10 are advanced), but they're not improving predictions
3. **Close to Random**: 52% accuracy is barely better than coin flipping (50%)
4. **The 15m Problem**: 15-minute predictions may be too noisy - price movement is dominated by randomness at this timeframe

### Top 10 Most Important Features (Ultra-Advanced Model)

1. **deriv4d_roc** (68) - Rate of change of 4-day derivative
2. **deriv24h_prime24h** (47) - 24h acceleration
3. **deriv3d_roc** (37) - Rate of change of 3-day derivative
4. **volatility_24** (36) - 24-period volatility
5. **medium_magnitude_std** (34) ⭐ *Advanced feature*
6. **weighted_persistence_24h** (32) ⭐ *Advanced feature*
7. **deriv30d_roc** (31) - Rate of change of 30-day derivative
8. **deriv7d_prime7d_jerk** (30) ⭐ *Ultra-advanced feature (3rd derivative)*
9. **momentum_state_7d_7d_vol_adj** (30) ⭐ *Ultra-advanced feature*
10. **deriv24h_prime4h_jerk** (29) ⭐ *Ultra-advanced feature*

## Why Is Accuracy Still Low?

### Fundamental Challenges

1. **Market Efficiency**: Bitcoin markets are reasonably efficient - if there were obvious patterns, they'd be arbitraged away
2. **Time Horizon Too Short**: 15-minute movements are dominated by noise, not signal
3. **Feature Redundancy**: 424 features likely have massive multicollinearity
4. **Missing Critical Information**: We don't have:
   - Volume/liquidity data
   - Order book depth
   - Funding rates
   - Open interest
   - Cross-exchange arbitrage opportunities
   - News/sentiment data
   - Macro indicators

5. **Wrong Target**: Predicting UP/DOWN ignores:
   - Magnitude of moves (a +0.01% move is very different from +5%)
   - Confidence levels (some periods are more predictable than others)
   - Transaction costs (need bigger moves to be profitable)

## Recommended Next Steps

### 1. Test Longer Horizons ✅ (HIGHEST PRIORITY)

15-minute predictions are likely too noisy. Test:
- **1 hour**: More signal, less noise
- **4 hours**: Clear trend direction
- **24 hours**: Daily trend prediction

**Hypothesis**: Accuracy should improve significantly at longer horizons.

### 2. Feature Selection (Aggressive Pruning)

We have 424 features but many are redundant. Reduce to 50-100 best:

```python
# Approaches:
1. Keep only top 50 by feature importance
2. Remove highly correlated features (|r| > 0.9)
3. Use SHAP for feature interaction analysis
4. Forward/backward stepwise selection
5. L1 regularization (LASSO) to auto-select
```

### 3. Filter Low-Volatility Periods

Train and predict only during high-volatility periods:

```python
# Only use data where volatility_24 > median
# Low volatility = random walk = unpredictable
```

**Hypothesis**: Accuracy could jump to 55-60% by filtering out noise.

### 4. Change the Target

Instead of binary UP/DOWN:

**Option A: Magnitude-Aware Classification**
```python
# 3 classes:
# 0 = DOWN (< -0.5%)
# 1 = FLAT (-0.5% to +0.5%)  
# 2 = UP (> +0.5%)
# Don't trade the FLAT zone
```

**Option B: Confidence-Weighted Prediction**
```python
# Only trade when model probability > 0.6
# This filters out uncertain predictions
```

**Option C: Regression + Sign**
```python
# Predict magnitude (regression)
# Then classify sign
# Only trade if predicted magnitude > threshold
```

### 5. Ensemble Methods

Instead of single LightGBM:

```python
# Combine multiple models:
1. LightGBM (current)
2. XGBoost
3. CatBoost
4. Random Forest
5. LSTM (temporal patterns)

# Vote or average predictions
```

### 6. Temporal Sequence Modeling (LSTM/Transformer)

Current approach treats each sample independently. Add temporal dependencies:

```python
# Input: Last 24 periods (6 hours @ 15m intervals)
# Output: Next period direction
# Model: LSTM or Transformer to capture sequences
```

### 7. Market Regime Adaptation

Different features work in different regimes:

```python
# Train 3 separate models:
1. Trending market model (follow momentum)
2. Ranging market model (mean reversion)  
3. High volatility model (reduce exposure)

# Use appropriate model based on current regime
```

### 8. Add External Data

If available:
- **Volume**: Essential for confirming momentum
- **Funding rates**: Indicates over-leverage
- **Bitcoin dominance**: Altcoin rotation signals
- **Macro events**: Fed meetings, economic data
- **On-chain metrics**: Exchange flows, whale movements

## Immediate Action Plan

### Phase 1: Test Longer Horizons (30 minutes)
```bash
# Test 1h, 4h, 24h horizons
python -m src.pipeline.train --horizon 1h
python -m src.pipeline.train --horizon 4h
python -m src.pipeline.train --horizon 24h
```

**Expected**: Accuracy should improve to 54-58% at 4h/24h horizons.

### Phase 2: Aggressive Feature Selection (1 hour)
```python
# Keep only top 50 features
# Remove correlated features
# Retrain
```

**Expected**: Accuracy should improve or stay same with 50 features vs 424.

### Phase 3: High-Volatility Filter (30 minutes)
```python
# Train only on top 50% volatility periods
# Test only on high volatility
```

**Expected**: Accuracy could jump to 55-60%.

### Phase 4: Ensemble Models (1 hour)
```python
# Add XGBoost, CatBoost
# Voting classifier
```

**Expected**: 1-2% accuracy improvement.

### Phase 5: LSTM Sequences (2 hours)
```python
# Build temporal model
# Input: 24-step sequences
# Compare vs tree models
```

**Expected**: May capture temporal patterns trees miss.

## Realistic Performance Targets

Based on published research on cryptocurrency prediction:

| Timeframe | Achievable Accuracy | Notes |
|-----------|---------------------|-------|
| **15 minutes** | 52-54% | Too noisy, low signal |
| **1 hour** | 54-57% | Moderate signal |
| **4 hours** | 56-60% | Good signal-to-noise |
| **24 hours** | 58-65% | Best predictability |

## Why 70%+ Is Unrealistic

1. **Market efficiency**: Easy patterns are arbitraged away
2. **Black swan events**: Unpredictable shocks (regulations, hacks, etc.)
3. **Regime changes**: Models trained on trending markets fail in ranging markets
4. **Liquidity cascades**: Large orders cause unpredictable price impact
5. **MEV/HFT**: High-frequency traders capture micro-inefficiencies

**Professional quant funds target**:
- **55-58% accuracy** with sophisticated infrastructure
- **Small edge** is amplified through:
  - High leverage
  - Thousands of trades
  - Microsecond execution
  - Market making
  - Statistical arbitrage

## Conclusion

We've built a sophisticated feature engineering system with 424 features including:
- ✅ Advanced derivatives
- ✅ Momentum regimes  
- ✅ Feature interactions
- ✅ Market microstructure
- ✅ Statistical arbitrage signals
- ✅ Regime-conditional features
- ✅ Temporal patterns

But 52% accuracy at 15-minute horizon suggests **we're hitting fundamental limits** of what's predictable at this timeframe.

**Next Steps**: Focus on:
1. ⏰ **Longer time horizons** (biggest expected gain)
2. 🎯 **Volatility filtering** (trade only predictable periods)
3. 🔍 **Feature selection** (reduce noise)
4. 🤖 **Ensemble methods** (combine multiple models)
5. 📊 **Better targets** (magnitude-aware or confidence-filtered)

These changes could realistically push us to **55-60% accuracy on 4h/24h horizons**, which is **actionable for a trading bot**.



