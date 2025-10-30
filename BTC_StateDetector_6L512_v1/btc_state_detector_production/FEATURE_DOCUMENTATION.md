# Feature Calculation Documentation

## Overview
This document explains how each of the 22 features is calculated from the 5-hour price/volume window.

---

## Input Data

For each prediction, we fetch the last **6 hours** of data (360 minutes) and use the most recent **5 hours** (300 minutes) for feature extraction.

**Required Columns:**
- `price`: Bitcoin price in USDT (float)
- `volume`: Trading volume (float)

**Data Format:**
```python
df = pd.DataFrame({
    'price': [108500.0, 108510.5, 108505.2, ...],  # 300 values
    'volume': [1234.5, 1456.8, 1123.4, ...]        # 300 values
})
```

---

## Feature Categories

### 1. Price Derivatives (12 features)

Derivatives measure the **rate of change** of price over different timeframes.

#### deriv_5m
**What:** Price change over last 5 minutes  
**Formula:**
```python
deriv_5m = (price[-1] - price[-5]) / price[-5] * 100
```
**Example:** If price 5 min ago was $108,000 and now is $108,500:
```
deriv_5m = (108500 - 108000) / 108000 * 100 = 0.46%
```
**Range:** Typically -2% to +2%  
**Meaning:** Positive = upward momentum, Negative = downward momentum

#### deriv_15m
**What:** Price change over last 15 minutes  
**Formula:**
```python
deriv_15m = (price[-1] - price[-15]) / price[-15] * 100
```
**Range:** Typically -5% to +5%  
**Meaning:** Captures short-term trend

#### deriv_30m
**What:** Price change over last 30 minutes  
**Formula:**
```python
deriv_30m = (price[-1] - price[-30]) / price[-30] * 100
```
**Range:** Typically -8% to +8%  
**Meaning:** Medium-term trend

#### deriv_1h
**What:** Price change over last 1 hour  
**Formula:**
```python
deriv_1h = (price[-1] - price[-60]) / price[-60] * 100
```
**Range:** Typically -10% to +10%  
**Meaning:** Hourly trend strength

#### deriv_4h
**What:** Price change over last 4 hours  
**Formula:**
```python
deriv_4h = (price[-1] - price[-240]) / price[-240] * 100
```
**Range:** Typically -15% to +15%  
**Meaning:** Longer-term trend direction

#### deriv_norm_5m, deriv_norm_15m, deriv_norm_30m, deriv_norm_1h, deriv_norm_4h
**What:** Normalized derivatives (relative to recent volatility)  
**Formula:**
```python
# Calculate recent volatility
returns = np.diff(prices[-60:]) / prices[-61:-1]  # Last hour
volatility = np.std(returns)

# Normalize derivative
deriv_norm_5m = deriv_5m / (volatility * 100) if volatility > 0 else 0
```
**Range:** Typically -10 to +10  
**Meaning:** Derivative strength relative to recent volatility  
**Why:** A 1% move means more in low volatility than high volatility

#### acceleration
**What:** Rate of change of the rate of change (2nd derivative)  
**Formula:**
```python
# First derivatives
deriv_now = (price[-1] - price[-5]) / price[-5]
deriv_prev = (price[-5] - price[-10]) / price[-10]

# Acceleration (change in derivative)
acceleration = (deriv_now - deriv_prev) / 5 * 100  # per minute
```
**Range:** Typically -0.5 to +0.5  
**Meaning:** Positive = accelerating upward, Negative = decelerating/reversing  
**Why:** Captures momentum changes before price changes significantly

#### roc_15m
**What:** Rate of Change over 15 minutes  
**Formula:**
```python
roc_15m = (price[-1] / price[-15] - 1) * 100
```
**Range:** Typically -5% to +5%  
**Meaning:** Similar to deriv_15m, alternative calculation method

---

### 2. Volatility Features (4 features)

Volatility measures how **choppy** or **stable** the price movement is.

#### volatility_5m
**What:** Standard deviation of returns over last 5 minutes  
**Formula:**
```python
prices_5m = prices[-5:]
returns_5m = np.diff(prices_5m) / prices_5m[:-1]
volatility_5m = np.std(returns_5m) * 100
```
**Example:** 
```
Prices: [108000, 108100, 108050, 108150, 108200]
Returns: [0.093%, -0.046%, 0.095%, 0.033%]
Volatility: std([0.093, -0.046, 0.095, 0.033]) = 0.065%
```
**Range:** 0% to 2%  
**Meaning:** Low = smooth move, High = choppy/jumpy

#### volatility_15m
**What:** Standard deviation of returns over last 15 minutes  
**Formula:**
```python
prices_15m = prices[-15:]
returns_15m = np.diff(prices_15m) / prices_15m[:-1]
volatility_15m = np.std(returns_15m) * 100
```
**Range:** 0% to 3%

#### volatility_30m
**What:** Standard deviation of returns over last 30 minutes  
**Formula:**
```python
prices_30m = prices[-30:]
returns_30m = np.diff(prices_30m) / prices_30m[:-1]
volatility_30m = np.std(returns_30m) * 100
```
**Range:** 0% to 4%

#### volatility_1h
**What:** Standard deviation of returns over last 1 hour  
**Formula:**
```python
prices_1h = prices[-60:]
returns_1h = np.diff(prices_1h) / prices_1h[:-1]
volatility_1h = np.std(returns_1h) * 100
```
**Range:** 0% to 5%  
**Meaning:** Captures overall choppiness of the hour

---

### 3. Deviation Features (2 features)

Deviation measures how far current price is from recent **average price**.

#### dev_from_avg_1h
**What:** Deviation from 1-hour average price  
**Formula:**
```python
avg_1h = np.mean(prices[-60:])
dev_from_avg_1h = (price[-1] - avg_1h) / avg_1h * 100
```
**Example:** If average price last hour was $108,000 and current is $108,500:
```
dev_from_avg_1h = (108500 - 108000) / 108000 * 100 = 0.46%
```
**Range:** Typically -2% to +2%  
**Meaning:** Positive = above average (potential overbought), Negative = below average (potential oversold)

#### dev_from_avg_4h
**What:** Deviation from 4-hour average price  
**Formula:**
```python
avg_4h = np.mean(prices[-240:])
dev_from_avg_4h = (price[-1] - avg_4h) / avg_4h * 100
```
**Range:** Typically -5% to +5%  
**Meaning:** Longer-term deviation, shows if in upper or lower part of 4h range

---

### 4. Volume Features (4 features)

Volume features measure **trading activity strength and trend**.

#### volume_change_5m
**What:** Volume trend over last 5 minutes  
**Formula:**
```python
if np.sum(volumes[-5:]) > 0:
    # Linear regression slope of volume over last 5 minutes
    x = np.arange(5)
    y = volumes[-5:]
    slope = np.polyfit(x, y, 1)[0]
    volume_change_5m = slope / np.mean(y) * 100 if np.mean(y) > 0 else 0
else:
    volume_change_5m = 0
```
**Range:** Typically -50% to +50%  
**Meaning:** Positive = volume increasing, Negative = volume decreasing  
**Why:** Increasing volume confirms price moves

#### volume_change_15m
**What:** Volume trend over last 15 minutes  
**Formula:**
```python
# Same as above but with volumes[-15:]
```
**Range:** Typically -40% to +40%

#### volume_change_30m
**What:** Volume trend over last 30 minutes  
**Formula:**
```python
# Same as above but with volumes[-30:]
```
**Range:** Typically -30% to +30%

#### volume_strength
**What:** Overall volume strength relative to 5-hour average  
**Formula:**
```python
avg_volume_5h = np.mean(volumes)
recent_volume = np.mean(volumes[-15:])  # Last 15 minutes

if avg_volume_5h > 0:
    volume_strength = (recent_volume - avg_volume_5h) / avg_volume_5h * 100
else:
    volume_strength = 0
```
**Example:** If 5h average volume is 1000 and last 15min average is 1500:
```
volume_strength = (1500 - 1000) / 1000 * 100 = 50%
```
**Range:** Typically -80% to +200%  
**Meaning:** Positive = higher than usual volume, Negative = lower than usual

---

## Feature Extraction Code

The features are extracted by `feature_extractor.py`:

```python
from feature_extractor import extract_features
import pandas as pd

# Prepare data (5 hours of 1-minute candles)
df = pd.DataFrame({
    'price': price_array,    # 300 values
    'volume': volume_array   # 300 values
})

# Extract all 22 features
features = extract_features(df)

# Result is a dictionary:
# {
#     'deriv_5m': 0.46,
#     'deriv_15m': 0.82,
#     'deriv_30m': 1.15,
#     ...
#     'volume_strength': 25.3
# }
```

---

## Feature Importance

Based on model analysis, the most important features are:

1. **deriv_1h** - Strongest predictor of direction
2. **volatility_1h** - Best indicator of sideways vs trending
3. **deriv_4h** - Captures longer-term trend
4. **acceleration** - Early warning of reversals
5. **volume_strength** - Confirms trend validity

Less important but still useful:
- Normalized derivatives (reduce noise)
- Short-term volatility (detects choppy markets)
- Volume trends (confirm momentum)

---

## Missing Data Handling

### If Volume is Zero or Missing:
```python
# All volume features set to 0
volume_change_5m = 0
volume_change_15m = 0
volume_change_30m = 0
volume_strength = 0
```

### If Price Data is Insufficient:
```python
# NaN values replaced with 0
features = {k: 0.0 if pd.isna(v) or np.isinf(v) else v 
            for k, v in features.items()}
```

---

## Feature Scaling

Before feeding to the model, features are **standardized**:

```python
from sklearn.preprocessing import StandardScaler

# During training, scaler is fitted
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# During prediction, same scaler is applied
X_scaled = scaler.transform(X_new)
```

**Formula:** `X_scaled = (X - mean) / std`

**Why:** Neural networks perform better when features are on similar scales.

---

## Feature Vector Example

Real example from October 23, 2025, 9:15 AM:

```python
{
    'deriv_5m': 0.12,
    'deriv_15m': 0.34,
    'deriv_30m': 0.52,
    'deriv_1h': -0.15,
    'deriv_4h': 0.28,
    'deriv_norm_5m': 1.5,
    'deriv_norm_15m': 4.2,
    'deriv_norm_30m': 6.5,
    'deriv_norm_1h': -1.9,
    'deriv_norm_4h': 3.5,
    'acceleration': -0.05,
    'roc_15m': 0.33,
    'volatility_5m': 0.08,
    'volatility_15m': 0.12,
    'volatility_30m': 0.15,
    'volatility_1h': 0.18,
    'dev_from_avg_1h': 0.25,
    'dev_from_avg_4h': 0.42,
    'volume_change_5m': 15.2,
    'volume_change_15m': 12.8,
    'volume_change_30m': 8.5,
    'volume_strength': 25.3
}
```

**Model Output:**
- Direction: NONE (99.9% confidence)
- Strength: 4.2/100

**Interpretation:** Low derivatives + low volatility → Sideways market

---

## Validation

To verify features are calculated correctly:

```python
# Check ranges
assert -20 < features['deriv_5m'] < 20
assert 0 <= features['volatility_1h'] < 10
assert -100 < features['dev_from_avg_1h'] < 100

# Check no NaN/inf
assert all(not pd.isna(v) and not np.isinf(v) for v in features.values())

# Check feature count
assert len(features) == 22
```

---

## Performance Impact

**Feature Extraction Time:** ~1 millisecond  
**Bottleneck:** Data fetching from network (~500-2000ms)  
**Optimization:** Feature calculation is fast enough; focus on data caching if needed

---

*Last Updated: October 23, 2025*  
*Model Version: Ultra-Deep v1.0*  
*Feature Set Version: 22 features v1.0*

