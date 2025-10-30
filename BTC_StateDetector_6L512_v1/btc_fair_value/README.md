# Bitcoin Fair Value Analysis System

## 🎯 What It Does

Calculates **empirical fair value**, **max**, and **min** for Bitcoin based on:
1. **Derivatives** (velocity and acceleration)
2. **Volume-Weighted Average** (VWAP)
3. **Statistical Bounds** (Bollinger Bands)
4. **Z-Score Analysis** (standard deviations from mean)

**Output**: Tells you if Bitcoin is overvalued, undervalued, or fairly priced **RIGHT NOW**.

## 📊 Example Output

```
Current Price:    $110,991.82
Fair Value:       $110,640.42
Empirical Max:    $111,211.23
Empirical Min:    $110,068.79

Deviation:        +0.32%
Position in Range: 80.8%

Status: 🔴 NEAR UPPER BOUND
Expected Move: DOWN (mean reversion)
```

## 🚀 Quick Start

### Run the Test

```bash
cd btc_fair_value
source ../btc_direction_predictor/venv/bin/activate
python test_fair_value.py
```

### Use in Your Code

```python
from fair_value_calculator import FairValueCalculator

# Load your data (must have 'price' and 'volume' columns)
df = load_your_data()  # Need 4+ hours of 1-minute data

# Calculate fair value
calculator = FairValueCalculator()
result = calculator.calculate_comprehensive_fair_value(df)

# Get values
current = result['current_price']
fair = result['fair_value']
max_price = result['empirical_max']
min_price = result['empirical_min']

print(f"Current: ${current:,.2f}")
print(f"Fair: ${fair:,.2f}")
print(f"Max: ${max_price:,.2f}")
print(f"Min: ${min_price:,.2f}")
print(f"Assessment: {result['assessment']}")
```

## 📊 Four Methods Combined

### 1. VWAP (Volume Weighted Average Price)
**What**: Average price weighted by volume  
**Why**: High-volume prices are more "real" than low-volume  
**Output**: Fair value based on trading activity

### 2. Statistical Bounds (Bollinger Bands)
**What**: Mean ± 2 standard deviations  
**Why**: 95% of prices fall within this range historically  
**Output**: Fair value (mean), Max (+2σ), Min (-2σ)

### 3. Derivative-Based Fair Value
**What**: Uses velocity (first derivative) to project  
**Why**: Current momentum indicates short-term fair value  
**Output**: Where price "should" be given current velocity

### 4. Z-Score Analysis
**What**: Standard deviations from mean  
**Why**: Shows extremeness of current price  
**Output**: Interpretation (OVERVALUED, FAIR, UNDERVALUED)

## 🎯 Interpretation Guide

### Assessment Types

**🔴 OVERVALUED**
- Current > Empirical Max
- Z-Score > +1.5
- **Action**: Consider selling or shorting
- **Expected**: Mean reversion down

**🟠 SLIGHTLY_OVERVALUED**
- Deviation > +1%
- Z-Score +0.5 to +1.5
- **Action**: Take profits, reduce longs
- **Expected**: Slight correction

**🟢 FAIR**
- Deviation ±1%
- Z-Score -0.5 to +0.5
- **Action**: Wait for better entry
- **Expected**: Ranging

**🔵 SLIGHTLY_UNDERVALUED**
- Deviation < -1%
- Z-Score -1.5 to -0.5
- **Action**: Scale into longs
- **Expected**: Slight bounce

**🟣 UNDERVALUED**
- Current < Empirical Min
- Z-Score < -1.5
- **Action**: Consider buying
- **Expected**: Mean reversion up

### Position in Range

**80-100% (Upper Bound)**
- High risk of reversal
- Take profits on longs
- Consider shorts
- Target: Fair value

**60-80% (Upper Middle)**
- Moderately extended
- Tighten stops on longs
- Wait for pullback
- Partial profit taking

**40-60% (Middle)**
- No clear edge
- Wait for extremes
- Watch for breakout
- Neutral stance

**20-40% (Lower Middle)**
- Moderately oversold
- Scale into longs
- Set tight stops
- Partial entry

**0-20% (Lower Bound)**
- High probability bounce
- Strong buy opportunity
- Full position
- Target: Fair value

## 🤖 Bot Integration Examples

### Example 1: Mean Reversion Trading

```python
result = calculator.calculate_comprehensive_fair_value(df)

if result['assessment'] == 'OVERVALUED':
    # Price above max - sell
    action = 'SELL'
    target = result['fair_value']
    stop_loss = result['empirical_max'] * 1.01
    
elif result['assessment'] == 'UNDERVALUED':
    # Price below min - buy
    action = 'BUY'
    target = result['fair_value']
    stop_loss = result['empirical_min'] * 0.99
    
else:
    # Fair - wait
    action = 'WAIT'
```

### Example 2: Position Sizing by Deviation

```python
result = calculator.calculate_comprehensive_fair_value(df)

deviation = abs(result['deviation_pct'])
base_size = 2.0  # 2% of capital

# More deviation = larger opportunity
if deviation > 2.0:
    position_size = base_size * 2.0  # 4%
elif deviation > 1.0:
    position_size = base_size * 1.5  # 3%
elif deviation > 0.5:
    position_size = base_size * 1.0  # 2%
else:
    position_size = 0  # Wait

if result['assessment'] in ['OVERVALUED', 'SLIGHTLY_OVERVALUED']:
    action = 'SELL'
elif result['assessment'] in ['UNDERVALUED', 'SLIGHTLY_UNDERVALUED']:
    action = 'BUY'
```

### Example 3: Combine with State Detection

```python
# Get fair value
fair_result = calculator.calculate_comprehensive_fair_value(df)

# Get current state (from state detection model)
state_result = detect_state(df)

# Combined logic
if (fair_result['assessment'] == 'UNDERVALUED' and 
    state_result['direction'] == 'UP' and 
    state_result['confidence'] > 0.9):
    # Undervalued + upward momentum + high confidence
    action = 'STRONG_BUY'
    
elif (fair_result['assessment'] == 'OVERVALUED' and 
      state_result['direction'] == 'DOWN' and 
      state_result['confidence'] > 0.9):
    # Overvalued + downward momentum + high confidence
    action = 'STRONG_SELL'
    
elif fair_result['assessment'] == 'FAIR':
    # Follow momentum only
    if state_result['direction'] == 'UP':
        action = 'BUY'
    elif state_result['direction'] == 'DOWN':
        action = 'SELL'
```

## 📊 Real-Time Monitoring

```python
import time

def monitor_fair_value(interval=300):
    """Monitor fair value every 5 minutes"""
    calculator = FairValueCalculator()
    
    while True:
        # Fetch data
        df = fetch_latest_data(hours=48)
        
        # Calculate
        result = calculator.calculate_comprehensive_fair_value(df)
        
        # Log
        print(f"[{datetime.now()}]")
        print(f"  Current: ${result['current_price']:,.2f}")
        print(f"  Fair: ${result['fair_value']:,.2f}")
        print(f"  Deviation: {result['deviation_pct']:+.2f}%")
        print(f"  Assessment: {result['assessment']}")
        print()
        
        # Alert on extremes
        if result['position_in_range_pct'] > 90:
            print("⚠️  ALERT: Near upper bound - reversal likely!")
        elif result['position_in_range_pct'] < 10:
            print("⚠️  ALERT: Near lower bound - bounce likely!")
        
        time.sleep(interval)

monitor_fair_value()
```

## 🎯 Key Features

### Derivatives Integration ✅
- Uses velocity (first derivative) for momentum
- Incorporates existing derivative columns if available
- Calculates simple derivatives if none exist
- Projects fair value based on current trajectory

### Empirical Max/Min ✅
- Based on 4-hour Bollinger Bands (2 std dev)
- Adaptive to current volatility
- 95% confidence interval
- Real-time updates

### Volume Confirmation ✅
- VWAP weights by trading volume
- High volume = more reliable prices
- Filters out low-liquidity noise
- Better fair value estimate

### Multiple Methods ✅
- 4 independent calculations
- Consensus via median (reduces outliers)
- Robust to individual method failures
- High reliability

## ⚙️ Configuration

### Adjust Time Windows

```python
# Shorter window (1 hour)
vwap_1h = calculator.calculate_vwap(df, period=60)

# Longer window (8 hours)
vwap_8h = calculator.calculate_vwap(df, period=480)

# Statistical bounds (24 hours)
stats_24h = calculator.calculate_statistical_bounds(df, period=1440)
```

### Adjust Standard Deviations

```python
# Tighter bounds (1.5 std dev)
stats_tight = calculator.calculate_statistical_bounds(df, std_dev=1.5)

# Wider bounds (3 std dev)
stats_wide = calculator.calculate_statistical_bounds(df, std_dev=3.0)
```

### Adjust Thresholds

```python
# More sensitive (0.5% deviation = overvalued)
if deviation > 0.5:
    assessment = 'OVERVALUED'

# Less sensitive (2.0% deviation = overvalued)
if deviation > 2.0:
    assessment = 'OVERVALUED'
```

## 📁 Files

- `fair_value_calculator.py` - Main calculator class
- `test_fair_value.py` - Test on current market
- `README.md` - This file

## 🎓 Understanding the Output

### Current Price
The latest price from your data feed

### Fair Value
Consensus of 4 methods:
- VWAP
- Statistical mean
- Derivative projection
- Z-score mean

### Empirical Max
Upper bound (mean + 2σ)  
95% chance price stays below this

### Empirical Min
Lower bound (mean - 2σ)  
95% chance price stays above this

### Deviation %
How far from fair value:
- Positive = overvalued
- Negative = undervalued
- ±1% = fair range

### Position in Range %
Where in the max-min range:
- 0% = at min (very undervalued)
- 50% = at fair (neutral)
- 100% = at max (very overvalued)

## 💡 Best Practices

### 1. Use with State Detection
Don't use fair value alone - combine with momentum/trend detection

### 2. Wait for Extremes
Trade at 80%+ or 20%- positions, not in middle

### 3. Check Volume
Low volume fair values are less reliable

### 4. Multiple Timeframes
Check 1h, 4h, and 24h fair values for confirmation

### 5. Set Stop Losses
Fair value can change - protect with stops

## ⚠️ Limitations

**Not Predictive**
- Shows current fair value
- Doesn't guarantee reversion timing
- Market can stay irrational

**Requires Data**
- Needs 4+ hours of data
- Volume data improves accuracy
- Derivative columns help

**Mean Reversion Assumption**
- Assumes prices revert to mean
- Strong trends can persist
- Use with trend detection

## 🎯 Recommended Usage

**For Mean Reversion Trading**:
1. Wait for position >80% or <20%
2. Check assessment (OVERVALUED/UNDERVALUED)
3. Confirm with state detection
4. Enter toward fair value
5. Exit at fair value or opposite extreme

**For Trend Following**:
1. Use fair value as support/resistance
2. Buy at fair value in uptrends
3. Sell at fair value in downtrends
4. Don't fight trends at extremes

**For Risk Management**:
1. Reduce positions at extremes
2. Increase at fair value
3. Tighten stops away from fair
4. Loosen stops near fair

---

*Combines derivatives, volume, statistics, and momentum*  
*For empirical fair value, max, and min calculation*  
*Use for mean reversion and value assessment*


