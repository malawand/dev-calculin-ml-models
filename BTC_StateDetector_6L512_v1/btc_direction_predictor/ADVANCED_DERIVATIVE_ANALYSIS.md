# Advanced Derivative Analysis

## Overview

This system goes beyond basic derivative values to extract sophisticated momentum patterns and relationships that are highly predictive of future price direction.

## Feature Categories

### 1. Velocity-Acceleration Alignment

**What it measures**: How velocity (first derivative) and acceleration (second derivative) align with each other.

**Key Insights**:
- **Strong Uptrend** (align = +1): Both velocity and acceleration are positive → Momentum is accelerating upward
- **Weakening Uptrend** (align = -1, vel+, accel-): Price still rising but decelerating → Potential top forming
- **Potential Reversal** (align = -1, vel-, accel+): Price falling but deceleration slowing → Potential bottom
- **Strong Downtrend** (align = +1): Both negative → Momentum accelerating downward

**Features**:
- `align_{timeframe}_{window}`: Alignment score (-1, 0, +1)
- `momentum_state_{timeframe}_{window}`: Categorical state (1=strong up, 2=weakening, 3=potential reversal, 4=strong down)
- `align_strength_{timeframe}_{window}`: Magnitude-weighted alignment

### 2. Momentum Regimes

**What it measures**: Current state of momentum (building, peaking, fading, reversing).

**Features**:
- `momentum_accel_{timeframe}`: Is momentum itself accelerating?
- `momentum_trend_{timeframe}`: Short-term direction of momentum
- `momentum_vol_{timeframe}`: How consistent/volatile is momentum?
- `momentum_consensus`: Do all timeframes agree on direction?
- `momentum_agreement`: How much do timeframes disagree? (low = high agreement)

**Why it matters**: Momentum regimes help identify:
- When trends are gaining strength (early entry opportunities)
- When momentum is peaked (take profit zones)
- When momentum is reversing (exit or reversal trades)

### 3. Cross-Timeframe Divergences

**What it measures**: When different timeframes show conflicting momentum.

**Key Insights**:
- **Bullish Divergence**: Short-term momentum negative, long-term positive → Pullback in uptrend
- **Bearish Divergence**: Short-term momentum positive, long-term negative → Rally in downtrend
- **Convergence**: All timeframes aligned → Strong directional conviction

**Features**:
- `short_term_momentum`: Average of 5m-1h derivatives
- `medium_term_momentum`: Average of 2h-24h derivatives  
- `long_term_momentum`: Average of 48h-30d derivatives
- `short_long_divergence`: Difference between short and long-term (flags major divergences)
- `short_long_div_magnitude`: How strong is the divergence?
- `momentum_convergence`: Overall alignment across all timeframes

**Why it matters**: Divergences often signal trend changes or exhaustion points.

### 4. Momentum Persistence

**What it measures**: How long momentum has stayed in the current direction.

**Features**:
- `persistence_{timeframe}`: Consecutive periods in same direction
- `weighted_persistence_{timeframe}`: Duration × magnitude

**Why it matters**:
- Long persistence in one direction → Strong trend OR potential exhaustion
- Can help distinguish between sustained trends vs. short-term noise

### 5. Derivative Curvature (Jerk)

**What it measures**: Third derivative - rate of change of acceleration.

**Features**:
- `{prime}_jerk`: How fast is acceleration changing?
- `{prime}_jerk_sign`: Direction of jerk

**Why it matters**: Helps detect inflection points where momentum shifts from accelerating to decelerating (or vice versa).

### 6. Momentum Strength Indicators

**What it measures**: Not just direction, but HOW STRONG the momentum is.

**Features**:
- `total_momentum_magnitude`: Sum of absolute momentum across all timeframes
- `net_momentum`: Positive momentum minus negative momentum
- `momentum_ratio`: Ratio of positive to negative momentum
- `momentum_concentration`: Is momentum focused or scattered?

**Why it matters**: Strong, focused momentum is more likely to continue. Weak, scattered momentum suggests chop/indecision.

### 7. Phase Cycle Features

**What it measures**: Where we are in the momentum cycle (building → peak → fading → trough).

**Features**:
- `cycle_position_{timeframe}`: Normalized position in cycle (z-score)
- `at_extreme_{timeframe}`: Binary flag for extreme positions (potential reversal zones)

**Why it matters**: Momentum at extremes often reverses. Identifying cycle position helps time entries/exits.

### 8. Multi-Scale Momentum Coherence

**What it measures**: How aligned momentum is across different time scales.

**Features**:
- `{scale}_coherence`: How many derivatives in this scale agree? (0-1)
- `{scale}_magnitude_std`: Are magnitudes similar or one dominant?

**Scales**:
- **Micro**: 15m, 30m, 1h (intraday chop)
- **Short**: 2h, 4h, 8h (day trading timeframes)
- **Medium**: 12h, 24h, 48h (swing trading)
- **Long**: 3d, 7d, 14d (position/trend trading)

**Why it matters**:
- High coherence = Strong directional conviction → High probability trades
- Low coherence = Market indecision/chop → Avoid or reduce size

## Interpretation Guide

### Strong Buy Signal
- Velocity-Acceleration: Strong Uptrend (align = +1)
- Cross-Timeframe: Convergence with positive momentum
- Persistence: Building momentum (moderate persistence)
- Coherence: High across all scales
- Cycle Position: Not at extreme high

### Strong Sell Signal
- Velocity-Acceleration: Strong Downtrend (align = +1, both negative)
- Cross-Timeframe: Convergence with negative momentum
- Persistence: Building downward momentum
- Coherence: High across all scales
- Cycle Position: Not at extreme low

### Potential Reversal (Bottom)
- Velocity-Acceleration: Weakening downtrend (vel-, accel+)
- Cross-Timeframe: Bullish divergence (short negative, long positive)
- Coherence: Low (market indecision)
- Cycle Position: At extreme low

### Potential Reversal (Top)
- Velocity-Acceleration: Weakening uptrend (vel+, accel-)
- Cross-Timeframe: Bearish divergence (short positive, long negative)  
- Coherence: Low
- Cycle Position: At extreme high

## Expected Impact

These advanced features should significantly improve model accuracy by:

1. **Better Trend Quality Assessment**: Not just detecting trends, but assessing their strength and sustainability
2. **Earlier Reversal Detection**: Velocity-acceleration divergences and curvature features catch inflection points early
3. **Reduced False Signals**: Coherence metrics help filter out noise and identify high-conviction setups
4. **Multi-Scale Context**: Understanding how different timeframes interact provides richer context

## Feature Importance Expectations

Based on trading principles, we expect these features to rank highly:
- Velocity-acceleration alignment (captures trend quality)
- Cross-timeframe divergences (catches reversals)
- Momentum coherence (filters noise)
- Momentum strength indicators (conviction metrics)



