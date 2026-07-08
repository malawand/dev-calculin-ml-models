# Stage 2: Build Labels — Market Regime Detection

## The Big Picture

This stage is **not** about predicting whether BTC will go up or down. Instead, we're labeling what **kind of market** we're in at each point in time. The ML model will learn to recognize these market "regimes" so your bots know **how** to trade, not just which direction.

Think of it like weather: you don't just want to know "rain or sun" — you want to know if it's a steady drizzle, a thunderstorm, or a calm sunny day. Each requires a different plan.

## The Four Regimes

### 0 — CHOP
The market is going sideways. Price bounces around without clear direction. This is the most common state. Trend-following strategies lose money here because every breakout fakes out. Your bots should either sit out or use mean-reversion tactics.

### 1 — TRENDING_UP
Sustained upward momentum. Price has gained at least 2% over the last 24 hours, recent momentum is stronger than the 7-day trend, and volatility is actually *contracting* (the trend is clean, not chaotic). This is where trend-following long positions work.

### 2 — TRENDING_DOWN
Mirror of TRENDING_UP. Price has lost at least 2% over 24 hours with accelerating downward momentum and contracting volatility. Clean downtrend — short positions or defensive exits.

### 3 — VOLATILE_EXPANSION
The market is making fast, large moves in either direction. Short-term volatility has spiked well above the background level. This is crash/pump territory. Normal strategies break down here — bots should reduce position sizes or switch to volatility-specific tactics.

## How Regimes Are Detected

Everything is computed from **past prices only** — we never look into the future. At any bar T, we only use prices from T and earlier.

### Features (all backward-looking)

| Feature | What it measures | Window |
|---|---|---|
| `rolling_return_24h` | Net price change over last 24 hours | 96 bars |
| `rolling_return_7d` | Net price change over last 7 days | 672 bars |
| `rolling_return_4h` | Net price change over last 4 hours | 16 bars |
| `realized_volatility_4h` | How choppy price has been recently | 16 bars |
| `realized_volatility_24h` | Background volatility level | 96 bars |

### Classification Rules

```
TRENDING_UP requires ALL of:
  - 24h return > +2%
  - 24h return > 7d return (momentum accelerating)
  - 4h volatility < 24h volatility (trend is clean)

TRENDING_DOWN requires ALL of:
  - 24h return < -2%
  - 24h return < 7d return
  - 4h volatility < 24h volatility

VOLATILE_EXPANSION requires ALL of:
  - |4h return| > 1.5% (fast large move)
  - 4h volatility > 1.5x 24h volatility (vol spiking)

CHOP:
  - Everything else
```

### Priority

When multiple regimes qualify simultaneously:
**VOLATILE_EXPANSION > TRENDING_UP > TRENDING_DOWN > CHOP**

For example, if a sharp upward spike meets both trending and volatile conditions, it's labeled VOLATILE_EXPANSION because the extreme volatility changes how you should trade it.

## Hysteresis (Noise Filtering)

Raw regime labels can flip rapidly — one bar is TRENDING_UP, the next is CHOP, then back. This noise is bad for training an ML model.

**Hysteresis rule:** A regime must persist for at least **4 consecutive bars** (1 hour) before it becomes the confirmed label. Any episode shorter than 4 bars gets absorbed into the surrounding regime.

This is why there are two regime columns:
- `regime_raw` — the unfiltered classification (can have rapid flips)
- `regime` — the confirmed label after hysteresis (this is what the model trains on)

## Output Columns

| Column | Type | Description |
|---|---|---|
| `regime_raw` | int (0-3) | Raw regime before hysteresis |
| `regime` | int (0-3) | Confirmed regime after hysteresis — **this is the training target** |
| `regime_name` | str | Human-readable name (CHOP, TRENDING_UP, etc.) |
| `regime_start` | int | Bar index where the current regime episode started |
| `bars_in_regime` | int | How many consecutive bars have been in the current regime |

## Warmup Period

The first ~672 bars (7 days of 15-minute data) are dropped because the 7-day rolling features can't be computed without enough history.

## Why This Matters for Your Bots

Instead of asking "should I go long or short?", the regime detector answers "what kind of market am I in?" Your bots can then:
- Use **trend-following** strategies during TRENDING_UP/DOWN
- Use **mean-reversion** strategies during CHOP
- **Reduce risk** or pause during VOLATILE_EXPANSION
- **Switch between strategy modes** automatically based on the detected regime

## Usage

```bash
# Label the exported data
python3 build_labels.py ../1_BTC_Data_Export/btc_data_15m.parquet

# Run tests
python3 -m pytest test_build_labels.py -v
```
