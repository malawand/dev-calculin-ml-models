#!/usr/bin/env python3
"""
Build hindsight regime labels from BTC price data (15-minute bars).

Regimes:
    0 = CHOP              — mean-reverting, no directional signal
    1 = TRENDING_UP       — sustained upward momentum with contracting vol
    2 = TRENDING_DOWN     — sustained downward momentum with contracting vol
    3 = VOLATILE_EXPANSION — fast large move with vol spike

All computations are backward-looking only (no future data leakage).
"""
from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np
import pandas as pd


REGIME_NAMES = {0: "CHOP", 1: "TRENDING_UP", 2: "TRENDING_DOWN", 3: "VOLATILE_EXPANSION"}

BARS_4H = 16
BARS_12H = 48
BARS_24H = 96
BARS_7D = 672

MIN_HYSTERESIS_BARS = 4

ANNUALIZE_FACTOR = np.sqrt(4 * 24 * 365)

# ---------------------------------------------------------------------------
# Default thresholds — tune these if too many bars land in CHOP.
# ---------------------------------------------------------------------------
DEFAULTS = {
    "trend_return_threshold": 0.008,           # 0.8% over 24h
    "trend_return_threshold_12h": 0.006,       # 0.6% over 12h (faster-trend path)
    "trend_volatility_ratio_max": 2.0,         # realized_volatility_4h < 2.0 * realized_volatility_24h
    "volatile_return_threshold": 0.01,         # 1% over 4h
    "volatile_volatility_ratio": 1.3,          # realized_volatility_4h > 1.3 * realized_volatility_24h
}


def _compute_regime_features(price: pd.Series) -> pd.DataFrame:
    """
    Compute the rolling features needed for regime classification.
    All features are backward-looking.
    """
    log_price = np.log(price)
    bar_logret = log_price.diff()

    rolling_return_4h = log_price - log_price.shift(BARS_4H)
    rolling_return_12h = log_price - log_price.shift(BARS_12H)
    rolling_return_24h = log_price - log_price.shift(BARS_24H)
    rolling_return_7d = log_price - log_price.shift(BARS_7D)

    realized_volatility_4h = bar_logret.rolling(window=BARS_4H, min_periods=BARS_4H).std() * ANNUALIZE_FACTOR
    realized_volatility_24h = bar_logret.rolling(window=BARS_24H, min_periods=BARS_24H).std() * ANNUALIZE_FACTOR

    return pd.DataFrame({
        "rolling_return_4h": rolling_return_4h,
        "rolling_return_12h": rolling_return_12h,
        "rolling_return_24h": rolling_return_24h,
        "rolling_return_7d": rolling_return_7d,
        "realized_volatility_4h": realized_volatility_4h,
        "realized_volatility_24h": realized_volatility_24h,
    }, index=price.index)


def _classify_raw(
    features: pd.DataFrame,
    trend_return_threshold: float = DEFAULTS["trend_return_threshold"],
    trend_return_threshold_12h: float = DEFAULTS["trend_return_threshold_12h"],
    trend_volatility_ratio_max: float = DEFAULTS["trend_volatility_ratio_max"],
    volatile_return_threshold: float = DEFAULTS["volatile_return_threshold"],
    volatile_volatility_ratio: float = DEFAULTS["volatile_volatility_ratio"],
) -> pd.Series:
    """
    Classify each bar into a raw regime (before hysteresis).
    Priority: VOLATILE_EXPANSION > TRENDING_UP > TRENDING_DOWN > CHOP.

    TRENDING_UP fires via EITHER:
      (A) 24h path: rr24h > thr AND rr7d >= 0 AND vol ratio OK   (sustained trend)
      (B) 12h path: rr12h > thr_12h AND rr24h > 0 AND vol ratio OK (fast emerging trend)
    TRENDING_DOWN mirrors.
    """
    rr4h = features["rolling_return_4h"]
    rr12h = features["rolling_return_12h"]
    rr24h = features["rolling_return_24h"]
    rr7d = features["rolling_return_7d"]
    volatility_4h = features["realized_volatility_4h"]
    volatility_24h = features["realized_volatility_24h"]

    vol_ratio_ok = volatility_4h < trend_volatility_ratio_max * volatility_24h

    is_volatile = (rr4h.abs() > volatile_return_threshold) & (volatility_4h > volatile_volatility_ratio * volatility_24h)

    is_trending_up_24h = (rr24h > trend_return_threshold) & (rr7d >= 0) & vol_ratio_ok
    is_trending_up_12h = (rr12h > trend_return_threshold_12h) & (rr24h > 0) & vol_ratio_ok
    is_trending_up = is_trending_up_24h | is_trending_up_12h

    is_trending_down_24h = (rr24h < -trend_return_threshold) & (rr7d <= 0) & vol_ratio_ok
    is_trending_down_12h = (rr12h < -trend_return_threshold_12h) & (rr24h < 0) & vol_ratio_ok
    is_trending_down = is_trending_down_24h | is_trending_down_12h

    regime = pd.Series(0, index=features.index, dtype=np.int8)
    regime[is_trending_down] = 2
    regime[is_trending_up] = 1
    regime[is_volatile] = 3

    return regime


def _apply_hysteresis(raw: pd.Series, min_bars: int = MIN_HYSTERESIS_BARS) -> pd.Series:
    """
    Remove regime episodes shorter than min_bars by absorbing them into the
    surrounding regime.

    Strategy: identify contiguous runs. Any run shorter than min_bars gets
    replaced by the regime of the preceding run (or the following run if
    it's the very first episode).
    """
    vals = raw.values.copy()
    n = len(vals)

    run_starts = [0]
    for i in range(1, n):
        if vals[i] != vals[i - 1]:
            run_starts.append(i)

    runs = []
    for j, start in enumerate(run_starts):
        end = run_starts[j + 1] if j + 1 < len(run_starts) else n
        runs.append((start, end, vals[start]))

    changed = True
    while changed:
        changed = False
        new_runs = []
        for start, end, regime in runs:
            length = end - start
            if length < min_bars and new_runs:
                prev_start, prev_end, prev_regime = new_runs[-1]
                new_runs[-1] = (prev_start, end, prev_regime)
                changed = True
            elif new_runs and new_runs[-1][2] == regime:
                prev_start, prev_end, prev_regime = new_runs[-1]
                new_runs[-1] = (prev_start, end, prev_regime)
                changed = True
            else:
                new_runs.append((start, end, regime))
        runs = new_runs

    if runs and (runs[0][1] - runs[0][0]) < min_bars and len(runs) > 1:
        first_start, first_end, _ = runs[0]
        second_start, second_end, second_regime = runs[1]
        runs = [(first_start, second_end, second_regime)] + runs[2:]

    result = np.empty(n, dtype=np.int8)
    for start, end, regime in runs:
        result[start:end] = regime

    return pd.Series(result, index=raw.index, dtype=np.int8)


def _compute_episode_info(regime: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Compute regime_start (bar index of current episode start) and
    bars_in_regime (consecutive bars in the current regime so far).
    """
    vals = regime.values
    n = len(vals)
    starts = np.empty(n, dtype=np.int64)
    counts = np.empty(n, dtype=np.int64)

    starts[0] = 0
    counts[0] = 1
    for i in range(1, n):
        if vals[i] != vals[i - 1]:
            starts[i] = i
            counts[i] = 1
        else:
            starts[i] = starts[i - 1]
            counts[i] = counts[i - 1] + 1

    return (
        pd.Series(starts, index=regime.index, dtype=np.int64),
        pd.Series(counts, index=regime.index, dtype=np.int64),
    )


def _print_regime_report(regime: pd.Series) -> None:
    """Print regime distribution, mean duration, and transition count."""
    total = len(regime)

    print("\nRegime Distribution:")
    print("=" * 55)
    for r in sorted(REGIME_NAMES.keys()):
        count = (regime == r).sum()
        pct = count / total * 100
        print(f"  {REGIME_NAMES[r]:>20s} ({r}): {count:>7,} bars ({pct:5.1f}%)")

    vals = regime.values
    transitions = np.sum(vals[1:] != vals[:-1])
    print(f"\n  Total transitions: {transitions:,}")

    run_starts = [0]
    for i in range(1, len(vals)):
        if vals[i] != vals[i - 1]:
            run_starts.append(i)

    episodes = {r: [] for r in REGIME_NAMES}
    for j, start in enumerate(run_starts):
        end = run_starts[j + 1] if j + 1 < len(run_starts) else len(vals)
        r = vals[start]
        episodes[r].append(end - start)

    print("\n  Mean episode duration:")
    for r in sorted(REGIME_NAMES.keys()):
        if episodes[r]:
            mean_bars = np.mean(episodes[r])
            mean_hours = mean_bars * 0.25
            print(f"    {REGIME_NAMES[r]:>20s}: {mean_bars:6.1f} bars ({mean_hours:5.1f} hours), {len(episodes[r])} episodes")
        else:
            print(f"    {REGIME_NAMES[r]:>20s}: no episodes")
    print()


def build_labels(
    df: pd.DataFrame,
    price_col: str = "price",
    trend_return_threshold: float = DEFAULTS["trend_return_threshold"],
    trend_return_threshold_12h: float = DEFAULTS["trend_return_threshold_12h"],
    trend_volatility_ratio_max: float = DEFAULTS["trend_volatility_ratio_max"],
    volatile_return_threshold: float = DEFAULTS["volatile_return_threshold"],
    volatile_volatility_ratio: float = DEFAULTS["volatile_volatility_ratio"],
    min_hysteresis_bars: int = MIN_HYSTERESIS_BARS,
) -> pd.DataFrame:
    """
    Compute backward-looking regime labels.

    Parameters
    ----------
    df : DataFrame with a price column and datetime index.
    price_col : Name of the close-price column.
    trend_return_threshold : Min 24h log-return magnitude for trending (default 0.008 = 0.8%).
    trend_return_threshold_12h : Min 12h log-return magnitude for the faster-trend path (default 0.006 = 0.6%).
    trend_volatility_ratio_max : Max realized_volatility_4h / realized_volatility_24h for a clean trend (default 2.0).
    volatile_return_threshold : Min |4h return| for volatile expansion (default 0.01 = 1%).
    volatile_volatility_ratio : Min realized_volatility_4h / realized_volatility_24h for volatile expansion (default 1.3).
    min_hysteresis_bars : Min episode length to be a confirmed regime (default 4).

    Returns
    -------
    DataFrame with original columns plus:
        regime_raw, regime, regime_name, regime_start, bars_in_regime
    Warmup rows (where features can't be computed) are dropped.
    """
    out = df.copy()
    features = _compute_regime_features(out[price_col])

    first_valid = features.dropna().index[0]
    out = out.loc[first_valid:]
    features = features.loc[first_valid:]

    out["regime_raw"] = _classify_raw(
        features,
        trend_return_threshold=trend_return_threshold,
        trend_return_threshold_12h=trend_return_threshold_12h,
        trend_volatility_ratio_max=trend_volatility_ratio_max,
        volatile_return_threshold=volatile_return_threshold,
        volatile_volatility_ratio=volatile_volatility_ratio,
    )
    out["regime"] = _apply_hysteresis(out["regime_raw"], min_bars=min_hysteresis_bars)
    out["regime_name"] = out["regime"].map(REGIME_NAMES)

    regime_start, bars_in_regime = _compute_episode_info(out["regime"])
    out["regime_start"] = regime_start
    out["bars_in_regime"] = bars_in_regime

    print(f"\nThresholds used:")
    print(f"  trend_return_threshold       = {trend_return_threshold}")
    print(f"  trend_return_threshold_12h   = {trend_return_threshold_12h}")
    print(f"  trend_volatility_ratio_max   = {trend_volatility_ratio_max}")
    print(f"  volatile_return_threshold    = {volatile_return_threshold}")
    print(f"  volatile_volatility_ratio    = {volatile_volatility_ratio}")
    print(f"  min_hysteresis_bars          = {min_hysteresis_bars}")

    _print_regime_report(out["regime"])

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build regime labels from Parquet price data")
    parser.add_argument("input", help="Path to input Parquet file")
    parser.add_argument("--output", default=None, help="Output Parquet path (default: input with _labeled suffix)")
    parser.add_argument("--price-col", default="price", help="Name of the price column")
    parser.add_argument("--trend-return-threshold", type=float, default=DEFAULTS["trend_return_threshold"],
                        help="Min 24h log-return magnitude for trending (default 0.008)")
    parser.add_argument("--trend-return-threshold-12h", type=float, default=DEFAULTS["trend_return_threshold_12h"],
                        help="Min 12h log-return for faster-trend path (default 0.006)")
    parser.add_argument("--trend-volatility-ratio-max", type=float, default=DEFAULTS["trend_volatility_ratio_max"],
                        help="Max realized_volatility_4h / realized_volatility_24h for a clean trend (default 2.0)")
    parser.add_argument("--volatile-return-threshold", type=float, default=DEFAULTS["volatile_return_threshold"],
                        help="Min |4h return| for VOLATILE_EXPANSION (default 0.01)")
    parser.add_argument("--volatile-volatility-ratio", type=float, default=DEFAULTS["volatile_volatility_ratio"],
                        help="Min realized_volatility_4h / realized_volatility_24h for VOLATILE_EXPANSION (default 1.3)")
    parser.add_argument("--min-hysteresis-bars", type=int, default=MIN_HYSTERESIS_BARS,
                        help="Min bars for a confirmed regime episode (default 4)")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    labeled = build_labels(
        df,
        price_col=args.price_col,
        trend_return_threshold=args.trend_return_threshold,
        trend_return_threshold_12h=args.trend_return_threshold_12h,
        trend_volatility_ratio_max=args.trend_volatility_ratio_max,
        volatile_return_threshold=args.volatile_return_threshold,
        volatile_volatility_ratio=args.volatile_volatility_ratio,
        min_hysteresis_bars=args.min_hysteresis_bars,
    )

    output = args.output or args.input.replace(".parquet", "_labeled.parquet")
    labeled.to_parquet(output, engine="pyarrow")
    print(f"Saved {len(labeled)} rows -> {output}")


if __name__ == "__main__":
    main()
