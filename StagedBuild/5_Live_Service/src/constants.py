"""Shared constants for the live regime detector."""

from __future__ import annotations

REGIME_INT_TO_NAME = {
    0: "CHOP",
    1: "TRENDING_UP",
    2: "TRENDING_DOWN",
    3: "VOLATILE_EXPANSION",
}

DEFAULT_FEATURE_COLS = [
    "weighted_norm_avg_16h_24h_48h",
    "weighted_deriv_24h_48h_7d",
    "norm_combined_avg",
]

DEFAULT_WINDOWS = [16, 32, 96, 192]

# Second derivative at w=192 needs 2 * 192 = 384 bars of 15-minute history.
MIN_BARS_FOR_FEATURES = 384
RECOMMENDED_LOOKBACK_DAYS = 8

BAR_STEP_SECONDS = 900  # 15 minutes

DEFAULT_PROMETHEUS_QUERIES = {
    "price": 'max by (symbol) (crypto_last_price{symbol="BTCUSDT"})',
    "weighted_norm_avg_16h_24h_48h": (
        'job:crypto_last_price:weighted_normalized_avg:16h:24h:48h{symbol="BTCUSDT"}'
    ),
    "weighted_deriv_24h_48h_7d": (
        'job:crypto_last_price:weighted_deriv:24h:48h:7d{symbol="BTCUSDT"}'
    ),
    "norm_combined_avg": (
        'job:crypto_last_price:normalized_combined_avg{symbol="BTCUSDT"}'
    ),
}
