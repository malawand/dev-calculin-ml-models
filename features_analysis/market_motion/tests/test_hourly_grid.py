"""Hourly UTC grid, causal OLS vector, prefix parity, and execution timestamps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_motion.exceptions import CortexMotionNotImplementedError, VectorSourceError
from market_motion.quality import bounded_ffill, reindex_hourly
from market_motion.source_guard import timing_skip_columns
from market_motion.vector_engine import build_vector, rolling_ols_slope


def _cfg(**overrides):
    cfg = {
        "asset": "BTCUSDT",
        "sampling_frequency": "1h",
        "timing": {
            "feature_available_lag_hours": 1,
            "decision_latency_hours": 0,
        },
        "vector": {
            "source": "price_ols_proxy",
            "version": "price_ols_proxy_v1",
            "use_log_price": True,
            "z_window_hours": 48,
            "z_min_obs": 12,
            "z_clip": 8.0,
            "velocity_delta_hours": 1,
            "min_ols_bars_for_slope": 3,
            "min_ols_bars_for_va": 8,
            "confirmation_delay_hours": 2,
            "execution_lag_hours": 1,
            "staleness_hours": {"crypto_last_price": 2},
            "horizons": {"2h": 3, "4h": 4, "7h": 7, "12h": 12},
        },
        "outcomes": {
            "version": "btc_outcomes_v1",
            "return_horizons_h": [1, 2, 4],
            "excursion_horizons_h": [4],
            "flat_threshold_pct": 0.1,
        },
    }
    cfg.update(overrides)
    return cfg


def _prices(n=200, start="2024-01-01"):
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    t = np.arange(n, dtype=float)
    px = 100_000 + 2_000 * np.sin(t / 18.0) + t * 2
    return pd.DataFrame({"time_bucket": idx, "close_price": px})


def _assert_row_close(a: pd.Series, b: pd.Series, cols: list[str]) -> None:
    for c in cols:
        va, vb = a[c], b[c]
        if pd.isna(va) and pd.isna(vb):
            continue
        if isinstance(va, (pd.Timestamp, np.datetime64)) or isinstance(vb, (pd.Timestamp, np.datetime64)):
            assert pd.Timestamp(va) == pd.Timestamp(vb), c
            continue
        if isinstance(va, str) or isinstance(vb, str):
            assert str(va) == str(vb), c
            continue
        if isinstance(va, (bool, np.bool_)) or isinstance(vb, (bool, np.bool_)):
            assert bool(va) == bool(vb), c
            continue
        assert abs(float(va) - float(vb)) < 1e-9, c


def test_reindex_is_hourly_unique_monotonic():
    raw = pd.DataFrame(
        {
            "time_bucket": pd.to_datetime(
                ["2024-01-01 00:00Z", "2024-01-01 00:00Z", "2024-01-01 02:00Z"]
            ),
            "x": [1.0, 2.0, 3.0],
        }
    )
    out = reindex_hourly(raw)
    assert out.index.is_monotonic_increasing
    assert out.index.tz is not None
    assert out.index.nunique() == len(out)
    assert len(out) == 3


def test_never_bfill_and_staleness():
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    s = pd.Series([1.0, np.nan, np.nan, np.nan, 2.0, np.nan], index=idx)
    filled, age, valid = bounded_ffill(s, max_hours=2)
    assert pd.isna(filled.iloc[3])
    assert valid.iloc[0]
    assert not valid.iloc[3]
    assert filled.iloc[1] == 1.0
    assert filled.iloc[4] == 2.0
    # Leading NaN would require a backward fill; that must not happen.
    s2 = pd.Series([np.nan, np.nan, 3.0], index=idx[:3])
    filled2, _, valid2 = bounded_ffill(s2, max_hours=2)
    assert pd.isna(filled2.iloc[0])
    assert pd.isna(filled2.iloc[1])
    assert not valid2.iloc[0]


def test_long_outage_never_stays_valid():
    idx = pd.date_range("2024-01-01", periods=12, freq="h", tz="UTC")
    s = pd.Series([1.0] + [np.nan] * 10 + [2.0], index=idx)
    filled, age, valid = bounded_ffill(s, max_hours=2)
    assert valid.iloc[0]
    assert valid.iloc[1]
    assert valid.iloc[2]
    assert not valid.iloc[3]
    assert not valid.iloc[9]
    assert valid.iloc[11]


def test_shift_hours_not_rows():
    df = _prices(30)
    df = df.iloc[[0, 1, 5, 6]]
    hourly = reindex_hourly(df)
    hourly["close_price"] = pd.to_numeric(hourly["close_price"])
    chg = hourly["close_price"].shift(-2) / hourly["close_price"] - 1
    assert pd.isna(chg.iloc[0])


def test_utc_day_boundary_alignment():
    df = _prices(8, start="2024-01-01 20:00")
    vec = build_vector(df, _cfg())
    midnight = pd.Timestamp("2024-01-02 00:00", tz="UTC")
    assert midnight in set(pd.to_datetime(vec["time_bucket"], utc=True))
    row = vec.loc[pd.to_datetime(vec["time_bucket"], utc=True) == midnight].iloc[0]
    assert row["observation_time_utc"] == midnight
    assert row["feature_available_time_utc"] == midnight + pd.Timedelta(hours=1)
    assert row["outcome_entry_time_utc"] == midnight + pd.Timedelta(hours=1)


def test_ols_is_causal():
    x = np.arange(10, dtype=float)
    s = rolling_ols_slope(x, 4)
    assert np.isnan(s[2])
    assert np.isfinite(s[3])


def test_ols_window_is_trailing_not_centered():
    x = np.zeros(10)
    x[9] = 100.0
    s = rolling_ols_slope(x, 4)
    assert s[8] == 0.0
    assert s[9] != 0.0


def test_vector_no_lookahead_on_z():
    df = _prices(80)
    vec = build_vector(df, _cfg())
    assert not bool(vec["2h_normalization_ready"].iloc[5])
    assert vec["2h_position_z"].notna().sum() > 10


def test_vector_identity_columns():
    vec = build_vector(_prices(40), _cfg())
    assert (vec["vector_source"] == "price_ols_proxy").all()
    assert (vec["vector_version"] == "price_ols_proxy_v1").all()


def test_short_window_va_not_stable():
    vec = build_vector(_prices(80), _cfg())
    assert (vec["2h_resolution_class"] == "slope_only").all()
    assert not bool(vec["2h_va_stable"].iloc[-1])
    assert bool(vec["12h_va_stable"].iloc[-1])
    assert vec["2h_velocity"].isna().all()
    assert vec["2h_acceleration"].isna().all()
    assert (vec["2h_ols_window_bars"] == 3).all()


def test_two_hour_ols_needs_finer_data_for_va():
    """2h at 1h sampling is 3 points: slope exists, V/A does not."""
    vec = build_vector(_prices(80), _cfg())
    assert int(vec["2h_ols_window_bars"].iloc[-1]) < 8
    assert vec["2h_resolution_class"].iloc[-1] == "slope_only"


def test_cortex_source_raises_and_does_not_fallback():
    cfg = _cfg()
    cfg["vector"]["source"] = "cortex_market_motion"
    with pytest.raises(CortexMotionNotImplementedError, match="price_ols_proxy"):
        build_vector(_prices(40), cfg)


def test_deprecated_cortex_derivative_name_raises():
    cfg = _cfg()
    cfg["vector"]["source"] = "cortex_derivative"
    with pytest.raises(CortexMotionNotImplementedError):
        build_vector(_prices(40), cfg)


def test_missing_source_refuses_default():
    cfg = _cfg()
    del cfg["vector"]["source"]
    with pytest.raises(VectorSourceError, match="required"):
        build_vector(_prices(20), cfg)


def test_execution_is_after_close():
    vec = build_vector(_prices(40), _cfg())
    lag = vec["feature_available_time_utc"] - vec["observation_time_utc"]
    assert (lag == pd.Timedelta(hours=1)).all()
    assert (vec["outcome_entry_time_utc"] == vec["decision_time_utc"]).all()
    assert (vec["outcome_entry_time_utc"] > vec["observation_time_utc"]).all()


def test_batch_vs_prefix_parity():
    df = _prices(90)
    cfg = _cfg()
    full = build_vector(df, cfg)
    prefix = build_vector(df.iloc[:70].copy(), cfg)
    skip = timing_skip_columns()
    cols = [c for c in full.columns if c not in skip]
    _assert_row_close(prefix[cols].iloc[-1], full[cols].iloc[69], cols)


def test_prefix_parity_with_missing_prices():
    df = _prices(80)
    df.loc[df.index[10:13], "close_price"] = np.nan
    cfg = _cfg()
    full = build_vector(df, cfg)
    prefix = build_vector(df.iloc[:50].copy(), cfg)
    skip = timing_skip_columns()
    cols = [c for c in full.columns if c not in skip]
    _assert_row_close(prefix[cols].iloc[-1], full[cols].iloc[49], cols)


def test_prefix_parity_normalization_warmup():
    df = _prices(40)
    cfg = _cfg()
    full = build_vector(df, cfg)
    prefix = build_vector(df.iloc[:20].copy(), cfg)
    assert not bool(prefix["12h_normalization_ready"].iloc[-1])
    assert bool(prefix["12h_normalization_ready"].iloc[-1]) == bool(
        full["12h_normalization_ready"].iloc[19]
    )


def test_prefix_parity_stale_source():
    df = _prices(60)
    df.loc[df.index[40:45], "close_price"] = np.nan
    cfg = _cfg()
    full = build_vector(df, cfg)
    prefix = build_vector(df.iloc[:50].copy(), cfg)
    assert full["price_feature_valid"].iloc[43] == prefix["price_feature_valid"].iloc[43]
    assert not bool(full["price_feature_valid"].iloc[43])
