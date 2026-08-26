"""Outcome v1: entry-aligned returns, close-path MFE/MAE, gaps, unlabeled tails."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_motion.outcome_engine import build_outcomes


def _cfg(entry_lag=1):
    return {
        "timing": {
            "feature_available_lag_hours": entry_lag,
            "decision_latency_hours": 0,
        },
        "vector": {"execution_lag_hours": entry_lag},
        "outcomes": {
            "version": "btc_outcomes_v1",
            "return_horizons_h": [1, 2, 4, 8, 12, 24, 48],
            "excursion_horizons_h": [4, 12, 24, 48],
            "flat_threshold_pct": 0.1,
            "path_convention": "hourly_close",
        },
    }


def _prices(n=80, start="2024-01-01"):
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    t = np.arange(n, dtype=float)
    px = 100_000 + 2_000 * np.sin(t / 18.0) + t * 2
    return pd.DataFrame({"time_bucket": idx, "close_price": px})


def test_returns_all_horizons_present():
    oc = build_outcomes(_prices(80), _cfg())
    for k in (1, 2, 4, 8, 12, 24, 48):
        assert f"return_{k}h" in oc.columns
        assert f"log_return_{k}h" in oc.columns
        assert f"label_up_down_flat_{k}h" in oc.columns


def test_unlabeled_tails_not_zero():
    oc = build_outcomes(_prices(20), _cfg(entry_lag=1))
    # entry lag 1 + 4h path => last labeled index is n - 1 - 1 - 4 = 14
    assert pd.isna(oc["return_4h"].iloc[-1])
    assert pd.isna(oc["return_4h"].iloc[-5])
    assert np.isfinite(oc["return_4h"].iloc[-6])
    assert oc["return_48h"].isna().all()


def test_no_same_close_execution_leakage():
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    px = np.array([100.0, 101.0, 99.0, 103.0, 98.0, 100.0])
    df = pd.DataFrame({"time_bucket": idx, "close_price": px})
    oc = build_outcomes(df, _cfg(entry_lag=1))
    leaked = (px[1] / px[0] - 1.0) * 100.0
    canonical = (px[2] / px[1] - 1.0) * 100.0
    assert abs(oc["return_1h"].iloc[0] - canonical) < 1e-9
    assert abs(oc["return_1h"].iloc[0] - leaked) > 0.01
    assert oc["outcome_entry_time_utc"].iloc[0] == idx[0] + pd.Timedelta(hours=1)
    assert oc["btc_close_at_entry"].iloc[0] == px[1]


def test_mfe_mae_hand_check_from_entry():
    idx = pd.date_range("2024-01-01", periods=7, freq="h", tz="UTC")
    px = np.array([100.0, 101.0, 99.0, 103.0, 98.0, 100.0, 100.0])
    df = pd.DataFrame({"time_bucket": idx, "close_price": px})
    oc = build_outcomes(df, _cfg(entry_lag=1))
    entry = px[1]
    path = (px[2:6] / entry - 1.0) * 100.0
    assert abs(oc["close_path_mfe_4h"].iloc[0] - float(path.max())) < 1e-9
    assert abs(oc["close_path_mae_4h"].iloc[0] - float(path.min())) < 1e-9
    assert oc["time_to_close_path_mfe_4h"].iloc[0] == int(path.argmax()) + 1
    assert oc["time_to_close_path_mae_4h"].iloc[0] == int(path.argmin()) + 1
    want_ret = (px[5] / entry - 1.0) * 100.0
    assert abs(oc["return_4h"].iloc[0] - want_ret) < 1e-9


def test_long_oriented_mfe_mae_signs():
    idx = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    px = np.array([100.0, 100.0, 102.0, 101.0, 99.0, 98.0, 100.0, 100.0])
    df = pd.DataFrame({"time_bucket": idx, "close_price": px})
    oc = build_outcomes(df, _cfg(entry_lag=0))
    assert oc["close_path_mfe_4h"].iloc[0] > 0
    assert oc["close_path_mae_4h"].iloc[0] < 0


def test_gap_inside_outcome_window_unlabeled():
    df = _prices(30)
    df.loc[df.index[10], "close_price"] = np.nan
    oc = build_outcomes(df, _cfg(entry_lag=1))
    # Observation i=5, entry=6, 4h path uses hours 7..10. Hour 10 is NaN.
    assert pd.isna(oc["return_4h"].iloc[5])
    assert pd.isna(oc["close_path_mfe_4h"].iloc[5])


def test_utc_midnight_return_alignment():
    df = _prices(10, start="2024-01-01 22:00")
    oc = build_outcomes(df, _cfg(entry_lag=1))
    t0 = pd.Timestamp("2024-01-01 22:00", tz="UTC")
    row = oc.loc[pd.to_datetime(oc["time_bucket"], utc=True) == t0].iloc[0]
    p_entry = df["close_price"].iloc[1]
    p_exit = df["close_price"].iloc[2]
    assert abs(row["return_1h"] - (p_exit / p_entry - 1) * 100) < 1e-9
