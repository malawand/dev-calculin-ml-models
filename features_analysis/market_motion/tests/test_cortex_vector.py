"""cortex_market_motion_v1: 8h not 7h, no OLS fallback, confirmation-time T from Cortex D."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_motion.cortex_client import collapse_raw_to_hourly
from market_motion.cortex_tape import hourly_from_raw
from market_motion.cortex_vector import build_cortex_vector, cortex_horizon_names
from market_motion.exceptions import CortexMotionNotImplementedError, CortexSourceError, VectorSourceError
from market_motion.feature_allowlist import select_model_feature_columns
from market_motion.manifest import csv_checksum, feature_configuration_hash
from market_motion.outcome_engine import build_outcomes
from market_motion.quality import bounded_ffill
from market_motion.source_guard import concat_vector_frames, load_inference_vector, timing_skip_columns
from market_motion.transition import EVENT_NONE, EVENT_PEAK, EVENT_TROUGH, confirm_local_extrema
from market_motion.vector_engine import build_vector


CORTEX_H = ["2h", "4h", "8h", "12h", "24h", "48h", "7d"]


def _cortex_cfg(**overrides):
    horizons = {
        name: {
            "query": f'job:crypto_last_price:deriv{name}{{symbol="BTCUSDT"}}',
            "horizon_hours": {"2h": 2, "4h": 4, "8h": 8, "12h": 12, "24h": 24, "48h": 48, "7d": 168}[name],
        }
        for name in CORTEX_H
    }
    cfg = {
        "asset": "BTCUSDT",
        "sampling_frequency": "1h",
        "timing": {"feature_available_lag_hours": 1, "decision_latency_hours": 0},
        "vector": {
            "source": "price_ols_proxy",
            "version": "price_ols_proxy_v1",
            "z_window_hours": 48,
            "z_min_obs": 12,
            "z_clip": 8.0,
            "velocity_delta_hours": 1,
            "confirmation_delay_hours": 2,
            "min_prominence_mad": 1.0,
            "min_duration_horizon_fraction": 0.25,
            "prominence_mad_window_hours": 168,
            "prominence_min_obs": 24,
            "execution_lag_hours": 1,
            "staleness_hours": {"crypto_last_price": 2, "cortex_d": 2},
            "horizons": {"2h": 3, "4h": 4, "7h": 7, "12h": 12},
        },
        "cortex_motion": {
            "version": "cortex_market_motion_v1",
            "source_query_version": "cortex_continuous_d_prod_a_preferred_v1",
            "max_staleness_hours": 2,
            "feature_available_lag_hours": 1,
            "timing_certainty": "uncertain_recording_rule",
            "horizons": horizons,
        },
        "outcomes": {
            "version": "btc_outcomes_v1",
            "return_horizons_h": [1, 24],
            "excursion_horizons_h": [4],
            "flat_threshold_pct": 0.1,
        },
    }
    cfg.update(overrides)
    return cfg


def _hourly(n=220, start="2024-01-01"):
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    t = np.arange(n, dtype=float)
    data = {"time_bucket": idx}
    periods = {"2h": 11, "4h": 19, "8h": 31, "12h": 47, "24h": 73, "48h": 97, "7d": 168}
    for name, p in periods.items():
        data[name] = np.sin(t / p) + 0.05 * np.sin(t / 5.0)
    return pd.DataFrame(data)


def _prices(n=220, start="2024-01-01"):
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


def test_cortex_horizons_are_eight_not_seven():
    names = cortex_horizon_names(_cortex_cfg())
    assert names == CORTEX_H
    assert "8h" in names
    assert "7h" not in names
    vec = build_cortex_vector(_hourly(), _cortex_cfg())
    assert any(c.startswith("8h_") for c in vec.columns)
    assert not any(c.startswith("7h_") for c in vec.columns)
    assert (vec["vector_source"] == "cortex_market_motion").all()
    assert (vec["vector_version"] == "cortex_market_motion_v1").all()
    assert (vec["source_query_version"] == "cortex_continuous_d_prod_a_preferred_v1").all()
    assert vec["feature_configuration_hash"].nunique() == 1


def test_seven_h_horizon_rejected():
    cfg = _cortex_cfg()
    cfg["cortex_motion"]["horizons"]["7h"] = {"query": "nope", "horizon_hours": 7}
    with pytest.raises(CortexSourceError, match="7h"):
        cortex_horizon_names(cfg)


def test_missing_required_cortex_source_fails():
    hourly = _hourly().drop(columns=["8h"])
    with pytest.raises(CortexSourceError, match="8h"):
        build_cortex_vector(hourly, _cortex_cfg())


def test_all_nan_cortex_source_fails_without_ols_fallback():
    hourly = _hourly()
    hourly["8h"] = np.nan
    prices = _prices()
    with pytest.raises(CortexSourceError, match="OLS will not be substituted"):
        build_cortex_vector(hourly, _cortex_cfg(), prices=prices)


def test_ols_engine_still_refuses_cortex_source():
    cfg = _cortex_cfg()
    cfg["vector"]["source"] = "cortex_market_motion"
    with pytest.raises(CortexMotionNotImplementedError):
        build_vector(_prices(), cfg)


def test_canonical_t_from_cortex_d_not_velocity():
    hourly = _hourly()
    vec = build_cortex_vector(hourly, _cortex_cfg())
    d = vec.set_index(pd.to_datetime(vec["time_bucket"], utc=True))["8h_derivative_raw"]
    want = confirm_local_extrema(
        d,
        delay_hours=2,
        min_prominence_mad=1.0,
        min_duration_hours=2,
        prominence_window_hours=168,
        prominence_min_obs=24,
    )
    np.testing.assert_equal(
        vec["8h_transition_state_raw"].to_numpy(dtype=float),
        want["transition_state_raw"].to_numpy(dtype=float),
    )
    assert "8h_velocity_transition_state_raw" in vec.columns
    features = select_model_feature_columns(vec, CORTEX_H)
    assert "8h_transition_state_raw" in features
    assert "8h_hours_since_confirmation" in features
    assert "8h_transition_raw" not in features
    assert not any("velocity_transition" in c for c in features)
    assert not any(c.startswith("7h_") for c in features)


def test_one_canonical_transition_and_age_per_horizon():
    vec = build_cortex_vector(_hourly(), _cortex_cfg())
    features = select_model_feature_columns(vec, CORTEX_H)
    for h in CORTEX_H:
        t_cols = [c for c in features if c.startswith(f"{h}_") and "transition" in c]
        age_cols = [c for c in features if c.startswith(f"{h}_") and "hours_since" in c]
        assert t_cols == [f"{h}_transition_state_raw"]
        assert age_cols == [f"{h}_hours_since_confirmation"]


def test_no_oracle_columns_on_cortex_vector():
    vec = build_cortex_vector(_hourly(), _cortex_cfg(), prices=_prices())
    assert not any(str(c).startswith("oracle_") for c in vec.columns)
    assert "waiting_cost" not in vec.columns


def test_unknown_initial_state_and_confirmation_time():
    vec = build_cortex_vector(_hourly(), _cortex_cfg())
    first = vec["8h_transition_event"].ne(EVENT_NONE)
    assert first.any()
    i = int(np.argmax(first.to_numpy()))
    assert vec["8h_transition_state_raw"].iloc[:i].isna().all()
    assert (vec["8h_transition_event"].iloc[:i] == EVENT_NONE).all()
    row = vec.iloc[i]
    assert row["8h_transition_event"] in {EVENT_PEAK, EVENT_TROUGH}
    obs = pd.Timestamp(row["observation_time_utc"])
    assert pd.Timestamp(row["feature_available_time_utc"]) == obs + pd.Timedelta(hours=1)
    assert pd.Timestamp(row["outcome_entry_time_utc"]) > obs


def test_no_same_close_execution():
    prices = _prices()
    vec = build_cortex_vector(_hourly(), _cortex_cfg(), prices=prices)
    oc = build_outcomes(prices, _cortex_cfg())
    joined = vec.merge(oc[["time_bucket", "btc_close_at_entry", "return_1h"]], on="time_bucket")
    events = joined[joined["8h_transition_event"].isin([EVENT_PEAK, EVENT_TROUGH])]
    assert len(events) > 0
    row = events.iloc[0]
    assert row["btc_close_at_entry"] != row["btc_close_at_t"]


def test_never_backward_fill_cortex_d():
    hourly = _hourly(40)
    hourly.loc[0:2, "2h"] = np.nan
    vec = build_cortex_vector(hourly, _cortex_cfg())
    assert pd.isna(vec["2h_derivative_raw"].iloc[0])
    assert pd.isna(vec["2h_derivative_raw"].iloc[1])
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    s = pd.Series([np.nan, np.nan, 3.0], index=idx[:3])
    filled, _, valid = bounded_ffill(s, max_hours=2)
    assert pd.isna(filled.iloc[0])
    assert not valid.iloc[0]


def test_bounded_staleness_and_availability_lag():
    hourly = _hourly(30)
    hourly.loc[10:16, "4h"] = np.nan
    vec = build_cortex_vector(hourly, _cortex_cfg())
    assert not bool(vec["4h_feature_valid"].iloc[13])
    assert (vec["feature_available_time_utc"] - vec["observation_time_utc"] == pd.Timedelta(hours=1)).all()
    assert vec["timing_certainty"].iloc[0] == "uncertain_recording_rule"


def test_batch_vs_prefix_parity_cortex():
    hourly = _hourly(90)
    cfg = _cortex_cfg()
    full = build_cortex_vector(hourly, cfg)
    prefix = build_cortex_vector(hourly.iloc[:70].copy(), cfg)
    skip = timing_skip_columns()
    cols = [c for c in full.columns if c not in skip]
    _assert_row_close(prefix[cols].iloc[-1], full[cols].iloc[69], cols)
    _assert_row_close(prefix[cols].iloc[20], full[cols].iloc[20], cols)


def test_duplicate_timestamps_prod_a_preferred():
    raw = pd.DataFrame(
        {
            "timestamp": [1_700_000_000, 1_700_000_000, 1_700_003_600],
            "value": [1.0, 3.0, 2.0],
            "series_name": ["8h", "8h", "8h"],
            "cluster": ["PROD-A-crypto-metrics", "PROD-B-crypto-metrics", "PROD-A-crypto-metrics"],
            "replica": ["", "", ""],
        }
    )
    s = collapse_raw_to_hourly(raw, "8h")
    assert float(s.iloc[0]) == 1.0
    assert float(s.iloc[0]) != 3.0
    assert float(s.iloc[1]) == 2.0
    hourly = hourly_from_raw(raw, ["8h"])
    assert not bool(hourly["8h_source_conflict"].iloc[0])
    assert hourly["8h"].iloc[0] == pytest.approx(1.0)


def test_hourly_alignment_preserves_gaps():
    raw = pd.DataFrame(
        {
            "timestamp": [1_700_000_000, 1_700_007_200],
            "value": [1.0, 2.0],
            "series_name": ["2h", "2h"],
            "replica": ["a", "a"],
        }
    )
    hourly = hourly_from_raw(raw, ["2h"])
    assert hourly["2h"].isna().sum() >= 1
    assert pd.isna(hourly["2h"].iloc[1])


def test_mixed_sources_and_versions_rejected(tmp_path):
    a = pd.DataFrame(
        {
            "vector_source": ["cortex_market_motion"],
            "vector_version": ["cortex_market_motion_v1"],
            "x": [1.0],
        }
    )
    b = pd.DataFrame(
        {
            "vector_source": ["price_ols_proxy"],
            "vector_version": ["price_ols_proxy_v1"],
            "x": [2.0],
        }
    )
    with pytest.raises(VectorSourceError, match="incompatible"):
        concat_vector_frames([a, b])
    mixed_ver = pd.DataFrame(
        {
            "vector_source": ["cortex_market_motion", "cortex_market_motion"],
            "vector_version": ["cortex_market_motion_v1", "cortex_market_motion_v0"],
            "2h_position_z": [0.1, 0.2],
        }
    )
    p = tmp_path / "vector_cortex_market_motion_v1.csv"
    mixed_ver.to_csv(p, index=False)
    with pytest.raises(VectorSourceError, match="mixed"):
        load_inference_vector(p)


def test_infer_loads_cortex_vector(tmp_path):
    vec = build_cortex_vector(_hourly(80), _cortex_cfg())
    p = tmp_path / "vector_cortex_market_motion_v1.csv"
    vec.to_csv(p, index=False)
    loaded = load_inference_vector(p)
    assert (loaded["vector_source"] == "cortex_market_motion").all()


def test_deterministic_checksums_same_source_tape():
    hourly = _hourly(60)
    cfg = _cortex_cfg()
    a = build_cortex_vector(hourly, cfg)
    b = build_cortex_vector(hourly.copy(), cfg)
    assert csv_checksum(a) == csv_checksum(b)
    assert feature_configuration_hash(cfg) == feature_configuration_hash(cfg)
    assert a["feature_configuration_hash"].iloc[0] == b["feature_configuration_hash"].iloc[0]
