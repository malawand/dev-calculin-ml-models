"""Confirmation-time state: 2:00 peak / 4:00 confirm, no backdating, oracle exclusion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_motion import TRANSITION_RAW_DOWN_AFTER_PEAK, TRANSITION_RAW_UP_AFTER_TROUGH
from market_motion.exceptions import VectorSourceError
from market_motion.feature_allowlist import (
    assert_no_oracle_columns,
    is_oracle_column,
    select_model_feature_columns,
)
from market_motion.oracle import ORACLE_FILE_NAME, build_oracle_table
from market_motion.outcome_engine import build_outcomes
from market_motion.source_guard import load_inference_vector, timing_skip_columns
from market_motion.transition import EVENT_NONE, EVENT_PEAK, EVENT_TROUGH, confirm_local_extrema
from market_motion.vector_engine import build_vector


def _peak_series():
    idx = pd.date_range("2024-01-01 00:00", periods=10, freq="h", tz="UTC")
    values = [1.0, 2.0, 5.0, 4.0, 3.0, 2.5, 2.0, 2.2, 2.8, 3.5]
    return pd.Series(values, index=idx)


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
            "horizons": {"2h": 3, "12h": 12},
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


def _prices(n=80, start="2024-01-01"):
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


def test_two_am_peak_confirmed_at_four_am():
    out = confirm_local_extrema(_peak_series(), delay_hours=2)
    idx = _peak_series().index
    two, three, four = idx[2], idx[3], idx[4]
    assert out["transition_event"].loc[two] == EVENT_NONE
    assert out["transition_event"].loc[three] == EVENT_NONE
    assert pd.isna(out["transition_state_raw"].loc[two])
    assert pd.isna(out["transition_state_raw"].loc[three])
    assert out["transition_event"].loc[four] == EVENT_PEAK
    assert out["transition_state_raw"].loc[four] == TRANSITION_RAW_DOWN_AFTER_PEAK
    assert out["confirmation_time"].loc[four] == four
    assert out["candidate_extremum_time"].loc[four] == two
    assert pd.isna(out["candidate_extremum_time"].loc[two])
    assert pd.isna(out["candidate_extremum_time"].loc[three])


def test_state_persists_and_event_is_once():
    out = confirm_local_extrema(_peak_series(), delay_hours=2)
    idx = _peak_series().index
    four, five = idx[4], idx[5]
    assert out["transition_event"].loc[five] == EVENT_NONE
    assert out["transition_state_raw"].loc[five] == TRANSITION_RAW_DOWN_AFTER_PEAK
    assert (out["transition_event"] == EVENT_PEAK).sum() == 1
    assert out["confirmation_time"].loc[five] == four


def test_trough_after_peak_flips_state():
    out = confirm_local_extrema(_peak_series(), delay_hours=2)
    idx = _peak_series().index
    trough_bar = idx[6]
    confirm_bar = idx[8]
    assert out["transition_event"].loc[confirm_bar] == EVENT_TROUGH
    assert out["transition_state_raw"].loc[confirm_bar] == TRANSITION_RAW_UP_AFTER_TROUGH
    assert out["candidate_extremum_time"].loc[confirm_bar] == trough_bar
    assert out["transition_event"].loc[idx[7]] == EVENT_NONE
    assert out["transition_state_raw"].loc[idx[7]] == TRANSITION_RAW_DOWN_AFTER_PEAK


def test_unknown_initial_state_not_guessed_or_backfilled():
    out = confirm_local_extrema(_peak_series(), delay_hours=2)
    first_event = out["transition_event"].ne(EVENT_NONE)
    first_i = int(np.argmax(first_event.to_numpy()))
    assert first_i == 4
    assert out["transition_state_raw"].iloc[:first_i].isna().all()
    assert (out["transition_event"].iloc[:first_i] == EVENT_NONE).all()


def test_no_backdating_when_history_grows():
    series = _peak_series()
    prefix = confirm_local_extrema(series.iloc[:4], delay_hours=2)
    full = confirm_local_extrema(series, delay_hours=2)
    for ts in series.index[:4]:
        assert prefix["transition_event"].loc[ts] == full["transition_event"].loc[ts]
        a, b = prefix["transition_state_raw"].loc[ts], full["transition_state_raw"].loc[ts]
        assert (pd.isna(a) and pd.isna(b)) or a == b
        assert pd.isna(full["candidate_extremum_time"].loc[ts])


def test_prefix_parity_includes_confirmation_fields():
    df = _prices(90)
    cfg = _cfg()
    full = build_vector(df, cfg)
    prefix = build_vector(df.iloc[:70].copy(), cfg)
    skip = timing_skip_columns()
    cols = [c for c in full.columns if c not in skip]
    _assert_row_close(prefix[cols].iloc[-1], full[cols].iloc[69], cols)
    earlier = [c for c in full.columns if c not in skip]
    _assert_row_close(prefix[earlier].iloc[20], full[earlier].iloc[20], earlier)


def test_confirmation_bar_execution_is_after_close():
    vec = build_vector(_prices(80), _cfg())
    events = vec[vec["2h_transition_event"].isin([EVENT_PEAK, EVENT_TROUGH])]
    assert len(events) > 0
    row = events.iloc[0]
    obs = pd.Timestamp(row["observation_time_utc"])
    avail = pd.Timestamp(row["feature_available_time_utc"])
    entry = pd.Timestamp(row["outcome_entry_time_utc"])
    assert avail == obs + pd.Timedelta(hours=1)
    assert entry == avail
    assert entry > obs
    oc = build_outcomes(_prices(80), _cfg())
    joined = vec.merge(oc[["time_bucket", "btc_close_at_entry", "return_1h"]], on="time_bucket")
    conf = joined[joined["2h_transition_event"].isin([EVENT_PEAK, EVENT_TROUGH])].iloc[0]
    assert conf["btc_close_at_entry"] != conf["btc_close_at_t"]
    assert pd.Timestamp(conf["outcome_entry_time_utc"]) > pd.Timestamp(conf["observation_time_utc"])


def test_oracle_columns_excluded_from_vector_and_allowlist():
    prices = _prices(80)
    cfg = _cfg()
    vec = build_vector(prices, cfg)
    oracle = build_oracle_table(prices, vec, cfg)
    assert "waiting_cost" in oracle.columns
    assert "oracle_extremum_time" in oracle.columns
    for col in (
        "oracle_extremum_time",
        "oracle_extremum_type",
        "return_from_oracle_extremum",
        "return_from_confirmation",
        "waiting_cost",
        "oracle_definition_version",
    ):
        assert col not in vec.columns
        assert is_oracle_column(col)
    features = select_model_feature_columns(vec)
    assert features
    assert not any(is_oracle_column(c) for c in features)
    assert not any("candidate_extremum" in c for c in features)
    assert not any(c.endswith("_confirmation_time") for c in features)
    assert not any(c.endswith("_transition_event") for c in features)
    leaked = vec.copy()
    leaked["waiting_cost"] = 0.0
    leaked["oracle_extremum_time"] = pd.NaT
    with pytest.raises(VectorSourceError, match="oracle"):
        select_model_feature_columns(leaked)


def test_every_oracle_column_is_rejected_by_guard():
    oracle_cols = [
        "oracle_extremum_time",
        "oracle_extremum_type",
        "return_from_oracle_extremum",
        "return_from_confirmation",
        "waiting_cost",
        "oracle_definition_version",
        "return_while_waiting",
    ]
    with pytest.raises(VectorSourceError, match="oracle"):
        assert_no_oracle_columns(oracle_cols)


def test_infer_refuses_oracle_file_and_columns(tmp_path):
    p = tmp_path / ORACLE_FILE_NAME
    pd.DataFrame(
        {
            "vector_source": ["price_ols_proxy"],
            "vector_version": ["price_ols_proxy_v1"],
            "oracle_extremum_time": ["2024-01-01T02:00:00Z"],
            "waiting_cost": [1.0],
        }
    ).to_csv(p, index=False)
    with pytest.raises(VectorSourceError, match="oracle"):
        load_inference_vector(p)
    q = tmp_path / "vector_price_ols_proxy_v1.csv"
    pd.DataFrame(
        {
            "vector_source": ["price_ols_proxy"],
            "vector_version": ["price_ols_proxy_v1"],
            "waiting_cost": [0.5],
            "row_vector_valid": [True],
        }
    ).to_csv(q, index=False)
    with pytest.raises(VectorSourceError, match="oracle"):
        load_inference_vector(q)


def test_vector_state_null_until_first_confirmation():
    vec = build_vector(_prices(80), _cfg())
    first = vec["2h_transition_event"].ne(EVENT_NONE)
    assert first.any()
    i = int(np.argmax(first.to_numpy()))
    assert vec["2h_transition_state_raw"].iloc[:i].isna().all()
    assert (vec["2h_transition_event"].iloc[:i] == EVENT_NONE).all()
    assert vec["2h_feature_valid"].iloc[:i].any()


def test_detector_delay_is_config_not_fit_on_full_sample():
    cfg = _cfg()
    vec = build_vector(_prices(80), cfg)
    assert (vec["confirmation_delay_hours"] == 2).all()
    assert vec["detector_version"].iloc[0] == "causal_d_extrema_delay2_prom1p0_durf0p25_v2"
    cfg["vector"]["confirmation_delay_hours"] = 3
    vec3 = build_vector(_prices(80), cfg)
    assert (vec3["confirmation_delay_hours"] == 3).all()
    assert "delay3" in str(vec3["detector_version"].iloc[0])


def test_canonical_transition_is_d_not_v():
    vec = build_vector(_prices(120), _cfg())
    d = vec.set_index(pd.to_datetime(vec["time_bucket"], utc=True))["12h_derivative_raw"]
    d_trans = confirm_local_extrema(
        d,
        delay_hours=2,
        min_prominence_mad=1.0,
        min_duration_hours=3,
        prominence_window_hours=168,
        prominence_min_obs=24,
    )
    got = vec["12h_transition_state_raw"].to_numpy(dtype=float)
    want_d = d_trans["transition_state_raw"].to_numpy(dtype=float)
    np.testing.assert_equal(got, want_d)
    assert "12h_velocity_transition_state_raw" in vec.columns
    assert "12h_velocity_transition_event" in vec.columns


def test_allowlist_keeps_one_transition_encoding():
    vec = build_vector(_prices(80), _cfg())
    features = select_model_feature_columns(vec)
    assert "2h_transition_state_raw" in features
    assert "2h_hours_since_confirmation" in features
    assert "2h_transition_raw" not in features
    assert "2h_transition_direction" not in features
    assert "2h_hours_since_turn" not in features
    assert not any("velocity_transition" in c for c in features)


def test_min_duration_blocks_fast_opposite_turn():
    series = _peak_series()
    blocked = confirm_local_extrema(series, delay_hours=2, min_duration_hours=10)
    idx = series.index
    assert blocked["transition_event"].loc[idx[4]] == EVENT_PEAK
    assert blocked["transition_event"].loc[idx[8]] == EVENT_NONE
    assert blocked["transition_state_raw"].loc[idx[8]] == TRANSITION_RAW_DOWN_AFTER_PEAK
    open_ = confirm_local_extrema(series, delay_hours=2, min_duration_hours=0)
    assert open_["transition_event"].loc[idx[8]] == EVENT_TROUGH


def test_small_reversal_fails_hysteresis():
    idx = pd.date_range("2024-01-01", periods=40, freq="h", tz="UTC")
    values = np.ones(40, dtype=float)
    values[:10] = np.linspace(0.0, 8.0, 10)
    values[10:] = np.linspace(7.99, 7.90, 30)
    s = pd.Series(values, index=idx)
    out = confirm_local_extrema(
        s,
        delay_hours=2,
        min_prominence_mad=1.0,
        min_duration_hours=0,
        prominence_window_hours=12,
        prominence_min_obs=8,
    )
    assert (out["transition_event"] == EVENT_PEAK).sum() <= 1
    assert (out["transition_event"] == EVENT_TROUGH).sum() == 0
