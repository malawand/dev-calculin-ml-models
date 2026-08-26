"""PROD-A preferred collapse, provenance, and continuous routing across labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_motion.cortex_tape import apply_live_cluster_selector, hourly_from_raw
from market_motion.cortex_vector import build_cortex_vector
from market_motion.exceptions import CortexSourceError
from market_motion.feature_allowlist import select_model_feature_columns
from market_motion.quality import bounded_ffill
from market_motion.source_collapse import (
    CANONICAL_CLUSTER,
    SELECTED_VALUE_RULE,
    collapse_timestamp_group,
)
from market_motion.source_guard import timing_skip_columns
from market_motion.tests.test_cortex_vector import (
    CORTEX_H,
    _assert_row_close,
    _cortex_cfg,
    _hourly,
    _prices,
)


def test_identical_duplicates_are_deterministic():
    rows = pd.DataFrame(
        {
            "value": [0.5, 0.5 + 1e-15],
            "cluster": ["PROD-A-crypto-metrics", "PROD-B-crypto-metrics"],
            "replica": ["r1", "r2"],
        }
    )
    rec = collapse_timestamp_group(rows)
    assert rec["source_conflict"] is False
    assert rec["value"] == pytest.approx(0.5)
    assert rec["selected_value_rule"] == SELECTED_VALUE_RULE
    assert rec["source_cluster"] == "PROD-A-crypto-metrics"


def test_prod_a_wins_when_present():
    rows = pd.DataFrame(
        {
            "value": [-0.02, 0.018],
            "cluster": ["PROD-A-crypto-metrics", "PROD-B-crypto-metrics"],
            "replica": ["PROD-A-10.2.20.100", "PROD-B-10.2.20.100"],
        }
    )
    rec = collapse_timestamp_group(rows)
    assert rec["source_cluster"] == CANONICAL_CLUSTER
    assert rec["value"] == pytest.approx(-0.02)
    assert rec["value"] != max(-0.02, 0.018)
    assert rec["source_conflict"] is False
    assert bool(rec["source_value_disagree"]) is True


def test_signed_disagreement_is_never_max():
    rows = pd.DataFrame(
        {
            "value": [-0.02, 0.018],
            "cluster": ["PROD-crypto-metrics", "prod-prom-cluster-1"],
            "replica": ["", ""],
        }
    )
    rec = collapse_timestamp_group(rows)
    assert rec["source_conflict"] is False
    assert rec["source_cluster"] == "PROD-crypto-metrics"
    assert rec["value"] == pytest.approx(-0.02)
    assert rec["value"] != max(-0.02, 0.018)


def test_historical_fallback_when_prod_a_absent():
    rows = pd.DataFrame(
        {
            "value": [0.22, 0.11],
            "cluster": ["prod-prom-cluster-1", "prod-calculin-tokyo-binance-scraper"],
            "replica": ["", ""],
        }
    )
    rec = collapse_timestamp_group(rows)
    assert rec["source_cluster"] == "prod-calculin-tokyo-binance-scraper"
    assert rec["value"] == pytest.approx(0.11)
    assert rec["value"] != max(0.22, 0.11)


def test_unapproved_only_is_true_conflict():
    rows = pd.DataFrame(
        {
            "value": [0.4, 0.5],
            "cluster": ["DEV-calculin-tokyo-binance-scraper-A", "unknown-cluster"],
            "replica": ["", ""],
        }
    )
    rec = collapse_timestamp_group(rows)
    assert rec["source_conflict"] is True
    assert np.isnan(rec["value"])
    assert rec["value"] != max(0.4, 0.5)


def test_hourly_selects_prod_a_and_records_disagreement():
    raw = pd.DataFrame(
        {
            "timestamp": [1_700_000_000, 1_700_000_000, 1_700_003_600],
            "value": [1.0, 3.0, 2.0],
            "series_name": ["8h", "8h", "8h"],
            "cluster": ["PROD-A-crypto-metrics", "PROD-B-crypto-metrics", "PROD-A-crypto-metrics"],
            "replica": ["rA", "rB", "rA"],
        }
    )
    hourly = hourly_from_raw(raw, ["8h"])
    assert hourly["8h"].iloc[0] == pytest.approx(1.0)
    assert hourly["8h"].iloc[0] != 3.0
    assert hourly["8h_source_cluster"].iloc[0] == "PROD-A-crypto-metrics"
    assert not bool(hourly["8h_source_conflict"].iloc[0])
    assert bool(hourly["8h_source_value_disagree"].iloc[0])
    assert hourly["8h"].iloc[1] == pytest.approx(2.0)
    assert hourly["8h_selected_value_rule"].iloc[0] == SELECTED_VALUE_RULE


def test_ffill_does_not_cross_optional_barrier():
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    s = pd.Series([1.0, np.nan, np.nan, np.nan, 5.0], index=idx)
    barrier = pd.Series([False, False, True, False, False], index=idx)
    filled, _, valid = bounded_ffill(s, max_hours=3, barrier=barrier)
    assert filled.iloc[1] == 1.0
    assert pd.isna(filled.iloc[2])
    assert not valid.iloc[2]


def test_true_conflict_is_not_forward_filled():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    times = pd.date_range(start, periods=6, freq="h", tz="UTC")
    rows = []
    for i, ts in enumerate(times):
        if i == 2:
            cluster, value = "unknown-cluster", 9.0
        else:
            cluster, value = "PROD-A-crypto-metrics", float(i)
        rows.append(
            {
                "timestamp": int(ts.timestamp()),
                "value": value,
                "series_name": "8h",
                "cluster": cluster,
                "replica": "",
            }
        )
    hourly = hourly_from_raw(pd.DataFrame(rows), ["8h"])
    assert bool(hourly["8h_source_conflict"].iloc[2])
    assert pd.isna(hourly["8h"].iloc[2])
    for name in CORTEX_H:
        if name != "8h":
            hourly[name] = hourly["8h"]
    vec = build_cortex_vector(hourly, _cortex_cfg())
    row = vec.iloc[2]
    assert pd.isna(row["8h_derivative_raw"])
    assert row["8h_invalidation_reason"] == "source_conflict"
    assert not bool(row["8h_feature_valid"])
    assert vec.iloc[1]["8h_derivative_raw"] == pytest.approx(1.0)
    assert vec.iloc[3]["8h_derivative_raw"] == pytest.approx(3.0)


def test_missing_hours_still_obey_staleness():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    times = list(pd.date_range(start, periods=10, freq="h", tz="UTC"))
    keep = times[:3] + times[8:]
    raw = pd.DataFrame(
        {
            "timestamp": [int(t.timestamp()) for t in keep],
            "value": [0.1, 0.2, 0.3, 0.4, 0.5],
            "series_name": ["8h"] * 5,
            "cluster": ["PROD-A-crypto-metrics"] * 5,
            "replica": [""] * 5,
        }
    )
    hourly = hourly_from_raw(raw, ["8h"])
    for name in CORTEX_H:
        if name != "8h":
            hourly[name] = hourly["8h"]
    cfg = _cortex_cfg()
    cfg["cortex_motion"]["max_staleness_hours"] = 2
    vec = build_cortex_vector(hourly, cfg)
    gap = pd.Timestamp("2024-01-01 06:00:00", tz="UTC")
    row = vec.loc[pd.to_datetime(vec["time_bucket"], utc=True) == gap].iloc[0]
    assert pd.isna(row["8h_derivative_raw"]) or row["8h_source_age_hours"] > 2 or not bool(row["8h_feature_valid"])
    assert row["8h_invalidation_reason"] in {"stale_or_missing_cortex_d", "cortex_d_undefined", "normalization_warmup", "velocity_undefined"}


def test_routing_label_change_does_not_reset_features_or_detector():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    times = pd.date_range(start, periods=80, freq="h", tz="UTC")
    rows = []
    for i, ts in enumerate(times):
        cluster = "prod-prom-cluster-1" if i < 40 else "prod-calculin-tokyo-binance-scraper"
        rows.append(
            {
                "timestamp": int(ts.timestamp()),
                "value": np.sin(i / 11.0),
                "series_name": "8h",
                "cluster": cluster,
                "replica": "",
            }
        )
    raw = pd.DataFrame(rows)
    hourly = hourly_from_raw(raw, ["8h"])
    assert bool(hourly["8h_source_boundary"].iloc[40])
    for name in CORTEX_H:
        if name != "8h":
            hourly[name] = np.sin(np.arange(len(hourly)) / 19.0)
    cfg = _cortex_cfg()
    vec = build_cortex_vector(hourly, cfg)
    before = vec.iloc[39]
    at = vec.iloc[40]
    after = vec.iloc[41]
    assert before["8h_normalization_ready"]
    assert at["8h_normalization_ready"]
    assert after["8h_normalization_ready"]
    assert at["8h_invalidation_reason"] != "source_boundary"
    assert at["8h_invalidation_reason"] != "normalization_warmup"
    assert bool(at["8h_feature_valid"])
    if pd.notna(before["8h_hours_since_confirmation"]) and pd.notna(at["8h_hours_since_confirmation"]):
        assert at["8h_hours_since_confirmation"] == before["8h_hours_since_confirmation"] + 1


def test_allowlist_excludes_provenance_metadata():
    hourly = _hourly(80)
    hourly["8h_source_cluster"] = "PROD-A-crypto-metrics"
    hourly["8h_source_replica"] = "PROD-A-10.2.20.100"
    hourly["8h_n_candidates"] = 1
    hourly["8h_source_conflict"] = False
    hourly["8h_source_value_disagree"] = False
    hourly["8h_source_boundary"] = False
    hourly["8h_source_segment_id"] = 0
    hourly["8h_selected_value_rule"] = SELECTED_VALUE_RULE
    vec = build_cortex_vector(hourly, _cortex_cfg())
    features = select_model_feature_columns(vec, CORTEX_H)
    banned = (
        "source_cluster",
        "source_conflict",
        "source_value_disagree",
        "source_boundary",
        "source_segment_id",
        "n_candidates",
        "source_replica",
        "selected_value_rule",
    )
    assert not any(any(c.endswith(f"_{b}") or c == b for b in banned) for c in features)
    assert "8h_transition_state_raw" in features
    assert (vec["8h_source_cluster"] == "PROD-A-crypto-metrics").all()


def test_prefix_parity_with_provenance():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    times = pd.date_range(start, periods=90, freq="h", tz="UTC")
    rows = []
    for i, ts in enumerate(times):
        cluster = "prod-prom-cluster-1" if i < 40 else "PROD-A-crypto-metrics"
        rows.append(
            {
                "timestamp": int(ts.timestamp()),
                "value": np.sin(i / 11.0),
                "series_name": "8h",
                "cluster": cluster,
                "replica": "",
            }
        )
    raw = pd.DataFrame(rows)
    hourly = hourly_from_raw(raw, ["8h"])
    for name in CORTEX_H:
        if name != "8h":
            hourly[name] = np.sin(np.arange(len(hourly)) / 19.0)
    cfg = _cortex_cfg()
    full = build_cortex_vector(hourly, cfg)
    prefix = build_cortex_vector(hourly.iloc[:70].copy(), cfg)
    skip = timing_skip_columns()
    cols = [c for c in full.columns if c not in skip]
    _assert_row_close(prefix[cols].iloc[-1], full[cols].iloc[69], cols)


def test_live_fetch_selector_injects_prod_a():
    q = 'job:crypto_last_price:deriv8h{symbol="BTCUSDT"}'
    out = apply_live_cluster_selector(q, "PROD-A-crypto-metrics")
    assert 'cluster="PROD-A-crypto-metrics"' in out
    assert "symbol" in out


def test_missing_horizon_still_not_ols():
    hourly = _hourly().drop(columns=["8h"])
    with pytest.raises(CortexSourceError, match="OLS will not be substituted"):
        build_cortex_vector(hourly, _cortex_cfg(), prices=_prices())
