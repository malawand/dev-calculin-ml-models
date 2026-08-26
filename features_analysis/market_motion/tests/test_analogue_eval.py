"""Walk-forward analogue evaluation: chronology, scoring, closed holdouts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_motion.analogue import match_query, prepare_analogue_frame, query_analogues
from market_motion.analogue_eval import (
    WALKFORWARD_DISCLAIMER,
    brier_score,
    daily_query_times,
    empirical_crps,
    run_walkforward_eval,
)
from market_motion.exceptions import VectorSourceError
from market_motion.periods import RETROSPECTIVE_HOLDOUT_START_UTC
from market_motion.tests.test_analogue import _outcomes, _vec


def test_empirical_crps_point_mass_is_mae():
    assert empirical_crps(np.array([1.0, 1.0, 1.0]), 1.0) == pytest.approx(0.0)
    assert empirical_crps(np.array([0.0, 0.0]), 1.0) == pytest.approx(1.0)


def test_empirical_crps_two_point_sample():
    # sample {0, 2}, y=1: E|X-y|=1, E|X-X'|=1, CRPS=1-0.5=0.5
    assert empirical_crps(np.array([0.0, 2.0]), 1.0) == pytest.approx(0.5)


def test_brier_perfect_and_wrong():
    assert brier_score(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(0.0)
    assert brier_score(np.array([0.0, 1.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)


def test_daily_schedule_is_fixed_hour_and_skips_holdout():
    times = pd.date_range("2024-12-01", "2026-02-01", freq="h", tz="UTC")
    qs = daily_query_times(
        times,
        query_hour_utc=0,
        warmup_hours=24,
        last_query=pd.Timestamp("2026-01-13 22:00:00", tz="UTC"),
    )
    assert qs
    assert all(t.hour == 0 for t in qs)
    assert all(t < RETROSPECTIVE_HOLDOUT_START_UTC for t in qs)
    assert qs[0] >= pd.Timestamp("2024-12-02 00:00:00", tz="UTC")


def test_match_uncond_pool_is_purged_and_excludes_query_return():
    vec = _vec(900, start="2024-01-01")
    oc = _outcomes(vec)
    frame, features = prepare_analogue_frame(vec, oc)
    q = pd.Timestamp("2024-01-28 12:00:00", tz="UTC")
    match = match_query(frame, q, features)
    assert match["abstain"] is False
    cutoff = q - pd.Timedelta(hours=26)
    assert match["cutoff_utc"] == cutoff
    rec = query_analogues(frame, q, features)
    assert rec["n_neighbors"] == match["n_neighbors"]
    assert rec["median_distance"] == pytest.approx(match["median_distance"])
    assert np.isfinite(match["actual_return"])
    assert match["n_neighbors"] <= match["n_pool"]
    assert match["n_pool"] == len(match["uncond_returns"])


def test_walkforward_refuses_holdout_query():
    vec = _vec(80, start="2026-03-01")
    oc = _outcomes(vec)
    frame, features = prepare_analogue_frame(vec, oc)
    with pytest.raises(VectorSourceError, match="holdout"):
        query_analogues(frame, pd.Timestamp("2026-03-01 00:00:00", tz="UTC"), features)


def test_walkforward_scores_development_only_and_past_uncond():
    vec = _vec(2000, start="2023-11-01")
    oc = _outcomes(vec)
    cfg = {"outcomes": {"round_trip_cost_pct": 0.04}, "analogue": {"k": 40, "bootstrap_draws": 40}}
    rec = run_walkforward_eval(vec, oc, cfg=cfg, query_hour_utc=0)
    assert WALKFORWARD_DISCLAIMER.split("\n")[0] in rec["disclaimer"]
    assert rec["n_evaluated"] > 0
    assert rec["vector_source"] == "cortex_market_motion"
    times = [pd.Timestamp(r["query_time_utc"]) for r in rec["queries"]]
    assert all(t.hour == 0 for t in times)
    assert all(t < RETROSPECTIVE_HOLDOUT_START_UTC for t in times)
    assert rec["mean_crps_analogue"] >= 0
    assert rec["mean_crps_uncond"] >= 0
    assert 0 <= rec["pct_queries_analogue_beats_uncond_crps"] <= 1
    for row in rec["queries"]:
        assert row["n_uncond"] >= row["n_neighbors"]
        assert np.isfinite(row["actual_return"])


def test_walkforward_keeps_ols_identity_separate():
    vec = _vec(2000, start="2023-11-01", source="price_ols_proxy", version="price_ols_proxy_v1")
    oc = _outcomes(vec)
    rec = run_walkforward_eval(vec, oc, cfg={"analogue": {"bootstrap_draws": 20}})
    assert rec["vector_source"] == "price_ols_proxy"
    assert rec["n_evaluated"] > 0


def _oc_all_horizons(vec):
    oc = _outcomes(vec)
    for h in (4, 8, 12):
        oc[f"return_{h}h"] = oc["return_24h"] * (h / 24.0)
    return oc


def test_documented_purge_lags():
    from market_motion.analogue import outcome_purge_cutoff, outcome_purge_lag_hours

    q = pd.Timestamp("2025-06-01 00:00:00", tz="UTC")
    expected = {4: 6, 8: 10, 12: 14, 24: 26}
    for h, lag in expected.items():
        assert outcome_purge_lag_hours(1, h) == lag
        assert outcome_purge_cutoff(q, entry_lag_hours=1, outcome_horizon_hours=h) == q - pd.Timedelta(hours=lag)


def test_horizon_purge_is_not_reused_from_24h():
    vec = _vec(900, start="2024-01-01")
    oc = _oc_all_horizons(vec)
    frame, features = prepare_analogue_frame(
        vec, oc, outcome_cols=["return_4h", "return_8h", "return_12h", "return_24h"]
    )
    q = pd.Timestamp("2024-01-28 12:00:00", tz="UTC")
    m4 = match_query(frame, q, features, outcome_col="return_4h", outcome_horizon_hours=4)
    m8 = match_query(frame, q, features, outcome_col="return_8h", outcome_horizon_hours=8)
    m12 = match_query(frame, q, features, outcome_col="return_12h", outcome_horizon_hours=12)
    m24 = match_query(frame, q, features, outcome_col="return_24h", outcome_horizon_hours=24)
    assert m4["purge_lag_hours"] == 6
    assert m8["purge_lag_hours"] == 10
    assert m12["purge_lag_hours"] == 14
    assert m24["purge_lag_hours"] == 26
    assert m4["n_pool"] > m8["n_pool"] > m12["n_pool"] > m24["n_pool"]
    for rec in (m4, m8, m12, m24):
        for nb in rec["neighbors_preview"]:
            assert pd.Timestamp(nb["time_utc"]) <= rec["cutoff_utc"]
            assert pd.Timestamp(nb["time_utc"]) < q


def test_skill_score_formula():
    from market_motion.analogue_eval import skill_score

    assert skill_score(0.8, 1.0) == pytest.approx(0.2)
    assert skill_score(1.2, 1.0) == pytest.approx(-0.2)


def test_multi_horizon_shares_query_dates_and_keeps_24h_target_label():
    from market_motion.analogue_eval import HORIZON_DISCLAIMER, run_multi_horizon_eval

    vec = _vec(2000, start="2023-11-01")
    oc = _oc_all_horizons(vec)
    rec = run_multi_horizon_eval(vec, oc, cfg={"analogue": {"bootstrap_draws": 20}})
    assert HORIZON_DISCLAIMER.split("\n")[0] in rec["disclaimer"]
    assert rec["approved_primary_target"] == "return_24h"
    assert rec["purge_lags_hours"] == {"4h": 6, "8h": 10, "12h": 14, "24h": 26}
    q4 = [r["query_time_utc"] for r in rec["query_tables"]["4h"]]
    q24 = [r["query_time_utc"] for r in rec["query_tables"]["24h"]]
    assert q4 == q24
    assert rec["horizons"]["24h"]["n_evaluated"] == rec["horizons"]["4h"]["n_evaluated"]
    assert "crps_skill_score" in rec["horizons"]["8h"]
    times = [pd.Timestamp(t) for t in q24]
    assert all(t.hour == 0 for t in times)
    assert all(t < RETROSPECTIVE_HOLDOUT_START_UTC for t in times)
