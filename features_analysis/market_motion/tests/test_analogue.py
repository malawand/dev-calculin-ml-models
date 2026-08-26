"""POC analogue matching: chronology, purge, allowlist, no holdout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_motion.analogue import (
    ANALOGUE_VERSION,
    development_eligible_mask,
    mixed_l1_distance,
    prepare_analogue_frame,
    query_analogues,
    run_analogue_poc,
)
from market_motion.exceptions import VectorSourceError
from market_motion.feature_allowlist import select_model_feature_columns
from market_motion.periods import (
    DEVELOPMENT_START_UTC,
    RETROSPECTIVE_HOLDOUT_START_UTC,
    last_observation_with_complete_outcome,
)
from market_motion.representation_sanity import run_representation_sanity
from market_motion.transition import EVENT_NONE, EVENT_PEAK, EVENT_TROUGH


def _vec(n=80, start="2024-03-01", source="cortex_market_motion", version="cortex_market_motion_v1"):
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    t = np.arange(n, dtype=float)
    data = {
        "time_bucket": idx,
        "vector_source": source,
        "vector_version": version,
        "source_query_version": "cortex_continuous_d_prod_a_preferred_v1",
        "detector_version": "causal_d_extrema_delay2_prom1p0_durf0p25_v2",
        "feature_configuration_hash": "abc",
        "row_vector_valid": True,
    }
    for h, p in {"2h": 11, "8h": 31, "24h": 73}.items():
        data[f"{h}_derivative_raw"] = np.sin(t / p)
        data[f"{h}_position_z"] = np.sin(t / p) * 0.5
        data[f"{h}_velocity"] = np.cos(t / p)
        data[f"{h}_acceleration"] = -np.sin(t / p)
        data[f"{h}_transition_state_raw"] = (np.sin(t / p) > 0).astype(float)
        data[f"{h}_hours_since_confirmation"] = np.arange(n) % 9
        data[f"{h}_transition_event"] = EVENT_NONE
        data[f"{h}_confirmation_time"] = idx
        data[f"{h}_candidate_extremum_time"] = idx - pd.Timedelta(hours=2)
        data[f"{h}_feature_valid"] = True
        data[f"{h}_invalidation_reason"] = ""
        data[f"{h}_source_cluster"] = "PROD-A-crypto-metrics"
    data["2h_transition_event"] = np.where((t % 20) == 4, EVENT_PEAK, EVENT_NONE)
    data["8h_transition_event"] = np.where((t % 28) == 6, EVENT_TROUGH, EVENT_NONE)
    data["cross_n_horizons_up"] = 2
    data["cross_n_horizons_valid"] = 3
    return pd.DataFrame(data)


def _outcomes(vec, col="return_24h"):
    n = len(vec)
    rng = np.random.default_rng(0)
    return pd.DataFrame({"time_bucket": vec["time_bucket"], col: rng.normal(0.1, 1.0, size=n)})


def test_allowlist_is_the_distance_feature_set():
    vec = _vec()
    oc = _outcomes(vec)
    frame, features = prepare_analogue_frame(vec, oc)
    allowed = select_model_feature_columns(vec)
    assert set(features) <= set(allowed)
    assert "return_24h" in frame.columns
    assert "return_24h" not in features
    assert "2h_source_cluster" not in features
    assert "2h_invalidation_reason" not in features
    assert "time_bucket" not in features
    assert "2h_transition_state_raw" in features
    assert "2h_position_z" in features


def test_prepare_refuses_outcomes_on_vector():
    vec = _vec()
    vec["return_24h"] = 0.1
    with pytest.raises(VectorSourceError, match="outcome"):
        prepare_analogue_frame(vec, _outcomes(vec))


def test_prepare_refuses_mixed_sources():
    a = _vec(10)
    b = _vec(10, source="price_ols_proxy", version="price_ols_proxy_v1")
    mixed = pd.concat([a, b], ignore_index=True)
    with pytest.raises(VectorSourceError, match="mixed"):
        prepare_analogue_frame(mixed, _outcomes(mixed))


def test_candidates_are_strictly_earlier_and_purged():
    vec = _vec(900, start="2024-01-01")
    oc = _outcomes(vec)
    frame, features = prepare_analogue_frame(vec, oc)
    q = pd.Timestamp("2024-01-28 12:00:00", tz="UTC")
    rec = query_analogues(frame, q, features)
    assert rec["abstain"] is False
    qtime = pd.Timestamp(rec["query_time_utc"])
    cutoff = qtime - pd.Timedelta(hours=26)
    for nb in rec["neighbors"]:
        nt = pd.Timestamp(nb["time_utc"])
        assert nt < qtime
        assert nt <= cutoff


def test_holdout_query_is_refused():
    vec = _vec(10, start="2026-03-01")
    oc = _outcomes(vec)
    frame, features = prepare_analogue_frame(vec, oc)
    with pytest.raises(VectorSourceError, match="holdout"):
        query_analogues(frame, pd.Timestamp("2026-03-01 00:00:00", tz="UTC"), features)


def test_holdout_rows_are_not_eligible():
    vec = _vec(80, start="2025-12-01")
    oc = _outcomes(vec)
    frame, _ = prepare_analogue_frame(vec, oc)
    mask = development_eligible_mask(frame)
    times = pd.to_datetime(frame["time_bucket"], utc=True)
    assert not bool((mask & (times >= RETROSPECTIVE_HOLDOUT_START_UTC)).any())
    last = last_observation_with_complete_outcome()
    assert last < RETROSPECTIVE_HOLDOUT_START_UTC
    assert DEVELOPMENT_START_UTC < last


def test_distance_is_not_max_and_is_finite():
    vec = _vec(30)
    oc = _outcomes(vec)
    frame, features = prepare_analogue_frame(vec, oc)
    d = mixed_l1_distance(frame.iloc[20], frame.iloc[:18], features)
    assert len(d) == 18
    assert np.isfinite(d).all()
    assert (d >= 0).all()


def test_abstain_when_history_is_too_short():
    vec = _vec(40, start="2023-11-01")
    oc = _outcomes(vec)
    frame, features = prepare_analogue_frame(vec, oc)
    rec = query_analogues(frame, pd.Timestamp("2023-11-01 08:00:00", tz="UTC"), features)
    assert rec["abstain"] is True


def test_run_poc_keeps_cortex_identity():
    vec = _vec(900, start="2024-01-01")
    oc = _outcomes(vec)
    out = run_analogue_poc(vec, oc, query_times=[pd.Timestamp("2024-01-28 00:00:00", tz="UTC")])
    assert out["vector_source"] == "cortex_market_motion"
    assert out["analogue_version"] == ANALOGUE_VERSION
    assert "PROVISIONAL POC" in out["disclaimer"]
    assert out["unconditional_development_return_24h"]["n"] > 0


def test_representation_sanity_confirmation_time():
    vec = _vec(80)
    for h in ("2h", "8h", "24h"):
        ev = vec[f"{h}_transition_event"].isin([EVENT_PEAK, EVENT_TROUGH])
        vec.loc[ev, f"{h}_hours_since_confirmation"] = 0
        vec.loc[ev, f"{h}_confirmation_time"] = vec.loc[ev, "time_bucket"]
    rec = run_representation_sanity(vec)
    assert rec["confirmation_time_rule_holds_in_sample"] is True
    assert rec["detector_status"] in {"POC-frozen", "needs_review"}
