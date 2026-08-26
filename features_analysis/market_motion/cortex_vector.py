"""cortex_market_motion_v1: T/Z/V/A/Q from continuous Cortex D. No OLS fallback."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import (
    TRANSITION_RAW_UP_AFTER_TROUGH,
    VECTOR_SOURCE_CORTEX_MOTION,
    VECTOR_VERSION_CORTEX_MOTION,
)
from .exceptions import CortexSourceError, VectorSourceError
from .feature_allowlist import assert_vector_has_no_oracle
from .manifest import feature_configuration_hash
from .quality import bounded_ffill
from .source_collapse import SELECTED_VALUE_RULE
from .transition import confirm_local_extrema, detector_version, horizon_min_duration_hours
from .vector_engine import _pick, _timing, robust_z

CORTEX_HORIZON_HOURS = {
    "2h": 2,
    "4h": 4,
    "8h": 8,
    "12h": 12,
    "24h": 24,
    "48h": 48,
    "7d": 168,
}

FORBIDDEN_OLS_HORIZONS = frozenset({"7h"})
SOURCE_QUERY_VERSION = "cortex_continuous_d_prod_a_preferred_v1"


def cortex_horizon_names(cfg: dict[str, Any]) -> list[str]:
    cm = cfg.get("cortex_motion") or {}
    horizons = cm.get("horizons") or {}
    names = list(horizons.keys())
    if not names:
        raise CortexSourceError("cortex_motion.horizons is required")
    bad = [h for h in names if h in FORBIDDEN_OLS_HORIZONS]
    if bad:
        raise CortexSourceError(
            f"cortex_market_motion_v1 must not use OLS-only horizon names {bad}; "
            "use 8h (deriv8h), not 7h"
        )
    if "8h" not in names:
        raise CortexSourceError("cortex_market_motion_v1 requires horizon 8h (not 7h)")
    return names


def _invalidation_reason(
    *,
    src_valid: pd.Series,
    d: pd.Series,
    z_ready: pd.Series,
    vel: pd.Series,
    conflict: pd.Series,
) -> pd.Series:
    """Routing-label changes are not an invalidation reason."""
    reasons = pd.Series("", index=d.index, dtype=object)
    reasons = reasons.mask(conflict.fillna(False) & d.isna(), "source_conflict")
    reasons = reasons.mask(~(conflict.fillna(False) & d.isna()) & ~src_valid.fillna(False), "stale_or_missing_cortex_d")
    reasons = reasons.mask(src_valid & d.isna() & ~conflict.fillna(False), "cortex_d_undefined")
    reasons = reasons.mask(src_valid & d.notna() & ~z_ready.fillna(False), "normalization_warmup")
    reasons = reasons.mask(src_valid & d.notna() & z_ready.fillna(False) & vel.isna(), "velocity_undefined")
    return reasons.where(reasons != "", other="")


def _map_z(d: pd.Series, z_win: int, z_min: int, z_clip: float):
    """Robust z over the full series. Routing labels do not restart warmup."""
    return robust_z(d, z_win, z_min, z_clip)


def build_cortex_vector(
    cortex_hourly: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    time_col: str = "time_bucket",
    prices: pd.DataFrame | None = None,
    price_col: str = "close_price",
) -> pd.DataFrame:
    """Build cortex_market_motion_v1 from hourly Cortex D columns. Never uses OLS D."""
    names = cortex_horizon_names(cfg)
    missing = [h for h in names if h not in cortex_hourly.columns]
    if missing:
        raise CortexSourceError(
            f"required Cortex D columns missing: {missing}. OLS will not be substituted."
        )
    vcfg = cfg.get("vector") or {}
    cm = cfg.get("cortex_motion") or {}
    work = cortex_hourly.copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=True)
    work = work.dropna(subset=[time_col]).sort_values(time_col)
    work = work.drop_duplicates(time_col, keep="last").set_index(time_col)
    if work.empty:
        raise CortexSourceError("no Cortex hourly rows")
    work = work.reindex(pd.date_range(work.index.min(), work.index.max(), freq="h", tz="UTC"))

    max_stale = int(cm.get("max_staleness_hours", vcfg.get("staleness_hours", {}).get("cortex_d", 2)))
    delay = int(_pick(vcfg, "confirmation_delay_hours", default=2))
    prom_k = float(_pick(vcfg, "min_prominence_mad", default=1.0))
    dur_frac = float(_pick(vcfg, "min_duration_horizon_fraction", default=0.25))
    prom_win = int(_pick(vcfg, "prominence_mad_window_hours", default=168))
    prom_min = int(_pick(vcfg, "prominence_mad_min_obs", default=24))
    z_win = int(_pick(vcfg, "z_window_hours"))
    z_min = int(_pick(vcfg, "z_min_obs"))
    z_clip = float(_pick(vcfg, "z_clip"))
    delta = int(_pick(vcfg, "velocity_delta_hours", default=1))
    feature_lag = int(cm.get("feature_available_lag_hours", _timing(cfg)[0]))
    decision_lag = int(_timing(cfg)[1])
    det_ver = detector_version(
        delay_hours=delay,
        min_prominence_mad=prom_k,
        min_duration_horizon_fraction=dur_frac,
    )
    query_ver = str(cm.get("source_query_version", SOURCE_QUERY_VERSION))

    px_f = pd.Series(np.nan, index=work.index)
    px_age = pd.Series(np.nan, index=work.index)
    px_valid = pd.Series(False, index=work.index)
    if prices is not None and not prices.empty:
        pr = prices.copy()
        pr[time_col] = pd.to_datetime(pr[time_col], utc=True)
        pr = pr.dropna(subset=[time_col]).drop_duplicates(time_col, keep="last").set_index(time_col)
        aligned = pd.to_numeric(pr.reindex(work.index)[price_col], errors="coerce")
        pmax = int((vcfg.get("staleness_hours") or {}).get("crypto_last_price", 2))
        px_f, px_age, px_valid = bounded_ffill(aligned, pmax)

    idx = work.index
    avail = idx + pd.Timedelta(hours=feature_lag)
    decision = avail + pd.Timedelta(hours=decision_lag)
    cols: dict[str, Any] = {
        "observation_time_utc": idx,
        "feature_available_time_utc": avail,
        "decision_time_utc": decision,
        "outcome_entry_time_utc": decision,
        "asset": cfg.get("asset", "BTCUSDT"),
        "sampling_frequency": cfg.get("sampling_frequency", "1h"),
        "vector_source": VECTOR_SOURCE_CORTEX_MOTION,
        "vector_version": VECTOR_VERSION_CORTEX_MOTION,
        "source_query_version": query_ver,
        "feature_configuration_hash": feature_configuration_hash(cfg),
        "detector_version": det_ver,
        "confirmation_delay_hours": delay,
        "min_prominence_mad": prom_k,
        "min_duration_horizon_fraction": dur_frac,
        "timing_certainty": str(cm.get("timing_certainty", "uncertain_recording_rule")),
        "source_semantic_certainty": str(
            cm.get("source_semantic_certainty", "owner_confirmed_routing_same_feature")
        ),
        "selected_value_rule": str(cm.get("collapse", {}).get("rule", SELECTED_VALUE_RULE)) if isinstance(cm.get("collapse"), dict) else SELECTED_VALUE_RULE,
        "btc_close_at_t": px_f,
        "price_source_age_hours": px_age,
        "price_feature_valid": px_valid,
        "price_max_staleness_hours": int((vcfg.get("staleness_hours") or {}).get("crypto_last_price", 2)),
        "feature_available_lag_hours": feature_lag,
        "decision_latency_hours": decision_lag,
    }

    for name in names:
        raw_d = pd.to_numeric(work[name], errors="coerce")
        if raw_d.notna().sum() == 0:
            raise CortexSourceError(
                f"Cortex series {name!r} is entirely missing. OLS will not be substituted."
            )
        conflict = (
            work[f"{name}_source_conflict"].fillna(False).astype(bool)
            if f"{name}_source_conflict" in work.columns
            else pd.Series(False, index=work.index)
        )
        cluster = (
            work[f"{name}_source_cluster"].astype("string")
            if f"{name}_source_cluster" in work.columns
            else pd.Series("", index=work.index, dtype="string")
        )
        replica = (
            work[f"{name}_source_replica"].astype("string")
            if f"{name}_source_replica" in work.columns
            else pd.Series("", index=work.index, dtype="string")
        )
        n_cand = (
            pd.to_numeric(work[f"{name}_n_candidates"], errors="coerce")
            if f"{name}_n_candidates" in work.columns
            else pd.Series(np.where(raw_d.notna(), 1, np.nan), index=work.index)
        )
        value_disagree = (
            work[f"{name}_source_value_disagree"].fillna(False).astype(bool)
            if f"{name}_source_value_disagree" in work.columns
            else pd.Series(False, index=work.index)
        )
        boundary = (
            work[f"{name}_source_boundary"].fillna(False).astype(bool)
            if f"{name}_source_boundary" in work.columns
            else pd.Series(False, index=work.index)
        )
        seg_ids = (
            pd.to_numeric(work[f"{name}_source_segment_id"], errors="coerce").fillna(-1).astype(int)
            if f"{name}_source_segment_id" in work.columns
            else pd.Series(0, index=work.index, dtype=int)
        )
        d_f, d_age, d_valid = bounded_ffill(raw_d, max_stale, barrier=conflict)
        z, z_ready = _map_z(d_f, z_win, z_min, z_clip)
        vel = (z - z.shift(delta)) / delta
        acc = (vel - vel.shift(delta)) / delta
        hours = int((cm.get("horizons") or {}).get(name, {}).get("horizon_hours", CORTEX_HORIZON_HOURS[name]))
        min_dur = horizon_min_duration_hours(hours, delay, dur_frac)
        detect_kw = dict(
            delay_hours=delay,
            min_prominence_mad=prom_k,
            min_duration_hours=min_dur,
            prominence_window_hours=prom_win,
            prominence_min_obs=prom_min,
        )
        trans = confirm_local_extrema(d_f, **detect_kw)
        vel_trans = confirm_local_extrema(vel, **detect_kw)
        src_valid = d_valid & d_f.notna()
        valid = src_valid & z_ready & vel.notna()
        reasons = _invalidation_reason(
            src_valid=src_valid,
            d=d_f,
            z_ready=z_ready,
            vel=vel,
            conflict=conflict,
        )
        p = f"{name}_"
        t_raw = trans["transition_state_raw"]
        hours_since = trans["hours_since_confirmation"]
        cols[f"{p}derivative_raw"] = d_f
        cols[f"{p}position_z"] = z
        cols[f"{p}velocity"] = vel
        cols[f"{p}acceleration"] = acc
        cols[f"{p}va_stable"] = True
        cols[f"{p}resolution_class"] = "cortex_source"
        cols[f"{p}horizon_hours"] = hours
        cols[f"{p}transition_state_raw"] = t_raw
        cols[f"{p}transition_raw"] = t_raw
        cols[f"{p}transition_direction"] = trans["transition_direction"]
        cols[f"{p}transition_event"] = trans["transition_event"]
        cols[f"{p}confirmation_time"] = trans["confirmation_time"]
        cols[f"{p}hours_since_confirmation"] = hours_since
        cols[f"{p}hours_since_turn"] = hours_since
        cols[f"{p}candidate_extremum_time"] = trans["candidate_extremum_time"]
        cols[f"{p}min_duration_hours"] = min_dur
        cols[f"{p}velocity_transition_state_raw"] = vel_trans["transition_state_raw"]
        cols[f"{p}velocity_transition_event"] = vel_trans["transition_event"]
        cols[f"{p}velocity_hours_since_confirmation"] = vel_trans["hours_since_confirmation"]
        cols[f"{p}source_present"] = raw_d.notna()
        cols[f"{p}source_age_hours"] = d_age
        cols[f"{p}normalization_ready"] = z_ready
        cols[f"{p}feature_valid"] = valid
        cols[f"{p}invalidation_reason"] = reasons.where(~valid, "")
        cols[f"{p}promql"] = str((cm.get("horizons") or {}).get(name, {}).get("query", ""))
        cols[f"{p}source_cluster"] = cluster
        cols[f"{p}source_replica"] = replica
        cols[f"{p}n_candidates"] = n_cand
        cols[f"{p}source_conflict"] = conflict
        cols[f"{p}source_value_disagree"] = value_disagree
        cols[f"{p}source_boundary"] = boundary
        cols[f"{p}source_segment_id"] = seg_ids
        cols[f"{p}selected_value_rule"] = SELECTED_VALUE_RULE

    if any(c.startswith("7h_") for c in cols):
        raise VectorSourceError("cortex vector must not contain 7h_ columns")

    n_up = sum(
        (cols[f"{h}_transition_state_raw"] == TRANSITION_RAW_UP_AFTER_TROUGH) & cols[f"{h}_feature_valid"]
        for h in names
    )
    n_valid = sum(cols[f"{h}_feature_valid"].astype(int) for h in names)
    cols["cross_n_horizons_up"] = n_up
    cols["cross_n_horizons_valid"] = n_valid
    cols["row_vector_valid"] = n_valid == len(names)
    out = pd.DataFrame(cols, index=idx)
    out.index.name = time_col
    result = out.reset_index()
    assert_vector_has_no_oracle(result, context="build_cortex_vector: ")
    return result
