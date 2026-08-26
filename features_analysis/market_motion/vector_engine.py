"""Causal Vector v1: T, Z, V, A, quality per horizon. No future data.

This builder implements **price_ols_proxy_v1** only: causal OLS on log close.
That is multi-horizon **price momentum**, not Cortex derivative-curve motion.

``cortex_market_motion_v1`` is reserved and raises. OLS is never used as a
fallback when Cortex continuous derivatives are missing.

Transition state becomes active at confirmation time, not at the hindsight
extremum. See ``transition.confirm_local_extrema``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import (
    TRANSITION_RAW_UP_AFTER_TROUGH,
    VECTOR_SOURCE_CORTEX_MOTION,
    VECTOR_SOURCE_OLS,
    VECTOR_VERSION_CORTEX_MOTION,
    VECTOR_VERSION_OLS,
)
from .exceptions import CortexMotionNotImplementedError, VectorSourceError
from .feature_allowlist import assert_vector_has_no_oracle
from .manifest import feature_configuration_hash
from .quality import bounded_ffill
from .transition import confirm_local_extrema, detector_version, horizon_min_duration_hours


def _pick(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _timing(cfg: dict[str, Any]) -> tuple[int, int]:
    tcfg = cfg.get("timing") or {}
    vcfg = cfg.get("vector") or {}
    feature_lag = int(
        _pick(
            tcfg,
            "feature_available_lag_hours",
            default=_pick(vcfg, "execution_lag_hours", default=1),
        )
    )
    decision_lag = int(_pick(tcfg, "decision_latency_hours", default=0))
    return feature_lag, decision_lag


def resolve_vector_identity(cfg: dict[str, Any]) -> tuple[str, str]:
    vcfg = cfg.get("vector") or {}
    source = vcfg.get("source")
    version = vcfg.get("version")
    if not source:
        raise VectorSourceError("vector.source is required; refusing to default")
    if source in {"cortex_derivative", VECTOR_SOURCE_CORTEX_MOTION}:
        raise CortexMotionNotImplementedError(
            "vector_source=cortex_market_motion is reserved for continuous Cortex "
            "derivative inputs. This engine builds price_ols_proxy_v1 only. "
            "OLS is not a fallback for missing Cortex D."
        )
    if source != VECTOR_SOURCE_OLS:
        raise VectorSourceError(
            f"Unknown vector source {source!r}. Allowed: {VECTOR_SOURCE_OLS!r} "
            f"(built) or {VECTOR_SOURCE_CORTEX_MOTION!r} (use build_cortex_vector)."
        )
    version = version or VECTOR_VERSION_OLS
    if version in {"market_motion_v1_price_ols_proxy", VECTOR_VERSION_OLS}:
        version = VECTOR_VERSION_OLS
    elif version in {
        "market_motion_v1_cortex_derivative",
        VECTOR_VERSION_CORTEX_MOTION,
        "cortex_derivative",
    }:
        raise CortexMotionNotImplementedError(
            f"vector_version={version!r} is cortex motion; use build_cortex_vector, not OLS"
        )
    else:
        raise VectorSourceError(f"Unknown vector_version {version!r}")
    return VECTOR_SOURCE_OLS, VECTOR_VERSION_OLS


def rolling_ols_slope(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing (causal) OLS slope. The window ending at t uses values[t-window+1 : t+1] only."""
    n = len(values)
    out = np.full(n, np.nan)
    if window < 2 or n < window:
        return out
    idx = np.arange(window, dtype=float)
    sum_x = idx.sum()
    sum_x2 = float(np.dot(idx, idx))
    denom = window * sum_x2 - sum_x * sum_x
    if denom == 0:
        return out
    from numpy.lib.stride_tricks import sliding_window_view

    views = sliding_window_view(values, window)
    bad = np.isnan(views).any(axis=1)
    filled = np.nan_to_num(views, nan=0.0)
    sum_y = filled.sum(axis=1)
    sum_xy = filled @ idx
    slope = (window * sum_xy - sum_x * sum_y) / denom
    slope[bad] = np.nan
    out[window - 1 :] = slope
    return out


def rolling_finite_count(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan)
    if window < 1 or n < window:
        return out
    from numpy.lib.stride_tricks import sliding_window_view

    views = sliding_window_view(np.isfinite(values).astype(float), window)
    out[window - 1 :] = views.sum(axis=1)
    return out


def robust_z(series: pd.Series, window: int, min_obs: int, clip: float) -> tuple[pd.Series, pd.Series]:
    """Z at t uses location/scale from strictly prior bars (t-1), then D_t."""
    past = series.shift(1)
    med = past.rolling(window, min_periods=min_obs).median()
    mad = (past - med).abs().rolling(window, min_periods=min_obs).median()
    scale = 1.4826 * mad + 1e-12
    z = (series - med) / scale
    ready = med.notna() & mad.notna() & (mad > 0)
    z = z.clip(-clip, clip)
    return z, ready


def _resolution_class(bars: int, min_slope: int, min_va: int) -> str:
    if bars < min_slope:
        return "insufficient"
    if bars < min_va:
        return "slope_only"
    return "va_stable"


def _invalidation_reason(
    *,
    price_valid: pd.Series,
    d: pd.Series,
    z_ready: pd.Series,
    ols_n: pd.Series,
    bars: int,
    va_stable: bool,
    vel: pd.Series,
) -> pd.Series:
    reasons = pd.Series("", index=d.index, dtype=object)
    reasons = reasons.mask(~price_valid.fillna(False), "stale_or_missing_price")
    reasons = reasons.mask(price_valid & d.isna() & (ols_n < bars), "ols_window_incomplete")
    reasons = reasons.mask(price_valid & d.isna() & (ols_n >= bars), "ols_undefined")
    reasons = reasons.mask(price_valid & d.notna() & ~z_ready.fillna(False), "normalization_warmup")
    if va_stable:
        reasons = reasons.mask(
            price_valid & d.notna() & z_ready.fillna(False) & vel.isna(),
            "velocity_undefined",
        )
    reasons = reasons.where(reasons != "", other="")
    return reasons


def build_vector(
    prices: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    time_col: str = "time_bucket",
    price_col: str = "close_price",
) -> pd.DataFrame:
    source, version = resolve_vector_identity(cfg)
    vcfg = cfg["vector"]
    work = prices.copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=True)
    work = work.dropna(subset=[time_col]).sort_values(time_col)
    work = work.drop_duplicates(time_col, keep="last").set_index(time_col)
    if work.empty:
        raise ValueError("no price rows")
    work = work.reindex(pd.date_range(work.index.min(), work.index.max(), freq="h", tz="UTC"))
    px = pd.to_numeric(work[price_col], errors="coerce")
    stale_map = vcfg.get("staleness_hours") or {}
    max_stale = int(_pick(stale_map, "crypto_last_price", default=vcfg.get("max_staleness_hours", 2)))
    px_f, px_age, px_valid = bounded_ffill(px, max_stale)
    work["btc_close_at_t"] = px_f
    work["price_source_age_hours"] = px_age
    work["price_feature_valid"] = px_valid

    series = np.log(px_f.replace(0, np.nan)) if vcfg.get("use_log_price", True) else px_f
    values = series.to_numpy(dtype=float)
    delta = int(_pick(vcfg, "velocity_delta_hours", default=1))
    z_win = int(_pick(vcfg, "z_window_hours"))
    z_min = int(_pick(vcfg, "z_min_obs"))
    z_clip = float(_pick(vcfg, "z_clip"))
    min_va = int(_pick(vcfg, "min_ols_bars_for_va", default=8))
    min_slope = int(_pick(vcfg, "min_ols_bars_for_slope", default=3))
    delay = int(_pick(vcfg, "confirmation_delay_hours", default=2))
    prom_k = float(_pick(vcfg, "min_prominence_mad", default=1.0))
    dur_frac = float(_pick(vcfg, "min_duration_horizon_fraction", default=0.25))
    prom_win = int(_pick(vcfg, "prominence_mad_window_hours", default=168))
    prom_min = int(_pick(vcfg, "prominence_mad_min_obs", default=24))
    feature_lag, decision_lag = _timing(cfg)
    det_ver = detector_version(
        delay_hours=delay,
        min_prominence_mad=prom_k,
        min_duration_horizon_fraction=dur_frac,
    )

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
        "vector_source": source,
        "vector_version": version,
        "source_query_version": "price_ols_log_close_v1",
        "feature_configuration_hash": feature_configuration_hash(cfg),
        "detector_version": det_ver,
        "confirmation_delay_hours": delay,
        "min_prominence_mad": prom_k,
        "min_duration_horizon_fraction": dur_frac,
        "btc_close_at_t": work["btc_close_at_t"],
        "price_source_age_hours": work["price_source_age_hours"],
        "price_feature_valid": work["price_feature_valid"],
        "price_max_staleness_hours": max_stale,
        "feature_available_lag_hours": feature_lag,
        "decision_latency_hours": decision_lag,
    }

    horizons: dict[str, int] = dict(vcfg["horizons"])
    empty_events = pd.Series("NONE", index=idx)
    empty_time = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")
    for name, bars in horizons.items():
        bars_i = int(bars)
        d = pd.Series(rolling_ols_slope(values, bars_i), index=work.index)
        ols_n = pd.Series(rolling_finite_count(values, bars_i), index=work.index)
        z, z_ready = robust_z(d, z_win, z_min, z_clip)
        vel = (z - z.shift(delta)) / delta
        acc = (vel - vel.shift(delta)) / delta
        res_class = _resolution_class(bars_i, min_slope, min_va)
        va_stable = res_class == "va_stable"
        if not va_stable:
            vel = pd.Series(np.nan, index=work.index)
            acc = pd.Series(np.nan, index=work.index)
        min_dur = horizon_min_duration_hours(bars_i, delay, dur_frac)
        detect_kw = dict(
            delay_hours=delay,
            min_prominence_mad=prom_k,
            min_duration_hours=min_dur,
            prominence_window_hours=prom_win,
            prominence_min_obs=prom_min,
        )
        if res_class == "insufficient":
            d = pd.Series(np.nan, index=work.index)
            z = pd.Series(np.nan, index=work.index)
            z_ready = pd.Series(False, index=work.index)
            trans = {
                "transition_state_raw": pd.Series(np.nan, index=work.index),
                "transition_direction": pd.Series(np.nan, index=work.index),
                "transition_event": empty_events.copy(),
                "confirmation_time": empty_time.copy(),
                "candidate_extremum_time": empty_time.copy(),
                "hours_since_confirmation": pd.Series(np.nan, index=work.index),
            }
        else:
            trans = confirm_local_extrema(d, **detect_kw)
        vel_trans = None
        if va_stable and res_class != "insufficient":
            vel_trans = confirm_local_extrema(vel, **detect_kw)
        # Live-safe validity does not require a confirmed turn. State stays null
        # until the first confirmation; D/Z remain usable.
        src_valid = px_valid & d.notna()
        valid = src_valid & z_ready
        if va_stable:
            valid = valid & vel.notna()
        reasons = _invalidation_reason(
            price_valid=px_valid,
            d=d,
            z_ready=z_ready,
            ols_n=ols_n,
            bars=bars_i,
            va_stable=va_stable,
            vel=vel,
        )
        p = f"{name}_"
        t_raw = trans["transition_state_raw"]
        hours_since = trans["hours_since_confirmation"]
        cols[f"{p}derivative_raw"] = d
        cols[f"{p}position_z"] = z
        cols[f"{p}velocity"] = vel
        cols[f"{p}acceleration"] = acc
        cols[f"{p}va_stable"] = va_stable
        cols[f"{p}ols_n_obs"] = ols_n
        cols[f"{p}ols_window_bars"] = bars_i
        cols[f"{p}resolution_class"] = res_class
        cols[f"{p}transition_state_raw"] = t_raw
        cols[f"{p}transition_raw"] = t_raw
        cols[f"{p}transition_direction"] = trans["transition_direction"]
        cols[f"{p}transition_event"] = trans["transition_event"]
        cols[f"{p}confirmation_time"] = trans["confirmation_time"]
        cols[f"{p}hours_since_confirmation"] = hours_since
        cols[f"{p}hours_since_turn"] = hours_since
        cols[f"{p}candidate_extremum_time"] = trans["candidate_extremum_time"]
        cols[f"{p}min_duration_hours"] = min_dur if res_class != "insufficient" else 0
        if vel_trans is not None:
            cols[f"{p}velocity_transition_state_raw"] = vel_trans["transition_state_raw"]
            cols[f"{p}velocity_transition_event"] = vel_trans["transition_event"]
            cols[f"{p}velocity_hours_since_confirmation"] = vel_trans["hours_since_confirmation"]
        else:
            cols[f"{p}velocity_transition_state_raw"] = pd.Series(np.nan, index=work.index)
            cols[f"{p}velocity_transition_event"] = empty_events.copy()
            cols[f"{p}velocity_hours_since_confirmation"] = pd.Series(np.nan, index=work.index)
        cols[f"{p}source_present"] = d.notna()
        cols[f"{p}source_age_hours"] = work["price_source_age_hours"]
        cols[f"{p}normalization_ready"] = z_ready
        cols[f"{p}feature_valid"] = valid
        cols[f"{p}invalidation_reason"] = reasons.where(~valid, "")

    n_up = sum(
        (cols[f"{h}_transition_state_raw"] == TRANSITION_RAW_UP_AFTER_TROUGH) & cols[f"{h}_feature_valid"]
        for h in horizons
    )
    n_valid = sum(cols[f"{h}_feature_valid"].astype(int) for h in horizons)
    cols["cross_n_horizons_up"] = n_up
    cols["cross_n_horizons_valid"] = n_valid
    cols["row_vector_valid"] = n_valid == len(horizons)
    out = pd.DataFrame(cols, index=idx)
    out.index.name = time_col
    result = out.reset_index()
    assert_vector_has_no_oracle(result, context="build_vector: ")
    return result


def vector_incremental_last_row(history_prices: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    """Replay helper: last row of batch vector on the history provided (must include t)."""
    vec = build_vector(history_prices, cfg)
    return vec.iloc[-1]


def vector_validity_by_horizon(vec: pd.DataFrame, horizons: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for h in horizons:
        col = f"{h}_feature_valid"
        if col not in vec.columns:
            continue
        v = vec[col].fillna(False).astype(bool)
        reasons = vec[f"{h}_invalidation_reason"] if f"{h}_invalidation_reason" in vec.columns else None
        rec: dict[str, Any] = {
            "n_valid": int(v.sum()),
            "n_invalid": int((~v).sum()),
            "valid_fraction": float(v.mean()) if len(v) else None,
            "va_stable": bool(vec[f"{h}_va_stable"].iloc[-1]) if f"{h}_va_stable" in vec.columns else None,
            "resolution_class": (
                str(vec[f"{h}_resolution_class"].dropna().iloc[-1])
                if f"{h}_resolution_class" in vec.columns and vec[f"{h}_resolution_class"].notna().any()
                else None
            ),
        }
        if reasons is not None:
            rec["invalidation_counts"] = (
                reasons.fillna("").replace("", pd.NA).dropna().value_counts().to_dict()
            )
        out[h] = rec
    return out
