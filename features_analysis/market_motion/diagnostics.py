"""Turn-frequency diagnostics. Research reporting only; not a feature source."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .transition import EVENT_PEAK, EVENT_TROUGH


def _event_mask(series: pd.Series) -> pd.Series:
    return series.isin({EVENT_PEAK, EVENT_TROUGH})


def _series_distribution(s: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return {"n": 0}
    q = x.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else None,
        "min": float(x.min()),
        "p05": float(q.loc[0.05]),
        "p25": float(q.loc[0.25]),
        "median": float(q.loc[0.5]),
        "p75": float(q.loc[0.75]),
        "p95": float(q.loc[0.95]),
        "max": float(x.max()),
    }


def horizon_representation_stats(vec: pd.DataFrame, horizon: str, *, time_col: str = "time_bucket") -> dict[str, Any]:
    """Coverage, quality, and turn stats. Does not look at 24h outcomes."""
    turns = horizon_turn_stats(vec, horizon, time_col=time_col)
    work = vec.copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=True)
    d_col = f"{horizon}_derivative_raw"
    present_col = f"{horizon}_source_present"
    age_col = f"{horizon}_source_age_hours"
    valid_col = f"{horizon}_feature_valid"
    ready_col = f"{horizon}_normalization_ready"
    reason_col = f"{horizon}_invalidation_reason"
    state_col = f"{horizon}_transition_state_raw"
    delay_col = f"{horizon}_min_duration_hours"
    cand_col = f"{horizon}_candidate_extremum_time"
    event_col = f"{horizon}_transition_event"
    z_col = f"{horizon}_position_z"
    v_col = f"{horizon}_velocity"
    rec: dict[str, Any] = dict(turns)
    rec["available_start_utc"] = str(work[time_col].min()) if len(work) else None
    rec["available_end_utc"] = str(work[time_col].max()) if len(work) else None
    if present_col in work.columns:
        present = work[present_col].fillna(False).astype(bool)
        rec["n_source_present"] = int(present.sum())
        rec["n_source_missing"] = int((~present).sum())
        rec["source_missing_fraction"] = float((~present).mean()) if len(work) else None
        if present.any():
            rec["source_first_utc"] = str(work.loc[present, time_col].min())
            rec["source_last_utc"] = str(work.loc[present, time_col].max())
    if valid_col in work.columns:
        valid = work[valid_col].fillna(False).astype(bool)
        rec["n_valid"] = int(valid.sum())
        rec["n_invalid"] = int((~valid).sum())
        rec["valid_fraction"] = float(valid.mean()) if len(work) else None
    if ready_col in work.columns:
        rec["n_normalization_warmup"] = int((~work[ready_col].fillna(False).astype(bool)).sum())
    if age_col in work.columns:
        age = pd.to_numeric(work[age_col], errors="coerce")
        rec["median_source_age_hours"] = float(age.median()) if age.notna().any() else None
        rec["n_stale_gt_2h"] = int((age > 2).fillna(False).sum())
    if reason_col in work.columns:
        rec["invalidation_counts"] = (
            work[reason_col].fillna("").replace("", pd.NA).dropna().value_counts().to_dict()
        )
    if state_col in work.columns:
        known = work[state_col].dropna()
        rec["n_unknown_state"] = int(work[state_col].isna().sum())
        rec["n_state_up_after_trough"] = int((known == 0).sum())
        rec["n_state_down_after_peak"] = int((known == 1).sum())
        rec["median_state_duration_hours"] = rec.get("median_hours_between_events")
    if event_col in work.columns and cand_col in work.columns:
        events = work.loc[_event_mask(work[event_col])]
        if len(events):
            delays = (
                pd.to_datetime(events[time_col], utc=True)
                - pd.to_datetime(events[cand_col], utc=True)
            ).dt.total_seconds() / 3600.0
            rec["median_confirmation_delay_hours"] = float(delays.median())
            rec["mean_confirmation_delay_hours"] = float(delays.mean())
    rec["position_z"] = _series_distribution(work[z_col]) if z_col in work.columns else {}
    rec["velocity"] = _series_distribution(work[v_col]) if v_col in work.columns else {}
    rec["derivative_raw"] = _series_distribution(work[d_col]) if d_col in work.columns else {}
    if delay_col in work.columns and work[delay_col].notna().any():
        rec["min_duration_hours"] = int(work[delay_col].dropna().iloc[0])
    return rec


def representation_diagnostics(vec: pd.DataFrame, horizons: list[str], *, time_col: str = "time_bucket") -> dict[str, Any]:
    by_h = {h: horizon_representation_stats(vec, h, time_col=time_col) for h in horizons}
    return {
        "by_horizon": by_h,
        "n_events_total": int(sum(v.get("n_events", 0) or 0 for v in by_h.values())),
        "canonical_signal": "cortex_derivative_raw",
        "note": (
            "Diagnostics only. Canonical T is a confirmed peak/trough of Cortex D. "
            "No 24h outcome relationship is used here. Cortex vs OLS, if present, is diagnostic."
        ),
    }


def horizon_turn_stats(vec: pd.DataFrame, horizon: str, *, time_col: str = "time_bucket") -> dict[str, Any]:
    event_col = f"{horizon}_transition_event"
    d_col = f"{horizon}_derivative_raw"
    hours_col = f"{horizon}_hours_since_confirmation"
    cand_col = f"{horizon}_candidate_extremum_time"
    if event_col not in vec.columns:
        return {"horizon": horizon, "n_events": 0}
    work = vec.copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=True)
    events = work.loc[_event_mask(work[event_col])].copy()
    n = int(len(events))
    n_rows = int(len(work))
    n_peak = int((events[event_col] == EVENT_PEAK).sum()) if n else 0
    n_trough = int((events[event_col] == EVENT_TROUGH).sum()) if n else 0
    hours_per_event = (n_rows / n) if n else None
    durations = events[time_col].diff().dt.total_seconds() / 3600.0 if n else pd.Series(dtype=float)
    moves: list[float] = []
    if n and d_col in work.columns and cand_col in events.columns:
        indexed = work.set_index(time_col)
        d = indexed[d_col]
        for _, rec in events.iterrows():
            ts_t = rec[time_col]
            ts_c = pd.to_datetime(rec[cand_col], utc=True)
            if ts_t in d.index and pd.notna(ts_c) and ts_c in d.index:
                a, b = d.loc[ts_c], d.loc[ts_t]
                if np.isfinite(a) and np.isfinite(b):
                    moves.append(float(abs(a - b)))
    known = work[hours_col].dropna() if hours_col in work.columns else pd.Series(dtype=float)
    return {
        "horizon": horizon,
        "n_rows": n_rows,
        "n_events": n,
        "n_peak_confirmed": n_peak,
        "n_trough_confirmed": n_trough,
        "hours_per_event": hours_per_event,
        "median_hours_between_events": float(durations.median()) if n > 1 else None,
        "mean_hours_between_events": float(durations.mean()) if n > 1 else None,
        "median_hours_since_confirmation": float(known.median()) if len(known) else None,
        "median_abs_d_move_candidate_to_confirm": float(np.median(moves)) if moves else None,
        "mean_abs_d_move_candidate_to_confirm": float(np.mean(moves)) if moves else None,
        "signal": "derivative_raw",
    }


def turn_diagnostics(vec: pd.DataFrame, horizons: list[str], *, time_col: str = "time_bucket") -> dict[str, Any]:
    by_h = {h: horizon_turn_stats(vec, h, time_col=time_col) for h in horizons}
    return {
        "by_horizon": by_h,
        "n_events_total": int(sum(v.get("n_events", 0) or 0 for v in by_h.values())),
        "canonical_signal": "derivative_raw",
        "note": (
            "transition_state_raw is a confirmed peak/trough of D on every horizon. "
            "Velocity turns, if present, are diagnostic only and are not in the model allowlist."
        ),
    }
