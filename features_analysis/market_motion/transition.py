"""Causal delayed local-extremum confirmation. Never backdates state.

A candidate extremum at hour c is confirmed at t = c + delay only if every bar
from c+1 through t stays strictly on the confirming side of the candidate, using
data through t only.

Canonical live T is a confirmed peak/trough of the **derivative curve D**.
Velocity turning points, if computed, are a separate experimental series.

v2 adds causal significance so tiny wiggles are not turns:
- confirmation still uses only data through t (timestamp rule frozen);
- after a confirmation, the opposite turn cannot fire until min_duration_hours;
- a later opposite extremum must also move at least k * trailing MAD(Δsignal)
  from the last confirmed extremum (hysteresis; scale from t−1 only).
The first confirmed turn uses the delay shape rule only, so slow curves are
not required to drop by a full-sample MAD in two hours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import (
    TRANSITION_DIRECTION_DOWN,
    TRANSITION_DIRECTION_UP,
    TRANSITION_RAW_DOWN_AFTER_PEAK,
    TRANSITION_RAW_UP_AFTER_TROUGH,
)

EVENT_NONE = "NONE"
EVENT_PEAK = "PEAK_CONFIRMED"
EVENT_TROUGH = "TROUGH_CONFIRMED"

DETECTOR_NAME = "causal_d_extrema"


def detector_version(
    *,
    delay_hours: int,
    min_prominence_mad: float,
    min_duration_horizon_fraction: float,
) -> str:
    k = str(float(min_prominence_mad)).replace(".", "p")
    frac = str(float(min_duration_horizon_fraction)).replace(".", "p")
    return f"{DETECTOR_NAME}_delay{int(delay_hours)}_prom{k}_durf{frac}_v2"


def horizon_min_duration_hours(bars: int, delay_hours: int, horizon_fraction: float) -> int:
    return max(int(delay_hours), int(bars * float(horizon_fraction)))


def trailing_change_mad(series: pd.Series, window: int, min_obs: int) -> pd.Series:
    """Robust scale of hour-to-hour changes using strictly prior bars."""
    past = series.diff().shift(1)
    med = past.rolling(window, min_periods=min_obs).median()
    mad = (past - med).abs().rolling(window, min_periods=min_obs).median()
    return 1.4826 * mad


def confirm_local_extrema(
    series: pd.Series,
    *,
    delay_hours: int,
    min_prominence_mad: float = 0.0,
    min_duration_hours: int = 0,
    prominence_window_hours: int = 168,
    prominence_min_obs: int = 24,
) -> dict[str, pd.Series]:
    """Confirm peaks/troughs with a causal delay. Initial state is null."""
    if delay_hours < 1:
        raise ValueError("confirmation_delay_hours must be >= 1 (no same-bar confirmation)")
    if min_duration_hours < 0:
        raise ValueError("min_duration_hours must be >= 0")
    values = series.to_numpy(dtype=float)
    index = series.index
    n = len(values)
    state = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    event = np.full(n, EVENT_NONE, dtype=object)
    confirm_stamp = pd.Series(pd.NaT, index=index, dtype="datetime64[ns, UTC]")
    extremum_stamp = pd.Series(pd.NaT, index=index, dtype="datetime64[ns, UTC]")
    scale = trailing_change_mad(series, prominence_window_hours, prominence_min_obs).to_numpy(dtype=float)

    last_state = np.nan
    last_confirm = pd.NaT
    last_ext = pd.NaT
    last_confirm_i = None
    last_ext_val = np.nan

    for t in range(delay_hours + 1, n):
        c = t - delay_hours
        window = values[c - 1 : t + 1]
        if np.isfinite(window).all():
            duration_ok = last_confirm_i is None or (t - last_confirm_i) >= min_duration_hours
            vc = float(values[c])
            right = values[c + 1 : t + 1]
            sc = scale[t]
            peak_shape = vc > values[c - 1] and bool(np.all(right < vc))
            trough_shape = vc < values[c - 1] and bool(np.all(right > vc))
            if min_prominence_mad <= 0 or not np.isfinite(last_ext_val):
                hysteresis_ok = True
            elif not np.isfinite(sc) or sc <= 0:
                hysteresis_ok = False
            else:
                hysteresis_ok = abs(vc - last_ext_val) >= min_prominence_mad * sc
            peak_ok = peak_shape and hysteresis_ok
            trough_ok = trough_shape and hysteresis_ok
            ts_t = index[t]
            ts_c = index[c]
            if duration_ok:
                if np.isnan(last_state):
                    if peak_ok and not trough_ok:
                        event[t] = EVENT_PEAK
                        last_state = TRANSITION_RAW_DOWN_AFTER_PEAK
                        last_confirm = ts_t
                        last_ext = ts_c
                        last_confirm_i = t
                        last_ext_val = vc
                    elif trough_ok and not peak_ok:
                        event[t] = EVENT_TROUGH
                        last_state = TRANSITION_RAW_UP_AFTER_TROUGH
                        last_confirm = ts_t
                        last_ext = ts_c
                        last_confirm_i = t
                        last_ext_val = vc
                elif last_state == TRANSITION_RAW_UP_AFTER_TROUGH:
                    if peak_ok:
                        event[t] = EVENT_PEAK
                        last_state = TRANSITION_RAW_DOWN_AFTER_PEAK
                        last_confirm = ts_t
                        last_ext = ts_c
                        last_confirm_i = t
                        last_ext_val = vc
                elif last_state == TRANSITION_RAW_DOWN_AFTER_PEAK:
                    if trough_ok:
                        event[t] = EVENT_TROUGH
                        last_state = TRANSITION_RAW_UP_AFTER_TROUGH
                        last_confirm = ts_t
                        last_ext = ts_c
                        last_confirm_i = t
                        last_ext_val = vc

        state[t] = last_state
        if last_state == TRANSITION_RAW_UP_AFTER_TROUGH:
            direction[t] = TRANSITION_DIRECTION_UP
        elif last_state == TRANSITION_RAW_DOWN_AFTER_PEAK:
            direction[t] = TRANSITION_DIRECTION_DOWN
        confirm_stamp.iloc[t] = last_confirm
        extremum_stamp.iloc[t] = last_ext

    hours_since = pd.Series(np.nan, index=index)
    known = confirm_stamp.notna()
    if known.any():
        hours_since.loc[known] = (
            index.to_series().loc[known] - confirm_stamp.loc[known]
        ).dt.total_seconds() / 3600.0

    return {
        "transition_state_raw": pd.Series(state, index=index),
        "transition_direction": pd.Series(direction, index=index),
        "transition_event": pd.Series(event, index=index),
        "confirmation_time": confirm_stamp,
        "candidate_extremum_time": extremum_stamp,
        "hours_since_confirmation": hours_since,
    }
