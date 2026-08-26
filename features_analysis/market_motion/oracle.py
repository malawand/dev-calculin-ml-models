"""Research-only oracle extrema. Never join into the live vector artifact.

Question answered here: how much of the move was already over by the time the
turn became confirmable? Hindsight peak/trough times are stored only in this
table. They are not inputs to the 24h model, the production vector, analogue
matching, or feature selection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .feature_allowlist import ORACLE_DEFINITION_VERSION, infer_horizons
from .outcome_engine import _future_window, _timing
from .transition import EVENT_PEAK, EVENT_TROUGH

ORACLE_FILE_NAME = "oracle_extrema_research_only.csv"


def _simple_return_from(
    px: np.ndarray,
    index: pd.DatetimeIndex,
    ts: pd.Timestamp,
    *,
    entry_lag: int,
    horizon_h: int,
) -> float:
    if ts is None or pd.isna(ts):
        return float("nan")
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if ts not in index:
        return float("nan")
    loc = int(index.get_loc(ts))
    entry = loc + entry_lag
    window = _future_window(px, entry, horizon_h)
    if window is None or not np.isfinite(px[entry]):
        return float("nan")
    return float((window[-1] / px[entry] - 1.0) * 100.0)


def build_oracle_table(
    prices: pd.DataFrame,
    vec: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    time_col: str = "time_bucket",
    price_col: str = "close_price",
    return_horizon_h: int = 24,
) -> pd.DataFrame:
    """One row per confirmed turn, per horizon. Physically separate from the vector."""
    feature_lag, decision_lag = _timing(cfg)
    entry_lag = feature_lag + decision_lag
    delay = int((cfg.get("vector") or {}).get("confirmation_delay_hours", 2))
    detector = str(vec["detector_version"].iloc[0]) if "detector_version" in vec.columns and len(vec) else ""

    work = prices.copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=True)
    work = work.dropna(subset=[time_col]).sort_values(time_col)
    work = work.drop_duplicates(time_col, keep="last").set_index(time_col)
    work = work.reindex(pd.date_range(work.index.min(), work.index.max(), freq="h", tz="UTC"))
    px = pd.to_numeric(work[price_col], errors="coerce").to_numpy(dtype=float)
    pindex = work.index

    v = vec.copy()
    v[time_col] = pd.to_datetime(v[time_col], utc=True)

    horizons = infer_horizons(v.columns)
    if not horizons:
        horizons = list((cfg.get("vector") or {}).get("horizons") or {})

    rows: list[dict[str, Any]] = []
    for h in horizons:
        event_col = f"{h}_transition_event"
        ext_col = f"{h}_candidate_extremum_time"
        conf_col = f"{h}_confirmation_time"
        if event_col not in v.columns:
            continue
        events = v[v[event_col].isin({EVENT_PEAK, EVENT_TROUGH})]
        for _, rec in events.iterrows():
            event = str(rec[event_col])
            conf_time = pd.to_datetime(rec[conf_col] if conf_col in rec.index else rec[time_col], utc=True)
            ext_time = pd.to_datetime(rec[ext_col], utc=True) if ext_col in rec.index else pd.NaT
            obs_time = pd.to_datetime(rec[time_col], utc=True)
            avail = obs_time + pd.Timedelta(hours=feature_lag)
            decision = avail + pd.Timedelta(hours=decision_lag)
            entry = decision
            ret_ext = _simple_return_from(px, pindex, ext_time, entry_lag=entry_lag, horizon_h=return_horizon_h)
            ret_conf = _simple_return_from(px, pindex, obs_time, entry_lag=entry_lag, horizon_h=return_horizon_h)
            waiting = ret_ext - ret_conf if np.isfinite(ret_ext) and np.isfinite(ret_conf) else float("nan")
            delay_h = (
                (conf_time - ext_time).total_seconds() / 3600.0
                if pd.notna(conf_time) and pd.notna(ext_time)
                else float(delay)
            )
            ext_type = "PEAK" if event == EVENT_PEAK else "TROUGH"
            rows.append(
                {
                    "horizon": h,
                    "oracle_extremum_time": ext_time,
                    "oracle_extremum_type": ext_type,
                    "confirmation_time": conf_time,
                    "confirmation_delay_hours": delay_h,
                    "observation_time_utc": obs_time,
                    "feature_available_time_utc": avail,
                    "decision_time_utc": decision,
                    "outcome_entry_time_utc": entry,
                    "return_from_oracle_extremum": ret_ext,
                    "return_from_confirmation": ret_conf,
                    "waiting_cost": waiting,
                    "oracle_definition_version": ORACLE_DEFINITION_VERSION,
                    "detector_version": detector,
                    "return_horizon_h": return_horizon_h,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "horizon",
                "oracle_extremum_time",
                "oracle_extremum_type",
                "confirmation_time",
                "confirmation_delay_hours",
                "observation_time_utc",
                "feature_available_time_utc",
                "decision_time_utc",
                "outcome_entry_time_utc",
                "return_from_oracle_extremum",
                "return_from_confirmation",
                "waiting_cost",
                "oracle_definition_version",
                "detector_version",
                "return_horizon_h",
            ]
        )
    return pd.DataFrame(rows)
