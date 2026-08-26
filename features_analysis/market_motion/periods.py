"""Frozen date-split helpers. Holdouts stay closed unless a later pass opens them."""

from __future__ import annotations

import pandas as pd

BURN_IN_END_UTC = pd.Timestamp("2023-10-31 23:00:00", tz="UTC")
DEVELOPMENT_START_UTC = pd.Timestamp("2023-11-01 00:00:00", tz="UTC")
RETROSPECTIVE_HOLDOUT_START_UTC = pd.Timestamp("2026-01-15 00:00:00", tz="UTC")
RETROSPECTIVE_HOLDOUT_END_UTC = pd.Timestamp("2026-06-30 23:00:00", tz="UTC")
PROSPECTIVE_LOCKED_START_UTC = pd.Timestamp("2026-09-01 00:00:00", tz="UTC")

DISCLAIMER = "PROVISIONAL POC — temporary Cortex-derived dataset"


def as_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def to_utc(series_or_ts) -> pd.DatetimeIndex | pd.Timestamp:
    return pd.to_datetime(series_or_ts, utc=True)


def in_development(times: pd.Series | pd.DatetimeIndex) -> pd.Series:
    t = pd.to_datetime(times, utc=True)
    return (t >= DEVELOPMENT_START_UTC) & (t < RETROSPECTIVE_HOLDOUT_START_UTC)


def in_retrospective_holdout(times: pd.Series | pd.DatetimeIndex) -> pd.Series:
    t = pd.to_datetime(times, utc=True)
    return (t >= RETROSPECTIVE_HOLDOUT_START_UTC) & (t <= RETROSPECTIVE_HOLDOUT_END_UTC)


def in_prospective_locked(times: pd.Series | pd.DatetimeIndex) -> pd.Series:
    t = pd.to_datetime(times, utc=True)
    return t >= PROSPECTIVE_LOCKED_START_UTC


def last_observation_with_complete_outcome(
    *,
    period_end_exclusive: pd.Timestamp = RETROSPECTIVE_HOLDOUT_START_UTC,
    entry_lag_hours: int = 1,
    outcome_horizon_hours: int = 24,
) -> pd.Timestamp:
    """Latest observation whose outcome window ends before period_end_exclusive."""
    return period_end_exclusive - pd.Timedelta(hours=entry_lag_hours + outcome_horizon_hours + 1)
