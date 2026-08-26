"""Hourly UTC grid, bounded forward-fill, quality flags. Never bfill."""

from __future__ import annotations

from typing import Any

import pandas as pd


def to_hourly_utc(index: pd.DatetimeIndex | pd.Series) -> pd.DatetimeIndex:
    ts = pd.to_datetime(index, utc=True, errors="coerce")
    if isinstance(ts, pd.Series):
        ts = pd.DatetimeIndex(ts)
    return ts


def unique_hourly_grid(times: pd.Series) -> pd.DatetimeIndex:
    t = to_hourly_utc(times).dropna().sort_values()
    t = t[~t.duplicated(keep="last")]
    if t.empty:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.date_range(t.min(), t.max(), freq="h", tz="UTC")


def reindex_hourly(df: pd.DataFrame, time_col: str = "time_bucket") -> pd.DataFrame:
    """One row per UTC hour, sorted, unique. Does not fill values."""
    out = df.copy()
    out[time_col] = to_hourly_utc(out[time_col])
    out = out.dropna(subset=[time_col]).sort_values(time_col)
    out = out.drop_duplicates(subset=[time_col], keep="last").set_index(time_col)
    grid = pd.date_range(out.index.min(), out.index.max(), freq="h", tz="UTC")
    out = out.reindex(grid)
    out.index.name = time_col
    return out


def bounded_ffill(
    series: pd.Series,
    max_hours: int,
    *,
    barrier: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Forward-fill at most max_hours. Returns (filled, source_age_hours, valid_mask).

    Age is hours since the last observed (non-null) source value. Gaps longer
    than max_hours are left as NaN and marked invalid. Never backward-fills.

    ``barrier`` hours (source conflicts) are not treated as ordinary scrape
    gaps: they stay missing and ffill cannot cross them.
    """
    s = series.copy()
    if barrier is None:
        barrier = pd.Series(False, index=s.index)
    else:
        barrier = barrier.reindex(s.index).fillna(False).astype(bool)
    s = s.where(~barrier)
    if barrier.any():
        filled_parts = []
        age_parts = []
        valid_parts = []
        group = barrier.cumsum()
        for _, sub in s.groupby(group, sort=False):
            f, a, v = _bounded_ffill_one(sub, max_hours)
            filled_parts.append(f)
            age_parts.append(a)
            valid_parts.append(v)
        filled = pd.concat(filled_parts).reindex(s.index)
        age = pd.concat(age_parts).reindex(s.index)
        valid = pd.concat(valid_parts).reindex(s.index).fillna(False)
        filled = filled.where(~barrier)
        valid = valid & ~barrier
        return filled, age, valid
    return _bounded_ffill_one(s, max_hours)


def _bounded_ffill_one(series: pd.Series, max_hours: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    present = series.notna()
    last_obs = pd.Series(series.index, index=series.index).where(present).ffill()
    age = (series.index.to_series() - last_obs).dt.total_seconds() / 3600.0
    filled = series.ffill(limit=max_hours)
    valid = present | ((~present) & filled.notna() & (age <= max_hours))
    filled = filled.where(valid)
    return filled, age, valid


def hourly_coverage_stats(index: pd.DatetimeIndex) -> dict[str, Any]:
    idx = to_hourly_utc(index).dropna().sort_values().unique()
    if len(idx) == 0:
        return {
            "n_index_rows": 0,
            "span_hours": 0,
            "expected_hourly_bars": 0,
            "coverage": None,
        }
    start, end = idx.min(), idx.max()
    expected = pd.date_range(start, end, freq="h", tz="UTC")
    return {
        "n_index_rows": int(len(idx)),
        "start": str(start),
        "end": str(end),
        "span_hours": int((end - start) / pd.Timedelta(hours=1)) + 1,
        "expected_hourly_bars": int(len(expected)),
        "coverage": float(len(idx) / len(expected)) if len(expected) else None,
    }


def quality_summary(
    df: pd.DataFrame,
    *,
    time_col: str = "time_bucket",
    valid_col: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compact quality block for manifests and baseline reports."""
    times = pd.to_datetime(df[time_col], utc=True, errors="coerce") if time_col in df.columns else df.index
    cov = hourly_coverage_stats(pd.DatetimeIndex(times))
    payload: dict[str, Any] = {
        "n_rows": int(len(df)),
        "coverage": cov,
        "null_fraction_by_column": {
            c: float(df[c].isna().mean())
            for c in df.columns
            if c != time_col and not str(c).endswith("_invalidation_reason")
        },
    }
    if valid_col and valid_col in df.columns:
        v = df[valid_col]
        payload["n_valid"] = int(v.fillna(False).astype(bool).sum())
        payload["n_invalid"] = int((~v.fillna(False).astype(bool)).sum())
    if extra:
        payload.update(extra)
    return payload
