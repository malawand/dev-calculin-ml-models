"""Cortex/Prometheus range queries with chunking. No bfill. No secret logging."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from .exceptions import CortexSourceError


def fetch_range(
    query: str,
    *,
    url: str,
    start_ts: int,
    end_ts: int,
    step: str = "1h",
    chunk_days: int = 180,
    timeout: int = 60,
    column: str = "value",
) -> pd.DataFrame:
    """Binary baseline helper. 0/1 bits may use max(); signed Cortex D must not."""
    raw, _log = fetch_range_raw(
        query,
        url=url,
        start_ts=start_ts,
        end_ts=end_ts,
        step=step,
        chunk_days=chunk_days,
        timeout=timeout,
        series_name=column,
    )
    if raw.empty:
        return pd.DataFrame(columns=["timestamp", column])
    collapsed = raw.groupby("timestamp", as_index=False)["value"].max()
    return collapsed.rename(columns={"value": column})


def fetch_range_raw(
    query: str,
    *,
    url: str,
    start_ts: int,
    end_ts: int,
    step: str = "1h",
    chunk_days: int = 180,
    timeout: int = 60,
    series_name: str = "value",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Preserve original Cortex timestamps and metric labels. Never fills."""
    frames: list[pd.DataFrame] = []
    current = int(start_ts)
    chunk_sec = int(chunk_days) * 24 * 3600
    chunks = 0
    http_errors = 0
    t0 = datetime.now(timezone.utc)
    retrieved_at = t0.isoformat()
    while current < end_ts:
        chunk_end = min(current + chunk_sec, end_ts)
        params = {"query": query, "start": current, "end": chunk_end, "step": step}
        resp = requests.get(url, params=params, timeout=timeout)
        chunks += 1
        if resp.status_code != 200:
            http_errors += 1
            raise CortexSourceError(
                f"Cortex HTTP {resp.status_code} for {series_name}: {resp.text[:500]}"
            )
        payload = resp.json()
        results = (payload.get("data") or {}).get("result") or []
        for series in results:
            metric = series.get("metric") or {}
            values = series.get("values") or []
            if not values:
                continue
            part = pd.DataFrame(values, columns=["timestamp", "value"])
            part["series_name"] = series_name
            part["promql"] = query
            part["__name__"] = metric.get("__name__", "")
            part["symbol"] = metric.get("symbol", "")
            part["cluster"] = metric.get("cluster", "")
            part["replica"] = metric.get("__replica__", metric.get("replica", ""))
            frames.append(part)
        current = chunk_end + 3600

    log: dict[str, Any] = {
        "series_name": series_name,
        "promql": query,
        "requested_start_unix": int(start_ts),
        "requested_end_unix": int(end_ts),
        "requested_start_utc": datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
        "requested_end_utc": datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat(),
        "step": step,
        "chunk_days": chunk_days,
        "n_chunks": chunks,
        "http_errors": http_errors,
        "retrieved_at_utc": retrieved_at,
        "elapsed_seconds": (datetime.now(timezone.utc) - t0).total_seconds(),
        "n_raw_rows": 0,
        "n_unique_timestamps": 0,
        "n_duplicate_timestamp_rows": 0,
        "n_replicas": 0,
        "returned_start_utc": None,
        "returned_end_utc": None,
        "missingness": None,
    }
    if not frames:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "value",
                "series_name",
                "promql",
                "__name__",
                "symbol",
                "cluster",
                "replica",
            ]
        ), log

    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = out["timestamp"].astype(int)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    n_raw = int(len(out))
    n_unique = int(out["timestamp"].nunique())
    log["n_raw_rows"] = n_raw
    log["n_unique_timestamps"] = n_unique
    log["n_duplicate_timestamp_rows"] = int(n_raw - n_unique)
    log["n_replicas"] = int(out["replica"].replace("", pd.NA).nunique(dropna=True))
    ts_min, ts_max = int(out["timestamp"].min()), int(out["timestamp"].max())
    log["returned_start_utc"] = datetime.fromtimestamp(ts_min, tz=timezone.utc).isoformat()
    log["returned_end_utc"] = datetime.fromtimestamp(ts_max, tz=timezone.utc).isoformat()
    expected = list(range(ts_min, ts_max + 1, 3600))
    have = set(out["timestamp"].unique().tolist())
    log["elapsed_seconds"] = (datetime.now(timezone.utc) - t0).total_seconds()
    log["n_expected_hourly_in_returned_span"] = len(expected)
    log["n_missing_hourly_in_returned_span"] = int(sum(1 for t in expected if t not in have))
    if expected:
        log["missingness"] = log["n_missing_hourly_in_returned_span"] / len(expected)
    return out, log


def fetch_named_queries(
    queries: dict[str, str],
    *,
    url: str,
    start: str,
    end: str,
    step: str = "1h",
    chunk_days: int = 180,
) -> pd.DataFrame:
    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
    end_ts = int(pd.Timestamp(end, tz="UTC").timestamp())
    merged: pd.DataFrame | None = None
    for name, promql in queries.items():
        part = fetch_range(
            promql,
            url=url,
            start_ts=start_ts,
            end_ts=end_ts,
            step=step,
            chunk_days=chunk_days,
            column=name,
        )
        if part.empty:
            continue
        merged = part if merged is None else merged.merge(part, on="timestamp", how="outer")
    if merged is None:
        raise CortexSourceError("No Cortex series returned")
    merged["time_bucket"] = pd.to_datetime(merged["timestamp"], unit="s", utc=True)
    return merged.drop(columns=["timestamp"]).sort_values("time_bucket")


def collapse_raw_to_hourly(raw: pd.DataFrame, series_name: str) -> pd.Series:
    """Priority-select hourly series. Does not fill gaps. Never max() of signed D."""
    from .source_collapse import collapse_series_hourly

    collapsed = collapse_series_hourly(raw, series_name)
    if collapsed.empty:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(collapsed["timestamp"], unit="s", utc=True)
    s = pd.Series(collapsed["value"].to_numpy(dtype=float), index=idx, name=series_name)
    return s.sort_index()
