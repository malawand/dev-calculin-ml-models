"""Fetch recent BTC indicator history from Prometheus for live inference."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from constants import BAR_STEP_SECONDS, DEFAULT_PROMETHEUS_QUERIES

MAX_SAMPLES_PER_REQUEST = 11_000
MAX_RETRIES = 5
INITIAL_BACKOFF_S = 1.0


def _col_name(prefix: str, metric_labels: dict[str, str]) -> str:
    ignore = {"__name__", "symbol"}
    extras = {k: v for k, v in metric_labels.items() if k not in ignore}
    if not extras:
        return prefix
    suffix = "_".join(str(v) for v in extras.values())
    return f"{prefix}__{suffix}"


def _coalesce_instance_columns(df: pd.DataFrame, prefixes: list[str]) -> pd.DataFrame:
    for prefix in prefixes:
        instance_cols = [c for c in df.columns if c.startswith(f"{prefix}__")]
        if not instance_cols:
            continue
        if len(instance_cols) == 1:
            df[prefix] = df[instance_cols[0]]
            df = df.drop(columns=instance_cols)
            continue
        df[prefix] = df[instance_cols].bfill(axis=1).iloc[:, 0]
        df = df.drop(columns=instance_cols)
    return df


def _query_range(
    url: str,
    query: str,
    start_ts: int,
    end_ts: int,
    step: int,
) -> list[dict[str, Any]]:
    params = {
        "query": query,
        "start": start_ts,
        "end": end_ts,
        "step": f"{step}s",
    }
    backoff = INITIAL_BACKOFF_S

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=120)
            if resp.status_code in {429, 500, 502, 503, 504}:
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            body = resp.json()
            if body.get("status") != "success":
                time.sleep(backoff)
                backoff *= 2
                continue
            return body.get("data", {}).get("result", [])
        except (requests.RequestException, ValueError):
            if attempt == MAX_RETRIES:
                raise
            time.sleep(backoff)
            backoff *= 2

    return []


def fetch_recent_bars(
    url: str,
    lookback_days: float,
    step: int = BAR_STEP_SECONDS,
    queries: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Pull the last ``lookback_days`` of 15-minute bars from Prometheus.

    Returns a wide DataFrame indexed by UTC timestamp with columns:
    price, weighted_norm_avg_16h_24h_48h, weighted_deriv_24h_48h_7d,
    norm_combined_avg.
    """
    queries = queries or DEFAULT_PROMETHEUS_QUERIES
    end_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = end_ts - int(lookback_days * 24 * 3600)
    total_seconds = end_ts - start_ts
    chunk_seconds = MAX_SAMPLES_PER_REQUEST * step

    all_series: dict[str, pd.Series] = {}

    for prefix, query in queries.items():
        n_chunks = max(1, math.ceil(total_seconds / chunk_seconds))
        chunk_start = start_ts

        for _ in range(n_chunks):
            chunk_end = min(chunk_start + chunk_seconds, end_ts)
            results = _query_range(url, query, chunk_start, chunk_end, step)

            for series in results:
                labels = series.get("metric", {})
                col = _col_name(prefix, labels)
                values = series.get("values", [])
                if not values:
                    continue
                idx = pd.to_datetime([v[0] for v in values], unit="s", utc=True)
                s = pd.Series([float(v[1]) for v in values], index=idx, name=col)
                if col in all_series:
                    all_series[col] = pd.concat([all_series[col], s])
                else:
                    all_series[col] = s

            chunk_start = chunk_end
            time.sleep(0.05)

    if not all_series:
        raise RuntimeError("Prometheus returned no data for any query.")

    for col in all_series:
        all_series[col] = all_series[col][~all_series[col].index.duplicated(keep="last")].sort_index()

    df = pd.DataFrame(all_series)
    df.index.name = "timestamp"
    df = df.sort_index()
    df = _coalesce_instance_columns(df, list(queries.keys()))
    return df
