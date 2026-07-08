#!/usr/bin/env python3
"""
Export Prometheus/Cortex time-series data to a wide-format Parquet file.

Handles the ~11,000-sample-per-request limit by automatically chunking the
time range, retries with exponential backoff, and shows tqdm progress.
"""
from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Default PromQL queries — each key becomes a column name prefix.
# Values are the raw PromQL expression sent to query_range.
# ---------------------------------------------------------------------------
DEFAULT_QUERIES: dict[str, str] = {
    "price": 'max by (symbol) (crypto_last_price{symbol="BTCUSDT"})',
    "weighted_norm_avg_16h_24h_48h": 'job:crypto_last_price:weighted_normalized_avg:16h:24h:48h{symbol="BTCUSDT"}',
    "weighted_deriv_24h_48h_7d": 'job:crypto_last_price:weighted_deriv:24h:48h:7d{symbol="BTCUSDT"}',
    "norm_combined_avg": 'job:crypto_last_price:normalized_combined_avg{symbol="BTCUSDT"}',
}

MAX_SAMPLES_PER_REQUEST = 11_000
MAX_RETRIES = 5
INITIAL_BACKOFF_S = 1.0


def _parse_dt(value: str) -> datetime:
    """Parse an ISO-ish date/datetime string into a timezone-aware UTC datetime."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime: {value!r}")


def _query_range(
    url: str,
    query: str,
    start_ts: int,
    end_ts: int,
    step: int,
) -> list[dict[str, Any]]:
    """
    Call /api/v1/query_range with retries + exponential backoff.
    Returns the list of result series from the response.
    """
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


def _col_name(prefix: str, metric_labels: dict[str, str]) -> str:
    """
    Build a column name from the query prefix and any distinguishing labels.
    For single-series results the prefix alone is used; for multi-series
    results the label values are appended.
    """
    ignore = {"__name__", "symbol"}
    extras = {k: v for k, v in metric_labels.items() if k not in ignore}
    if not extras:
        return prefix
    suffix = "_".join(f"{v}" for v in extras.values())
    return f"{prefix}__{suffix}"


def _coalesce_instance_columns(df: pd.DataFrame, prefixes: list[str]) -> pd.DataFrame:
    """
    When a PromQL query returns multiple series (one per job instance),
    the DataFrame ends up with columns like:
        weighted_deriv_24h_48h_7d__PROD-A-10.1.20.100_...
        weighted_deriv_24h_48h_7d__PROD-A-10.2.20.100_...
        weighted_deriv_24h_48h_7d__PROD-B-10.2.20.100_...

    This function merges them into a single column per prefix by taking
    the first non-null value across instances at each timestamp.
    """
    for prefix in prefixes:
        instance_cols = [c for c in df.columns if c.startswith(f"{prefix}__")]
        if len(instance_cols) <= 1:
            continue

        print(f"\n  Coalescing {len(instance_cols)} instance columns -> '{prefix}'")
        for c in instance_cols:
            non_null = df[c].notna().sum()
            print(f"    {c}: {non_null:,} non-null ({non_null / len(df) * 100:.1f}%)")

        df[prefix] = df[instance_cols].bfill(axis=1).iloc[:, 0]
        non_null = df[prefix].notna().sum()
        print(f"    -> merged '{prefix}': {non_null:,} non-null ({non_null / len(df) * 100:.1f}%)")
        df = df.drop(columns=instance_cols)

    return df


def export(
    url: str,
    queries: dict[str, str],
    start: datetime,
    end: datetime,
    step: int,
    output: str,
) -> pd.DataFrame:
    """
    Export all queries into a single wide DataFrame and save as Parquet.
    """
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())
    total_seconds = end_ts - start_ts
    chunk_seconds = MAX_SAMPLES_PER_REQUEST * step

    all_series: dict[str, pd.Series] = {}

    for prefix, query in queries.items():
        n_chunks = max(1, math.ceil(total_seconds / chunk_seconds))
        chunk_start = start_ts

        print(f"\n[{prefix}] {query}")
        print(f"  {n_chunks} chunk(s), {step}s step")

        for _ in tqdm(range(n_chunks), desc=prefix, unit="chunk"):
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
        raise RuntimeError("No data returned for any query.")

    for col in all_series:
        all_series[col] = all_series[col][~all_series[col].index.duplicated(keep="last")].sort_index()

    df = pd.DataFrame(all_series)
    df.index.name = "timestamp"
    df = df.sort_index()

    df = _coalesce_instance_columns(df, list(queries.keys()))

    df.to_parquet(output, engine="pyarrow")
    print(f"\nSaved {len(df)} rows x {len(df.columns)} cols -> {output}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Prometheus data to Parquet")
    parser.add_argument("--start-date", default="2025-03-11 00:15:18")
    parser.add_argument("--end-date", default="2026-04-15 00:15:18")
    parser.add_argument("--step", type=int, default=3600, help="Query step in seconds")
    parser.add_argument("--url", default="http://10.1.20.60:9009/prometheus/api/v1/query_range")
    parser.add_argument("--output", default="btc_data.parquet")
    args = parser.parse_args()

    start = _parse_dt(args.start_date)
    end = _parse_dt(args.end_date)

    print(f"Range : {start.isoformat()} -> {end.isoformat()}")
    print(f"Step  : {args.step}s")
    print(f"URL   : {args.url}")
    print(f"Output: {args.output}")

    export(
        url=args.url,
        queries=DEFAULT_QUERIES,
        start=start,
        end=end,
        step=args.step,
        output=args.output,
    )


if __name__ == "__main__":
    main()
