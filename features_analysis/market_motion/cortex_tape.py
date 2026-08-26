"""Retrieve and persist raw Cortex continuous derivatives. Fill happens only after save."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import cortex_url
from .cortex_client import fetch_range_raw
from .exceptions import CortexSourceError
from .source_collapse import hourly_from_raw as collapse_hourly_from_raw


def cortex_motion_queries(cfg: dict[str, Any]) -> dict[str, str]:
    cm = cfg.get("cortex_motion") or {}
    out: dict[str, str] = {}
    for name, spec in (cm.get("horizons") or {}).items():
        if isinstance(spec, dict):
            out[name] = str(spec["query"])
        else:
            out[name] = str(spec)
    return out


def apply_live_cluster_selector(promql: str, cluster: str | None) -> str:
    """Restrict a PromQL selector to one Cortex cluster for live/future fetch."""
    if not cluster:
        return promql
    token = f'cluster="{cluster}"'
    if token in promql:
        return promql
    if "{" in promql:
        return promql.replace("{", "{" + token + ",", 1)
    return f"{promql}{{{token}}}"


def research_only_queries(cfg: dict[str, Any]) -> dict[str, str]:
    cm = cfg.get("cortex_motion") or {}
    extra = cm.get("research_only_queries") or {}
    return {str(k): str(v) for k, v in extra.items()}


def fetch_cortex_motion_raw(cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Historical query_range. Returns unlabeled raw rows plus per-series logs."""
    cm = cfg.get("cortex_motion") or {}
    period = cm.get("period") or cfg.get("period") or {}
    start = str(period["start"])
    end = str(period["end"])
    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
    end_ts = int(pd.Timestamp(end, tz="UTC").timestamp())
    url = str(cm.get("url") or cortex_url(cfg))
    step = str(cm.get("step", (cfg.get("cortex") or {}).get("step", "1h")))
    chunk_days = int(cm.get("chunk_days", (cfg.get("cortex") or {}).get("chunk_days", 90)))
    queries = {**cortex_motion_queries(cfg), **research_only_queries(cfg)}
    live_cluster = cm.get("live_cluster")
    queries = {name: apply_live_cluster_selector(q, live_cluster) for name, q in queries.items()}
    if not queries:
        raise CortexSourceError("no Cortex motion queries configured")
    frames: list[pd.DataFrame] = []
    logs: list[dict[str, Any]] = []
    for name, promql in queries.items():
        part, log = fetch_range_raw(
            promql,
            url=url,
            start_ts=start_ts,
            end_ts=end_ts,
            step=step,
            chunk_days=chunk_days,
            series_name=name,
        )
        logs.append(log)
        if not part.empty:
            frames.append(part)
    if not frames:
        raise CortexSourceError("Cortex returned no continuous derivative history")
    raw = pd.concat(frames, ignore_index=True)
    raw["time_utc"] = pd.to_datetime(raw["timestamp"], unit="s", utc=True)
    return raw, logs


def hourly_from_raw(raw: pd.DataFrame, horizon_names: list[str], *, required: bool = True) -> pd.DataFrame:
    """Unfilled hourly grid with source provenance. Never max() of signed D."""
    return collapse_hourly_from_raw(raw, horizon_names, required=required)


def hourly_wide_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """All series_name values, unfilled hourly grid, including research-only primes."""
    names = [str(n) for n in raw["series_name"].dropna().unique().tolist()]
    if not names:
        raise CortexSourceError("raw Cortex tape has no series_name values")
    return hourly_from_raw(raw, names, required=False)
