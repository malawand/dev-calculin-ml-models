"""Parameterized binary baseline: same statistics as the original four scripts."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from . import BASELINE_VERSION
from .config import clickhouse_settings, cortex_url
from .cortex_client import fetch_named_queries
from .prices import fetch_hourly_closes
from .quality import bounded_ffill, quality_summary, reindex_hourly


FEATURE_RENAME = {
    "poly_transition_weighted24h48h7d": "poly_transition(_weighted24h48h7d)",
}


def fetch_baseline_features(cfg: dict[str, Any]) -> pd.DataFrame:
    period = cfg["period"]
    cortex = cfg["cortex"]
    raw = fetch_named_queries(
        cortex["queries"],
        url=cortex_url(cfg),
        start=period["start"],
        end=period["end"],
        step=cortex.get("step", "1h"),
        chunk_days=int(cortex.get("chunk_days", 180)),
    )
    hourly = reindex_hourly(raw, "time_bucket")
    max_h = int(cortex.get("max_staleness_hours", 2))
    cols = list(cortex["queries"].keys())
    for col in cols:
        filled, age, valid = bounded_ffill(hourly[col], max_h)
        hourly[col] = filled
        hourly[f"{col}_source_age_hours"] = age
        hourly[f"{col}_feature_valid"] = valid
    # Rows where any required feature is invalid are dropped (same as original dropna).
    ok = hourly[cols].notna().all(axis=1)
    hourly = hourly.loc[ok].reset_index()
    return hourly


def align_prices(features: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    period = cfg["period"]
    ch = clickhouse_settings(cfg)
    prices = fetch_hourly_closes(
        host=ch["host"],
        port=ch["port"],
        username=ch["username"],
        password=ch["password"],
        table=ch["table"],
        symbol=ch["symbol"],
        start=period["start"],
        end=period["end"],
    )
    feat = features.copy()
    feat["time_bucket"] = pd.to_datetime(feat["time_bucket"], utc=True)
    prices["time_bucket"] = pd.to_datetime(prices["time_bucket"], utc=True)
    return feat.merge(prices, on="time_bucket", how="left")


def load_aligned_csv(path: str | Path) -> pd.DataFrame:
    """Replay baseline from a previously fetched aligned CSV. Not synthetic."""
    df = pd.read_csv(path)
    if "time_bucket" not in df.columns:
        raise ValueError(f"{path} must contain time_bucket")
    df["time_bucket"] = pd.to_datetime(df["time_bucket"], utc=True)
    reverse = {v: k for k, v in FEATURE_RENAME.items()}
    df = df.rename(columns=reverse)
    return df


def truth_table(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    table = pd.DataFrame(
        product([0.0, 1.0], repeat=len(feature_columns)),
        columns=feature_columns,
    )
    observed = (
        df.groupby(feature_columns, dropna=False).size().reset_index(name="occurrence_count")
    )
    results = table.merge(observed, on=feature_columns, how="left")
    results["occurrence_count"] = results["occurrence_count"].fillna(0).astype(int)
    results["ever_occurred"] = results["occurrence_count"] > 0
    return results


def classify_moves(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    hours: int = 24,
    threshold: float = 0.1,
    time_col: str = "time_bucket",
    price_col: str = "close_price",
    log_return: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce count_moves.py: hourly reindex, shift(-hours) is hours not rows."""
    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=True, errors="coerce")
    work = (
        work.dropna(subset=[time_col])
        .sort_values(time_col)
        .drop_duplicates(subset=[time_col], keep="last")
        .set_index(time_col)
    )
    for c in feature_columns + [price_col]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.reindex(pd.date_range(work.index.min(), work.index.max(), freq="h", tz="UTC"))
    ok = work[feature_columns].notna().all(axis=1)
    work["state"] = None
    work.loc[ok, "state"] = work.loc[ok, feature_columns].astype(int).astype(str).agg("".join, axis=1)
    px = work[price_col]
    if log_return:
        import numpy as np

        work["change"] = (np.log(px.shift(-hours) / px)) * 100
    else:
        work["change"] = (px.shift(-hours) / px - 1) * 100
    work["result"] = None
    work.loc[work["change"] > threshold, "result"] = "UP"
    work.loc[work["change"] < -threshold, "result"] = "DOWN"
    work.loc[work["change"].abs() <= threshold, "result"] = "FLAT"
    rows = work.dropna(subset=["state", "result"])
    summary_rows = []
    states = ["".join(t) for t in product("01", repeat=len(feature_columns))]
    for s in states:
        g = rows[rows["state"] == s]
        n = len(g)
        rec = {"state": s, "UP": 0, "DOWN": 0, "FLAT": 0, "TOTAL": n, "UP_pct": None, "DOWN_pct": None, "FLAT_pct": None}
        if n:
            rec["UP"] = int((g["result"] == "UP").sum())
            rec["DOWN"] = int((g["result"] == "DOWN").sum())
            rec["FLAT"] = int((g["result"] == "FLAT").sum())
            rec["UP_pct"] = round(100 * rec["UP"] / n, 1)
            rec["DOWN_pct"] = round(100 * rec["DOWN"] / n, 1)
            rec["FLAT_pct"] = round(100 * rec["FLAT"] / n, 1)
        summary_rows.append(rec)
    n = len(rows)
    all_rec = {
        "state": "ALL",
        "UP": int((rows["result"] == "UP").sum()) if n else 0,
        "DOWN": int((rows["result"] == "DOWN").sum()) if n else 0,
        "FLAT": int((rows["result"] == "FLAT").sum()) if n else 0,
        "TOTAL": n,
        "UP_pct": round(100 * (rows["result"] == "UP").mean(), 1) if n else None,
        "DOWN_pct": round(100 * (rows["result"] == "DOWN").mean(), 1) if n else None,
        "FLAT_pct": round(100 * (rows["result"] == "FLAT").mean(), 1) if n else None,
    }
    summary_rows.append(all_rec)
    return rows.reset_index().rename(columns={"index": time_col}), pd.DataFrame(summary_rows)


def write_baseline_report(summary: pd.DataFrame, dest: Path, meta: dict[str, Any]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            **meta,
            "baseline_version": meta.get("baseline_version", BASELINE_VERSION),
        },
        "quality": meta.get("quality"),
        "states": summary.to_dict(orient="records"),
    }
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md = dest.with_suffix(".md")
    lines = [
        "# Binary baseline report",
        "",
        f"- Baseline version: {meta.get('baseline_version', BASELINE_VERSION)}",
        f"- Lookahead hours: {meta.get('hours')}",
        f"- FLAT threshold: ±{meta.get('flat_threshold_pct')}%",
        f"- Features: {', '.join(meta.get('features', []))}",
        "",
        "| state | UP | DOWN | FLAT | TOTAL | UP% | DOWN% | FLAT% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary.to_dict(orient="records"):
        lines.append(
            f"| {r['state']} | {r['UP']} | {r['DOWN']} | {r['FLAT']} | {r['TOTAL']} "
            f"| {r['UP_pct']} | {r['DOWN_pct']} | {r['FLAT_pct']} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
