"""Deterministic Cortex source selection. Never max() signed values.

Policy version: prod_a_preferred_v1

Owner clarification: cluster/replica labels are routing/provenance for the
same feature family, not different derivative formulas.

- Live retrieval selects cluster=PROD-A-crypto-metrics.
- Historical tape prefers PROD-A when present, else the next approved
  production delivery source at that hour.
- Routing-label changes are not gaps and do not reset Z/V/A/T.
- True conflict only when no approved source can be selected.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .exceptions import CortexSourceError
from .quality import reindex_hourly

SELECTED_VALUE_RULE = "prod_a_preferred_v1"
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-12

# Owner-approved historical fallback. Do not reorder without reporting.
CLUSTER_PRIORITY: tuple[str, ...] = (
    "PROD-A-crypto-metrics",
    "PROD-B-crypto-metrics",
    "PROD-crypto-metrics",
    "prod-calculin-tokyo-binance-scraper",
    "prod-prom-cluster-1",
    "DEV-crypto-metrics",
    "unlabeled",
)
CANONICAL_CLUSTER = CLUSTER_PRIORITY[0]
CLUSTER_RANK = {name: i for i, name in enumerate(CLUSTER_PRIORITY)}
UNAPPROVED_RANK = len(CLUSTER_PRIORITY)

PROVENANCE_SUFFIXES = (
    "source_cluster",
    "source_replica",
    "n_candidates",
    "source_conflict",
    "source_value_disagree",
    "source_boundary",
    "source_segment_id",
    "selected_value_rule",
)


def values_agree(values: np.ndarray, *, rtol: float, atol: float) -> bool:
    finite = values[np.isfinite(values)]
    if len(finite) <= 1:
        return True
    ref = float(np.min(finite))
    return bool(np.all(np.abs(finite - ref) <= (atol + rtol * np.abs(ref))))


def normalize_cluster(val: Any) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)) or str(val).strip() == "":
        return "unlabeled"
    return str(val)


def cluster_rank(cluster: str) -> int:
    return CLUSTER_RANK.get(normalize_cluster(cluster), UNAPPROVED_RANK)


def is_approved_cluster(cluster: str) -> bool:
    return cluster_rank(cluster) < UNAPPROVED_RANK


def _empty_rec(n: int = 0) -> dict[str, Any]:
    return {
        "n_candidates": n,
        "source_cluster": "",
        "source_replica": "",
        "selected_value_rule": SELECTED_VALUE_RULE,
        "source_conflict": False,
        "source_value_disagree": False,
        "value": float("nan"),
    }


def collapse_timestamp_group(
    rows: pd.DataFrame,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> dict[str, Any]:
    """Select the highest-priority approved source. Never max() of signed D."""
    n = int(len(rows))
    rec = _empty_rec(n)
    if n == 0:
        return rec
    vals = pd.to_numeric(rows["value"], errors="coerce")
    clusters = [normalize_cluster(x) for x in rows["cluster"]] if "cluster" in rows.columns else ["unlabeled"] * n
    replicas = [
        "" if pd.isna(x) or str(x) == "nan" else str(x)
        for x in (rows["replica"] if "replica" in rows.columns else [""] * n)
    ]
    work = pd.DataFrame(
        {
            "value": vals.to_numpy(dtype=float),
            "cluster": clusters,
            "replica": replicas,
            "rank": [cluster_rank(c) for c in clusters],
            "ord": np.arange(n),
        }
    )
    finite = work["value"].to_numpy(dtype=float)
    rec["source_value_disagree"] = bool(
        (not values_agree(finite, rtol=rtol, atol=atol)) and np.isfinite(finite).sum() >= 2
    )
    approved = work.loc[work["rank"].lt(UNAPPROVED_RANK) & np.isfinite(work["value"].to_numpy(dtype=float))]
    if approved.empty:
        rec["source_conflict"] = bool(np.isfinite(finite).any())
        rec["source_cluster"] = "|".join(sorted(set(clusters)))
        rec["source_replica"] = "|".join(sorted({r for r in replicas if r}))
        return rec
    approved = approved.sort_values(["rank", "replica", "ord"], kind="mergesort")
    win = approved.iloc[0]
    rec["value"] = float(win["value"])
    rec["source_cluster"] = str(win["cluster"])
    rec["source_replica"] = str(win["replica"])
    rec["source_conflict"] = False
    return rec


def collapse_series_hourly(
    raw: pd.DataFrame,
    series_name: str,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> pd.DataFrame:
    """Hourly selected value plus provenance. Does not fill gaps."""
    empty_cols = [
        "timestamp",
        "value",
        "source_cluster",
        "source_replica",
        "n_candidates",
        "source_conflict",
        "source_value_disagree",
        "selected_value_rule",
    ]
    part = raw.loc[raw["series_name"] == series_name].copy()
    if part.empty:
        return pd.DataFrame(columns=empty_cols)
    if "cluster" not in part.columns:
        part["cluster"] = ""
    if "replica" not in part.columns:
        part["replica"] = ""
    part["cluster"] = part["cluster"].map(normalize_cluster)
    part["replica"] = part["replica"].fillna("").replace("nan", "").astype(str)
    part["value"] = pd.to_numeric(part["value"], errors="coerce")
    part["_rank"] = part["cluster"].map(cluster_rank)
    part["_ord"] = np.arange(len(part))
    n = part.groupby("timestamp", sort=True).size()
    vmin = part.groupby("timestamp", sort=True)["value"].min()
    vmax = part.groupby("timestamp", sort=True)["value"].max()
    disagree = (vmax - vmin) > (atol + rtol * vmin.abs())
    disagree = disagree.fillna(False) & vmin.notna() & vmax.notna()

    approved = part.loc[part["_rank"].lt(UNAPPROVED_RANK) & part["value"].notna()].copy()
    if approved.empty:
        out = pd.DataFrame(
            {
                "timestamp": n.index.astype(int),
                "value": np.nan,
                "source_cluster": "",
                "source_replica": "",
                "n_candidates": n.to_numpy(dtype=int),
                "source_conflict": True,
                "source_value_disagree": disagree.reindex(n.index).fillna(False).to_numpy(dtype=bool),
                "selected_value_rule": SELECTED_VALUE_RULE,
            }
        )
        return out
    approved = approved.sort_values(["timestamp", "_rank", "replica", "_ord"], kind="mergesort")
    selected = approved.groupby("timestamp", sort=True).first()
    all_ts = n.index
    value = selected["value"].reindex(all_ts)
    cluster = selected["cluster"].reindex(all_ts)
    replica = selected["replica"].reindex(all_ts)
    has_finite = part.assign(_ok=part["value"].notna()).groupby("timestamp")["_ok"].any()
    conflict = value.isna() & has_finite.reindex(all_ts).fillna(False)
    out = pd.DataFrame(
        {
            "timestamp": all_ts.astype(int),
            "value": value.to_numpy(dtype=float),
            "source_cluster": cluster.fillna("").astype(str).to_numpy(),
            "source_replica": replica.fillna("").astype(str).to_numpy(),
            "n_candidates": n.reindex(all_ts).fillna(0).to_numpy(dtype=int),
            "source_conflict": conflict.fillna(False).to_numpy(dtype=bool),
            "source_value_disagree": disagree.reindex(all_ts).fillna(False).to_numpy(dtype=bool),
            "selected_value_rule": SELECTED_VALUE_RULE,
        }
    )
    return out


def mark_source_boundaries(cluster: pd.Series, present: pd.Series) -> pd.Series:
    """Diagnostic: first hour of a new routing label. Does not invalidate features."""
    lin = cluster.where(present)
    last_val: str | None = None
    out = pd.Series(False, index=cluster.index)
    for idx, val in lin.items():
        if pd.isna(val) or val == "":
            continue
        sval = str(val)
        if last_val is not None and sval != last_val:
            out.loc[idx] = True
        last_val = sval
    return out


def segment_ids(cluster: pd.Series, present: pd.Series, conflict: pd.Series) -> pd.Series:
    """Diagnostic routing-era id. Not used to reset Z/V/A/T."""
    ids = np.full(len(cluster), -1, dtype=int)
    last: str | None = None
    sid = 0
    started = False
    lin = cluster.to_numpy()
    pres = present.fillna(False).to_numpy()
    conf = conflict.fillna(False).to_numpy()
    for i in range(len(cluster)):
        if conf[i]:
            ids[i] = sid if started else -1
            continue
        if not pres[i]:
            ids[i] = sid if started else -1
            continue
        cur = str(lin[i]) if pd.notna(lin[i]) and str(lin[i]) != "" else "unlabeled"
        if started and last is not None and cur != last:
            sid += 1
        started = True
        last = cur
        ids[i] = sid
    return pd.Series(ids, index=cluster.index)


def hourly_from_raw(
    raw: pd.DataFrame,
    horizon_names: list[str],
    *,
    required: bool = True,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> pd.DataFrame:
    """Unfilled hourly grid with provenance. OLS is never substituted."""
    pieces: list[pd.DataFrame] = []
    missing: list[str] = []
    for name in horizon_names:
        collapsed = collapse_series_hourly(raw, name, rtol=rtol, atol=atol)
        if collapsed.empty:
            missing.append(name)
            continue
        collapsed["time_bucket"] = pd.to_datetime(collapsed["timestamp"], unit="s", utc=True)
        collapsed = collapsed.drop(columns=["timestamp"])
        renamed = collapsed.rename(
            columns={
                "value": name,
                "source_cluster": f"{name}_source_cluster",
                "source_replica": f"{name}_source_replica",
                "n_candidates": f"{name}_n_candidates",
                "source_conflict": f"{name}_source_conflict",
                "source_value_disagree": f"{name}_source_value_disagree",
                "selected_value_rule": f"{name}_selected_value_rule",
            }
        )
        pieces.append(renamed.set_index("time_bucket"))
    if required and missing:
        raise CortexSourceError(
            f"no hourly values for required Cortex horizon(s) {missing!r}; OLS will not be substituted"
        )
    if not pieces:
        raise CortexSourceError("no Cortex hourly series after collapse")
    wide = pieces[0]
    for extra in pieces[1:]:
        wide = wide.join(extra, how="outer")
    wide.index.name = "time_bucket"
    out = wide.reset_index()
    out = reindex_hourly(out, "time_bucket").reset_index()
    for name in horizon_names:
        if name not in out.columns:
            continue
        present = out[name].notna()
        conf_col = f"{name}_source_conflict"
        if conf_col not in out.columns:
            out[conf_col] = False
        out[conf_col] = out[conf_col].fillna(False).astype(bool)
        if f"{name}_source_value_disagree" not in out.columns:
            out[f"{name}_source_value_disagree"] = False
        out[f"{name}_source_value_disagree"] = out[f"{name}_source_value_disagree"].fillna(False).astype(bool)
        cl_col = f"{name}_source_cluster"
        if cl_col not in out.columns:
            out[cl_col] = ""
        out[f"{name}_source_boundary"] = mark_source_boundaries(out[cl_col], present)
        out[f"{name}_source_segment_id"] = segment_ids(out[cl_col], present, out[conf_col])
        if f"{name}_n_candidates" in out.columns:
            out[f"{name}_n_candidates"] = pd.to_numeric(out[f"{name}_n_candidates"], errors="coerce")
        if f"{name}_selected_value_rule" in out.columns:
            out[f"{name}_selected_value_rule"] = out[f"{name}_selected_value_rule"].fillna(SELECTED_VALUE_RULE)
    return out
