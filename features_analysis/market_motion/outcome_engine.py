"""Outcome v1: multi-horizon simple/log returns, close-path MFE/MAE, derived labels.

Features at observation t use the close at t; they are not executable at that close.
Canonical outcomes begin at ``outcome_entry_time_utc`` (feature available + decision
latency). Incomplete windows stay unlabeled (NaN), not zero. Missing future hours
are not bridged.

MFE/MAE are **close-path** (hourly closes). They underestimate intrahour extremes
and are not tick-level executable excursions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import OUTCOME_VERSION


def _pick(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _timing(cfg: dict[str, Any]) -> tuple[int, int]:
    tcfg = cfg.get("timing") or {}
    vcfg = cfg.get("vector") or {}
    feature_lag = int(
        _pick(
            tcfg,
            "feature_available_lag_hours",
            "feature_available_lag_hours",
            default=_pick(vcfg, "execution_lag_hours", "execution_lag_hours", default=1),
        )
    )
    decision_lag = int(_pick(tcfg, "decision_latency_hours", "decision_latency_hours", default=0))
    return feature_lag, decision_lag


def _future_window(px: np.ndarray, entry: int, k: int) -> np.ndarray | None:
    """k hourly closes after entry. Requires a complete finite path; no bridging."""
    end = entry + k
    if entry < 0 or end >= len(px):
        return None
    if not np.isfinite(px[entry]):
        return None
    window = px[entry + 1 : end + 1]
    if window.size < k or not np.isfinite(window).all():
        return None
    return window


def unlabeled_tail_counts(oc: pd.DataFrame, horizons: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in horizons:
        col = f"return_{k}h"
        if col not in oc.columns:
            continue
        na = oc[col].isna()
        out[col] = {
            "n_unlabeled": int(na.sum()),
            "n_labeled": int((~na).sum()),
            "unlabeled_fraction": float(na.mean()) if len(oc) else None,
        }
    return out


def build_outcomes(
    prices: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    time_col: str = "time_bucket",
    price_col: str = "close_price",
) -> pd.DataFrame:
    oc = cfg["outcomes"]
    feature_lag, decision_lag = _timing(cfg)
    entry_lag = feature_lag + decision_lag
    work = prices.copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=True)
    work = work.dropna(subset=[time_col]).sort_values(time_col)
    work = work.drop_duplicates(time_col, keep="last").set_index(time_col)
    work = work.reindex(pd.date_range(work.index.min(), work.index.max(), freq="h", tz="UTC"))
    px = pd.to_numeric(work[price_col], errors="coerce").to_numpy(dtype=float)
    n = len(px)
    out = pd.DataFrame(index=work.index)
    out["observation_time_utc"] = out.index
    out["feature_available_time_utc"] = out.index + pd.Timedelta(hours=feature_lag)
    out["decision_time_utc"] = out["feature_available_time_utc"] + pd.Timedelta(
        hours=decision_lag
    )
    out["outcome_entry_time_utc"] = out["decision_time_utc"]
    out["outcome_version"] = oc.get("version", OUTCOME_VERSION)
    out["path_convention"] = oc.get("path_convention", "hourly_close")
    out["outcome_entry_lag_hours"] = entry_lag
    out["btc_close_at_t"] = px
    entry_px = np.full(n, np.nan)
    if entry_lag == 0:
        entry_px = px.copy()
    elif entry_lag < n:
        entry_px[: n - entry_lag] = px[entry_lag:]
    out["btc_close_at_entry"] = entry_px

    ret_h = [int(h) for h in _pick(oc, "return_horizons_h", "return_horizons_h")]
    mfe_h = [int(h) for h in _pick(oc, "excursion_horizons_h", "excursion_horizons_h")]
    thr = float(_pick(oc, "flat_threshold_pct", "flat_threshold_pct", default=0.1))

    for k in ret_h:
        simple = np.full(n, np.nan)
        logret = np.full(n, np.nan)
        for i in range(n):
            e = i + entry_lag
            w = _future_window(px, e, k)
            if w is None:
                continue
            simple[i] = (w[-1] / px[e] - 1.0) * 100.0
            logret[i] = float(np.log(w[-1] / px[e]) * 100.0)
        out[f"return_{k}h"] = simple
        out[f"log_return_{k}h"] = logret
        lab = np.full(n, None, dtype=object)
        ok = np.isfinite(simple)
        lab[ok & (simple > thr)] = "UP"
        lab[ok & (simple < -thr)] = "DOWN"
        lab[ok & np.isfinite(simple) & (np.abs(simple) <= thr)] = "FLAT"
        out[f"label_up_down_flat_{k}h"] = lab

    for k in mfe_h:
        mfe = np.full(n, np.nan)
        mae = np.full(n, np.nan)
        t_mfe = np.full(n, np.nan)
        t_mae = np.full(n, np.nan)
        for i in range(n):
            e = i + entry_lag
            w = _future_window(px, e, k)
            if w is None:
                continue
            path = (w / px[e] - 1.0) * 100.0
            imax = int(np.argmax(path))
            imin = int(np.argmin(path))
            mfe[i] = float(path[imax])
            mae[i] = float(path[imin])
            t_mfe[i] = imax + 1
            t_mae[i] = imin + 1
        out[f"close_path_mfe_{k}h"] = mfe
        out[f"close_path_mae_{k}h"] = mae
        out[f"time_to_close_path_mfe_{k}h"] = t_mfe
        out[f"time_to_close_path_mae_{k}h"] = t_mae

    out.index.name = time_col
    return out.reset_index()
