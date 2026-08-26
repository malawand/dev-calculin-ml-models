"""Representation diagnostics and spot checks. No 24h-return scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .transition import EVENT_PEAK, EVENT_TROUGH


def _ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True)


def spot_events(vec: pd.DataFrame, horizon: str, event: str, *, n: int = 8) -> pd.DataFrame:
    event_col = f"{horizon}_transition_event"
    d_col = f"{horizon}_derivative_raw"
    cand_col = f"{horizon}_candidate_extremum_time"
    age_col = f"{horizon}_hours_since_confirmation"
    cols = [
        "time_bucket",
        event_col,
        d_col,
        cand_col,
        age_col,
        f"{horizon}_transition_state_raw",
        f"{horizon}_feature_valid",
    ]
    have = [c for c in cols if c in vec.columns]
    rows = vec.loc[vec[event_col] == event, have].copy() if event_col in vec.columns else pd.DataFrame()
    if rows.empty:
        return rows
    return rows.head(n)


def longest_states(vec: pd.DataFrame, horizon: str, state: float, *, n: int = 6) -> pd.DataFrame:
    state_col = f"{horizon}_transition_state_raw"
    age_col = f"{horizon}_hours_since_confirmation"
    if state_col not in vec.columns or age_col not in vec.columns:
        return pd.DataFrame()
    work = vec.loc[vec[state_col] == state].copy()
    if work.empty:
        return work
    work = work.sort_values(age_col, ascending=False)
    keep = [
        "time_bucket",
        state_col,
        age_col,
        f"{horizon}_derivative_raw",
        f"{horizon}_transition_event",
        f"{horizon}_feature_valid",
    ]
    return work[[c for c in keep if c in work.columns]].head(n)


def horizon_disagreements(vec: pd.DataFrame, short: str, long: str, *, n: int = 12) -> pd.DataFrame:
    s_col = f"{short}_transition_state_raw"
    l_col = f"{long}_transition_state_raw"
    if s_col not in vec.columns or l_col not in vec.columns:
        return pd.DataFrame()
    mask = vec[s_col].notna() & vec[l_col].notna() & (vec[s_col] != vec[l_col])
    keep = [
        "time_bucket",
        s_col,
        l_col,
        f"{short}_derivative_raw",
        f"{long}_derivative_raw",
        f"{short}_hours_since_confirmation",
        f"{long}_hours_since_confirmation",
    ]
    out = vec.loc[mask, [c for c in keep if c in vec.columns]]
    return out.head(n)


def source_gap_rows(vec: pd.DataFrame, horizon: str, *, n: int = 12) -> pd.DataFrame:
    present = f"{horizon}_source_present"
    if present not in vec.columns:
        return pd.DataFrame()
    keep = [
        "time_bucket",
        present,
        f"{horizon}_source_age_hours",
        f"{horizon}_derivative_raw",
        f"{horizon}_feature_valid",
        f"{horizon}_invalidation_reason",
    ]
    out = vec.loc[~vec[present].fillna(False), [c for c in keep if c in vec.columns]]
    return out.head(n)


def cortex_vs_ols_disagreement(
    cortex: pd.DataFrame,
    ols: pd.DataFrame,
    *,
    cortex_horizon: str,
    ols_horizon: str,
    n: int = 12,
) -> dict[str, Any]:
    """Diagnostic only. Does not combine sources or pick a winner on 24h returns."""
    time_col = "time_bucket"
    c = cortex.copy()
    o = ols.copy()
    c[time_col] = _ts(c[time_col])
    o[time_col] = _ts(o[time_col])
    ct = f"{cortex_horizon}_transition_state_raw"
    ot = f"{ols_horizon}_transition_state_raw"
    cd = f"{cortex_horizon}_derivative_raw"
    od = f"{ols_horizon}_derivative_raw"
    if ct not in c.columns or ot not in o.columns:
        return {"n_overlap": 0, "note": "required columns missing"}
    left = c[[time_col, ct, cd]].rename(columns={ct: "cortex_t", cd: "cortex_d"})
    right = o[[time_col, ot, od]].rename(columns={ot: "ols_t", od: "ols_d"})
    m = left.merge(right, on=time_col, how="inner")
    both_t = m["cortex_t"].notna() & m["ols_t"].notna()
    t_dis = both_t & (m["cortex_t"] != m["ols_t"])
    both_d = m["cortex_d"].notna() & m["ols_d"].notna()
    sign_dis = both_d & (np.sign(m["cortex_d"].to_numpy()) != np.sign(m["ols_d"].to_numpy()))
    sample = m.loc[t_dis, [time_col, "cortex_t", "ols_t", "cortex_d", "ols_d"]].head(n)
    return {
        "cortex_horizon": cortex_horizon,
        "ols_horizon": ols_horizon,
        "n_overlap": int(len(m)),
        "n_both_state_known": int(both_t.sum()),
        "n_state_disagree": int(t_dis.sum()),
        "state_disagree_fraction": float(t_dis.sum() / both_t.sum()) if both_t.any() else None,
        "n_both_d_known": int(both_d.sum()),
        "n_d_sign_disagree": int(sign_dis.sum()),
        "d_sign_disagree_fraction": float(sign_dis.sum() / both_d.sum()) if both_d.any() else None,
        "sample_state_disagreements": sample.to_dict(orient="records"),
        "note": "Diagnostic only. Do not concatenate or select using the 24h target.",
    }


def write_spot_tables(vec: pd.DataFrame, dest: Path, horizons: list[str]) -> dict[str, str]:
    dest.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    short = horizons[0] if horizons else "2h"
    long = "7d" if "7d" in horizons else horizons[-1]
    frames = {
        "spot_peaks.csv": pd.concat(
            [spot_events(vec, h, EVENT_PEAK).assign(horizon=h) for h in horizons],
            ignore_index=True,
        ),
        "spot_troughs.csv": pd.concat(
            [spot_events(vec, h, EVENT_TROUGH).assign(horizon=h) for h in horizons],
            ignore_index=True,
        ),
        "spot_long_rising.csv": pd.concat(
            [longest_states(vec, h, 0.0).assign(horizon=h) for h in horizons],
            ignore_index=True,
        ),
        "spot_long_falling.csv": pd.concat(
            [longest_states(vec, h, 1.0).assign(horizon=h) for h in horizons],
            ignore_index=True,
        ),
        "spot_horizon_disagreement.csv": horizon_disagreements(vec, short, long),
        "spot_source_gaps.csv": pd.concat(
            [source_gap_rows(vec, h).assign(horizon=h) for h in horizons],
            ignore_index=True,
        ),
    }
    for name, frame in frames.items():
        path = dest / name
        frame.to_csv(path, index=False)
        written[name] = str(path)
    return written


def write_horizon_plots(vec: pd.DataFrame, dest: Path, horizons: list[str]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    dest.mkdir(parents=True, exist_ok=True)
    t = _ts(vec["time_bucket"])
    paths: list[str] = []
    for h in horizons:
        d_col = f"{h}_derivative_raw"
        e_col = f"{h}_transition_event"
        s_col = f"{h}_transition_state_raw"
        if d_col not in vec.columns:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        axes[0].plot(t, vec[d_col], color="steelblue", linewidth=0.8, label=f"{h} D")
        peaks = vec[e_col] == EVENT_PEAK if e_col in vec.columns else pd.Series(False, index=vec.index)
        troughs = vec[e_col] == EVENT_TROUGH if e_col in vec.columns else pd.Series(False, index=vec.index)
        if peaks.any():
            axes[0].scatter(t[peaks], vec.loc[peaks, d_col], color="crimson", s=12, label="peak confirm", zorder=3)
        if troughs.any():
            axes[0].scatter(t[troughs], vec.loc[troughs, d_col], color="seagreen", s=12, label="trough confirm", zorder=3)
        axes[0].set_ylabel("Cortex D")
        axes[0].legend(loc="upper left", fontsize=8)
        axes[0].set_title(f"{h} Cortex derivative and confirmed turns (detector v2)")
        if s_col in vec.columns:
            axes[1].step(t, vec[s_col], where="post", color="black", linewidth=0.8)
            axes[1].set_ylabel("T (0 up / 1 down / null)")
            axes[1].set_ylim(-0.2, 1.2)
        axes[1].set_xlabel("UTC")
        fig.tight_layout()
        path = dest / f"diag_{h}_turns.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(str(path))
    return paths
