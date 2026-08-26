"""POC analogue matching. Development period only. Never mixes Cortex and OLS.

PROVISIONAL POC — temporary Cortex-derived dataset

Distance is an equal-weight mixed L1 on the explicit model-feature allowlist:
binary T mismatch plus robust-scaled absolute differences of continuous
features. Scaling uses only eligible past candidates at the query hour.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .exceptions import VectorSourceError
from .feature_allowlist import infer_horizons, select_model_feature_columns
from .periods import (
    DEVELOPMENT_START_UTC,
    DISCLAIMER,
    RETROSPECTIVE_HOLDOUT_START_UTC,
    as_utc,
    in_development,
    in_prospective_locked,
    in_retrospective_holdout,
    last_observation_with_complete_outcome,
)
from .source_guard import assert_single_vector_identity, outcome_columns

ANALOGUE_VERSION = "mixed_l1_equal_weight_poc_v1"
OUTCOME_COL = "return_24h"

BINARY_SUFFIX = "transition_state_raw"


def analogue_defaults(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = (cfg or {}).get("analogue") or {}
    return {
        "version": str(raw.get("version", ANALOGUE_VERSION)),
        "k": int(raw.get("k", 40)),
        "min_neighbors": int(raw.get("min_neighbors", 12)),
        "min_effective_support": int(raw.get("min_effective_support", 8)),
        "min_separation_hours": int(raw.get("min_separation_hours", 24)),
        "entry_lag_hours": int(raw.get("entry_lag_hours", 1)),
        "outcome_horizon_hours": int(raw.get("outcome_horizon_hours", 24)),
        "bootstrap_draws": int(raw.get("bootstrap_draws", 400)),
        "n_stride_queries": int(raw.get("n_stride_queries", 48)),
        "disclaimer": DISCLAIMER,
    }


def _times(df: pd.DataFrame, time_col: str = "time_bucket") -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(df[time_col], utc=True))


def outcome_purge_lag_hours(entry_lag_hours: int, outcome_horizon_hours: int) -> int:
    """Hours a candidate must sit before the query so its outcome window has finished.

    Entry is the next close (default +1h). The outcome then runs ``outcome_horizon_hours``.
    One extra hour makes the last outcome close strictly earlier than the query bar:

    ``cutoff = query - (entry_lag + outcome_horizon + 1)``

    With the default 1h entry lag that is 6h / 10h / 14h / 26h for 4h / 8h / 12h / 24h.
    """
    return int(entry_lag_hours) + int(outcome_horizon_hours) + 1


def prepare_analogue_frame(
    vector: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    time_col: str = "time_bucket",
    outcome_col: str = OUTCOME_COL,
    outcome_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Join vector to outcome returns. Distance features are allowlist-only."""
    assert_single_vector_identity(vector, context="analogue vector: ")
    leaked = outcome_columns(vector)
    if leaked:
        raise VectorSourceError(f"analogue vector must not contain outcome columns: {leaked!r}")
    hs = infer_horizons(vector.columns)
    features = select_model_feature_columns(vector, hs)
    if not features:
        raise VectorSourceError("analogue allowlist selected no feature columns")
    banned = (
        "source_cluster",
        "source_replica",
        "source_conflict",
        "source_boundary",
        "invalidation_reason",
        "transition_event",
        "confirmation_time",
        "candidate_extremum",
        "oracle",
    )
    for c in features:
        if any(b in c for b in banned):
            raise VectorSourceError(f"metadata leaked into analogue features: {c}")
    cols = list(outcome_cols or [outcome_col])
    missing = [c for c in cols if c not in outcomes.columns]
    if missing:
        raise VectorSourceError(f"outcomes table missing required columns: {missing!r}")
    oc = outcomes[[time_col, *cols]].copy()
    oc[time_col] = pd.to_datetime(oc[time_col], utc=True)
    work = vector.copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=True)
    work = work.merge(oc, on=time_col, how="left")
    work = work.sort_values(time_col).reset_index(drop=True)
    return work, features


def development_eligible_mask(
    df: pd.DataFrame,
    *,
    time_col: str = "time_bucket",
    outcome_col: str = OUTCOME_COL,
    entry_lag_hours: int = 1,
    outcome_horizon_hours: int = 24,
) -> pd.Series:
    times = _times(df, time_col)
    last = last_observation_with_complete_outcome(
        period_end_exclusive=RETROSPECTIVE_HOLDOUT_START_UTC,
        entry_lag_hours=entry_lag_hours,
        outcome_horizon_hours=outcome_horizon_hours,
    )
    valid = df["row_vector_valid"].fillna(False).astype(bool) if "row_vector_valid" in df.columns else pd.Series(True, index=df.index)
    labeled = pd.to_numeric(df[outcome_col], errors="coerce").notna()
    return in_development(times) & (times <= last) & valid & labeled


def assert_not_holdout(times: pd.DatetimeIndex, *, context: str = "query") -> None:
    if bool(in_retrospective_holdout(times).any()) or bool(in_prospective_locked(times).any()):
        raise VectorSourceError(
            f"{context} times fall in a closed holdout; analogue POC may not open or score holdouts"
        )


def _split_feature_kinds(columns: list[str]) -> tuple[list[str], list[str]]:
    binary = [c for c in columns if c.endswith(BINARY_SUFFIX)]
    continuous = [c for c in columns if c not in binary]
    return binary, continuous


def _robust_scale(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(train, axis=0)
    mad = np.nanmedian(np.abs(train - med), axis=0)
    scale = np.where(np.isfinite(mad) & (mad > 1e-12), mad, 1.0)
    med = np.where(np.isfinite(med), med, 0.0)
    return (train - med) / scale, (test - med) / scale


def mixed_l1_distance(
    query_row: pd.Series,
    candidates: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    """Equal-weight mixed L1. Scale continuous columns from candidates only."""
    binary, continuous = _split_feature_kinds(feature_cols)
    n = len(candidates)
    dist = np.zeros(n, dtype=float)
    n_terms = 0
    if binary:
        q_b = query_row[binary].to_numpy(dtype=float)
        c_b = candidates[binary].to_numpy(dtype=float)
        q_b2 = np.broadcast_to(q_b, c_b.shape)
        both = np.isfinite(q_b2) & np.isfinite(c_b)
        mismatch = np.zeros(c_b.shape, dtype=float)
        mismatch[both] = (q_b2[both] != c_b[both]).astype(float)
        dist = dist + mismatch.sum(axis=1)
        n_terms += int(np.isfinite(q_b).sum())
    if continuous:
        q_c = query_row[continuous].to_numpy(dtype=float).reshape(1, -1)
        c_c = candidates[continuous].to_numpy(dtype=float)
        usable = np.isfinite(q_c.reshape(-1)) & np.isfinite(c_c).any(axis=0)
        if usable.any():
            c_u = c_c[:, usable]
            q_u = q_c[:, usable]
            c_s, q_s = _robust_scale(c_u, q_u)
            delta = np.abs(c_s - q_s)
            both = np.isfinite(c_s) & np.isfinite(q_s)
            delta = np.where(both, delta, 0.0)
            dist = dist + delta.sum(axis=1)
            n_terms += int(usable.sum())
    if n_terms == 0:
        return np.full(n, np.nan)
    return dist / float(n_terms)


def _min_separation_pick(
    times: pd.DatetimeIndex,
    distances: np.ndarray,
    *,
    k: int,
    min_separation_hours: int,
) -> np.ndarray:
    order = np.argsort(distances, kind="mergesort")
    chosen: list[int] = []
    chosen_times: list[pd.Timestamp] = []
    gap = pd.Timedelta(hours=min_separation_hours)
    for idx in order:
        if not np.isfinite(distances[idx]):
            continue
        t = times[idx]
        if any(abs(t - ct) < gap for ct in chosen_times):
            continue
        chosen.append(int(idx))
        chosen_times.append(t)
        if len(chosen) >= k:
            break
    return np.asarray(chosen, dtype=int)


def summarize_returns(values: np.ndarray, *, bootstrap_draws: int = 400, rng: np.random.Generator | None = None) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = int(len(x))
    rec: dict[str, Any] = {"n": n, "unit": "percent"}
    if n == 0:
        rec["abstain"] = True
        return rec
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    rec.update(
        {
            "mean": float(x.mean()),
            "median": float(q[2]),
            "std": float(x.std(ddof=1)) if n > 1 else 0.0,
            "p05": float(q[0]),
            "p25": float(q[1]),
            "p75": float(q[3]),
            "p95": float(q[4]),
            "iqr": float(q[3] - q[1]),
            "probability_positive": float((x > 0).mean()),
            "probability_negative": float((x < 0).mean()),
        }
    )
    gen = rng or np.random.default_rng(0)
    if n >= 8 and bootstrap_draws > 0:
        draws = gen.choice(x, size=(bootstrap_draws, n), replace=True)
        meds = np.median(draws, axis=1)
        rec["median_bootstrap_p05"] = float(np.quantile(meds, 0.05))
        rec["median_bootstrap_p95"] = float(np.quantile(meds, 0.95))
        rec["uncertainty_note"] = "bootstrap 90% interval on the analogue-set median; neighbors are dependent"
    else:
        rec["uncertainty_note"] = "too few neighbours for bootstrap; report IQR only"
    return rec


def outcome_purge_cutoff(
    query_time: pd.Timestamp,
    *,
    entry_lag_hours: int = 1,
    outcome_horizon_hours: int = 24,
) -> pd.Timestamp:
    """Latest candidate time whose outcome window ends before the query hour."""
    return as_utc(query_time) - pd.Timedelta(
        hours=outcome_purge_lag_hours(entry_lag_hours, outcome_horizon_hours)
    )


def match_query(
    frame: pd.DataFrame,
    query_time: pd.Timestamp,
    feature_cols: list[str],
    *,
    time_col: str = "time_bucket",
    outcome_col: str = OUTCOME_COL,
    outcome_horizon_hours: int | None = None,
    cfg: dict[str, Any] | None = None,
    eligible: pd.Series | None = None,
) -> dict[str, Any]:
    """Select analogue neighbors and the past-only unconditional pool for one query.

    Distance is mixed L1 with k and 24h neighbour separation from analogue defaults.
    The outcome-window purge follows ``outcome_horizon_hours`` and is not reused
    from the 24h target unless that horizon is the one being scored.
    """
    params = analogue_defaults(cfg)
    horizon = int(params["outcome_horizon_hours"] if outcome_horizon_hours is None else outcome_horizon_hours)
    qtime = as_utc(query_time)
    assert_not_holdout(pd.DatetimeIndex([qtime]), context="query")
    times = _times(frame, time_col)
    if eligible is None:
        eligible = development_eligible_mask(
            frame,
            time_col=time_col,
            outcome_col=outcome_col,
            entry_lag_hours=int(params["entry_lag_hours"]),
            outcome_horizon_hours=horizon,
        )
    empty = {
        "query_time_utc": qtime,
        "abstain": True,
        "reason": "query_not_in_eligible_development_set",
        "n_pool": 0,
        "n_neighbors": 0,
        "neighbor_returns": np.array([], dtype=float),
        "neighbor_distances": np.array([], dtype=float),
        "uncond_returns": np.array([], dtype=float),
        "actual_return": float("nan"),
        "median_distance": None,
        "max_distance": None,
        "n_eligible_candidates_before_knn": 0,
        "neighbors_preview": [],
        "k_requested": int(params["k"]),
        "min_separation_hours": int(params["min_separation_hours"]),
        "params": params,
    }
    query_rows = frame.loc[eligible.to_numpy() & (times == qtime)]
    if query_rows.empty:
        return empty
    query = query_rows.iloc[0]
    actual = float(pd.to_numeric(query[outcome_col], errors="coerce"))
    cutoff = outcome_purge_cutoff(
        qtime,
        entry_lag_hours=int(params["entry_lag_hours"]),
        outcome_horizon_hours=horizon,
    )
    cand_mask = eligible.to_numpy() & (times <= cutoff)
    if not bool(cand_mask.any()):
        empty["reason"] = "no_earlier_eligible_candidates"
        empty["actual_return"] = actual
        return empty
    use_cols = [c for c in (time_col, outcome_col, *feature_cols) if c in frame.columns]
    candidates = frame.loc[cand_mask, use_cols]
    distances = mixed_l1_distance(query, candidates, feature_cols)
    cand_times = _times(candidates, time_col)
    picked = _min_separation_pick(
        cand_times,
        distances,
        k=int(params["k"]),
        min_separation_hours=int(params["min_separation_hours"]),
    )
    n_raw = int(np.isfinite(distances).sum())
    n_sel = int(len(picked))
    abstain = n_sel < int(params["min_neighbors"]) or n_sel < int(params["min_effective_support"])
    uncond = pd.to_numeric(candidates[outcome_col], errors="coerce").to_numpy(dtype=float)
    neighbor_returns = np.array([], dtype=float)
    neighbor_distances = np.array([], dtype=float)
    preview: list[dict[str, Any]] = []
    if n_sel:
        sel = candidates.iloc[picked]
        neighbor_distances = np.asarray(distances[picked], dtype=float)
        neighbor_returns = pd.to_numeric(sel[outcome_col], errors="coerce").to_numpy(dtype=float)
        for i, (_, row) in enumerate(sel.iterrows()):
            preview.append(
                {
                    "time_utc": str(as_utc(row[time_col])),
                    "distance": float(neighbor_distances[i]),
                    "return_24h_pct": None if not np.isfinite(neighbor_returns[i]) else float(neighbor_returns[i]),
                }
            )
    return {
        "query_time_utc": qtime,
        "abstain": bool(abstain),
        "reason": "too_few_similar_historical_examples" if abstain else None,
        "n_pool": int(len(candidates)),
        "n_neighbors": n_sel,
        "neighbor_returns": neighbor_returns,
        "neighbor_distances": neighbor_distances,
        "uncond_returns": uncond,
        "actual_return": actual,
        "median_distance": None if n_sel == 0 else float(np.median(neighbor_distances)),
        "max_distance": None if n_sel == 0 else float(np.max(neighbor_distances)),
        "n_eligible_candidates_before_knn": n_raw,
        "neighbors_preview": preview,
        "k_requested": int(params["k"]),
        "min_separation_hours": int(params["min_separation_hours"]),
        "params": params,
        "query_row": query,
        "cutoff_utc": cutoff,
        "outcome_col": outcome_col,
        "outcome_horizon_hours": horizon,
        "purge_lag_hours": outcome_purge_lag_hours(int(params["entry_lag_hours"]), horizon),
    }


def query_analogues(
    frame: pd.DataFrame,
    query_time: pd.Timestamp,
    feature_cols: list[str],
    *,
    time_col: str = "time_bucket",
    outcome_col: str = OUTCOME_COL,
    outcome_horizon_hours: int | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    match = match_query(
        frame,
        query_time,
        feature_cols,
        time_col=time_col,
        outcome_col=outcome_col,
        outcome_horizon_hours=outcome_horizon_hours,
        cfg=cfg,
    )
    source, version = assert_single_vector_identity(frame)
    params = match["params"]
    returns = match["neighbor_returns"]
    dist_sum = summarize_returns(returns, bootstrap_draws=int(params["bootstrap_draws"]))
    reason = match["reason"]
    if match["abstain"] and reason is None:
        reason = "too_few_similar_historical_examples"
    return {
        "disclaimer": DISCLAIMER,
        "analogue_version": params["version"],
        "vector_source": source,
        "vector_version": version,
        "query_time_utc": str(match["query_time_utc"]),
        "feature_columns": list(feature_cols),
        "n_eligible_candidates_before_knn": match["n_eligible_candidates_before_knn"],
        "n_neighbors": match["n_neighbors"],
        "effective_support": match["n_neighbors"],
        "k_requested": match["k_requested"],
        "min_separation_hours": match["min_separation_hours"],
        "median_distance": match["median_distance"],
        "max_distance": match["max_distance"],
        "neighbors": match["neighbors_preview"][:15],
        "conditional_return_24h": dist_sum,
        "abstain": bool(match["abstain"]),
        "reason": reason if match["abstain"] else None,
    }


def unconditional_development_returns(
    frame: pd.DataFrame,
    *,
    time_col: str = "time_bucket",
    outcome_col: str = OUTCOME_COL,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = analogue_defaults(cfg)
    mask = development_eligible_mask(
        frame,
        time_col=time_col,
        outcome_col=outcome_col,
        entry_lag_hours=int(params["entry_lag_hours"]),
        outcome_horizon_hours=int(params["outcome_horizon_hours"]),
    )
    values = pd.to_numeric(frame.loc[mask, outcome_col], errors="coerce").to_numpy(dtype=float)
    rec = summarize_returns(values, bootstrap_draws=int(params["bootstrap_draws"]))
    rec["disclaimer"] = DISCLAIMER
    rec["period"] = {
        "start_utc": str(DEVELOPMENT_START_UTC),
        "end_exclusive_utc": str(RETROSPECTIVE_HOLDOUT_START_UTC),
        "n_eligible_rows": int(mask.sum()),
    }
    return rec


def stride_query_times(frame: pd.DataFrame, eligible: pd.Series, n_queries: int, time_col: str = "time_bucket") -> list[pd.Timestamp]:
    times = _times(frame, time_col)[eligible.to_numpy()]
    if len(times) == 0:
        return []
    n_queries = max(1, min(int(n_queries), len(times)))
    idx = np.linspace(0, len(times) - 1, n_queries, dtype=int)
    # skip the first ~10% so each query has a non-trivial past
    start = max(0, int(0.12 * len(times)))
    idx = idx[idx >= start]
    if len(idx) == 0:
        idx = np.array([len(times) - 1], dtype=int)
    return [pd.Timestamp(times[i]) for i in idx]


def run_analogue_poc(
    vector: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    query_times: list[pd.Timestamp] | None = None,
    extra_example_times: list[pd.Timestamp] | None = None,
    cfg: dict[str, Any] | None = None,
    time_col: str = "time_bucket",
) -> dict[str, Any]:
    params = analogue_defaults(cfg)
    frame, features = prepare_analogue_frame(vector, outcomes, time_col=time_col)
    eligible = development_eligible_mask(
        frame,
        time_col=time_col,
        entry_lag_hours=int(params["entry_lag_hours"]),
        outcome_horizon_hours=int(params["outcome_horizon_hours"]),
    )
    baseline = unconditional_development_returns(frame, time_col=time_col, cfg=cfg)
    times = list(query_times or stride_query_times(frame, eligible, int(params["n_stride_queries"]), time_col=time_col))
    for extra in extra_example_times or []:
        ts = as_utc(extra)
        if ts not in times:
            times.append(ts)
    cards = [query_analogues(frame, t, features, time_col=time_col, cfg=cfg) for t in times]
    used = [c for c in cards if not c.get("abstain")]
    cond_med = [c["conditional_return_24h"]["median"] for c in used if c.get("conditional_return_24h", {}).get("n", 0)]
    cond_ppos = [c["conditional_return_24h"]["probability_positive"] for c in used if c.get("conditional_return_24h", {}).get("n", 0)]
    source, version = assert_single_vector_identity(vector)
    return {
        "disclaimer": DISCLAIMER,
        "analogue_version": params["version"],
        "vector_source": source,
        "vector_version": version,
        "distance": {
            "name": "equal-weight mixed L1",
            "binary": "per-horizon transition_state_raw mismatch",
            "continuous": "robust median/MAD scale from eligible past candidates, then L1",
            "features": features,
            "future_exclusion": (
                "candidates must be strictly earlier than the query and their "
                "24h outcome window must end before the query observation time; "
                "holdout rows are never candidates or queries"
            ),
        },
        "unconditional_development_return_24h": baseline,
        "n_queries": len(cards),
        "n_abstain": int(sum(1 for c in cards if c.get("abstain"))),
        "n_with_support": len(used),
        "batch_conditional_median_of_medians": None if not cond_med else float(np.median(cond_med)),
        "batch_mean_p_positive": None if not cond_ppos else float(np.mean(cond_ppos)),
        "unconditional_median": baseline.get("median"),
        "unconditional_p_positive": baseline.get("probability_positive"),
        "queries": cards,
    }
