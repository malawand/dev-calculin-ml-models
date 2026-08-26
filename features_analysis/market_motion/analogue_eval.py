"""Development-only walk-forward scoring of the existing analogue matcher.

PROVISIONAL DEVELOPMENT-ONLY POC
Not locked evidence
Temporary Cortex-derived dataset

Does not change k, radius, feature weights, the vector, or the detector.
Holdouts stay closed. Cortex and OLS remain separate experiments.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd

from .analogue import (
    analogue_defaults,
    development_eligible_mask,
    match_query,
    outcome_purge_lag_hours,
    prepare_analogue_frame,
)
from .exceptions import VectorSourceError
from .periods import (
    DEVELOPMENT_START_UTC,
    RETROSPECTIVE_HOLDOUT_START_UTC,
    as_utc,
    in_prospective_locked,
    in_retrospective_holdout,
    last_observation_with_complete_outcome,
)
from .source_guard import assert_single_vector_identity

WALKFORWARD_DISCLAIMER = (
    "PROVISIONAL DEVELOPMENT-ONLY POC\n"
    "Not locked evidence\n"
    "Temporary Cortex-derived dataset"
)

HORIZON_DISCLAIMER = (
    "EXPLORATORY DEVELOPMENT-ONLY POC\n"
    "Multiple outcome horizons examined\n"
    "Temporary Cortex-derived dataset\n"
    "Not locked evidence"
)

QUERY_HOUR_UTC = 0
WARMUP_MULTIPLIER = 1  # first query after k * min_separation_hours of candidate span
SCHEDULE_HORIZON_HOURS = 24  # freeze daily query dates to the 24h walk-forward
EVAL_OUTCOME_HORIZONS = (4, 8, 12, 24)

QUERY_HOUR_UTC = 0
WARMUP_MULTIPLIER = 1  # first query after k * min_separation_hours of candidate span


def empirical_crps(sample: np.ndarray, y: float) -> float:
    """Ensemble CRPS: E|X-y| - 0.5 E|X-X'|."""
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0 or not np.isfinite(y):
        return float("nan")
    n = int(x.size)
    term1 = float(np.mean(np.abs(x - y)))
    xs = np.sort(x)
    i = np.arange(1, n + 1, dtype=float)
    pairwise = float(np.dot(2.0 * i - n - 1.0, xs))
    term2 = pairwise / float(n * n)
    return term1 - term2


def brier_score(prob: np.ndarray, event: np.ndarray) -> float:
    p = np.asarray(prob, dtype=float)
    e = np.asarray(event, dtype=float)
    ok = np.isfinite(p) & np.isfinite(e)
    if not bool(ok.any()):
        return float("nan")
    return float(np.mean((p[ok] - e[ok]) ** 2))


def _bootstrap_mean_ci(values: np.ndarray, *, draws: int = 400, rng: np.random.Generator | None = None) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    rec = {"n": float(x.size), "mean": float("nan"), "p05": float("nan"), "p95": float("nan")}
    if x.size == 0:
        return rec
    rec["mean"] = float(x.mean())
    if x.size < 8 or draws <= 0:
        return rec
    gen = rng or np.random.default_rng(0)
    draws_m = gen.choice(x, size=(draws, x.size), replace=True).mean(axis=1)
    rec["p05"] = float(np.quantile(draws_m, 0.05))
    rec["p95"] = float(np.quantile(draws_m, 0.95))
    return rec


def skill_score(model: float, baseline: float) -> float:
    """1 - model/baseline. Positive means the analogue beat the baseline. Not comparable as raw CRPS."""
    if not np.isfinite(model) or not np.isfinite(baseline) or baseline == 0:
        return float("nan")
    return float(1.0 - model / baseline)


def round_trip_cost_pct(cfg: dict[str, Any] | None = None) -> float:
    oc = (cfg or {}).get("outcomes") or {}
    return float(oc.get("round_trip_cost_pct", 0.04))


def daily_query_times(
    eligible_times: pd.DatetimeIndex,
    *,
    query_hour_utc: int = QUERY_HOUR_UTC,
    warmup_hours: int,
    last_query: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Fixed clock-hour daily schedule inside the development eligible set."""
    if len(eligible_times) == 0:
        return []
    first = eligible_times.min()
    start = first + pd.Timedelta(hours=int(warmup_hours))
    start = max(as_utc(start), as_utc(DEVELOPMENT_START_UTC))
    last = min(as_utc(last_query), as_utc(eligible_times.max()))
    if start > last:
        return []
    days = pd.date_range(start.normalize(), last.normalize(), freq="D", tz="UTC")
    wanted = pd.DatetimeIndex([d + pd.Timedelta(hours=int(query_hour_utc)) for d in days])
    eligible_idx = pd.DatetimeIndex(pd.to_datetime(eligible_times, utc=True))
    hit = wanted.isin(eligible_idx)
    out: list[pd.Timestamp] = []
    for ts, ok in zip(wanted, hit):
        ts = as_utc(ts)
        if not bool(ok):
            continue
        if ts < start or ts > last:
            continue
        stamp = pd.DatetimeIndex([ts])
        if bool(np.asarray(in_retrospective_holdout(stamp)).any()):
            continue
        if bool(np.asarray(in_prospective_locked(stamp)).any()):
            continue
        out.append(ts)
    return out


def _tertile_labels(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    x = pd.Series(np.asarray(values, dtype=float))
    labels = ["low", "middle", "high"]
    try:
        cats = pd.qcut(x, 3, labels=labels, duplicates="drop")
    except ValueError:
        cats = pd.Series(["middle"] * len(x), index=x.index)
    return cats.astype(str).to_numpy(), labels


def _group_rows(df: pd.DataFrame, col: str, labels: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lab in labels:
        part = df.loc[df[col] == lab]
        if part.empty:
            continue
        rows.append(
            {
                "group": lab,
                "n": int(len(part)),
                "mean_predicted_p_gt_cost": float(part["p_analogue_gt_cost"].mean()),
                "realized_rate_gt_cost": float(part["event_gt_cost"].mean()),
                "realized_mean_return_pct": float(part["actual_return"].mean()),
                "realized_median_return_pct": float(part["actual_return"].median()),
                "realized_positive_rate": float((part["actual_return"] > 0).mean()),
                "mean_crps_analogue": float(part["crps_analogue"].mean()),
                "mean_crps_uncond": float(part["crps_uncond"].mean()),
                "mean_crps_diff": float((part["crps_analogue"] - part["crps_uncond"]).mean()),
                "pct_analogue_beats_crps": float((part["crps_analogue"] < part["crps_uncond"]).mean()),
                "mean_median_distance": float(part["median_distance"].mean()),
            }
        )
    return rows


def _monotonic_increasing(values: list[float]) -> bool:
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def interpret_walkforward(summary: dict[str, Any]) -> dict[str, Any]:
    """Plain-language flags. Not a locked-test decision."""
    p_groups = summary.get("probability_groups") or []
    d_groups = summary.get("distance_groups") or []
    realized_p = [g["realized_rate_gt_cost"] for g in p_groups]
    realized_r = [g["realized_mean_return_pct"] for g in p_groups]
    pred_p = [g["mean_predicted_p_gt_cost"] for g in p_groups]
    crps_diff = summary.get("mean_crps_difference_analogue_minus_uncond")
    brier_diff = summary.get("mean_brier_difference_analogue_minus_uncond")
    crps_better = crps_diff is not None and crps_diff < 0
    brier_better = brier_diff is not None and brier_diff < 0
    outcomes_follow_p = bool(p_groups) and _monotonic_increasing(realized_p) and (max(realized_p) - min(realized_p) >= 0.02)
    returns_follow_p = bool(p_groups) and _monotonic_increasing(realized_r)
    close_skill = None
    far_skill = None
    for g in d_groups:
        if g["group"] == "low":
            close_skill = -g["mean_crps_diff"]
        if g["group"] == "high":
            far_skill = -g["mean_crps_diff"]
    closer_better = (
        close_skill is not None
        and far_skill is not None
        and close_skill > far_skill
        and close_skill > 0
    )
    useful = bool(outcomes_follow_p and crps_better)
    unpromising = (not outcomes_follow_p) and (not crps_better) and (not brier_better)
    if useful:
        pattern = "useful_pattern"
        headline = (
            "Low-confidence states predicted lower exceed-cost probability and subsequently "
            "had fewer such outcomes; high-confidence states predicted higher probability "
            "and subsequently had more. Analogue CRPS was lower than the past-only unconditional."
        )
    elif unpromising:
        pattern = "unpromising_pattern"
        headline = (
            "Predicted probabilities varied, but realized outcomes did not follow them in a "
            "stable way. CRPS and Brier were no better than the past-only unconditional baseline."
        )
    else:
        pattern = "mixed"
        headline = (
            "Some diagnostics moved in a useful direction and others did not. "
            "This is not locked evidence."
        )
    return {
        "pattern": pattern,
        "headline": headline,
        "predicted_p_spread": None if len(pred_p) < 2 else float(max(pred_p) - min(pred_p)),
        "realized_hit_rate_follows_predicted_p": outcomes_follow_p,
        "realized_mean_return_follows_predicted_p": returns_follow_p,
        "analogue_crps_better_than_uncond": crps_better,
        "analogue_brier_better_than_uncond": brier_better,
        "closer_analogues_more_useful": closer_better,
        "recommend_radius_abstention_next": bool(closer_better and not useful),
    }


def run_walkforward_eval(
    vector: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    cfg: dict[str, Any] | None = None,
    time_col: str = "time_bucket",
    query_hour_utc: int = QUERY_HOUR_UTC,
    outcome_horizon_hours: int = 24,
    outcome_col: str | None = None,
    query_times: list[pd.Timestamp] | None = None,
    disclaimer: str | None = None,
    progress_label: str | None = None,
) -> dict[str, Any]:
    params = analogue_defaults(cfg)
    cost = round_trip_cost_pct(cfg)
    horizon = int(outcome_horizon_hours)
    ocol = outcome_col or f"return_{horizon}h"
    frame, features = prepare_analogue_frame(vector, outcomes, time_col=time_col, outcome_col=ocol)
    eligible = development_eligible_mask(
        frame,
        time_col=time_col,
        outcome_col=ocol,
        entry_lag_hours=int(params["entry_lag_hours"]),
        outcome_horizon_hours=horizon,
    )
    times = pd.DatetimeIndex(pd.to_datetime(frame[time_col], utc=True))
    last_q = last_observation_with_complete_outcome(
        period_end_exclusive=RETROSPECTIVE_HOLDOUT_START_UTC,
        entry_lag_hours=int(params["entry_lag_hours"]),
        outcome_horizon_hours=SCHEDULE_HORIZON_HOURS,
    )
    warmup_hours = (
        int(params["k"]) * int(params["min_separation_hours"]) * WARMUP_MULTIPLIER
        + outcome_purge_lag_hours(int(params["entry_lag_hours"]), SCHEDULE_HORIZON_HOURS)
    )
    if query_times is None:
        schedule_col = "return_24h" if "return_24h" in frame.columns else ocol
        schedule_h = SCHEDULE_HORIZON_HOURS if schedule_col == "return_24h" else horizon
        schedule_eligible = development_eligible_mask(
            frame,
            time_col=time_col,
            outcome_col=schedule_col,
            entry_lag_hours=int(params["entry_lag_hours"]),
            outcome_horizon_hours=schedule_h,
        )
        query_times = daily_query_times(
            times[schedule_eligible.to_numpy()],
            query_hour_utc=query_hour_utc,
            warmup_hours=warmup_hours,
            last_query=last_q,
        )
    holdout_q = in_retrospective_holdout(pd.DatetimeIndex(query_times))
    locked_q = in_prospective_locked(pd.DatetimeIndex(query_times))
    if bool(holdout_q.any()) or bool(locked_q.any()):
        raise VectorSourceError("walk-forward queries fell in a closed holdout")

    purge_lag = outcome_purge_lag_hours(int(params["entry_lag_hours"]), horizon)
    rows: list[dict[str, Any]] = []
    n_abstain = 0
    n_skipped_unscored = 0
    n_q = len(query_times)
    label = progress_label or f"{horizon}h"
    for i, qtime in enumerate(query_times):
        if n_q and (i + 1) % 50 == 0:
            print(f"walk-forward {label} {i + 1}/{n_q}", file=sys.stderr)
        match = match_query(
            frame,
            qtime,
            features,
            time_col=time_col,
            outcome_col=ocol,
            outcome_horizon_hours=horizon,
            cfg=cfg,
            eligible=eligible,
        )
        y = match["actual_return"]
        if match["abstain"] or not np.isfinite(y) or match["n_neighbors"] == 0:
            n_abstain += int(bool(match["abstain"]))
            if not match["abstain"]:
                n_skipped_unscored += 1
            continue
        a_ret = match["neighbor_returns"]
        u_ret = match["uncond_returns"]
        a_ret = a_ret[np.isfinite(a_ret)]
        u_ret = u_ret[np.isfinite(u_ret)]
        if a_ret.size == 0 or u_ret.size == 0:
            n_skipped_unscored += 1
            continue
        p_a = float((a_ret > cost).mean())
        p_u = float((u_ret > cost).mean())
        event = float(y > cost)
        crps_a = empirical_crps(a_ret, y)
        crps_u = empirical_crps(u_ret, y)
        rows.append(
            {
                "query_time_utc": str(as_utc(qtime)),
                "actual_return": float(y),
                "event_gt_cost": event,
                "p_analogue_gt_cost": p_a,
                "p_uncond_gt_cost": p_u,
                "brier_analogue": (p_a - event) ** 2,
                "brier_uncond": (p_u - event) ** 2,
                "crps_analogue": crps_a,
                "crps_uncond": crps_u,
                "crps_diff": crps_a - crps_u,
                "n_neighbors": int(match["n_neighbors"]),
                "n_uncond": int(np.isfinite(u_ret).sum()),
                "median_distance": float(match["median_distance"]),
            }
        )

    scored = pd.DataFrame(rows)
    source, version = assert_single_vector_identity(vector)
    summary: dict[str, Any] = {
        "disclaimer": disclaimer or WALKFORWARD_DISCLAIMER,
        "analogue_version": params["version"],
        "vector_source": source,
        "vector_version": version,
        "outcome_col": ocol,
        "outcome_horizon_hours": horizon,
        "purge_lag_hours": purge_lag,
        "purge_rule": f"candidate t <= query t - {purge_lag}h (entry {params['entry_lag_hours']}h + outcome {horizon}h + 1h)",
        "schedule": {
            "query_hour_utc": int(query_hour_utc),
            "warmup_hours": warmup_hours,
            "schedule_horizon_hours": SCHEDULE_HORIZON_HOURS,
            "first_query_utc": None if not query_times else str(query_times[0]),
            "last_query_utc": None if not query_times else str(query_times[-1]),
            "n_scheduled_eligible": len(query_times),
        },
        "round_trip_cost_pct": cost,
        "n_evaluated": int(len(scored)),
        "n_abstain": int(n_abstain),
        "n_skipped_unscored": int(n_skipped_unscored),
        "feature_count": len(features),
        "k": int(params["k"]),
        "min_separation_hours": int(params["min_separation_hours"]),
        "period": {
            "start_utc": str(DEVELOPMENT_START_UTC),
            "end_exclusive_utc": str(RETROSPECTIVE_HOLDOUT_START_UTC),
            "last_observation_with_complete_outcome_utc": str(last_q),
        },
        "queries": rows,
    }
    if scored.empty:
        summary["interpretation"] = {
            "pattern": "no_scored_queries",
            "headline": "No scored development queries.",
        }
        return summary

    crps_diff = scored["crps_diff"].to_numpy(dtype=float)
    brier_diff = (scored["brier_analogue"] - scored["brier_uncond"]).to_numpy(dtype=float)
    crps_ci = _bootstrap_mean_ci(crps_diff, draws=int(params["bootstrap_draws"]))
    brier_ci = _bootstrap_mean_ci(brier_diff, draws=int(params["bootstrap_draws"]))
    beat = (scored["crps_analogue"] < scored["crps_uncond"]).to_numpy(dtype=float)
    beat_ci = _bootstrap_mean_ci(beat, draws=int(params["bootstrap_draws"]))

    p_labels, p_names = _tertile_labels(scored["p_analogue_gt_cost"].to_numpy())
    d_labels, d_names = _tertile_labels(scored["median_distance"].to_numpy())
    scored = scored.copy()
    scored["p_group"] = p_labels
    scored["d_group"] = d_labels
    p_groups = _group_rows(scored, "p_group", p_names)
    d_groups = _group_rows(scored, "d_group", d_names)

    cal = _calibration_table(scored["p_analogue_gt_cost"].to_numpy(), scored["event_gt_cost"].to_numpy())

    summary.update(
        {
            "mean_crps_analogue": float(scored["crps_analogue"].mean()),
            "mean_crps_uncond": float(scored["crps_uncond"].mean()),
            "median_crps_analogue": float(scored["crps_analogue"].median()),
            "median_crps_uncond": float(scored["crps_uncond"].median()),
            "mean_crps_difference_analogue_minus_uncond": float(crps_diff.mean()),
            "median_crps_difference_analogue_minus_uncond": float(np.median(crps_diff)),
            "mean_crps_difference_bootstrap90": crps_ci,
            "pct_queries_analogue_beats_uncond_crps": float(beat.mean()),
            "pct_queries_analogue_beats_uncond_crps_bootstrap90": beat_ci,
            "mean_brier_analogue": float(scored["brier_analogue"].mean()),
            "mean_brier_uncond": float(scored["brier_uncond"].mean()),
            "mean_brier_difference_analogue_minus_uncond": float(brier_diff.mean()),
            "mean_brier_difference_bootstrap90": brier_ci,
            "mean_predicted_p_gt_cost_analogue": float(scored["p_analogue_gt_cost"].mean()),
            "mean_predicted_p_gt_cost_uncond": float(scored["p_uncond_gt_cost"].mean()),
            "realized_rate_gt_cost": float(scored["event_gt_cost"].mean()),
            "realized_mean_return_pct": float(scored["actual_return"].mean()),
            "spearman_predicted_p_vs_actual_return": _spearman(
                scored["p_analogue_gt_cost"].to_numpy(), scored["actual_return"].to_numpy()
            ),
            "spearman_predicted_p_vs_event": _spearman(
                scored["p_analogue_gt_cost"].to_numpy(), scored["event_gt_cost"].to_numpy()
            ),
            "probability_groups": p_groups,
            "distance_groups": d_groups,
            "calibration": cal,
            "crps_skill_score": skill_score(float(scored["crps_analogue"].mean()), float(scored["crps_uncond"].mean())),
            "brier_skill_score": skill_score(float(scored["brier_analogue"].mean()), float(scored["brier_uncond"].mean())),
        }
    )
    summary["interpretation"] = interpret_walkforward(summary)
    return summary


def _calibration_table(pred: np.ndarray, event: np.ndarray, n_bins: int = 5) -> list[dict[str, Any]]:
    s = pd.DataFrame({"p": pred, "e": event})
    s = s[np.isfinite(s["p"]) & np.isfinite(s["e"])]
    if s.empty:
        return []
    try:
        s["bin"] = pd.qcut(s["p"], n_bins, duplicates="drop")
    except ValueError:
        s["bin"] = "all"
    rows: list[dict[str, Any]] = []
    for i, (lab, part) in enumerate(s.groupby("bin", observed=True)):
        rows.append(
            {
                "bin": str(lab),
                "bin_index": i,
                "n": int(len(part)),
                "mean_predicted": float(part["p"].mean()),
                "realized_rate": float(part["e"].mean()),
            }
        )
    return rows


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return None
    rx = pd.Series(x[ok]).rank().to_numpy()
    ry = pd.Series(y[ok]).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def walkforward_markdown(rec: dict[str, Any]) -> str:
    lines = [
        "# Analogue walk-forward evaluation",
        "",
        "```text",
        rec.get("disclaimer", WALKFORWARD_DISCLAIMER),
        "```",
        "",
        f"- vector: `{rec.get('vector_source')}` / `{rec.get('vector_version')}`",
        f"- matcher: `{rec.get('analogue_version')}` (unchanged; k={rec.get('k')}, separation={rec.get('min_separation_hours')}h)",
        f"- evaluated queries: {rec.get('n_evaluated')}",
        f"- abstain: {rec.get('n_abstain')}",
        f"- scheduled eligible daily queries: {(rec.get('schedule') or {}).get('n_scheduled_eligible')}",
        f"- round-trip cost: {rec.get('round_trip_cost_pct')} percent",
        "",
        "## Interpretation",
        "",
        str((rec.get("interpretation") or {}).get("headline") or ""),
        "",
        f"- pattern: `{(rec.get('interpretation') or {}).get('pattern')}`",
        f"- realized hit-rate follows predicted P(R>cost): {(rec.get('interpretation') or {}).get('realized_hit_rate_follows_predicted_p')}",
        f"- analogue CRPS better than past-only unconditional: {(rec.get('interpretation') or {}).get('analogue_crps_better_than_uncond')}",
        f"- analogue Brier better than past-only unconditional: {(rec.get('interpretation') or {}).get('analogue_brier_better_than_uncond')}",
        f"- closer analogues more useful: {(rec.get('interpretation') or {}).get('closer_analogues_more_useful')}",
        f"- recommend radius/abstention next (do not implement this pass): {(rec.get('interpretation') or {}).get('recommend_radius_abstention_next')}",
        "",
        "## Primary: CRPS",
        "",
        f"- mean CRPS analogue: {rec.get('mean_crps_analogue')}",
        f"- mean CRPS past-only unconditional: {rec.get('mean_crps_uncond')}",
        f"- mean CRPS difference (analogue − uncond): {rec.get('mean_crps_difference_analogue_minus_uncond')}",
        f"- median CRPS difference: {rec.get('median_crps_difference_analogue_minus_uncond')}",
        f"- bootstrap 90% on mean difference: {rec.get('mean_crps_difference_bootstrap90')}",
        f"- share of queries where analogue CRPS is lower: {rec.get('pct_queries_analogue_beats_uncond_crps')}",
        "",
        "## Secondary: P(R_24h > round-trip cost) and Brier",
        "",
        f"- mean analogue P(R>cost): {rec.get('mean_predicted_p_gt_cost_analogue')}",
        f"- mean past-only uncond P(R>cost): {rec.get('mean_predicted_p_gt_cost_uncond')}",
        f"- realized rate: {rec.get('realized_rate_gt_cost')}",
        f"- mean Brier analogue: {rec.get('mean_brier_analogue')}",
        f"- mean Brier uncond: {rec.get('mean_brier_uncond')}",
        f"- mean Brier difference (analogue − uncond): {rec.get('mean_brier_difference_analogue_minus_uncond')}",
        "",
        "## Probability groups (tertiles of analogue P(R>cost))",
        "",
    ]
    for g in rec.get("probability_groups") or []:
        lines.append(
            f"- {g['group']}: n={g['n']} pred P={g['mean_predicted_p_gt_cost']:.3f} "
            f"realized P(R>cost)={g['realized_rate_gt_cost']:.3f} "
            f"mean return={g['realized_mean_return_pct']:.3f}% "
            f"CRPS diff={g['mean_crps_diff']:.4f}"
        )
    lines.extend(["", "## Distance groups (tertiles of median neighbour distance)", ""])
    for g in rec.get("distance_groups") or []:
        lines.append(
            f"- {g['group']} distance: n={g['n']} mean d={g['mean_median_distance']:.3f} "
            f"CRPS diff={g['mean_crps_diff']:.4f} beat rate={g['pct_analogue_beats_crps']:.3f}"
        )
    lines.extend(["", "## Calibration", ""])
    for c in rec.get("calibration") or []:
        lines.append(
            f"- {c['bin']}: n={c['n']} mean predicted={c['mean_predicted']:.3f} "
            f"realized={c['realized_rate']:.3f}"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Past-only unconditional uses the same eligible candidate pool as the analogue search (outcome window ended before the query).",
            "- Daily 24h outcomes overlap; bootstrap intervals treat queries as independent and are optimistic.",
            "- Matcher was not tuned on these scores.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def shared_daily_query_times(
    vector: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    cfg: dict[str, Any] | None = None,
    time_col: str = "time_bucket",
    query_hour_utc: int = QUERY_HOUR_UTC,
) -> list[pd.Timestamp]:
    """Daily 00:00 UTC schedule frozen to the 24h walk-forward eligibility window."""
    params = analogue_defaults(cfg)
    frame, _ = prepare_analogue_frame(vector, outcomes, time_col=time_col, outcome_col="return_24h")
    eligible = development_eligible_mask(
        frame,
        time_col=time_col,
        outcome_col="return_24h",
        entry_lag_hours=int(params["entry_lag_hours"]),
        outcome_horizon_hours=SCHEDULE_HORIZON_HOURS,
    )
    times = pd.DatetimeIndex(pd.to_datetime(frame[time_col], utc=True))
    last_q = last_observation_with_complete_outcome(
        period_end_exclusive=RETROSPECTIVE_HOLDOUT_START_UTC,
        entry_lag_hours=int(params["entry_lag_hours"]),
        outcome_horizon_hours=SCHEDULE_HORIZON_HOURS,
    )
    warmup_hours = (
        int(params["k"]) * int(params["min_separation_hours"]) * WARMUP_MULTIPLIER
        + outcome_purge_lag_hours(int(params["entry_lag_hours"]), SCHEDULE_HORIZON_HOURS)
    )
    return daily_query_times(
        times[eligible.to_numpy()],
        query_hour_utc=query_hour_utc,
        warmup_hours=warmup_hours,
        last_query=last_q,
    )


def _strip_queries(rec: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in rec.items() if k != "queries"}


def run_multi_horizon_eval(
    vector: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    cfg: dict[str, Any] | None = None,
    time_col: str = "time_bucket",
    query_hour_utc: int = QUERY_HOUR_UTC,
    horizons: tuple[int, ...] = EVAL_OUTCOME_HORIZONS,
) -> dict[str, Any]:
    query_times = shared_daily_query_times(
        vector, outcomes, cfg=cfg, time_col=time_col, query_hour_utc=query_hour_utc
    )
    source, version = assert_single_vector_identity(vector)
    params = analogue_defaults(cfg)
    by_h: dict[str, Any] = {}
    query_tables: dict[str, list[dict[str, Any]]] = {}
    for h in horizons:
        rec = run_walkforward_eval(
            vector,
            outcomes,
            cfg=cfg,
            time_col=time_col,
            query_hour_utc=query_hour_utc,
            outcome_horizon_hours=h,
            outcome_col=f"return_{h}h",
            query_times=query_times,
            disclaimer=HORIZON_DISCLAIMER,
            progress_label=f"{h}h",
        )
        query_tables[f"{h}h"] = rec.get("queries") or []
        by_h[f"{h}h"] = _strip_queries(rec)
    return {
        "disclaimer": HORIZON_DISCLAIMER,
        "analogue_version": params["version"],
        "vector_source": source,
        "vector_version": version,
        "k": int(params["k"]),
        "min_separation_hours": int(params["min_separation_hours"]),
        "shared_schedule": {
            "query_hour_utc": int(query_hour_utc),
            "n_scheduled": len(query_times),
            "first_query_utc": None if not query_times else str(query_times[0]),
            "last_query_utc": None if not query_times else str(query_times[-1]),
            "note": "Same daily 00:00 UTC dates as the 24h walk-forward. Outcome purge is horizon-specific.",
        },
        "horizons": by_h,
        "query_tables": query_tables,
        "purge_lags_hours": {f"{h}h": outcome_purge_lag_hours(int(params["entry_lag_hours"]), h) for h in horizons},
        "approved_primary_target": "return_24h",
        "note": (
            "24h remains the approved research target. Other horizons are exploratory. "
            "Do not compare raw CRPS across horizons; use CRPS skill and CRPS difference."
        ),
    }


def multi_horizon_markdown(bundle: dict[str, Any]) -> str:
    lines = [
        "# Analogue horizon-sensitivity evaluation",
        "",
        "```text",
        bundle.get("disclaimer", HORIZON_DISCLAIMER),
        "```",
        "",
        f"- vector: `{bundle.get('vector_source')}` / `{bundle.get('vector_version')}`",
        f"- matcher: `{bundle.get('analogue_version')}` k={bundle.get('k')} neighbour-separation={bundle.get('min_separation_hours')}h (unchanged)",
        f"- shared daily queries: {(bundle.get('shared_schedule') or {}).get('n_scheduled')}",
        f"- purge lags: {bundle.get('purge_lags_hours')}",
        f"- approved primary target: `{bundle.get('approved_primary_target')}` (not replaced)",
        "",
        str(bundle.get("note") or ""),
        "",
        "| Horizon | n | Abstain | Purge | CRPS analogue | CRPS uncond | CRPS Δ | CRPS skill | Beat % | Brier skill | Rank monotonic | Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for key in ("4h", "8h", "12h", "24h"):
        rec = (bundle.get("horizons") or {}).get(key) or {}
        if not rec:
            continue
        interp = rec.get("interpretation") or {}
        lines.append(
            f"| {key} | {rec.get('n_evaluated')} | {rec.get('n_abstain')} | "
            f"{rec.get('purge_lag_hours')}h | "
            f"{rec.get('mean_crps_analogue'):.4f} | {rec.get('mean_crps_uncond'):.4f} | "
            f"{rec.get('mean_crps_difference_analogue_minus_uncond'):.4f} | "
            f"{rec.get('crps_skill_score'):.4f} | "
            f"{100 * float(rec.get('pct_queries_analogue_beats_uncond_crps') or 0):.1f}% | "
            f"{rec.get('brier_skill_score'):.4f} | "
            f"{interp.get('realized_hit_rate_follows_predicted_p')} | "
            f"{rec.get('spearman_predicted_p_vs_actual_return')} |"
        )
    for key in ("4h", "8h", "12h", "24h"):
        rec = (bundle.get("horizons") or {}).get(key) or {}
        if not rec:
            continue
        lines.extend(["", f"## {key}", "", walkforward_markdown(rec)])
    return "\n".join(lines) + "\n"
