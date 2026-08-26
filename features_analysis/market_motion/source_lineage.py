"""Source-lineage audit for Cortex continuous D. Representation only; no 24h scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .source_collapse import DEFAULT_ATOL, DEFAULT_RTOL, values_agree
from .transition import EVENT_PEAK, EVENT_TROUGH


def _label(val: Any) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)) or str(val) == "":
        return "unlabeled"
    return str(val)


def audit_horizon(raw: pd.DataFrame, series_name: str, *, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL) -> dict[str, Any]:
    part = raw.loc[raw["series_name"] == series_name].copy()
    if part.empty:
        return {"series_name": series_name, "n_raw_rows": 0}
    part["time_utc"] = pd.to_datetime(part.get("time_utc", pd.to_datetime(part["timestamp"], unit="s", utc=True)), utc=True)
    part["cluster"] = part["cluster"].map(_label) if "cluster" in part.columns else "unlabeled"
    part["replica"] = part["replica"].map(lambda x: _label(x) if _label(x) != "unlabeled" else "") if "replica" in part.columns else ""
    part["value"] = pd.to_numeric(part["value"], errors="coerce")

    identities = []
    for keys, g in part.groupby(["cluster", "replica", "__name__"] if "__name__" in part.columns else ["cluster", "replica"]):
        if "__name__" in part.columns:
            cluster, replica, metric = keys
        else:
            cluster, replica = keys
            metric = ""
        identities.append(
            {
                "cluster": cluster,
                "replica": replica,
                "metric": metric,
                "n_rows": int(len(g)),
                "start_utc": str(g["time_utc"].min()),
                "end_utc": str(g["time_utc"].max()),
            }
        )

    overlap_rows = []
    sign_dis = 0
    value_dis = 0
    n_multi = 0
    abs_diffs: list[float] = []
    rel_diffs: list[float] = []
    for ts, rows in part.groupby("timestamp"):
        clusters = sorted(set(rows["cluster"]))
        if len(rows) <= 1:
            continue
        n_multi += 1
        vals = rows["value"].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        agree = values_agree(finite, rtol=rtol, atol=atol)
        if len(finite) >= 2:
            ad = float(np.nanmax(finite) - np.nanmin(finite))
            scale = max(float(np.nanmax(np.abs(finite))), 1e-12)
            rd = ad / scale
            abs_diffs.append(ad)
            rel_diffs.append(rd)
            if not agree:
                value_dis += 1
            signs = set(np.sign(finite[finite != 0]))
            if len(signs) > 1:
                sign_dis += 1
        overlap_rows.append(
            {
                "time_utc": str(pd.to_datetime(rows["time_utc"].iloc[0], utc=True)),
                "n_rows": int(len(rows)),
                "clusters": clusters,
                "values": [None if not np.isfinite(v) else float(v) for v in vals],
                "agree": bool(agree),
                "sign_disagree": bool(len(set(np.sign(finite[np.isfinite(finite) & (finite != 0)]))) > 1) if len(finite) else False,
            }
        )

    per_hour = part.groupby("time_utc").agg(
        cluster=("cluster", lambda s: s.iloc[0] if s.nunique() == 1 else "MULTI"),
        value=("value", "mean"),
    ).sort_index()
    transitions = []
    prev_c = None
    prev_v = np.nan
    prev_t = None
    d1 = per_hour["value"].diff().abs()
    for ts, rec in per_hour.iterrows():
        cur = rec["cluster"]
        val = rec["value"]
        if prev_c is not None and cur != prev_c:
            trail = d1.loc[:prev_t].dropna().tail(168)
            typical = float(trail.median()) if len(trail) else None
            jump = float(abs(val - prev_v)) if np.isfinite(val) and np.isfinite(prev_v) else None
            ratio = (jump / typical) if typical and typical > 0 and jump is not None else None
            transitions.append(
                {
                    "time_utc": str(ts),
                    "from_cluster": prev_c,
                    "to_cluster": cur,
                    "from_value": None if not np.isfinite(prev_v) else float(prev_v),
                    "to_value": None if not np.isfinite(val) else float(val),
                    "abs_jump": jump,
                    "trailing_median_abs_change": typical,
                    "jump_over_typical": ratio,
                    "routing_change": True,
                    "large_d_jump": bool(ratio is not None and ratio >= 5.0),
                    "note": "Owner-confirmed routing/provenance change for the same feature family.",
                }
            )
        prev_c, prev_v, prev_t = cur, val, ts

    return {
        "series_name": series_name,
        "n_raw_rows": int(len(part)),
        "n_unique_hours": int(part["timestamp"].nunique()),
        "identities": identities,
        "n_hours_with_multiple_rows": n_multi,
        "n_material_value_disagreements": value_dis,
        "n_sign_disagreements": sign_dis,
        "overlap_abs_diff": {
            "max": float(np.max(abs_diffs)) if abs_diffs else None,
            "median": float(np.median(abs_diffs)) if abs_diffs else None,
        },
        "overlap_rel_diff": {
            "max": float(np.max(rel_diffs)) if rel_diffs else None,
            "median": float(np.median(rel_diffs)) if rel_diffs else None,
        },
        "overlaps": overlap_rows,
        "source_identity_transitions": transitions,
        "n_source_identity_transitions": len(transitions),
        "semantic_certainty": "owner_confirmed_routing_same_feature",
        "interpretation": (
            "Cluster and replica labels identify Cortex routing/delivery, not different "
            "derivative formulas. Owner-confirmed same feature family. "
            "Historical hours are selected with prod_a_preferred_v1."
        ),
    }


def annotate_turns_near_boundaries(
    audit: dict[str, Any],
    vec: pd.DataFrame,
    horizons: list[str],
    *,
    window_hours: int = 2,
) -> dict[str, Any]:
    """Count confirmed D turns on and near source-boundary hours. Representation only."""
    times = pd.to_datetime(vec["time_bucket"], utc=True)
    by_h = audit.setdefault("by_horizon", {})
    for h in horizons:
        bcol = f"{h}_source_boundary"
        ecol = f"{h}_transition_event"
        rec = by_h.setdefault(h, {})
        if bcol not in vec.columns or ecol not in vec.columns:
            rec["turns_near_boundaries"] = {"note": "vector columns missing"}
            continue
        boundary = vec[bcol].fillna(False).astype(bool)
        event = vec[ecol].astype(str)
        is_turn = event.isin([EVENT_PEAK, EVENT_TROUGH])
        n_on = int((boundary & is_turn).sum())
        near = pd.Series(False, index=vec.index)
        b_times = times.loc[boundary]
        for t in b_times:
            near |= (times - t).abs() <= pd.Timedelta(hours=window_hours)
        n_near = int((near & is_turn).sum())
        rec["turns_near_boundaries"] = {
            "window_hours": window_hours,
            "n_boundary_hours": int(boundary.sum()),
            "n_confirmed_turns_on_boundary_hour": n_on,
            "n_confirmed_turns_within_window": n_near,
            "policy": "routing labels are provenance; detector/Z/V/A are continuous across handoffs",
        }
    return audit


def annotate_selection(audit: dict[str, Any], hourly: pd.DataFrame, horizons: list[str]) -> dict[str, Any]:
    """Summarize which approved source was selected by hour. Representation only."""
    from .source_collapse import CANONICAL_CLUSTER, CLUSTER_PRIORITY, UNAPPROVED_RANK, cluster_rank

    times = pd.to_datetime(hourly["time_bucket"], utc=True)
    by_h = audit.setdefault("by_horizon", {})
    for h in horizons:
        col = f"{h}_source_cluster"
        val_col = h
        rec = by_h.setdefault(h, {})
        if col not in hourly.columns:
            continue
        selected = hourly[col].fillna("").astype(str).replace("", "unlabeled")
        present = hourly[val_col].notna() if val_col in hourly.columns else selected.ne("")
        counts: dict[str, int] = {}
        for name, n in selected.loc[present].value_counts().items():
            counts[str(name)] = int(n)
        rec["selected_hours_by_cluster"] = counts
        rec["n_hours_selected"] = int(present.sum())
        prod_a = (selected == CANONICAL_CLUSTER) & present
        rec["prod_a_coverage"] = {
            "n_hours": int(prod_a.sum()),
            "first_utc": str(times.loc[prod_a].iloc[0]) if bool(prod_a.any()) else None,
            "last_utc": str(times.loc[prod_a].iloc[-1]) if bool(prod_a.any()) else None,
        }
        if bool(prod_a.any()):
            pa_times = times.loc[prod_a].sort_values()
            deltas = pa_times.diff().dropna()
            long = deltas[deltas > pd.Timedelta(hours=2)]
            rec["prod_a_coverage"]["n_internal_gaps_gt_2h"] = int(len(long))
            if len(long):
                gap_idx = long.idxmax()
                rec["prod_a_coverage"]["longest_internal_gap_hours"] = float(long.max() / pd.Timedelta(hours=1))
                rec["prod_a_coverage"]["longest_internal_gap_after_utc"] = str(times.loc[gap_idx] - long.max())
        spans = []
        if bool(present.any()):
            run_c = None
            run_start = None
            run_n = 0
            for t, c, p in zip(times, selected, present, strict=False):
                if not p:
                    if run_c is not None:
                        spans.append({"cluster": run_c, "start_utc": str(run_start), "end_utc": str(prev_t), "n_hours": run_n})
                        run_c = None
                    continue
                if c != run_c:
                    if run_c is not None:
                        spans.append({"cluster": run_c, "start_utc": str(run_start), "end_utc": str(prev_t), "n_hours": run_n})
                    run_c = c
                    run_start = t
                    run_n = 1
                else:
                    run_n += 1
                prev_t = t
            if run_c is not None:
                spans.append({"cluster": run_c, "start_utc": str(run_start), "end_utc": str(prev_t), "n_hours": run_n})
        rec["selected_spans"] = spans
        rec["fallback_hours"] = {
            k: v for k, v in counts.items() if k != CANONICAL_CLUSTER
        }
        rec["unapproved_selected_hours"] = int(
            sum(v for k, v in counts.items() if cluster_rank(k) >= UNAPPROVED_RANK)
        )
        rec["priority"] = list(CLUSTER_PRIORITY)
        disagree_col = f"{h}_source_value_disagree"
        if disagree_col in hourly.columns:
            rec["n_overlap_value_disagreements_selected"] = int(hourly[disagree_col].fillna(False).sum())
        conf_col = f"{h}_source_conflict"
        if conf_col in hourly.columns:
            rec["n_true_conflicts"] = int(hourly[conf_col].fillna(False).sum())
    return audit


def audit_raw_tape(raw: pd.DataFrame, horizons: list[str], **kwargs: Any) -> dict[str, Any]:
    by_h = {h: audit_horizon(raw, h, **kwargs) for h in horizons}
    return {
        "policy": "prod_a_preferred_v1",
        "rtol": kwargs.get("rtol", DEFAULT_RTOL),
        "atol": kwargs.get("atol", DEFAULT_ATOL),
        "semantic_certainty": "owner_confirmed_routing_same_feature",
        "canonical_cluster": "PROD-A-crypto-metrics",
        "note": (
            "Cluster/replica labels are owner-confirmed routing metadata for the same "
            "feature family. Historical hours prefer PROD-A-crypto-metrics, then the "
            "documented production fallback list. Signed values are never collapsed with max(). "
            "True conflict only when no approved source can be selected. "
            "Detector, Z, V, and A are continuous across routing-label changes. "
            "The upstream derivative formula text remains unavailable in this repository."
        ),
        "by_horizon": by_h,
    }


def audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Cortex source-lineage audit",
        "",
        f"**Policy:** `{audit.get('policy')}`  ",
        f"**Semantic certainty:** {audit.get('semantic_certainty')}  ",
        f"**Tolerance:** rtol={audit.get('rtol')}, atol={audit.get('atol')}",
        "",
        audit.get("note", ""),
        "",
        "Recording-rule / generator formula text is still not in this repository. "
        "Cluster labels are routing metadata, not formula versions.",
        "",
        f"**Canonical live cluster:** `{audit.get('canonical_cluster', 'PROD-A-crypto-metrics')}`",
        "",
    ]
    for h, rec in (audit.get("by_horizon") or {}).items():
        lines.append(f"## {h}")
        lines.append("")
        lines.append(f"- raw rows: {rec.get('n_raw_rows')}; unique hours: {rec.get('n_unique_hours')}")
        lines.append(f"- multi-row hours: {rec.get('n_hours_with_multiple_rows')}")
        lines.append(f"- material value disagreements: {rec.get('n_material_value_disagreements')}")
        lines.append(f"- sign disagreements: {rec.get('n_sign_disagreements')}")
        lines.append(f"- lineage transitions: {rec.get('n_source_identity_transitions')}")
        near = rec.get("turns_near_boundaries") or {}
        if near:
            lines.append(
                f"- confirmed turns on routing-boundary hour: {near.get('n_confirmed_turns_on_boundary_hour')}; "
                f"within ±{near.get('window_hours')}h: {near.get('n_confirmed_turns_within_window')}"
            )
        pa = rec.get("prod_a_coverage") or {}
        if pa:
            lines.append(
                f"- PROD-A selected hours: {pa.get('n_hours')}; span {pa.get('first_utc')} → {pa.get('last_utc')}"
            )
            if pa.get("longest_internal_gap_hours") is not None:
                lines.append(
                    f"- longest PROD-A internal gap: {pa.get('longest_internal_gap_hours'):.0f}h "
                    f"after {pa.get('longest_internal_gap_after_utc')}"
                )
        if rec.get("selected_hours_by_cluster"):
            lines.append(f"- selected hours by cluster: `{rec['selected_hours_by_cluster']}`")
        if rec.get("n_true_conflicts") is not None:
            lines.append(f"- true conflicts (no approved source): {rec.get('n_true_conflicts')}")
        if rec.get("n_overlap_value_disagreements_selected") is not None:
            lines.append(
                f"- overlapping hours with value disagreement (still selected): "
                f"{rec.get('n_overlap_value_disagreements_selected')}"
            )
        lines.append("")
        spans = rec.get("selected_spans") or []
        if spans:
            lines.append("| selected cluster | hours | start | end |")
            lines.append("|---|---:|---|---|")
            for sp in spans:
                lines.append(
                    f"| `{sp['cluster']}` | {sp['n_hours']} | {sp['start_utc']} | {sp['end_utc']} |"
                )
            lines.append("")
        ov = rec.get("overlaps") or []
        if ov:
            lines.append("| overlap UTC | clusters | n | agree | sign disagree | values |")
            lines.append("|---|---|---:|---|---|---|")
            for o in ov:
                vals = o.get("values") or []
                val_s = ", ".join(f"{v:.6g}" if isinstance(v, (int, float)) else str(v) for v in vals)
                clusters = ", ".join(f"`{c}`" for c in (o.get("clusters") or []))
                lines.append(
                    f"| {o.get('time_utc')} | {clusters} | {o.get('n_rows')} | "
                    f"{o.get('agree')} | {o.get('sign_disagree')} | {val_s} |"
                )
            lines.append("")
        lines.append("| cluster | replica | rows | start | end |")
        lines.append("|---|---|---:|---|---|")
        for ident in rec.get("identities") or []:
            lines.append(
                f"| `{ident['cluster']}` | `{ident['replica'] or ''}` | {ident['n_rows']} | "
                f"{ident['start_utc']} | {ident['end_utc']} |"
            )
        lines.append("")
        trans = rec.get("source_identity_transitions") or []
        if trans:
            lines.append("| time UTC | from | to | jump | vs typical | large D jump |")
            lines.append("|---|---|---|---:|---:|---|")
            for t in trans:
                jot = t.get("jump_over_typical")
                jot_s = "" if jot is None else f"{jot:.1f}×"
                jump = t.get("abs_jump")
                jump_s = "" if jump is None else f"{jump:.6g}"
                lines.append(
                    f"| {t['time_utc']} | `{t['from_cluster']}` | `{t['to_cluster']}` | "
                    f"{jump_s} | {jot_s} | {t.get('large_d_jump')} |"
                )
            lines.append("")
    return "\n".join(lines) + "\n"
