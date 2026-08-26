"""CLI: baseline | vector | outcomes | dataset | infer | visualize-vector."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from . import BASELINE_VERSION, VECTOR_VERSION_CORTEX_MOTION, VECTOR_VERSION_OLS
from .baseline import (
    align_prices,
    classify_moves,
    fetch_baseline_features,
    load_aligned_csv,
    truth_table,
    write_baseline_report,
)
from .config import load_config
from .cortex_tape import fetch_cortex_motion_raw, hourly_from_raw, hourly_wide_from_raw
from .cortex_vector import build_cortex_vector, cortex_horizon_names
from .diagnostics import representation_diagnostics, turn_diagnostics
from .feature_allowlist import select_model_feature_columns
from .manifest import sha256_file, write_manifest
from .oracle import ORACLE_FILE_NAME, build_oracle_table
from .outcome_engine import build_outcomes, unlabeled_tail_counts
from .quality import quality_summary
from .representation import cortex_vs_ols_disagreement, write_horizon_plots, write_spot_tables
from .source_lineage import annotate_selection, annotate_turns_near_boundaries, audit_markdown, audit_raw_tape
from .source_guard import inference_row_is_actionable, load_inference_vector
from .vector_engine import build_vector, vector_validity_by_horizon


def _run_dir(root: Path, run_id: str) -> Path:
    d = root / "artifacts" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_baseline(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    repo = Path(args.repo)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("baseline_%Y%m%dT%H%M%SZ")
    out = _run_dir(Path(args.out_root), run_id)
    cols = list(cfg["cortex"]["queries"].keys())
    if args.aligned_csv:
        aligned = load_aligned_csv(args.aligned_csv)
        features = aligned.drop(columns=["close_price"], errors="ignore")
        feat_path = out / "baseline_features.csv"
        features.to_csv(feat_path, index=False)
        aligned_path = out / "baseline_cortex_transition_baseline_v1.csv"
        aligned.to_csv(aligned_path, index=False)
        inputs = {"aligned_csv": Path(args.aligned_csv)}
    else:
        features = fetch_baseline_features(cfg)
        feat_path = out / "baseline_features.csv"
        features.to_csv(feat_path, index=False)
        aligned = align_prices(features, cfg)
        aligned_path = out / "baseline_cortex_transition_baseline_v1.csv"
        aligned.to_csv(aligned_path, index=False)
        inputs = {}
    tt = truth_table(features, cols)
    tt.to_csv(out / "truth_table.csv", index=False)
    b = cfg["baseline"]
    rows, summary = classify_moves(
        aligned,
        cols,
        hours=int(b["hours"]),
        threshold=float(b["flat_threshold_pct"]),
        log_return=bool(b.get("log_return", False)),
    )
    summary.to_csv(out / "baseline_state_counts.csv", index=False)
    q = quality_summary(
        aligned,
        time_col="time_bucket",
        extra={"baseline_version": b.get("version", BASELINE_VERSION)},
    )
    (out / "quality_summary.json").write_text(json.dumps(q, indent=2, default=str) + "\n")
    write_baseline_report(
        summary,
        out / "baseline_report.json",
        {
            "hours": b["hours"],
            "flat_threshold_pct": b["flat_threshold_pct"],
            "features": cols,
            "run_id": run_id,
            "baseline_version": b.get("version", BASELINE_VERSION),
            "quality": q,
        },
    )
    write_manifest(
        out / "manifest.json",
        run_id=run_id,
        repo=repo,
        config_path=Path(args.config),
        config=cfg,
        inputs=inputs,
        outputs={
            "features": feat_path,
            "aligned": aligned_path,
            "truth_table": out / "truth_table.csv",
            "state_counts": out / "baseline_state_counts.csv",
            "quality": out / "quality_summary.json",
        },
        extra={"command": "baseline", "baseline_version": b.get("version", BASELINE_VERSION), "quality": q},
    )
    print(json.dumps({"run_id": run_id, "out": str(out), "states": len(summary), "rows": len(aligned)}, indent=2))


def cmd_vector(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    prices = __import__("pandas").read_csv(args.prices)
    vec = build_vector(prices, cfg)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    vec.to_csv(dest, index=False)
    n_valid = int(vec["row_vector_valid"].sum()) if "row_vector_valid" in vec.columns else 0
    print(f"wrote {dest} rows={len(vec)} valid={n_valid}")


def cmd_outcomes(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    prices = __import__("pandas").read_csv(args.prices)
    oc = build_outcomes(prices, cfg)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    oc.to_csv(dest, index=False)
    print(f"wrote {dest} rows={len(oc)}")


def cmd_dataset(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    prices = __import__("pandas").read_csv(args.prices)
    vec = build_vector(prices, cfg)
    oc = build_outcomes(prices, cfg)
    time_col = "time_bucket"
    run_id = args.run_id or datetime.now(timezone.utc).strftime("dataset_%Y%m%dT%H%M%SZ")
    out = _run_dir(Path(args.out_root), run_id)
    vec_path = out / "vector_price_ols_proxy_v1.csv"
    oc_path = out / "outcomes_v1.csv"
    vec.to_csv(vec_path, index=False)
    oc.to_csv(oc_path, index=False)
    oracle = build_oracle_table(prices, vec, cfg)
    oracle_path = out / ORACLE_FILE_NAME
    oracle.to_csv(oracle_path, index=False)
    drop_cols = [c for c in ("btc_close_at_t", "observation_time_utc") if c in oc.columns]
    view = vec.merge(oc.drop(columns=drop_cols, errors="ignore"), on=time_col, how="left")
    view_path = out / "research_view.csv"
    view.to_csv(view_path, index=False)
    model_cols = select_model_feature_columns(vec, list(cfg["vector"]["horizons"].keys()))
    allowlist_path = out / "model_feature_allowlist.json"
    allowlist_path.write_text(
        json.dumps({"model_feature_allowlist": model_cols}, indent=2) + "\n"
    )
    horizons = list(cfg["vector"]["horizons"].keys())
    ret_h = list(cfg["outcomes"]["return_horizons_h"])
    turns = turn_diagnostics(vec, horizons)
    turns_path = out / "turn_diagnostics.json"
    turns_path.write_text(json.dumps(turns, indent=2, default=str) + "\n")
    q = quality_summary(
        vec,
        time_col=time_col,
        valid_col="row_vector_valid",
        extra={
            "vector_source": str(vec["vector_source"].iloc[0]) if len(vec) else None,
            "vector_version": str(vec["vector_version"].iloc[0]) if len(vec) else None,
            "detector_version": str(vec["detector_version"].iloc[0]) if len(vec) else None,
            "vector_validity_by_horizon": vector_validity_by_horizon(vec, horizons),
            "unlabeled_outcome_tails": unlabeled_tail_counts(oc, ret_h),
            "turn_diagnostics": turns,
        },
    )
    (out / "quality_summary.json").write_text(json.dumps(q, indent=2, default=str) + "\n")
    write_manifest(
        out / "manifest.json",
        run_id=run_id,
        repo=Path(args.repo),
        config_path=Path(args.config),
        config=cfg,
        inputs={"prices": Path(args.prices)},
        outputs={
            "vector": vec_path,
            "outcomes": oc_path,
            "view": view_path,
            "oracle_research_only": oracle_path,
            "model_feature_allowlist": allowlist_path,
            "turn_diagnostics": turns_path,
            "quality": out / "quality_summary.json",
        },
        extra={
            "command": "dataset",
            "note": (
                "Live inference must load vector_price_ols_proxy_v1.csv only, never "
                "research_view.csv or oracle_extrema_research_only.csv. "
                "Oracle extrema are research-only and are not merged into the vector."
            ),
            "model_feature_allowlist": model_cols,
            "quality": q,
        },
    )
    print(json.dumps({"run_id": run_id, "out": str(out), "rows": len(vec), "quality": q["n_rows"]}, indent=2))


def cmd_cortex_dataset(args: argparse.Namespace) -> None:
    """Build cortex_market_motion_v1 from a preserved raw query_range tape."""
    cfg = load_config(args.config)
    run_id = args.run_id or "dataset_cortex_market_motion_v1_prod_a_preferred"
    out = _run_dir(Path(args.out_root), run_id)
    names = cortex_horizon_names(cfg)
    raw_path = Path(args.raw_csv) if args.raw_csv else out / "cortex_continuous_derivatives_raw.csv"
    logs: list = []
    if args.fetch or not raw_path.exists():
        raw, logs = fetch_cortex_motion_raw(cfg)
        raw.to_csv(raw_path, index=False)
        (out / "cortex_retrieval_log.json").write_text(json.dumps(logs, indent=2, default=str) + "\n")
    else:
        raw = __import__("pandas").read_csv(raw_path)
        log_path = out / "cortex_retrieval_log.json"
        if log_path.exists():
            logs = json.loads(log_path.read_text())

    before_turns = None
    prev_turns = out / "turn_diagnostics.json"
    if prev_turns.exists():
        before_turns = json.loads(prev_turns.read_text())
    else:
        legacy_turns = Path(args.out_root) / "artifacts" / "dataset_cortex_market_motion_v1_tape" / "turn_diagnostics.json"
        if legacy_turns.exists():
            before_turns = json.loads(legacy_turns.read_text())

    audit = audit_raw_tape(raw, names)

    hourly_all = hourly_wide_from_raw(raw)
    hourly_all_path = out / "cortex_continuous_derivatives_hourly.csv"
    hourly_all.to_csv(hourly_all_path, index=False)
    hourly = hourly_from_raw(raw, names, required=True)
    prices = None
    inputs: dict = {"raw": raw_path}
    if args.prices:
        prices = __import__("pandas").read_csv(args.prices)
        inputs["prices"] = Path(args.prices)
    vec = build_cortex_vector(hourly, cfg, prices=prices)
    vec_path = out / f"vector_{VECTOR_VERSION_CORTEX_MOTION}.csv"
    vec.to_csv(vec_path, index=False)

    audit = annotate_selection(audit, hourly, names)
    audit = annotate_turns_near_boundaries(audit, vec, names)
    audit_path = out / "source_lineage_audit.json"
    audit_md_path = out / "source_lineage_audit.md"
    audit_path.write_text(json.dumps(audit, indent=2, default=str) + "\n")
    audit_md_path.write_text(audit_markdown(audit))

    oracle = __import__("pandas").DataFrame()
    oracle_path = out / ORACLE_FILE_NAME
    if prices is not None:
        oracle = build_oracle_table(prices, vec, cfg)
        oracle.to_csv(oracle_path, index=False)
        oc = build_outcomes(prices, cfg)
        oc_path = out / "outcomes_v1.csv"
        oc.to_csv(oc_path, index=False)
        drop_cols = [c for c in ("btc_close_at_t", "observation_time_utc") if c in oc.columns]
        view = vec.merge(oc.drop(columns=drop_cols, errors="ignore"), on="time_bucket", how="left")
        view_path = out / "research_view.csv"
        view.to_csv(view_path, index=False)
    else:
        oc_path = None
        view_path = None

    model_cols = select_model_feature_columns(vec, names)
    allowlist_path = out / "model_feature_allowlist.json"
    allowlist_path.write_text(json.dumps({"model_feature_allowlist": model_cols}, indent=2) + "\n")
    turns = turn_diagnostics(vec, names)
    representation = representation_diagnostics(vec, names)
    diag = {**representation, "turn_diagnostics": turns, "retrieval": logs}
    if args.ols_vector:
        ols = __import__("pandas").read_csv(args.ols_vector, low_memory=False)
        diag["cortex_vs_ols"] = {
            "7d_vs_7d": cortex_vs_ols_disagreement(vec, ols, cortex_horizon="7d", ols_horizon="7d"),
            "8h_vs_7h": cortex_vs_ols_disagreement(vec, ols, cortex_horizon="8h", ols_horizon="7h"),
        }
        inputs["ols_vector_diagnostic_only"] = Path(args.ols_vector)
    turns_path = out / "turn_diagnostics.json"
    turns_path.write_text(json.dumps(turns, indent=2, default=str) + "\n")
    rep_path = out / "representation_diagnostics.json"
    rep_path.write_text(json.dumps(diag, indent=2, default=str) + "\n")
    spots = write_spot_tables(vec, out / "spot_checks", names)
    plots = write_horizon_plots(vec, out / "spot_checks", names)
    q = quality_summary(
        vec,
        time_col="time_bucket",
        valid_col="row_vector_valid",
        extra={
            "vector_source": str(vec["vector_source"].iloc[0]) if len(vec) else None,
            "vector_version": str(vec["vector_version"].iloc[0]) if len(vec) else None,
            "source_query_version": str(vec["source_query_version"].iloc[0]) if len(vec) else None,
            "detector_version": str(vec["detector_version"].iloc[0]) if len(vec) else None,
            "feature_configuration_hash": str(vec["feature_configuration_hash"].iloc[0]) if len(vec) else None,
            "timing_certainty": str(vec["timing_certainty"].iloc[0]) if "timing_certainty" in vec.columns and len(vec) else None,
            "raw_sha256": sha256_file(raw_path) if raw_path.exists() else None,
            "vector_validity_by_horizon": vector_validity_by_horizon(vec, names),
            "turn_diagnostics": turns,
            "representation": representation,
            "spot_checks": spots,
            "plots": plots,
            "retrieval": logs,
            "horizons": names,
            "source_lineage_audit": str(audit_path),
            "preserved_segmented_artifact": "artifacts/dataset_cortex_market_motion_v1_tape",
            "turn_counts_before_prod_a_preferred": before_turns,
            "n_source_conflicts": {
                h: int(vec[f"{h}_source_conflict"].fillna(False).sum())
                for h in names
                if f"{h}_source_conflict" in vec.columns
            },
            "n_source_boundaries": {
                h: int(vec[f"{h}_source_boundary"].fillna(False).sum())
                for h in names
                if f"{h}_source_boundary" in vec.columns
            },
            "note": (
                "Raw tape is unfilled. Signed Cortex D uses prod_a_preferred_v1, never max(). "
                "Cluster/replica are routing provenance for the same feature family. "
                "Routing-label changes do not reset Z/V/A/T or restart normalization warmup. "
                "True conflict only when no approved source can be selected. "
                "Prime/weighted series stay on the hourly tape and are not Vector v1. "
                "OLS comparison, if present, is diagnostic only. "
                "The previous segmented agree_or_conflict_v1 tape is preserved at "
                "artifacts/dataset_cortex_market_motion_v1_tape."
            ),
        },
    )
    (out / "quality_summary.json").write_text(json.dumps(q, indent=2, default=str) + "\n")
    outputs = {
        "raw": raw_path,
        "hourly_unfilled": hourly_all_path,
        "source_lineage_audit_json": audit_path,
        "source_lineage_audit_md": audit_md_path,
        "vector": vec_path,
        "model_feature_allowlist": allowlist_path,
        "turn_diagnostics": turns_path,
        "representation_diagnostics": rep_path,
        "quality": out / "quality_summary.json",
    }
    if oc_path is not None:
        outputs["outcomes"] = oc_path
        outputs["view"] = view_path
        outputs["oracle_research_only"] = oracle_path
    write_manifest(
        out / "manifest.json",
        run_id=run_id,
        repo=Path(args.repo),
        config_path=Path(args.config),
        config=cfg,
        inputs=inputs,
        outputs=outputs,
        extra={
            "command": "cortex-dataset",
            "vector_source": VECTOR_VERSION_CORTEX_MOTION,
            "note": (
                "Live inference must load vector_cortex_market_motion_v1.csv only. "
                "Do not mix with price_ols_proxy_v1 or cortex_transition_baseline_v1. "
                "Oracle extrema are research-only."
            ),
            "model_feature_allowlist": model_cols,
            "quality": q,
        },
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "out": str(out),
                "rows": len(vec),
                "valid": int(vec["row_vector_valid"].sum()) if "row_vector_valid" in vec.columns else None,
                "horizons": names,
            },
            indent=2,
        )
    )


def cmd_infer(args: argparse.Namespace) -> None:
    vec = load_inference_vector(args.vector)
    last = vec.iloc[-1]
    actionable = inference_row_is_actionable(last)
    print(
        json.dumps(
            {
                "vector_source": str(last.get("vector_source")),
                "vector_version": str(last.get("vector_version")),
                "row_vector_valid": bool(actionable),
                "action": "predict" if actionable else "no_prediction",
                "reason": None if actionable else "required_inputs_invalid",
            },
            indent=2,
        )
    )


def cmd_representation_sanity(args: argparse.Namespace) -> None:
    pd = __import__("pandas")
    from .representation_sanity import run_representation_sanity

    vec = pd.read_csv(args.vector, low_memory=False)
    rec = run_representation_sanity(vec)
    run_id = args.run_id or "representation_sanity_poc"
    out = _run_dir(Path(args.out_root), run_id)
    path = out / "representation_sanity.json"
    path.write_text(json.dumps(rec, indent=2, default=str) + "\n")
    print(json.dumps({"run_id": run_id, "out": str(path), "detector_status": rec.get("detector_status"), "defects": rec.get("defects")}, indent=2))


def cmd_analogue_eval(args: argparse.Namespace) -> None:
    pd = __import__("pandas")
    from .analogue_eval import WALKFORWARD_DISCLAIMER, run_walkforward_eval, walkforward_markdown

    cfg = load_config(args.config)
    vec = pd.read_csv(args.vector, low_memory=False)
    oc = pd.read_csv(args.outcomes, low_memory=False)
    rec = run_walkforward_eval(vec, oc, cfg=cfg)
    source = str(rec.get("vector_source") or "unknown")
    run_id = args.run_id or f"analogue_eval_{source}"
    out = _run_dir(Path(args.out_root), run_id)
    json_path = out / "analogue_walkforward.json"
    md_path = out / "ANALOGUE_WALKFORWARD.md"
    csv_path = out / "analogue_walkforward_queries.csv"
    json_path.write_text(json.dumps({k: v for k, v in rec.items() if k != "queries"}, indent=2, default=str) + "\n")
    md_path.write_text(walkforward_markdown(rec))
    queries = rec.get("queries") or []
    if queries:
        pd.DataFrame(queries).to_csv(csv_path, index=False)
    write_manifest(
        out / "manifest.json",
        run_id=run_id,
        repo=Path(args.repo),
        config_path=Path(args.config),
        config=cfg,
        inputs={"vector": Path(args.vector), "outcomes": Path(args.outcomes)},
        outputs={"eval_json": json_path, "eval_md": md_path, "eval_csv": csv_path},
        extra={
            "command": "analogue-eval",
            "disclaimer": WALKFORWARD_DISCLAIMER,
            "note": "Development-only walk-forward. Matcher unchanged. Holdouts were not opened or scored.",
            "n_evaluated": rec.get("n_evaluated"),
            "n_abstain": rec.get("n_abstain"),
            "interpretation": rec.get("interpretation"),
        },
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "out": str(out),
                "disclaimer": WALKFORWARD_DISCLAIMER,
                "n_evaluated": rec.get("n_evaluated"),
                "n_abstain": rec.get("n_abstain"),
                "mean_crps_analogue": rec.get("mean_crps_analogue"),
                "mean_crps_uncond": rec.get("mean_crps_uncond"),
                "mean_crps_difference_analogue_minus_uncond": rec.get(
                    "mean_crps_difference_analogue_minus_uncond"
                ),
                "pct_queries_analogue_beats_uncond_crps": rec.get("pct_queries_analogue_beats_uncond_crps"),
                "interpretation": rec.get("interpretation"),
            },
            indent=2,
            default=str,
        )
    )


def cmd_analogue_eval_horizons(args: argparse.Namespace) -> None:
    pd = __import__("pandas")
    from .analogue_eval import HORIZON_DISCLAIMER, multi_horizon_markdown, run_multi_horizon_eval

    cfg = load_config(args.config)
    vec = pd.read_csv(args.vector, low_memory=False)
    oc = pd.read_csv(args.outcomes, low_memory=False)
    rec = run_multi_horizon_eval(vec, oc, cfg=cfg)
    source = str(rec.get("vector_source") or "unknown")
    run_id = args.run_id or f"analogue_eval_horizons_{source}"
    out = _run_dir(Path(args.out_root), run_id)
    tables = rec.pop("query_tables", {}) or {}
    json_path = out / "analogue_horizons.json"
    md_path = out / "ANALOGUE_HORIZONS.md"
    json_path.write_text(json.dumps(rec, indent=2, default=str) + "\n")
    md_path.write_text(multi_horizon_markdown({**rec, "query_tables": tables}))
    outputs = {"eval_json": json_path, "eval_md": md_path}
    for name, rows in tables.items():
        if not rows:
            continue
        csv_path = out / f"analogue_horizon_{name}_queries.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        outputs[f"queries_{name}"] = csv_path
    write_manifest(
        out / "manifest.json",
        run_id=run_id,
        repo=Path(args.repo),
        config_path=Path(args.config),
        config=cfg,
        inputs={"vector": Path(args.vector), "outcomes": Path(args.outcomes)},
        outputs=outputs,
        extra={
            "command": "analogue-eval-horizons",
            "disclaimer": HORIZON_DISCLAIMER,
            "note": "Exploratory multi-horizon walk-forward. Matcher unchanged. 24h remains the approved target. Holdouts closed.",
            "shared_schedule": rec.get("shared_schedule"),
            "purge_lags_hours": rec.get("purge_lags_hours"),
        },
    )
    skills = {
        h: (rec.get("horizons") or {}).get(h, {}).get("crps_skill_score")
        for h in ("4h", "8h", "12h", "24h")
    }
    print(
        json.dumps(
            {
                "run_id": run_id,
                "out": str(out),
                "disclaimer": HORIZON_DISCLAIMER,
                "n_scheduled": (rec.get("shared_schedule") or {}).get("n_scheduled"),
                "crps_skill_by_horizon": skills,
                "n_evaluated": {
                    h: (rec.get("horizons") or {}).get(h, {}).get("n_evaluated")
                    for h in ("4h", "8h", "12h", "24h")
                },
            },
            indent=2,
            default=str,
        )
    )


def cmd_analogue_poc(args: argparse.Namespace) -> None:
    pd = __import__("pandas")
    from .analogue import run_analogue_poc
    from .periods import DISCLAIMER
    from .transition import EVENT_PEAK, EVENT_TROUGH

    cfg = load_config(args.config)
    vec = pd.read_csv(args.vector, low_memory=False)
    oc = pd.read_csv(args.outcomes, low_memory=False)
    extra = [__import__("pandas").Timestamp(t) for t in (args.example_time or [])]
    extra = [t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC") for t in extra]
    if not extra:
        times = pd.to_datetime(vec["time_bucket"], utc=True)
        from .periods import DEVELOPMENT_START_UTC, last_observation_with_complete_outcome

        last = last_observation_with_complete_outcome()
        in_dev = (times >= DEVELOPMENT_START_UTC) & (times <= last)
        for h, col in (("2h", "2h_transition_event"), ("7d", "7d_transition_event"), ("24h", "24h_transition_event")):
            if col not in vec.columns:
                continue
            ev = vec[col].astype(str).isin({EVENT_PEAK, EVENT_TROUGH}) & in_dev
            if not bool(ev.any()):
                continue
            extra.append(pd.Timestamp(times[ev].iloc[len(times[ev]) // 2]))
    rec = run_analogue_poc(vec, oc, extra_example_times=extra, cfg=cfg)
    source = str(rec.get("vector_source") or "unknown")
    run_id = args.run_id or f"analogue_poc_{source}"
    out = _run_dir(Path(args.out_root), run_id)
    json_path = out / "analogue_poc.json"
    md_path = out / "ANALOGUE_POC.md"
    json_path.write_text(json.dumps(rec, indent=2, default=str) + "\n")
    md_path.write_text(_analogue_markdown(rec))
    write_manifest(
        out / "manifest.json",
        run_id=run_id,
        repo=Path(args.repo),
        config_path=Path(args.config),
        config=cfg,
        inputs={"vector": Path(args.vector), "outcomes": Path(args.outcomes)},
        outputs={"analogue_json": json_path, "analogue_md": md_path},
        extra={
            "command": "analogue-poc",
            "disclaimer": DISCLAIMER,
            "note": "Development period only. Holdouts were not opened or scored.",
        },
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "out": str(out),
                "disclaimer": DISCLAIMER,
                "n_queries": rec.get("n_queries"),
                "n_abstain": rec.get("n_abstain"),
                "unconditional_median": rec.get("unconditional_median"),
                "batch_conditional_median_of_medians": rec.get("batch_conditional_median_of_medians"),
            },
            indent=2,
        )
    )


def cmd_visualize_vector(args: argparse.Namespace) -> None:
    from .visualize_vector import run_visualization

    summary = run_visualization(Path(args.vector), Path(args.out))
    printable = {k: v for k, v in summary.items() if k != "canvas"}
    print(json.dumps(printable, indent=2, default=str))


def _analogue_markdown(rec: dict) -> str:
    from .periods import DISCLAIMER

    un = rec.get("unconditional_development_return_24h") or {}
    lines = [
        f"# Analogue POC",
        "",
        f"**{DISCLAIMER}**",
        "",
        f"- vector: `{rec.get('vector_source')}` / `{rec.get('vector_version')}`",
        f"- distance: `{rec.get('analogue_version')}`",
        f"- queries: {rec.get('n_queries')} (abstain {rec.get('n_abstain')})",
        "",
        "## Unconditional development 24h return (%)",
        "",
        f"- n={un.get('n')}; median={un.get('median')}; P(+ )={un.get('probability_positive')}",
        f"- p05={un.get('p05')}; p25={un.get('p25')}; p75={un.get('p75')}; p95={un.get('p95')}",
        "",
        "## Batch analogue vs unconditional",
        "",
        f"- median of analogue medians: {rec.get('batch_conditional_median_of_medians')}",
        f"- mean analogue P(+): {rec.get('batch_mean_p_positive')}",
        f"- unconditional median: {rec.get('unconditional_median')}",
        f"- unconditional P(+): {rec.get('unconditional_p_positive')}",
        "",
        "## Example queries",
        "",
    ]
    for card in rec.get("queries") or []:
        cond = card.get("conditional_return_24h") or {}
        lines.append(f"### {card.get('query_time_utc')}")
        lines.append("")
        lines.append(
            f"- abstain={card.get('abstain')} reason={card.get('reason')} "
            f"neighbors={card.get('n_neighbors')} effective={card.get('effective_support')} "
            f"median_distance={card.get('median_distance')}"
        )
        if cond.get("n"):
            lines.append(
                f"- analogue 24h %: median={cond.get('median')} "
                f"P(+)={cond.get('probability_positive')} "
                f"p05={cond.get('p05')} p95={cond.get('p95')}"
            )
        nbs = card.get("neighbors") or []
        if nbs:
            lines.append("- nearest: " + "; ".join(f"{n['time_utc']} d={n['distance']:.3f} r={n['return_24h_pct']}" for n in nbs[:5]))
        lines.append("")
    dist = rec.get("distance") or {}
    lines.extend(
        [
            "## How similarity is measured",
            "",
            str(dist.get("name")),
            str(dist.get("binary")),
            str(dist.get("continuous")),
            "",
            "## How future information is excluded",
            "",
            str(dist.get("future_exclusion")),
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="market_motion")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--repo", default="..")
    p.add_argument("--out-root", default=".")
    p.add_argument("--run-id", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("baseline", help="Cortex transition baseline v1 (live fetch or local aligned CSV)")
    b.add_argument(
        "--aligned-csv",
        default=None,
        help="Previously fetched Cortex+price CSV. Skips live Cortex/ClickHouse.",
    )
    b.set_defaults(func=cmd_baseline)

    v = sub.add_parser("vector", help="Build price_ols_proxy_v1 from an hourly price CSV")
    v.add_argument("--prices", required=True)
    v.add_argument("--out", default=f"artifacts/vector_{VECTOR_VERSION_OLS}.csv")
    v.set_defaults(func=cmd_vector)

    o = sub.add_parser("outcomes", help="Build Outcome v1 from an hourly price CSV")
    o.add_argument("--prices", required=True)
    o.add_argument("--out", default="artifacts/outcomes_v1.csv")
    o.set_defaults(func=cmd_outcomes)

    d = sub.add_parser("dataset", help="Vector + outcomes as separate tables plus a joined research view")
    d.add_argument("--prices", required=True)
    d.set_defaults(func=cmd_dataset)

    cd = sub.add_parser(
        "cortex-dataset",
        help="Build cortex_market_motion_v1 from Cortex query_range (raw saved before fill)",
    )
    cd.add_argument("--raw-csv", default=None, help="Existing raw tape; skips fetch when the file exists")
    cd.add_argument("--prices", default="btc_last_price.csv")
    cd.add_argument(
        "--ols-vector",
        default=None,
        help="Optional price_ols_proxy_v1 CSV for diagnostic disagreement tables only",
    )
    cd.add_argument("--fetch", action="store_true", help="Re-query Cortex even if raw CSV exists")
    cd.set_defaults(func=cmd_cortex_dataset)

    inf = sub.add_parser("infer", help="Load a vector artifact only; refuse research_view / invalid rows")
    inf.add_argument("--vector", required=True)
    inf.set_defaults(func=cmd_infer)

    rs = sub.add_parser("representation-sanity", help="Lightweight detector review; no 24h scoring")
    rs.add_argument("--vector", required=True)
    rs.set_defaults(func=cmd_representation_sanity)

    ap = sub.add_parser("analogue-poc", help="Phase 4 analogue POC on the development period only")
    ap.add_argument("--vector", required=True)
    ap.add_argument("--outcomes", required=True)
    ap.add_argument("--example-time", action="append", default=[], help="Extra UTC query timestamp")
    ap.set_defaults(func=cmd_analogue_poc)

    ae = sub.add_parser(
        "analogue-eval",
        help="Development-only walk-forward CRPS/Brier of the frozen analogue matcher",
    )
    ae.add_argument("--vector", required=True)
    ae.add_argument("--outcomes", required=True)
    ae.set_defaults(func=cmd_analogue_eval)

    ah = sub.add_parser(
        "analogue-eval-horizons",
        help="Exploratory 4h/8h/12h/24h walk-forward; matcher and query dates unchanged",
    )
    ah.add_argument("--vector", required=True)
    ah.add_argument("--outcomes", required=True)
    ah.set_defaults(func=cmd_analogue_eval_horizons)

    viz = sub.add_parser(
        "visualize-vector",
        help="Full-history Cortex vector dashboard (visualization only; no scoring)",
    )
    viz.add_argument(
        "--vector",
        default="artifacts/dataset_cortex_market_motion_v1_prod_a_preferred/vector_cortex_market_motion_v1.csv",
    )
    viz.add_argument("--out", default="artifacts/vector_visualization_poc")
    viz.set_defaults(func=cmd_visualize_vector)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
