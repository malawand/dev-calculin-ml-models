"""Lightweight representation-only review of detector v2. No 24h scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .feature_allowlist import infer_horizons
from .periods import DISCLAIMER, as_utc
from .transition import EVENT_PEAK, EVENT_TROUGH

HORIZON_ORDER = ("2h", "4h", "7h", "8h", "12h", "24h", "48h", "7d")


def _horizon_hours(name: str) -> int:
    if name.endswith("d"):
        return int(name[:-1]) * 24
    return int(name[:-1])


def _event_mask(s: pd.Series) -> pd.Series:
    return s.astype(str).isin({EVENT_PEAK, EVENT_TROUGH})


def inspect_random_windows(vec: pd.DataFrame, rng: np.random.Generator, n: int = 4) -> list[dict[str, Any]]:
    times = pd.to_datetime(vec["time_bucket"], utc=True)
    valid = vec["row_vector_valid"].fillna(False).astype(bool) if "row_vector_valid" in vec.columns else pd.Series(True, index=vec.index)
    valid_times = pd.to_datetime(vec.loc[valid, "time_bucket"], utc=True)
    if valid_times.empty:
        return []
    step = max(1, len(valid_times) // (n + 2))
    starts = valid_times.iloc[::step][:n]
    out = []
    horizons = [h for h in infer_horizons(vec.columns) if f"{h}_transition_state_raw" in vec.columns]
    for ts in starts[:n]:
        window = (times >= ts) & (times < ts + pd.Timedelta(days=7))
        rec: dict[str, Any] = {"window_start_utc": str(ts), "n_rows": int(window.sum())}
        rec["n_valid"] = int((window & valid).sum())
        rec["horizon_state_changes"] = {}
        rec["horizon_events"] = {}
        for h in horizons:
            st = vec.loc[window, f"{h}_transition_state_raw"]
            rec["horizon_state_changes"][h] = int(st.nunique(dropna=True))
            ev = vec.loc[window, f"{h}_transition_event"] if f"{h}_transition_event" in vec.columns else pd.Series(dtype=object)
            rec["horizon_events"][h] = int(_event_mask(ev).sum()) if len(ev) else 0
        out.append(rec)
    return out


def inspect_confirmed_turns(vec: pd.DataFrame, n_each: int = 4) -> dict[str, Any]:
    times = pd.to_datetime(vec["time_bucket"], utc=True)
    horizons = [h for h in HORIZON_ORDER if f"{h}_transition_event" in vec.columns]
    by_h: dict[str, Any] = {}
    defects: list[str] = []
    for h in horizons:
        ev = vec[f"{h}_transition_event"]
        mask = _event_mask(ev)
        n = int(mask.sum())
        rec: dict[str, Any] = {"n_events": n}
        if n == 0:
            by_h[h] = rec
            continue
        idx = np.linspace(0, n - 1, min(n_each, n), dtype=int)
        event_rows = vec.loc[mask].iloc[idx]
        samples = []
        delays = []
        hours_at_event = []
        for _, row in event_rows.iterrows():
            t = as_utc(row["time_bucket"])
            confirm = pd.to_datetime(row.get(f"{h}_confirmation_time"), utc=True, errors="coerce")
            cand = pd.to_datetime(row.get(f"{h}_candidate_extremum_time"), utc=True, errors="coerce")
            hrs = row.get(f"{h}_hours_since_confirmation")
            samples.append(
                {
                    "time_utc": str(t),
                    "event": str(row[f"{h}_transition_event"]),
                    "confirmation_time": None if pd.isna(confirm) else str(confirm),
                    "candidate_extremum_time": None if pd.isna(cand) else str(cand),
                    "hours_since_confirmation": None if pd.isna(hrs) else float(hrs),
                    "state": None if pd.isna(row.get(f"{h}_transition_state_raw")) else float(row[f"{h}_transition_state_raw"]),
                }
            )
            if pd.notna(confirm) and confirm != t:
                defects.append(f"{h} confirmation_time {confirm} != event row {t}")
            if pd.notna(hrs) and float(hrs) != 0:
                defects.append(f"{h} hours_since_confirmation={hrs} at event {t} (expected 0)")
            if pd.notna(cand) and pd.notna(confirm):
                delays.append((confirm - cand) / pd.Timedelta(hours=1))
            hours_at_event.append(0.0 if pd.isna(hrs) else float(hrs))
        rec["samples"] = samples
        rec["median_hours_between_events"] = None
        ev_times = times[mask].sort_values()
        if len(ev_times) >= 2:
            rec["median_hours_between_events"] = float(ev_times.diff().dt.total_seconds().dropna().median() / 3600.0)
        rec["confirmation_delay_hours_sample"] = None if not delays else float(np.median(delays))
        rec["hours_since_confirmation_at_event_max"] = max(hours_at_event) if hours_at_event else None
        by_h[h] = rec
    order = [h for h in horizons if by_h.get(h, {}).get("median_hours_between_events") is not None]
    medians = [by_h[h]["median_hours_between_events"] for h in order]
    faster_ok = all(medians[i] <= medians[i + 1] + 1e-9 for i in range(len(medians) - 1)) if len(medians) >= 2 else False
    # 2h vs 4h can be noisy and similar; require 2h < 24h < 7d as the obvious check
    def _med(name: str) -> float | None:
        return by_h.get(name, {}).get("median_hours_between_events")

    obvious = True
    if _med("2h") is not None and _med("24h") is not None and _med("2h") >= _med("24h"):
        obvious = False
        defects.append("2h turns are not faster than 24h turns")
    if _med("24h") is not None and _med("7d") is not None and _med("24h") >= _med("7d"):
        obvious = False
        defects.append("24h turns are not faster than 7d turns")
    return {
        "by_horizon": by_h,
        "median_hours_between_events_in_schema_order": {h: _med(h) for h in order},
        "short_horizons_generally_faster": bool(faster_ok or obvious),
        "obvious_short_vs_long_ok": obvious,
        "defects": defects,
    }


def inspect_missing_periods(vec: pd.DataFrame) -> dict[str, Any]:
    horizons = [h for h in infer_horizons(vec.columns) if f"{h}_invalidation_reason" in vec.columns]
    rec: dict[str, Any] = {"by_horizon": {}, "defects": []}
    for h in horizons:
        reason = vec[f"{h}_invalidation_reason"].fillna("").astype(str)
        stale = reason.eq("stale_or_missing_cortex_d") | reason.eq("cortex_d_undefined")
        events = _event_mask(vec[f"{h}_transition_event"]) if f"{h}_transition_event" in vec.columns else pd.Series(False, index=vec.index)
        valid = vec[f"{h}_feature_valid"].fillna(False).astype(bool) if f"{h}_feature_valid" in vec.columns else pd.Series(True, index=vec.index)
        n_stale = int(stale.sum())
        n_events_on_stale = int((stale & events).sum())
        n_events_on_invalid = int((~valid & events).sum())
        rec["by_horizon"][h] = {
            "n_stale_or_missing": n_stale,
            "n_confirmed_turns_on_stale_or_missing": n_events_on_stale,
            "n_confirmed_turns_while_feature_invalid": n_events_on_invalid,
        }
        if n_events_on_stale:
            rec["defects"].append(f"{h}: {n_events_on_stale} confirmed turns on stale/missing D")
    return rec


def run_representation_sanity(vec: pd.DataFrame, *, seed: int = 7) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    turns = inspect_confirmed_turns(vec)
    missing = inspect_missing_periods(vec)
    windows = inspect_random_windows(vec, rng)
    defects = list(turns.get("defects") or []) + list(missing.get("defects") or [])
    # hours_since_confirmation == 0 at event is the confirmation-time rule
    confirm_ok = not any("hours_since_confirmation" in d or "confirmation_time" in d for d in turns.get("defects") or [])
    status = "POC-frozen" if not defects else "needs_review"
    return {
        "disclaimer": DISCLAIMER,
        "detector_version": str(vec["detector_version"].iloc[0]) if "detector_version" in vec.columns and len(vec) else None,
        "detector_status": status,
        "detector_status_meaning": (
            "fixed for this proof of concept — not approved as the final detector"
            if status == "POC-frozen"
            else "obvious defect found; do not POC-freeze"
        ),
        "n_rows": int(len(vec)),
        "random_windows": windows,
        "confirmed_turns": turns,
        "missing_data": missing,
        "confirmation_time_rule_holds_in_sample": confirm_ok,
        "short_horizons_faster_than_long": bool(turns.get("obvious_short_vs_long_ok")),
        "defects": defects,
        "note": "Representation only. No 24h outcome relationship was examined.",
    }
