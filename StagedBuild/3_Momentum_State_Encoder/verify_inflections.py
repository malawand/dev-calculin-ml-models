#!/usr/bin/env python3
"""
Verification harness for the momentum state encoder.

The question this script answers:

    "When the encoder labels a bar as DERIVATIVE_TROUGH (or PEAK),
     is the FEATURE actually at a trough (or peak) at that bar?"

This is independent of any price-related question. We only ask:

    1. (Mechanical) Did d1 actually flip sign from neg -> pos
       within the lookback window at every claimed TROUGH event?
       (This must be 100% — it is what the encoder is defined to detect.)

    2. (Local-extremum) Is the feature value at each TROUGH event a
       local minimum compared to the surrounding bars in a small
       window? Symmetric check for PEAK -> local maximum.

    3. (Visual) Plot the feature curve directly with the encoder's
       TROUGH/PEAK markers on top, so a human can sanity-check by
       eye that the dots sit at the wave's actual troughs/peaks.

Outputs:
    plots/verify_<feat>_w<window>.png      — full-dataset overlay
    plots/verify_<feat>_w<window>_<slice>.png — zoomed overlay
    A printed summary table of mechanical and local-extremum agreement.
"""
from __future__ import annotations

import argparse
import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from momentum_state import DERIVATIVE_PEAK, DERIVATIVE_TROUGH, CROSSING_LOOKBACK_BARS


PLOTS_DIR = "plots"
DEFAULT_INPUT = "btc_data_15m_mstate.parquet"
DEFAULT_FEATURES = [
    "weighted_norm_avg_16h_24h_48h",
    "weighted_deriv_24h_48h_7d",
    "norm_combined_avg",
]
DEFAULT_WINDOWS = [16, 96]
LOCAL_EXTREMUM_HALF_WINDOW = 8  # ±2h at 15m bars


# ---------------------------------------------------------------------------
# 1. Mechanical verification
# ---------------------------------------------------------------------------
def mechanical_check(df: pd.DataFrame, feat: str, window: int) -> dict:
    """
    For every TROUGH event the encoder produced, confirm that the FAST
    derivative ``d1_fast`` actually contains a negative -> non-negative
    flip within the last CROSSING_LOOKBACK_BARS bars. Same for PEAK with
    pos -> non-positive.

    (TROUGH/PEAK fire on d1_fast, NOT on d1 — d1 is reserved for the
    steady-state classifications.)
    """
    d1_fast_col = f"{feat}_d1_fast_{window}"
    if d1_fast_col not in df.columns:
        # Older parquet — fall back to windowed d1.
        d1_fast_col = f"{feat}_d1_{window}"
    d1 = df[d1_fast_col].values
    state = df[f"{feat}_mstate_{window}"].values

    trough_idx = np.where(state == DERIVATIVE_TROUGH)[0]
    peak_idx = np.where(state == DERIVATIVE_PEAK)[0]

    def has_flip(idx: int, direction: str) -> bool:
        # Look at transitions ending in {idx-2 -> idx-1, idx-1 -> idx}.
        for k in range(CROSSING_LOOKBACK_BARS - 1):
            i = idx - 1 - k
            j = i + 1
            if i < 0 or j >= len(d1):
                continue
            a, b = d1[i], d1[j]
            if np.isnan(a) or np.isnan(b):
                continue
            if direction == "up" and a < 0 and b >= 0:
                return True
            if direction == "down" and a > 0 and b <= 0:
                return True
        return False

    trough_ok = sum(has_flip(i, "up") for i in trough_idx)
    peak_ok = sum(has_flip(i, "down") for i in peak_idx)

    return {
        "trough_total": len(trough_idx),
        "trough_with_d1_flip": trough_ok,
        "trough_pct": (trough_ok / len(trough_idx) * 100) if len(trough_idx) else float("nan"),
        "peak_total": len(peak_idx),
        "peak_with_d1_flip": peak_ok,
        "peak_pct": (peak_ok / len(peak_idx) * 100) if len(peak_idx) else float("nan"),
    }


# ---------------------------------------------------------------------------
# 2. Local-extremum verification
# ---------------------------------------------------------------------------
def local_extremum_check(
    df: pd.DataFrame,
    feat: str,
    window: int,
    half_window: int | None = None,
) -> dict:
    """
    A bar is a "local minimum" of the feature if the feature value at the
    bar is the minimum among bars [t - half_window, t + half_window].
    Symmetric for local maximum. We are *retrospectively validating* the
    encoder's claim, so it is OK to look both backward AND forward here —
    this script never feeds those values back into the encoder.

    We also report a more lenient "near-local-extremum" rate where the
    bar is within ``tolerance`` of the actual extremum value.
    """
    feature = df[feat].values.astype(float)
    state = df[f"{feat}_mstate_{window}"].values

    # Expected detection lag is ~K bars (K = inflection window). We look
    # for a local minimum/maximum of f in [t - 2K, t]; that's the band
    # where the actual extremum should sit.
    from momentum_state import _inflection_window_for
    k = _inflection_window_for(window)
    look_back = 2 * k
    look_forward = max(1, k // 2)

    n = len(feature)

    trough_idx = np.where(state == DERIVATIVE_TROUGH)[0]
    peak_idx = np.where(state == DERIVATIVE_PEAK)[0]

    def has_local_min_in_band(t: int) -> bool:
        lo, hi = max(0, t - look_back), min(n, t + look_forward + 1)
        seg = feature[lo:hi]
        if len(seg) < 3 or np.isnan(seg).any():
            return False
        argmin = lo + int(np.argmin(seg))
        # Is argmin a strict local minimum within seg (not at boundary)?
        if 0 < argmin - lo < len(seg) - 1:
            return (feature[argmin] < feature[argmin - 1]) and (feature[argmin] < feature[argmin + 1])
        # Boundary case: still accept if it's the minimum.
        return True

    def has_local_max_in_band(t: int) -> bool:
        lo, hi = max(0, t - look_back), min(n, t + look_forward + 1)
        seg = feature[lo:hi]
        if len(seg) < 3 or np.isnan(seg).any():
            return False
        argmax = lo + int(np.argmax(seg))
        if 0 < argmax - lo < len(seg) - 1:
            return (feature[argmax] > feature[argmax - 1]) and (feature[argmax] > feature[argmax + 1])
        return True

    def near_local_min(t: int, tolerance_pct: float = 0.15) -> bool:
        lo, hi = max(0, t - look_back), min(n, t + look_forward + 1)
        seg = feature[lo:hi]
        if len(seg) < 3 or np.isnan(seg).any():
            return False
        seg_min, seg_max = seg.min(), seg.max()
        span = seg_max - seg_min
        if span == 0:
            return True
        return (feature[t] - seg_min) <= tolerance_pct * span

    def near_local_max(t: int, tolerance_pct: float = 0.15) -> bool:
        lo, hi = max(0, t - look_back), min(n, t + look_forward + 1)
        seg = feature[lo:hi]
        if len(seg) < 3 or np.isnan(seg).any():
            return False
        seg_min, seg_max = seg.min(), seg.max()
        span = seg_max - seg_min
        if span == 0:
            return True
        return (seg_max - feature[t]) <= tolerance_pct * span

    trough_strict = sum(has_local_min_in_band(i) for i in trough_idx)
    trough_near = sum(near_local_min(i) for i in trough_idx)
    peak_strict = sum(has_local_max_in_band(i) for i in peak_idx)
    peak_near = sum(near_local_max(i) for i in peak_idx)

    return {
        "trough_total": len(trough_idx),
        "trough_strict_min": trough_strict,
        "trough_near_min": trough_near,
        "peak_total": len(peak_idx),
        "peak_strict_max": peak_strict,
        "peak_near_max": peak_near,
        "look_back_bars": look_back,
        "look_forward_bars": look_forward,
    }


# ---------------------------------------------------------------------------
# 3. Visual overlay
# ---------------------------------------------------------------------------
def plot_feature_with_events(
    df: pd.DataFrame,
    feat: str,
    window: int,
    out_path: str,
    title_suffix: str = "",
) -> None:
    """
    Two-panel plot:
      - Top:   the feature wave with TROUGH (gold) / PEAK (orange)
               markers placed AT the bar where the encoder fired.
      - Bottom: the first derivative d1 with the same markers, plus a
               horizontal zero line so it's obvious the markers sit
               at d1 sign-change points.

    No price involved — this is purely a self-consistency check.
    """
    f = df[feat]
    d1_col = f"{feat}_d1_{window}"
    d1_fast_col = f"{feat}_d1_fast_{window}"
    have_fast = d1_fast_col in df.columns

    d1 = df[d1_col]
    d1_fast = df[d1_fast_col] if have_fast else d1
    state = df[f"{feat}_mstate_{window}"]

    trough_mask = state == DERIVATIVE_TROUGH
    peak_mask = state == DERIVATIVE_PEAK

    fig, axes = plt.subplots(3 if have_fast else 2, 1, figsize=(14, 9 if have_fast else 7), sharex=True)
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2] if have_fast else None

    ax1.plot(df.index, f, color="steelblue", linewidth=0.9)
    ax1.axhline(0, color="black", linewidth=0.4, alpha=0.5)
    ax1.scatter(
        df.index[trough_mask], f[trough_mask],
        color="gold", edgecolor="black", linewidth=0.4,
        s=22, zorder=5, label=f"DERIVATIVE_TROUGH ({trough_mask.sum():,})",
    )
    ax1.scatter(
        df.index[peak_mask], f[peak_mask],
        color="darkorange", edgecolor="black", linewidth=0.4,
        s=22, zorder=5, label=f"DERIVATIVE_PEAK ({peak_mask.sum():,})",
    )
    ax1.set_ylabel("feature value")
    ax1.set_title(f"{feat}  |  window={window}b  {title_suffix}")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.92)

    # Show d1_fast (what the events actually fire on) on its own panel.
    ax2.plot(df.index, d1_fast, color="goldenrod", linewidth=0.7, alpha=0.9)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.scatter(df.index[trough_mask], d1_fast[trough_mask],
                color="gold", edgecolor="black", linewidth=0.4, s=22, zorder=5)
    ax2.scatter(df.index[peak_mask], d1_fast[peak_mask],
                color="darkorange", edgecolor="black", linewidth=0.4, s=22, zorder=5)
    ax2.set_ylabel(f"d1_fast (K=inflection)")

    if ax3 is not None:
        ax3.plot(df.index, d1, color="orange", linewidth=0.7, alpha=0.9)
        ax3.axhline(0, color="black", linewidth=0.6)
        ax3.scatter(df.index[trough_mask], d1[trough_mask],
                    color="gold", edgecolor="black", linewidth=0.4, s=18, zorder=5)
        ax3.scatter(df.index[peak_mask], d1[peak_mask],
                    color="darkorange", edgecolor="black", linewidth=0.4, s=18, zorder=5)
        ax3.set_ylabel(f"d1 (windowed, w={window})")
        ax3.set_xlabel("time")
    else:
        ax2.set_xlabel("time")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt_pct(num: int, den: int) -> str:
    if den == 0:
        return "    n/a"
    return f"{num/den*100:6.2f}%"


def _print_table(rows: list[list[str]], header: list[str]) -> None:
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = "  ".join(f"{h:<{w}}" for h, w in zip(header, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(f"{c:<{w}}" for c, w in zip(r, widths)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify the encoder finds troughs/peaks in the feature itself")
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    p.add_argument("--windows", nargs="+", type=int, default=DEFAULT_WINDOWS)
    p.add_argument("--slice", nargs=2, metavar=("START", "END"),
                   default=["2025-03-18", "2025-03-20"])
    p.add_argument("--half-window", type=int, default=LOCAL_EXTREMUM_HALF_WINDOW,
                   help="±N bars to define a local extremum (default 8 = ±2h)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} rows from {args.input}\n")

    # ---- 1. Mechanical correctness ----
    print("=" * 78)
    print("MECHANICAL CHECK")
    print("(Did d1 actually sign-flip within last 3 bars at every event?)")
    print("This MUST be 100% — anything less is a bug in the encoder.")
    print("=" * 78)
    rows = []
    for feat in args.features:
        for w in args.windows:
            r = mechanical_check(df, feat, w)
            rows.append([
                feat[:30], str(w),
                f"{r['trough_total']:,}", _fmt_pct(r['trough_with_d1_flip'], r['trough_total']),
                f"{r['peak_total']:,}", _fmt_pct(r['peak_with_d1_flip'], r['peak_total']),
            ])
    _print_table(rows, ["feature", "win", "trough_n", "trough_d1_ok",
                        "peak_n", "peak_d1_ok"])

    # ---- 2. Local extremum check ----
    print()
    print("=" * 78)
    print("LOCAL EXTREMUM CHECK (band scaled by inflection window K)")
    print("Strict: a true local minimum/maximum exists in [t - 2K, t + K/2].")
    print("Near:   feature value at event is within 15% of band's [min, max] span.")
    print("=" * 78)
    rows = []
    for feat in args.features:
        for w in args.windows:
            r = local_extremum_check(df, feat, w, half_window=args.half_window)
            rows.append([
                feat[:30], str(w),
                f"{r['trough_total']:,}",
                _fmt_pct(r['trough_strict_min'], r['trough_total']),
                _fmt_pct(r['trough_near_min'], r['trough_total']),
                f"{r['peak_total']:,}",
                _fmt_pct(r['peak_strict_max'], r['peak_total']),
                _fmt_pct(r['peak_near_max'], r['peak_total']),
            ])
    _print_table(rows, ["feature", "win", "trough_n", "trough_strict",
                        "trough_near", "peak_n", "peak_strict", "peak_near"])

    # ---- 3. Visual overlay ----
    print()
    print("=" * 78)
    print("VISUAL OVERLAYS")
    print("=" * 78)

    # Full dataset overlay for one window only (full plots get huge).
    for feat in args.features:
        plot_feature_with_events(
            df, feat, max(args.windows),
            out_path=os.path.join(PLOTS_DIR, f"verify_{feat}_w{max(args.windows)}_full.png"),
            title_suffix="(entire dataset)",
        )

    # Zoomed slice overlays for every feature × window pair.
    if args.slice:
        start, end = args.slice
        sliced = df.loc[start:end]
        if not sliced.empty:
            slice_tag = f"{start}_to_{end}".replace(":", "")
            print(f"\nZoom slice: {sliced.index[0]} -> {sliced.index[-1]} ({len(sliced):,} bars)")
            for feat in args.features:
                for w in args.windows:
                    plot_feature_with_events(
                        sliced, feat, w,
                        out_path=os.path.join(PLOTS_DIR, f"verify_{feat}_w{w}_{slice_tag}.png"),
                        title_suffix=f"(slice {start} -> {end})",
                    )

    print("\nDone. Open plots/verify_*_*.png to inspect manually.")


if __name__ == "__main__":
    main()
