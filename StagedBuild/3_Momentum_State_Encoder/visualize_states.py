#!/usr/bin/env python3
"""
Visualization companion to ``momentum_state.py``.

Reads ``btc_data_15m_mstate.parquet`` (the encoder output) and writes a set
of PNG files into ``plots/`` so you can see what the encoder produced
without opening an interactive matplotlib session.

Three plot types are produced:

  1. plots/state_distribution.png
        Stacked horizontal bar chart of how the 9 momentum states are
        distributed for each (feature, window) pair across the entire
        dataset.

  2. plots/<feat>_w<window>_full.png
        Four-panel diagnostic of a single (feature, window) pair across
        a chosen slice of the timeline (default: the entire dataset).

  3. plots/<feat>_w<window>_<slice>.png
        Same four panels but zoomed into a specific date range — useful
        for sanity-checking the encoder against a known event.

Usage:
    python3 visualize_states.py
        Generates all defaults using the parquet at the standard path.

    python3 visualize_states.py --slice 2025-03-18 2025-03-20
        Adds a zoomed plot of that exact date range.
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend so this works over SSH/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm

from momentum_state import STATE_NAMES, quick_visualize


PLOTS_DIR = "plots"
DEFAULT_INPUT = "btc_data_15m_mstate.parquet"

DEFAULT_FEATURES = [
    "weighted_norm_avg_16h_24h_48h",
    "weighted_deriv_24h_48h_7d",
    "norm_combined_avg",
]
DEFAULT_WINDOWS = [16, 32, 96, 192]

STATE_COLORS = {
    0: "lightgray",       # NEUTRAL
    1: "darkred",         # NEGATIVE_ACCELERATING
    2: "lightcoral",      # NEGATIVE_DECELERATING
    3: "black",           # CROSSING_DOWN
    4: "lightgreen",      # POSITIVE_FLATTENING
    5: "mediumseagreen",  # POSITIVE_DECELERATING
    6: "darkgreen",       # POSITIVE_ACCELERATING
    7: "gold",            # DERIVATIVE_TROUGH
    8: "darkorange",      # DERIVATIVE_PEAK
}


# ---------------------------------------------------------------------------
# 1) State distribution stacked bar
# ---------------------------------------------------------------------------
def plot_state_distribution(
    df: pd.DataFrame,
    features: List[str],
    windows: List[int],
    out_path: str,
) -> None:
    rows = []
    labels = []
    for feat in features:
        for w in windows:
            col = f"{feat}_mstate_{w}"
            counts = df[col].value_counts(normalize=True).reindex(range(9), fill_value=0)
            rows.append(counts.values)
            labels.append(f"{feat[:24]}\n(w={w})")
    pct = np.array(rows) * 100  # shape (n_pairs, 9)

    fig, ax = plt.subplots(figsize=(11, max(4, 0.45 * len(labels))))
    left = np.zeros(pct.shape[0])
    for s in range(9):
        ax.barh(
            range(pct.shape[0]),
            pct[:, s],
            left=left,
            color=STATE_COLORS[s],
            label=STATE_NAMES[s],
            edgecolor="white",
            linewidth=0.5,
        )
        left += pct[:, s]

    ax.set_yticks(range(pct.shape[0]))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("share of bars (%)")
    ax.set_title("Momentum state distribution per (feature, window)")
    ax.legend(loc="lower right", fontsize=7, ncols=2, framealpha=0.95)
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# 2) Four-panel diagnostic over an arbitrary slice
# ---------------------------------------------------------------------------
def plot_four_panel(
    df: pd.DataFrame,
    feat: str,
    window: int,
    out_path: str,
    title_suffix: str = "",
) -> None:
    d1_col = f"{feat}_d1_{window}"
    d2_col = f"{feat}_d2_{window}"
    mstate_col = f"{feat}_mstate_{window}"
    warn_cross_col = f"{feat}_pre_cross_warning_{window}"
    warn_trough_col = f"{feat}_pre_trough_warning_{window}"

    cmap = ListedColormap([STATE_COLORS[k] for k in range(9)])
    norm = BoundaryNorm(np.arange(-0.5, 9.5, 1), cmap.N)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df.index, df[feat], color="steelblue", linewidth=0.8)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_title(f"{feat}  |  window={window} bars  {title_suffix}")
    axes[0].set_ylabel("value")

    axes[1].plot(df.index, df[d1_col], color="orange", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel(f"d1 ({window}b)")

    axes[2].plot(df.index, df[d2_col], color="purple", linewidth=0.8)
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].set_ylabel(f"d2 ({window}b)")

    states = df[mstate_col].values
    axes[3].scatter(df.index, states, c=states, cmap=cmap, norm=norm, s=6)
    axes[3].set_yticks(list(STATE_NAMES.keys()))
    axes[3].set_yticklabels([STATE_NAMES[k] for k in sorted(STATE_NAMES.keys())], fontsize=7)
    axes[3].set_ylabel("mstate")
    axes[3].set_xlabel("time")

    cross_warn_times = df.index[df[warn_cross_col].values]
    trough_warn_times = df.index[df[warn_trough_col].values]
    for ax in axes:
        for t in cross_warn_times:
            ax.axvline(t, color="crimson", alpha=0.10, linewidth=0.6, linestyle="--")
        for t in trough_warn_times:
            ax.axvline(t, color="goldenrod", alpha=0.18, linewidth=0.6, linestyle=":")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# 3) Price + colored regime ribbon (single plot showing one feature only)
# ---------------------------------------------------------------------------
def plot_price_with_state_ribbon(
    df: pd.DataFrame,
    feat: str,
    window: int,
    out_path: str,
    title_suffix: str = "",
) -> None:
    """
    A simpler narrative plot: BTC price on top with a thin colored ribbon
    underneath that encodes the momentum state of one (feature, window).

    Easier to scan than the four-panel plot when looking for "did the
    encoder fire correctly during this episode?".
    """
    if "price" not in df.columns:
        # Encoder doesn't require price, so this plot is best-effort.
        return
    mstate_col = f"{feat}_mstate_{window}"
    cmap = ListedColormap([STATE_COLORS[k] for k in range(9)])
    norm = BoundaryNorm(np.arange(-0.5, 9.5, 1), cmap.N)

    fig, (ax_price, ax_band) = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    ax_price.plot(df.index, df["price"], color="black", linewidth=0.9)
    ax_price.set_ylabel("BTC price (USD)")
    ax_price.set_title(
        f"BTC price + momentum state ribbon for {feat}  (w={window})  {title_suffix}"
    )

    states = df[mstate_col].values.reshape(1, -1)
    ax_band.imshow(
        states,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        extent=[
            matplotlib.dates.date2num(df.index[0].to_pydatetime()),
            matplotlib.dates.date2num(df.index[-1].to_pydatetime()),
            0, 1,
        ],
        interpolation="nearest",
    )
    ax_band.set_yticks([])
    ax_band.set_xlabel("time")
    ax_band.xaxis_date()

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=STATE_COLORS[k]) for k in range(9)
    ]
    ax_price.legend(
        legend_handles,
        [STATE_NAMES[k] for k in range(9)],
        loc="upper left",
        ncols=3,
        fontsize=7,
        framealpha=0.92,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate momentum-state visualization PNGs")
    p.add_argument("--input", default=DEFAULT_INPUT,
                   help=f"Input parquet (default: {DEFAULT_INPUT})")
    p.add_argument("--features", nargs="+", default=DEFAULT_FEATURES,
                   help="Feature columns to visualize")
    p.add_argument("--windows", nargs="+", type=int, default=DEFAULT_WINDOWS,
                   help="Windows to visualize")
    p.add_argument("--slice", nargs=2, metavar=("START", "END"),
                   default=["2025-03-18", "2025-03-20"],
                   help="Date range for the zoomed plot (inclusive). "
                        "Pick a memorable event to manually verify.")
    p.add_argument("--full-window", type=int, default=96,
                   help="Window used for the full-dataset 4-panel + ribbon plots")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} rows from {args.input}")
    print(f"Range: {df.index[0]}  ->  {df.index[-1]}")
    print(f"Writing plots to ./{PLOTS_DIR}/")

    # 1) state distribution
    plot_state_distribution(
        df, args.features, args.windows,
        out_path=os.path.join(PLOTS_DIR, "state_distribution.png"),
    )

    # 2) full-dataset 4-panel + price ribbon for each feature at full-window
    for feat in args.features:
        plot_four_panel(
            df, feat, args.full_window,
            out_path=os.path.join(PLOTS_DIR, f"{feat}_w{args.full_window}_full.png"),
            title_suffix="(entire dataset)",
        )
        plot_price_with_state_ribbon(
            df, feat, args.full_window,
            out_path=os.path.join(PLOTS_DIR, f"{feat}_w{args.full_window}_ribbon.png"),
            title_suffix="(entire dataset)",
        )

    # 3) zoomed slice — the user-supplied date range
    if args.slice:
        start, end = args.slice
        sliced = df.loc[start:end]
        if sliced.empty:
            print(f"WARNING: slice {start} -> {end} returned 0 rows; skipping zoom.")
        else:
            slice_tag = f"{start}_to_{end}".replace(":", "")
            print(f"Zoom slice: {sliced.index[0]} -> {sliced.index[-1]} "
                  f"({len(sliced):,} bars)")
            for feat in args.features:
                plot_four_panel(
                    sliced, feat, args.full_window,
                    out_path=os.path.join(PLOTS_DIR, f"{feat}_w{args.full_window}_{slice_tag}.png"),
                    title_suffix=f"(slice: {start} -> {end})",
                )
                plot_price_with_state_ribbon(
                    sliced, feat, args.full_window,
                    out_path=os.path.join(PLOTS_DIR, f"{feat}_w{args.full_window}_{slice_tag}_ribbon.png"),
                    title_suffix=f"(slice: {start} -> {end})",
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
