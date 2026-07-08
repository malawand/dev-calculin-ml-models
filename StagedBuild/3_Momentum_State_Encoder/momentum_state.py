#!/usr/bin/env python3
"""
Momentum State Encoder for BTC market regime detection (Task 1.3).

Converts any continuous feature into a compact, causal, 9-class momentum
state. The vocabulary is biased toward *leading* signals — events on the
first derivative are top priority because by the time the feature itself
crosses zero, the move has already happened.

State codes (integer values used everywhere):

    8 = DERIVATIVE_PEAK         f' changed pos->neg in last 3 bars   (sell event)
    7 = DERIVATIVE_TROUGH       f' changed neg->pos in last 3 bars   (buy event)
    6 = POSITIVE_ACCELERATING   f > threshold  AND  f' > 0  AND  f'' > 0
    5 = POSITIVE_DECELERATING   f > threshold  AND  f' > 0  AND  f'' < 0   (warning)
    4 = POSITIVE_FLATTENING     f > 0          AND  |f'| < epsilon         (pre-cross)
    3 = CROSSING_DOWN           f changed pos->neg in last 3 bars   (lagging confirm)
    2 = NEGATIVE_DECELERATING   f < -threshold AND  f' < 0  AND  f'' > 0   (pre-trough)
    1 = NEGATIVE_ACCELERATING   f < -threshold AND  f' < 0  AND  f'' < 0
    0 = NEUTRAL                 |f| < threshold (fallback)

Priority (highest wins):
    DERIVATIVE_TROUGH (7)  ===  primary buy signal
    DERIVATIVE_PEAK   (8)  ===  primary sell signal
    CROSSING_DOWN     (3)  ===  late confirm
    POSITIVE_ACCELERATING > POSITIVE_DECELERATING > POSITIVE_FLATTENING
    NEGATIVE_ACCELERATING > NEGATIVE_DECELERATING
    NEUTRAL (fallback)

Why DERIVATIVE_TROUGH/PEAK are top priority:
    Watching f's sign tells you a regime AFTER it has already begun.
    Watching f' flip its sign tells you a regime is STARTING — the
    feature has stopped getting worse and started getting better
    (or vice versa). That's the actionable moment.

All computation is strictly backward-looking. No shift(-N), no future-aware
operations.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Public state vocabulary
# -----------------------------------------------------------------------------
NEUTRAL = 0
NEGATIVE_ACCELERATING = 1
NEGATIVE_DECELERATING = 2
CROSSING_DOWN = 3
POSITIVE_FLATTENING = 4
POSITIVE_DECELERATING = 5
POSITIVE_ACCELERATING = 6
DERIVATIVE_TROUGH = 7
DERIVATIVE_PEAK = 8

STATE_NAMES: Dict[int, str] = {
    0: "NEUTRAL",
    1: "NEGATIVE_ACCELERATING",
    2: "NEGATIVE_DECELERATING",
    3: "CROSSING_DOWN",
    4: "POSITIVE_FLATTENING",
    5: "POSITIVE_DECELERATING",
    6: "POSITIVE_ACCELERATING",
    7: "DERIVATIVE_TROUGH",
    8: "DERIVATIVE_PEAK",
}

DEFAULT_WINDOWS: List[int] = [16, 32, 96, 192]  # 4h, 8h, 24h, 48h at 15m bars

CROSSING_LOOKBACK_BARS: int = 3  # detect sign change within last 3 bars (t-2, t-1, t)

THRESHOLD_PCTL: float = 30.0  # per-feature threshold = 30th pct of |f| on train
EPSILON_PCTL: float = 10.0    # per-(feat, window) epsilon = 10th pct of |f'| on train
TRAIN_FRACTION: float = 0.70  # first 70% of data (chronological) is the "training" split

# ---------------------------------------------------------------------------
# Fast-inflection window
# ---------------------------------------------------------------------------
# DERIVATIVE_TROUGH / DERIVATIVE_PEAK / pre_trough_warning are detected on a
# SHORT-WINDOW first derivative (d1_fast) instead of the regime window's d1.
# Reason: a windowed slope (f.diff(W)/W) only crosses zero ~W/2 bars AFTER
# the actual local min/max of f. A short window keeps the trough event
# pinned near the actual turning point at the cost of slightly more noise.
#
# inflection_window = clamp(w // INFLECTION_WINDOW_RATIO_DENOM, MIN, MAX)
# defaults give:
#   w=16  -> K=2   (~1 bar lag)
#   w=32  -> K=4   (~2 bar lag)
#   w=96  -> K=12  (~6 bar lag)
#   w=192 -> K=24  (~12 bar lag)
# Steady-state classifications (POS_ACC, POS_DEC, POS_FLAT, NEG_*) keep
# using the full-window d1 — they want long-window smoothing.
INFLECTION_WINDOW_RATIO_DENOM: int = 8
MIN_INFLECTION_WINDOW: int = 2
MAX_INFLECTION_WINDOW: int = 24


def _inflection_window_for(w: int) -> int:
    """Return the small window K used for fast d1 at regime window w."""
    return max(MIN_INFLECTION_WINDOW,
               min(MAX_INFLECTION_WINDOW, w // INFLECTION_WINDOW_RATIO_DENOM))


# -----------------------------------------------------------------------------
# Encoder
# -----------------------------------------------------------------------------
class MomentumStateEncoder:
    """
    Fit-then-transform encoder that turns feature columns into
    causal 7-class momentum states at multiple time windows.

    Parameters
    ----------
    feature_cols : sequence of column names to encode.
    windows : list of lookback windows in bars. Defaults to [16, 32, 96, 192]
              which correspond to 4h, 8h, 24h, 48h at 15-minute bars.
    threshold_pctl : percentile of |feature| used to pick the dead-zone threshold
                     on the training split.
    epsilon_pctl : percentile of |first-derivative| used to pick the flatness
                   tolerance on the training split (per window).
    train_fraction : chronological fraction of input used to estimate
                     thresholds/epsilons. The remaining tail is held out and
                     never influences the dead-zones.
    """

    def __init__(
        self,
        feature_cols: Sequence[str],
        windows: Sequence[int] = DEFAULT_WINDOWS,
        threshold_pctl: float = THRESHOLD_PCTL,
        epsilon_pctl: float = EPSILON_PCTL,
        train_fraction: float = TRAIN_FRACTION,
    ) -> None:
        self.feature_cols: List[str] = list(feature_cols)
        self.windows: List[int] = list(windows)
        self.threshold_pctl = float(threshold_pctl)
        self.epsilon_pctl = float(epsilon_pctl)
        self.train_fraction = float(train_fraction)

        self.thresholds: Dict[str, float] = {}
        self.epsilons: Dict[str, Dict[int, float]] = {}
        # epsilons_fast[feat][w] = 10th pct of |d1_fast| over training,
        # where d1_fast uses the small inflection window K for window w.
        self.epsilons_fast: Dict[str, Dict[int, float]] = {}
        self._fitted: bool = False

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "MomentumStateEncoder":
        """
        Estimate per-feature thresholds and per-(feature, window) epsilons
        using only the first ``train_fraction`` of the data chronologically.
        """
        if not 0 < self.train_fraction <= 1:
            raise ValueError("train_fraction must be in (0, 1].")

        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"feature_cols missing from df: {missing}")

        n = len(df)
        cutoff = max(1, int(round(n * self.train_fraction)))
        train = df.iloc[:cutoff]

        self.thresholds = {}
        self.epsilons = {}
        self.epsilons_fast = {}
        for feat in self.feature_cols:
            series = train[feat].astype(float)
            abs_series = series.abs().dropna()
            if abs_series.empty:
                raise ValueError(f"Training slice is empty for feature '{feat}'.")
            self.thresholds[feat] = float(np.percentile(abs_series.values, self.threshold_pctl))

            self.epsilons[feat] = {}
            self.epsilons_fast[feat] = {}
            for w in self.windows:
                d1 = series.diff(w) / w
                abs_d1 = d1.abs().dropna()
                if abs_d1.empty:
                    self.epsilons[feat][w] = 0.0
                else:
                    self.epsilons[feat][w] = float(np.percentile(abs_d1.values, self.epsilon_pctl))

                k = _inflection_window_for(w)
                d1_fast = series.diff(k) / k
                abs_d1_fast = d1_fast.abs().dropna()
                if abs_d1_fast.empty:
                    self.epsilons_fast[feat][w] = 0.0
                else:
                    self.epsilons_fast[feat][w] = float(
                        np.percentile(abs_d1_fast.values, self.epsilon_pctl)
                    )

        self._fitted = True
        return self

    # -------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a new DataFrame with the derivative, state, duration, and
        pre-cross warning columns appended. Input is never mutated.
        """
        if not self._fitted:
            raise RuntimeError("MomentumStateEncoder.fit() must be called before transform().")

        out = df.copy()
        for feat in self.feature_cols:
            series = out[feat].astype(float)
            thr = self.thresholds[feat]
            for w in self.windows:
                k = _inflection_window_for(w)
                d1 = series.diff(w) / w
                d2 = d1.diff(w) / w
                d1_fast = series.diff(k) / k
                eps = self.epsilons[feat][w]
                eps_fast = self.epsilons_fast[feat][w]

                state = _classify_state(
                    series, d1, d2, thr, eps,
                    d1_fast=d1_fast,
                )

                duration = _run_length(state)
                pre_cross = _pre_cross_warning(series, state, thr)
                pre_trough = _pre_trough_warning(d1_fast, eps_fast)

                out[f"{feat}_d1_{w}"] = d1
                out[f"{feat}_d2_{w}"] = d2
                out[f"{feat}_d1_fast_{w}"] = d1_fast
                out[f"{feat}_mstate_{w}"] = state.astype(np.int8)
                out[f"{feat}_mstate_duration_{w}"] = duration.astype(np.int32)
                out[f"{feat}_pre_cross_warning_{w}"] = pre_cross.astype(bool)
                out[f"{feat}_pre_trough_warning_{w}"] = pre_trough.astype(bool)

        return out

    # ------------------------------------------------------------- fit_transform
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def state_name(state_int: int) -> str:
        """Map the integer state code to its human-readable name."""
        if state_int not in STATE_NAMES:
            raise KeyError(f"Unknown state code: {state_int}")
        return STATE_NAMES[state_int]


# -----------------------------------------------------------------------------
# Core classification helpers (module-level so tests can hit them directly)
# -----------------------------------------------------------------------------
def _classify_state(
    series: pd.Series,
    d1: pd.Series,
    d2: pd.Series,
    threshold: float,
    epsilon: float,
    d1_fast: pd.Series | None = None,
) -> pd.Series:
    """
    Classify each bar into one of the 9 momentum states.

    Steady-state classifications use the regime-window first/second
    derivatives ``d1`` and ``d2``:
        POSITIVE_ACCELERATING (6) > POSITIVE_DECELERATING (5) > POSITIVE_FLATTENING (4)
        NEGATIVE_ACCELERATING (1) > NEGATIVE_DECELERATING (2)
        NEUTRAL (0) fallback.

    Event states use the FAST first derivative ``d1_fast`` so the
    inflection event fires near the actual local extremum of f rather
    than ~W/2 bars later. ``CROSSING_DOWN`` still fires on f's own sign
    flip (it's the late confirm). When ``d1_fast`` is None it falls
    back to ``d1`` (backward compatible).

        DERIVATIVE_TROUGH (7)  ===  primary buy event   (top priority)
        DERIVATIVE_PEAK   (8)  ===  primary sell event (top priority)
        CROSSING_DOWN     (3)  ===  late confirm
    """
    idx = series.index
    f = series.values.astype(float)
    d1_v = d1.values.astype(float)
    d2_v = d2.values.astype(float)
    d1_fast_v = d1_fast.values.astype(float) if d1_fast is not None else d1_v

    valid_f = ~np.isnan(f)
    valid_d1 = ~np.isnan(d1_v)
    valid_d2 = ~np.isnan(d2_v)

    state = np.full(len(f), NEUTRAL, dtype=np.int8)

    # Apply in priority order from LOW to HIGH so that later writes overwrite
    # earlier ones and the highest priority wins.

    #  POSITIVE_FLATTENING (4): f > 0 AND |d1| < eps
    mask_pos_flat = valid_f & valid_d1 & (f > 0) & (np.abs(d1_v) < epsilon)
    state[mask_pos_flat] = POSITIVE_FLATTENING

    #  NEGATIVE_DECELERATING (2): f < -thr AND d1 < 0 AND d2 > 0
    mask_neg_dec = valid_f & valid_d1 & valid_d2 & (f < -threshold) & (d1_v < 0) & (d2_v > 0)
    state[mask_neg_dec] = NEGATIVE_DECELERATING

    #  POSITIVE_DECELERATING (5): f > thr AND d1 > 0 AND d2 < 0
    mask_pos_dec = valid_f & valid_d1 & valid_d2 & (f > threshold) & (d1_v > 0) & (d2_v < 0)
    state[mask_pos_dec] = POSITIVE_DECELERATING

    #  NEGATIVE_ACCELERATING (1): f < -thr AND d1 < 0 AND d2 < 0
    mask_neg_acc = valid_f & valid_d1 & valid_d2 & (f < -threshold) & (d1_v < 0) & (d2_v < 0)
    state[mask_neg_acc] = NEGATIVE_ACCELERATING

    #  POSITIVE_ACCELERATING (6): f > thr AND d1 > 0 AND d2 > 0
    mask_pos_acc = valid_f & valid_d1 & valid_d2 & (f > threshold) & (d1_v > 0) & (d2_v > 0)
    state[mask_pos_acc] = POSITIVE_ACCELERATING

    #  CROSSING_DOWN (3) — old-style sign flip on f itself. Late confirm,
    #  but kept because downstream models may still want it.
    cross = _crossing_mask(f, direction="down")
    state[cross] = CROSSING_DOWN

    #  DERIVATIVE_PEAK (8) — d1_FAST went pos -> neg in last CROSSING_LOOKBACK_BARS.
    #  Uses the small-window fast derivative so the event fires near the
    #  actual local maximum of f, not W/2 bars later.
    peak = _crossing_mask(d1_fast_v, direction="down")
    state[peak] = DERIVATIVE_PEAK

    #  DERIVATIVE_TROUGH (7) — d1_FAST went neg -> pos in last CROSSING_LOOKBACK_BARS.
    #  Highest priority of all states.
    trough = _crossing_mask(d1_fast_v, direction="up")
    state[trough] = DERIVATIVE_TROUGH

    # Bars where features couldn't be evaluated at all remain NEUTRAL with
    # nothing overridden; callers can drop the first max(windows) rows if
    # they want "pure" output.

    return pd.Series(state, index=idx, dtype=np.int8)


def _crossing_mask(arr: np.ndarray, direction: str = "down") -> np.ndarray:
    """
    Boolean array marking bars where the input array changed sign within
    the last CROSSING_LOOKBACK_BARS bars (inclusive of the current bar).

    direction:
        "down" => positive -> non-positive flip
        "up"   => negative -> non-negative flip

    Strictly causal — no future values touched.
    """
    n = len(arr)
    out = np.zeros(n, dtype=bool)
    if n < 2:
        return out

    arr = np.asarray(arr, dtype=float)
    prev = arr[:-1]
    curr = arr[1:]
    with np.errstate(invalid="ignore"):
        if direction == "down":
            flip = (prev > 0) & (curr <= 0)
        elif direction == "up":
            flip = (prev < 0) & (curr >= 0)
        else:
            raise ValueError(f"direction must be 'down' or 'up', got {direction!r}")

    # `flip[i]` is the crossing between bar i and bar i+1.
    # For bar t, we look at transitions ending in {t-2, t-1, t}, i.e. flips
    # indexed by {t-1-k for k in 0..lookback-1}. Mark bar t True if any apply.
    lookback = CROSSING_LOOKBACK_BARS - 1  # number of transitions to inspect
    for k in range(lookback):
        shifted = np.zeros(n, dtype=bool)
        end_src = n - 1 - k
        start_dst = k + 1
        if end_src > 0:
            shifted[start_dst:n] = flip[:end_src]
        out = out | shifted

    return out


def _crossing_down_mask(f: np.ndarray) -> np.ndarray:
    """Backward-compatible alias — kept so older imports keep working."""
    return _crossing_mask(f, direction="down")


def _run_length(state: pd.Series) -> pd.Series:
    """How many consecutive bars the state has held the current value so far.

    Resets to 1 whenever the state changes. Entirely backward-looking.
    """
    vals = state.values
    n = len(vals)
    out = np.ones(n, dtype=np.int32)
    for i in range(1, n):
        out[i] = out[i - 1] + 1 if vals[i] == vals[i - 1] else 1
    return pd.Series(out, index=state.index, dtype=np.int32)


def _pre_cross_warning(
    series: pd.Series,
    state: pd.Series,
    threshold: float,
) -> pd.Series:
    """
    Pre-cross warning: fires when a feature is above zero, is decelerating
    or flattening, AND is already close enough to zero to matter.

    This catches the "this rally is fading" moment one to several bars
    before f actually crosses zero downward.
    """
    f = series.values.astype(float)
    s = state.values
    valid = ~np.isnan(f)
    dec_or_flat = (s == POSITIVE_DECELERATING) | (s == POSITIVE_FLATTENING)
    above_zero = f > 0
    near_zero = f < 2 * threshold
    warn = valid & dec_or_flat & above_zero & near_zero
    return pd.Series(warn, index=series.index, dtype=bool)


def _pre_trough_warning(
    d1_fast: pd.Series,
    epsilon_fast: float,
) -> pd.Series:
    """
    Pre-trough warning: fires when the FAST first derivative is negative
    but already close to zero, i.e. the decline is running out of steam
    and a DERIVATIVE_TROUGH event is imminent.

    Operates on the small-window ``d1_fast`` (rather than the windowed
    d1) so the warning has an actual lead time over the trough event.

    Symmetric in spirit to ``_pre_cross_warning``, but on f' instead of f.
    """
    d = d1_fast.values.astype(float)
    valid = ~np.isnan(d)
    below_zero = d < 0
    near_zero = d > -2 * epsilon_fast
    warn = valid & below_zero & near_zero
    return pd.Series(warn, index=d1_fast.index, dtype=bool)


# -----------------------------------------------------------------------------
# Visualization (optional — matplotlib only imported when called)
# -----------------------------------------------------------------------------
def quick_visualize(
    df: pd.DataFrame,
    feat: str,
    window: int,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Four-panel diagnostic plot for a single (feature, window) pair:
      1) raw feature value with zero line
      2) first derivative
      3) second derivative
      4) momentum state as a color-coded step plot

    Vertical dashed lines are drawn wherever ``<feat>_pre_cross_warning_<window>``
    is True. Use this on your real indicator data to sanity-check that the
    encoder's states line up with what you see visually.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    d1_col = f"{feat}_d1_{window}"
    d2_col = f"{feat}_d2_{window}"
    mstate_col = f"{feat}_mstate_{window}"
    warn_cross_col = f"{feat}_pre_cross_warning_{window}"
    warn_trough_col = f"{feat}_pre_trough_warning_{window}"

    for col in (feat, d1_col, d2_col, mstate_col, warn_cross_col, warn_trough_col):
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' in DataFrame — run transform() first.")

    state_colors = [
        "lightgray",       # 0 NEUTRAL
        "darkred",         # 1 NEGATIVE_ACCELERATING
        "lightcoral",      # 2 NEGATIVE_DECELERATING
        "black",           # 3 CROSSING_DOWN
        "lightgreen",      # 4 POSITIVE_FLATTENING
        "mediumseagreen",  # 5 POSITIVE_DECELERATING
        "darkgreen",       # 6 POSITIVE_ACCELERATING
        "gold",            # 7 DERIVATIVE_TROUGH (key buy signal)
        "darkorange",      # 8 DERIVATIVE_PEAK   (key sell signal)
    ]
    cmap = ListedColormap(state_colors)
    norm = BoundaryNorm(np.arange(-0.5, len(state_colors) + 0.5, 1), cmap.N)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    ax0 = axes[0]
    ax0.plot(df.index, df[feat], color="steelblue", linewidth=0.9)
    ax0.axhline(0, color="black", linewidth=0.5)
    ax0.set_title(f"{feat}  |  window={window} bars")
    ax0.set_ylabel("value")

    axes[1].plot(df.index, df[d1_col], color="orange", linewidth=0.9)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel(f"d1 ({window}b)")

    axes[2].plot(df.index, df[d2_col], color="purple", linewidth=0.9)
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].set_ylabel(f"d2 ({window}b)")

    ax3 = axes[3]
    states = df[mstate_col].values
    ax3.scatter(df.index, states, c=states, cmap=cmap, norm=norm, s=4)
    ax3.set_yticks(list(STATE_NAMES.keys()))
    ax3.set_yticklabels([STATE_NAMES[k] for k in sorted(STATE_NAMES.keys())], fontsize=8)
    ax3.set_ylabel("mstate")
    ax3.set_xlabel("time")

    cross_warn_times = df.index[df[warn_cross_col].values]
    trough_warn_times = df.index[df[warn_trough_col].values]
    for ax in axes:
        for t in cross_warn_times:
            ax.axvline(t, color="crimson", alpha=0.12, linewidth=0.8, linestyle="--")
        for t in trough_warn_times:
            ax.axvline(t, color="goldenrod", alpha=0.18, linewidth=0.8, linestyle=":")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"Saved plot -> {save_path}")
    if show:
        plt.show()
    return fig, axes


# -----------------------------------------------------------------------------
# CLI: encode the labeled parquet with the three indicator features
# -----------------------------------------------------------------------------
def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Encode momentum states from a Parquet file")
    p.add_argument(
        "input",
        nargs="?",
        default="/Users/mazenlawand/Documents/Calculin ML Models/StagedBuild/2_Build_Labels/btc_data_15m_labeled.parquet",
        help="Path to input Parquet",
    )
    p.add_argument(
        "--output",
        default="btc_data_15m_mstate.parquet",
        help="Path to write transformed parquet",
    )
    p.add_argument(
        "--features",
        nargs="+",
        default=["weighted_norm_avg_16h_24h_48h", "weighted_deriv_24h_48h_7d", "norm_combined_avg"],
        help="Feature columns to encode",
    )
    p.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=DEFAULT_WINDOWS,
        help="Windows in bars (default: 16 32 96 192 -> 4h 8h 24h 48h)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} rows from {args.input}")
    print(f"Encoding features: {args.features}")
    print(f"Windows (bars):    {args.windows}")

    enc = MomentumStateEncoder(feature_cols=args.features, windows=args.windows)
    out = enc.fit_transform(df)

    print("\nLearned thresholds (30th pct of |f| on first 70%):")
    for feat, thr in enc.thresholds.items():
        print(f"  {feat:<40s}: {thr:.5f}")
    print("\nLearned epsilons (windowed d1) per (feature, window):")
    for feat in args.features:
        for w in args.windows:
            print(f"  {feat:<40s} w={w:>4d}: {enc.epsilons[feat][w]:.6f}")

    print("\nLearned epsilons_fast (fast d1, K = inflection_window) per (feature, window):")
    for feat in args.features:
        for w in args.windows:
            k = _inflection_window_for(w)
            print(f"  {feat:<40s} w={w:>4d} K={k:>3d}: {enc.epsilons_fast[feat][w]:.6f}")

    print("\nState distribution per (feature, window):")
    for feat in args.features:
        for w in args.windows:
            col = f"{feat}_mstate_{w}"
            counts = out[col].value_counts().sort_index()
            total = counts.sum()
            pretty = ", ".join(f"{STATE_NAMES[k]}={v/total*100:.1f}%" for k, v in counts.items())
            print(f"  {col}: {pretty}")

    out.to_parquet(args.output, engine="pyarrow")
    print(f"\nSaved {len(out):,} rows -> {args.output}")


if __name__ == "__main__":
    main()
