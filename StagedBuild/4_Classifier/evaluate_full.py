#!/usr/bin/env python3
"""
Full-dataset evaluation for the Stage 4 detector.

Runs the trained model over the ENTIRE Stage 3 parquet (train + val +
test), then reports detection metrics per split.

IMPORTANT — bias warning
------------------------
The model has SEEN the train split during training and used the val
split for early stopping. Per-split metrics on those slices are
optimistic and only useful for sanity checks (e.g. "does train accuracy
look way too perfect, hinting at leakage?"). The TEST split is the only
honest, unbiased number.

For unbiased performance across the entire timeline, use walk-forward
retraining (separate script, not yet implemented).

Outputs:
  Stdout: per-split metric blocks + an overall block with caveats.
  plots/timeline_full.png        : true vs predicted regime over time + price.
  plots/confusion_full_<split>.png for each split.
  full_predictions.parquet       : per-bar probabilities + truth for every bar.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from data import (
    DEFAULT_INPUT_PARQUET,
    DEFAULT_TRAIN_FRAC,
    DEFAULT_VAL_FRAC,
    NUM_REGIMES,
    REGIME_INT_TO_NAME,
    select_feature_columns,
)

ARTIFACTS_DIR = Path(__file__).parent / "model_artifacts"
PLOTS_DIR = Path(__file__).parent / "plots"
FULL_PRED_PATH = Path(__file__).parent / "full_predictions.parquet"

REGIME_COLORS = {
    0: "lightgray",  # CHOP
    1: "#2ca02c",    # TRENDING_UP
    2: "#d62728",    # TRENDING_DOWN
    3: "#ff7f0e",    # VOLATILE_EXPANSION
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-dataset evaluation of the Stage 4 detector.")
    p.add_argument("--data", type=str, default=str(DEFAULT_INPUT_PARQUET))
    p.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    p.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    p.add_argument("--transition-tolerance", type=int, default=4)
    return p.parse_args()


def load_full_dataset(parquet_path: str, train_frac: float, val_frac: float) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path).sort_index()
    feature_cols = select_feature_columns(df)
    before = len(df)
    df = df.dropna(subset=feature_cols + ["regime"])
    after = len(df)
    if before != after:
        print(f"Dropped {before - after:,} warm-up rows with NaN features.")
    df["regime"] = df["regime"].astype(int)

    n = len(df)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    df["split"] = "test"
    df.iloc[:n_train, df.columns.get_loc("split")] = "train"
    df.iloc[n_train:n_train + n_val, df.columns.get_loc("split")] = "val"
    return df


def score_block(name: str, y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, args: argparse.Namespace) -> dict:
    print("=" * 76)
    print(f"Split: {name.upper()}  (rows={len(y_true):,})")
    print("=" * 76)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    ll = log_loss(y_true, proba, labels=list(range(NUM_REGIMES)))
    print(f"accuracy    : {acc*100:5.1f}%")
    print(f"macro F1    : {macro_f1:.4f}")
    print(f"logloss     : {ll:.4f}")

    target_names = [REGIME_INT_TO_NAME[i] for i in range(NUM_REGIMES)]
    print()
    print(classification_report(
        y_true, y_pred,
        labels=list(range(NUM_REGIMES)),
        target_names=target_names,
        digits=3, zero_division=0,
    ))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_REGIMES)))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    # Plot confusion matrix.
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_REGIMES))
    ax.set_yticks(range(NUM_REGIMES))
    ax.set_xticklabels(target_names, rotation=30, ha="right")
    ax.set_yticklabels(target_names)
    ax.set_xlabel("Predicted regime")
    ax.set_ylabel("True regime")
    ax.set_title(f"Confusion matrix ({name}) — row-normalized")
    for i in range(NUM_REGIMES):
        for j in range(NUM_REGIMES):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]*100:.1f}%\n({cm[i, j]:,})",
                    ha="center", va="center", fontsize=8, color=color)
    fig.tight_layout()
    p = PLOTS_DIR / f"confusion_full_{name}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"wrote {p}")

    # Transition accuracy.
    transitions = np.where(np.diff(y_true) != 0)[0] + 1
    tol = args.transition_tolerance
    if len(transitions) > 0:
        caught = 0
        for t in transitions:
            new_regime = y_true[t]
            lo = max(0, t - tol)
            hi = min(len(y_pred), t + tol + 1)
            if (y_pred[lo:hi] == new_regime).any():
                caught += 1
        trans_rate = caught / len(transitions)
        print(f"transitions : {len(transitions):,}  caught(±{tol}): {caught:,}  ({trans_rate*100:.1f}%)")
    else:
        trans_rate = 0.0
        print("transitions : none in this split")

    # Confidence -> accuracy curve.
    confidence = proba.max(axis=1)
    correct = (y_true == y_pred).astype(int)
    bins = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.001]
    out = pd.DataFrame({"conf": confidence, "correct": correct})
    out["bucket"] = pd.cut(out["conf"], bins=bins, include_lowest=True, right=False)
    grp = out.groupby("bucket", observed=True).agg(
        n=("correct", "size"), acc=("correct", "mean"),
    )
    grp["share"] = grp["n"] / grp["n"].sum()
    print(f"\nconfidence -> accuracy:")
    print(f"  {'bucket':<14s} {'n':>9s} {'share':>7s} {'accuracy':>10s}")
    for idx, row in grp.iterrows():
        print(f"  {str(idx):<14s} {int(row['n']):>9,d} {row['share']*100:>6.1f}% {row['acc']*100:>9.1f}%")
    print()

    return {
        "n": int(len(y_true)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "logloss": float(ll),
        "transition_catch_rate": float(trans_rate),
    }


def plot_timeline(df: pd.DataFrame, y_pred: np.ndarray, args: argparse.Namespace) -> None:
    """Draw price + true regime ribbon + predicted regime ribbon over time.

    Uses ``imshow`` for the ribbons rather than overlapping ``fill_between``s
    so that 100k+ bars render correctly without color-stacking artifacts.
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1]})

    cmap = ListedColormap([REGIME_COLORS[i] for i in range(NUM_REGIMES)])

    # Top: BTC price.
    ax_price = axes[0]
    ax_price.plot(df.index, df["price"], color="black", linewidth=0.6)
    ax_price.set_ylabel("BTC price")
    ax_price.set_title("Detector predictions across the entire dataset")
    ax_price.grid(alpha=0.3)

    train_end = df[df["split"] == "train"].index.max()
    val_end = df[df["split"] == "val"].index.max()
    for ax in axes:
        ax.axvline(train_end, color="black", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.axvline(val_end, color="black", linestyle="--", linewidth=0.7, alpha=0.6)

    train_mid = df[df["split"] == "train"].index[len(df[df["split"] == "train"]) // 2]
    val_mid = df[df["split"] == "val"].index[len(df[df["split"] == "val"]) // 2]
    test_mid = df[df["split"] == "test"].index[len(df[df["split"] == "test"]) // 2]
    y_top = df["price"].max()
    ax_price.text(train_mid, y_top, "TRAIN (model saw this)", ha="center", va="top",
                  fontsize=9, color="dimgray")
    ax_price.text(val_mid, y_top, "VAL (early-stopping)", ha="center", va="top",
                  fontsize=9, color="dimgray")
    ax_price.text(test_mid, y_top, "TEST (unbiased)", ha="center", va="top",
                  fontsize=9, color="black", weight="bold")

    # Convert datetimes to matplotlib floats for imshow extent.
    x = mdates.date2num(df.index.to_pydatetime())
    extent = (x.min(), x.max(), 0, 1)

    y_true = df["regime"].values.astype(int).reshape(1, -1)
    y_pred_2d = y_pred.astype(int).reshape(1, -1)

    ax_true = axes[1]
    ax_true.imshow(y_true, aspect="auto", cmap=cmap, vmin=0, vmax=NUM_REGIMES - 1,
                   extent=extent, interpolation="nearest")
    ax_true.set_yticks([])
    ax_true.set_ylabel("TRUE\nregime")

    ax_pred = axes[2]
    ax_pred.imshow(y_pred_2d, aspect="auto", cmap=cmap, vmin=0, vmax=NUM_REGIMES - 1,
                   extent=extent, interpolation="nearest")
    ax_pred.set_yticks([])
    ax_pred.set_ylabel("PRED\nregime")

    legend_handles = [Patch(facecolor=REGIME_COLORS[i], label=REGIME_INT_TO_NAME[i])
                      for i in range(NUM_REGIMES)]
    ax_true.legend(handles=legend_handles, loc="upper right", ncol=4,
                   fontsize=8, framealpha=0.95, bbox_to_anchor=(1, 1.6))

    ax_pred.xaxis_date()
    ax_pred.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_pred.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    fig.autofmt_xdate()
    fig.tight_layout()
    p = PLOTS_DIR / "timeline_full.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"wrote {p}")

    # Also a "matches vs misses" lane for quick visual error scan.
    fig2, ax = plt.subplots(figsize=(15, 1.6))
    matches = (df["regime"].values == y_pred).astype(int).reshape(1, -1)
    match_cmap = ListedColormap(["#d62728", "#7eb37e"])  # red=miss, green=match
    ax.imshow(matches, aspect="auto", cmap=match_cmap, vmin=0, vmax=1,
              extent=extent, interpolation="nearest")
    ax.set_yticks([])
    ax.set_title("Predicted regime correctness over time   "
                 "(green=match with hindsight label, red=mismatch)")
    ax.axvline(mdates.date2num(train_end), color="black", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.axvline(mdates.date2num(val_end), color="black", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    fig2.tight_layout()
    p2 = PLOTS_DIR / "timeline_correctness.png"
    fig2.savefig(p2, dpi=130)
    plt.close(fig2)
    print(f"wrote {p2}")


def main() -> None:
    args = parse_args()
    PLOTS_DIR.mkdir(exist_ok=True)

    booster = lgb.Booster(model_file=str(ARTIFACTS_DIR / "model.txt"))
    feature_cols = json.loads((ARTIFACTS_DIR / "feature_cols.json").read_text())

    df = load_full_dataset(args.data, args.train_frac, args.val_frac)
    print(f"\nFull dataset rows : {len(df):,}")
    print(f"Range             : {df.index.min()} -> {df.index.max()}")
    print(f"Split sizes       : train={int((df['split']=='train').sum()):,}  "
          f"val={int((df['split']=='val').sum()):,}  test={int((df['split']=='test').sum()):,}")

    X = df[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(np.int8)

    proba_full = booster.predict(X)
    y_pred_full = proba_full.argmax(axis=1)
    y_true_full = df["regime"].values

    # Save full predictions.
    out = pd.DataFrame(
        proba_full,
        columns=[f"prob_{REGIME_INT_TO_NAME[i]}" for i in range(NUM_REGIMES)],
        index=df.index,
    )
    out["pred_int"] = y_pred_full
    out["pred_name"] = [REGIME_INT_TO_NAME[int(p)] for p in y_pred_full]
    out["confidence"] = proba_full.max(axis=1)
    out["true_int"] = y_true_full
    out["true_name"] = [REGIME_INT_TO_NAME[int(t)] for t in y_true_full]
    out["split"] = df["split"].values
    out.to_parquet(FULL_PRED_PATH)
    print(f"wrote {FULL_PRED_PATH}\n")

    # Per-split metrics.
    metrics_per_split = {}
    for name in ("train", "val", "test"):
        mask = (df["split"] == name).values
        if not mask.any():
            continue
        if name in ("train", "val"):
            print("\n[!] BIAS WARNING — this split was used during fitting; "
                  "treat the numbers as a sanity check, not honest performance.")
        metrics_per_split[name] = score_block(
            name=name,
            y_true=y_true_full[mask],
            y_pred=y_pred_full[mask],
            proba=proba_full[mask],
            args=args,
        )

    # Overall block (with caveat).
    print("\n[!] OVERALL block below mixes biased train/val with unbiased test;")
    print("    do not quote it as 'performance across the dataset' on its own.")
    score_block(
        name="overall_BIASED",
        y_true=y_true_full,
        y_pred=y_pred_full,
        proba=proba_full,
        args=args,
    )

    # Timeline plot.
    df_with_price = df[["price", "regime", "split"]].copy()
    plot_timeline(df_with_price, y_pred_full, args)

    # Headline summary.
    print("\n" + "=" * 76)
    print("HEADLINE per-split summary")
    print("=" * 76)
    print(f"{'split':<8s} {'rows':>8s} {'acc':>7s} {'macroF1':>9s} {'logloss':>9s} {'trans±4':>9s}")
    for name in ("train", "val", "test"):
        m = metrics_per_split.get(name)
        if not m:
            continue
        flag = "" if name == "test" else " (biased)"
        print(f"{name:<8s} {m['n']:>8,d} {m['accuracy']*100:>6.1f}% {m['macro_f1']:>9.4f} "
              f"{m['logloss']:>9.4f} {m['transition_catch_rate']*100:>8.1f}%{flag}")
    print()
    print("Reminder: only TEST is honest. TRAIN and VAL show the model on")
    print("data it has already seen — useful for spotting overfit gaps,")
    print("not for reporting headline performance.")


if __name__ == "__main__":
    main()
