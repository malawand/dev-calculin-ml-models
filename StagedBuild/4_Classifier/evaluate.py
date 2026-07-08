#!/usr/bin/env python3
"""
Stage 4 evaluator — turns the per-row predictions saved by ``train.py``
into the metrics that actually matter for a regime classifier:

  * Confusion matrix + per-class precision / recall / F1.
  * Confidence-vs-accuracy table (does high confidence really mean
    high accuracy?).
  * Transition accuracy — at the bars where the regime CHANGED, did the
    model see the change within +/- N bars?
  * Top feature importances.

Run AFTER train.py:
    python3 evaluate.py
    python3 evaluate.py --split val      # if you want val instead of test

Outputs:
  Stdout: human-readable report
  plots/confusion_matrix_<split>.png
  plots/feature_importance.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from data import NUM_REGIMES, REGIME_INT_TO_NAME

ARTIFACTS_DIR = Path(__file__).parent / "model_artifacts"
PLOTS_DIR = Path(__file__).parent / "plots"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Stage 4 regime classifier.")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--top-k-importances", type=int, default=25)
    p.add_argument("--transition-tolerance", type=int, default=4,
                   help="A transition is 'caught' if predicted within +/- N bars.")
    return p.parse_args()


def fmt_pct(x: float) -> str:
    return f"{x*100:5.1f}%"


def evaluate(split: str, args: argparse.Namespace) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    pred_path = ARTIFACTS_DIR / f"predictions_{split}.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"Run train.py first; missing {pred_path}.")

    pred = pd.read_parquet(pred_path).sort_index()
    y_true = pred["y_true"].astype(int).values
    y_pred = pred["y_pred"].astype(int).values
    proba_cols = [f"prob_{REGIME_INT_TO_NAME[i]}" for i in range(NUM_REGIMES)]
    proba = pred[proba_cols].values

    print("=" * 74)
    print(f"Evaluation on the {split.upper()} split")
    print(f"Range: {pred.index.min()}  ->  {pred.index.max()}")
    print(f"Bars : {len(pred):,}")
    print("=" * 74)

    # ------------------------------------------------------------------
    # Headline
    # ------------------------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\nOverall accuracy : {fmt_pct(acc)}")
    print(f"Macro F1         : {macro_f1:.4f}")
    print(f"  (Macro F1 averages F1 across the 4 classes EQUALLY — so a")
    print(f"   model that nails CHOP and ignores VOLATILE_EXPANSION will")
    print(f"   look bad here even if its overall accuracy is high.)")

    # ------------------------------------------------------------------
    # Per-class precision / recall / F1
    # ------------------------------------------------------------------
    print("\nPer-class report:")
    print("  precision = of the bars I predicted as X, what fraction were really X?")
    print("  recall    = of the bars that were really X, what fraction did I predict as X?")
    target_names = [REGIME_INT_TO_NAME[i] for i in range(NUM_REGIMES)]
    print(classification_report(
        y_true, y_pred,
        labels=list(range(NUM_REGIMES)),
        target_names=target_names,
        digits=3,
        zero_division=0,
    ))

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_REGIMES)))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    print("\nConfusion matrix (rows=true, cols=predicted, normalized by row):")
    header = "true\\pred         " + "  ".join(f"{n[:10]:>10s}" for n in target_names)
    print(header)
    for i, name in enumerate(target_names):
        row = "  ".join(f"{cm_norm[i, j]*100:9.1f}%" for j in range(NUM_REGIMES))
        print(f"{name:<18s}{row}")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_REGIMES))
    ax.set_yticks(range(NUM_REGIMES))
    ax.set_xticklabels(target_names, rotation=30, ha="right")
    ax.set_yticklabels(target_names)
    ax.set_xlabel("Predicted regime")
    ax.set_ylabel("True regime")
    ax.set_title(f"Confusion matrix ({split}) — row-normalized")
    for i in range(NUM_REGIMES):
        for j in range(NUM_REGIMES):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]*100:.1f}%\n({cm[i, j]:,})",
                    ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    cm_path = PLOTS_DIR / f"confusion_matrix_{split}.png"
    fig.savefig(cm_path, dpi=130)
    plt.close(fig)
    print(f"\nWrote {cm_path}")

    # ------------------------------------------------------------------
    # Confidence vs accuracy
    # ------------------------------------------------------------------
    confidence = proba.max(axis=1)
    print("\nConfidence vs accuracy (does high model confidence => correct?):")
    bins = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.001]
    correct = (y_true == y_pred).astype(int)
    out = pd.DataFrame({"conf": confidence, "correct": correct})
    out["bucket"] = pd.cut(out["conf"], bins=bins, include_lowest=True, right=False)
    grp = out.groupby("bucket", observed=True).agg(
        n=("correct", "size"), acc=("correct", "mean"),
    )
    grp["share"] = grp["n"] / grp["n"].sum()
    print(f"  {'bucket':<14s} {'n':>9s} {'share':>7s} {'accuracy':>10s}")
    for idx, row in grp.iterrows():
        print(f"  {str(idx):<14s} {int(row['n']):>9,d} {row['share']*100:>6.1f}% {fmt_pct(row['acc']):>10s}")

    # ------------------------------------------------------------------
    # Transition accuracy
    # ------------------------------------------------------------------
    print(f"\nTransition accuracy (tolerance: +/- {args.transition_tolerance} bars):")
    print("  At each bar where the TRUE regime changed, did the model")
    print("  predict the NEW regime within +/- N bars?")
    y_true_arr = y_true
    y_pred_arr = y_pred
    transitions = np.where(np.diff(y_true_arr) != 0)[0] + 1
    tol = args.transition_tolerance
    if len(transitions) == 0:
        print("  No regime transitions in this split.")
    else:
        caught = 0
        for t in transitions:
            new_regime = y_true_arr[t]
            lo = max(0, t - tol)
            hi = min(len(y_pred_arr), t + tol + 1)
            if (y_pred_arr[lo:hi] == new_regime).any():
                caught += 1
        print(f"  Transitions in split    : {len(transitions):,}")
        print(f"  Transitions caught (±{tol}): {caught:,}  ({fmt_pct(caught / len(transitions))})")

    # Per-regime transition catch
    per_regime = {i: [0, 0] for i in range(NUM_REGIMES)}  # [caught, total]
    for t in transitions:
        new_regime = int(y_true_arr[t])
        lo = max(0, t - tol)
        hi = min(len(y_pred_arr), t + tol + 1)
        per_regime[new_regime][1] += 1
        if (y_pred_arr[lo:hi] == new_regime).any():
            per_regime[new_regime][0] += 1
    print("  By new-regime class:")
    for r in range(NUM_REGIMES):
        c, n = per_regime[r]
        if n == 0:
            continue
        print(f"    -> {REGIME_INT_TO_NAME[r]:<22s} {c:>5,}/{n:<5,}  {fmt_pct(c / n)}")

    # ------------------------------------------------------------------
    # Feature importance (uses the saved model)
    # ------------------------------------------------------------------
    model_path = ARTIFACTS_DIR / "model.txt"
    feat_path = ARTIFACTS_DIR / "feature_cols.json"
    if model_path.exists() and feat_path.exists():
        booster = lgb.Booster(model_file=str(model_path))
        feat_cols = json.loads(feat_path.read_text())
        imp_gain = booster.feature_importance(importance_type="gain")
        imp = pd.DataFrame({"feature": feat_cols, "gain": imp_gain})
        imp = imp.sort_values("gain", ascending=False).head(args.top_k_importances)

        print(f"\nTop {len(imp)} features by total gain:")
        for _, row in imp.iterrows():
            print(f"  {row['feature']:<55s} {row['gain']:>14,.0f}")

        fig, ax = plt.subplots(figsize=(9, 0.32 * len(imp) + 1.5))
        ax.barh(imp["feature"][::-1], imp["gain"][::-1], color="steelblue")
        ax.set_title("Top features by total gain")
        ax.set_xlabel("LightGBM gain (higher = more useful)")
        fig.tight_layout()
        imp_path = PLOTS_DIR / "feature_importance.png"
        fig.savefig(imp_path, dpi=130)
        plt.close(fig)
        print(f"\nWrote {imp_path}")


def main() -> None:
    args = parse_args()
    evaluate(args.split, args)


if __name__ == "__main__":
    main()
