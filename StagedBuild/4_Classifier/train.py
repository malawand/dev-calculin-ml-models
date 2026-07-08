#!/usr/bin/env python3
"""
Stage 4 trainer — fits a LightGBM multi-class model that predicts the
hindsight regime from Stage 3's momentum-state features.

Pipeline:
  1. Load Stage 3 parquet via data.load_dataset().
  2. Apply inverse-frequency class weights so the rare regime
     (VOLATILE_EXPANSION) isn't ignored.
  3. Train a LightGBM `multiclass` model with early stopping on val.
  4. Save the model, the feature list, the per-row predictions on
     val + test, and the class weights into ``model_artifacts/``.
  5. Print val + test metrics so you immediately see whether the run
     was sane.

Run:
    python3 train.py
    python3 train.py --num-leaves 63 --learning-rate 0.05

Outputs (model_artifacts/):
    model.txt                   LightGBM model in its native text format.
    feature_cols.json           Ordered list of features the model expects.
    class_weights.json          Per-class weight dict used at training.
    predictions_val.parquet     Per-row val predictions (probs + argmax).
    predictions_test.parquet    Per-row test predictions.
    train_log.txt               Captured stdout/stderr from training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

from data import (
    DEFAULT_INPUT_PARQUET,
    NUM_REGIMES,
    REGIME_INT_TO_NAME,
    class_weights_inverse_freq,
    load_dataset,
)

ARTIFACTS_DIR = Path(__file__).parent / "model_artifacts"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage 4 regime classifier (LightGBM).")
    p.add_argument("--data", type=str, default=str(DEFAULT_INPUT_PARQUET),
                   help="Path to Stage 3 parquet.")
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--n-estimators", type=int, default=2000,
                   help="Max boosting rounds; early stopping will usually halt earlier.")
    p.add_argument("--min-data-in-leaf", type=int, default=200)
    p.add_argument("--feature-fraction", type=float, default=0.85)
    p.add_argument("--bagging-fraction", type=float, default=0.85)
    p.add_argument("--bagging-freq", type=int, default=5)
    p.add_argument("--lambda-l2", type=float, default=0.1)
    p.add_argument("--early-stopping", type=int, default=100,
                   help="Stop if val_logloss hasn't improved for N rounds.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-class-weights", action="store_true",
                   help="Disable inverse-frequency class weights.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    log_lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        log_lines.append(msg)

    log("=" * 74)
    log("Stage 4 — Regime classifier training")
    log("=" * 74)

    ds = load_dataset(args.data)
    log(ds.summary())

    if args.no_class_weights:
        cw = {k: 1.0 for k in REGIME_INT_TO_NAME}
        log("Class weights: DISABLED (every class weighted 1.0).")
    else:
        cw = class_weights_inverse_freq(ds.y_train)
        log("Class weights (inverse frequency, normalized to mean 1):")
        for k, v in sorted(cw.items()):
            log(f"  {REGIME_INT_TO_NAME[k]:<22s} weight={v:.3f}")
        log("")

    sample_weight_train = ds.y_train.map(cw).astype(float).values
    sample_weight_val = ds.y_val.map(cw).astype(float).values

    params = dict(
        objective="multiclass",
        num_class=NUM_REGIMES,
        metric=["multi_logloss"],
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_data_in_leaf=args.min_data_in_leaf,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        bagging_freq=args.bagging_freq,
        lambda_l2=args.lambda_l2,
        verbose=-1,
        seed=args.seed,
        deterministic=True,
    )

    train_data = lgb.Dataset(ds.X_train, label=ds.y_train, weight=sample_weight_train)
    val_data = lgb.Dataset(ds.X_val, label=ds.y_val, weight=sample_weight_val,
                           reference=train_data)

    log(f"LightGBM params: {params}")
    log(f"Training... (max {args.n_estimators} rounds, early stop after {args.early_stopping}).")

    model = lgb.train(
        params,
        train_data,
        num_boost_round=args.n_estimators,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(args.early_stopping, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )

    log("")
    log(f"Best iteration : {model.best_iteration}")
    log(f"Best val score : {model.best_score['val']['multi_logloss']:.5f}")
    log("")

    proba_val = model.predict(ds.X_val, num_iteration=model.best_iteration)
    proba_test = model.predict(ds.X_test, num_iteration=model.best_iteration)
    pred_val = proba_val.argmax(axis=1)
    pred_test = proba_test.argmax(axis=1)

    val_acc = accuracy_score(ds.y_val, pred_val)
    val_f1 = f1_score(ds.y_val, pred_val, average="macro")
    val_logloss = log_loss(ds.y_val, proba_val, labels=list(range(NUM_REGIMES)))
    test_acc = accuracy_score(ds.y_test, pred_test)
    test_f1 = f1_score(ds.y_test, pred_test, average="macro")
    test_logloss = log_loss(ds.y_test, proba_test, labels=list(range(NUM_REGIMES)))

    log("=" * 74)
    log(f"VAL : acc={val_acc:.4f}  macro_F1={val_f1:.4f}  logloss={val_logloss:.4f}")
    log(f"TEST: acc={test_acc:.4f}  macro_F1={test_f1:.4f}  logloss={test_logloss:.4f}")
    log("=" * 74)

    model.save_model(str(ARTIFACTS_DIR / "model.txt"),
                     num_iteration=model.best_iteration)

    with open(ARTIFACTS_DIR / "feature_cols.json", "w") as f:
        json.dump(ds.feature_cols, f, indent=2)
    with open(ARTIFACTS_DIR / "class_weights.json", "w") as f:
        json.dump(cw, f, indent=2)
    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump({
            "best_iteration": int(model.best_iteration),
            "val_acc": float(val_acc),
            "val_macro_f1": float(val_f1),
            "val_logloss": float(val_logloss),
            "test_acc": float(test_acc),
            "test_macro_f1": float(test_f1),
            "test_logloss": float(test_logloss),
        }, f, indent=2)

    def proba_frame(idx: pd.Index, y_true: pd.Series, proba: np.ndarray) -> pd.DataFrame:
        out = pd.DataFrame(proba, columns=[f"prob_{REGIME_INT_TO_NAME[i]}" for i in range(NUM_REGIMES)],
                           index=idx)
        out["y_true"] = y_true.values
        out["y_pred"] = proba.argmax(axis=1)
        out["y_pred_name"] = [REGIME_INT_TO_NAME[int(p)] for p in out["y_pred"]]
        out["confidence"] = proba.max(axis=1)
        return out

    proba_frame(ds.val_index, ds.y_val, proba_val).to_parquet(
        ARTIFACTS_DIR / "predictions_val.parquet")
    proba_frame(ds.test_index, ds.y_test, proba_test).to_parquet(
        ARTIFACTS_DIR / "predictions_test.parquet")

    with open(ARTIFACTS_DIR / "train_log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n")

    log(f"Artifacts saved to {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()
