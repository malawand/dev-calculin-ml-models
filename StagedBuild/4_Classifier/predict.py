#!/usr/bin/env python3
"""
Stage 4 inference helper.

Loads the trained classifier (from ``model_artifacts/``) and produces
regime predictions for a parquet of features that has the same schema
as Stage 3's output.

Use for:
  * Standalone scoring of a fresh chunk of bars.
  * Wrapping into a tiny live service later (the classes here are the
    same ones a real-time loop would call).

Usage:
    python3 predict.py --input ../3_Momentum_State_Encoder/btc_data_15m_mstate.parquet \
                       --output predictions.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from data import NUM_REGIMES, REGIME_INT_TO_NAME

ARTIFACTS_DIR = Path(__file__).parent / "model_artifacts"


class RegimeClassifier:
    """Thin convenience wrapper around the saved LightGBM booster."""

    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
        self.booster = lgb.Booster(model_file=str(artifacts_dir / "model.txt"))
        self.feature_cols = json.loads((artifacts_dir / "feature_cols.json").read_text())

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"Input df is missing {len(missing)} feature columns. "
                f"First 5: {missing[:5]}"
            )
        X = df[self.feature_cols].copy()
        for c in X.columns:
            if X[c].dtype == bool:
                X[c] = X[c].astype(np.int8)
        return self.booster.predict(X)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        proba = self.predict_proba(df)
        out = pd.DataFrame(
            proba,
            columns=[f"prob_{REGIME_INT_TO_NAME[i]}" for i in range(NUM_REGIMES)],
            index=df.index,
        )
        out["pred_int"] = proba.argmax(axis=1)
        out["pred_name"] = [REGIME_INT_TO_NAME[int(i)] for i in out["pred_int"]]
        out["confidence"] = proba.max(axis=1)
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score Stage 3 features with the Stage 4 classifier.")
    p.add_argument("--input", required=True, help="Path to a parquet with Stage 3 features.")
    p.add_argument("--output", required=True, help="Where to write predictions parquet.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input).sort_index()

    clf = RegimeClassifier()
    nan_mask = df[clf.feature_cols].isna().any(axis=1)
    if nan_mask.any():
        print(f"Dropping {nan_mask.sum():,} rows with NaN features (likely warm-up).")
        df = df[~nan_mask]

    pred = clf.predict(df)
    pred.to_parquet(args.output)

    print(f"Wrote {len(pred):,} predictions to {args.output}")
    print("Distribution of predicted regimes:")
    print(pred["pred_name"].value_counts())


if __name__ == "__main__":
    main()
