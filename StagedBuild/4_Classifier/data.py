#!/usr/bin/env python3
"""
Shared data utilities for Stage 4 (regime classifier).

Two responsibilities:
  1. Decide which columns are SAFE to use as model inputs and which are
     LEAKAGE (the columns that define the regime label).
  2. Split a chronological time series into train / val / test slices
     without ever shuffling.

Both decisions matter more than the model choice. A leaked column gives
"perfect" accuracy that vanishes the moment you go live; a randomly
shuffled split lets information from the future leak into the past.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants the rest of the stage relies on
# ---------------------------------------------------------------------------

# Path to Stage 3 output. Override with DATA_PATH env var if you keep
# the parquet somewhere else.
DEFAULT_INPUT_PARQUET = Path(
    "/Users/mazenlawand/Documents/Calculin ML Models/StagedBuild/3_Momentum_State_Encoder/btc_data_15m_mstate.parquet"
)

# The four regimes. Match Stage 2's vocabulary exactly.
REGIME_INT_TO_NAME = {
    0: "CHOP",
    1: "TRENDING_UP",
    2: "TRENDING_DOWN",
    3: "VOLATILE_EXPANSION",
}
REGIME_NAME_TO_INT = {v: k for k, v in REGIME_INT_TO_NAME.items()}
NUM_REGIMES = len(REGIME_INT_TO_NAME)

# Columns that DEFINE the regime label. If we trained on these, the
# classifier would just memorize the labelling rules. Drop them.
LEAKY_COLUMNS: List[str] = [
    "regime",
    "regime_raw",
    "regime_name",
    "regime_start",
    "bars_in_regime",
    # Stage 2 may emit these; drop just in case.
    "rolling_return_4h",
    "rolling_return_12h",
    "rolling_return_24h",
    "rolling_return_7d",
    "realized_volatility_4h",
    "realized_volatility_24h",
]

# Identity columns we never feed in but want to keep alongside predictions.
NON_FEATURE_COLUMNS: List[str] = ["price"]

# Chronological split fractions. NEVER shuffle.
DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC = 0.15
# test_frac is implicit = 1 - train - val.


# ---------------------------------------------------------------------------
# Public dataclass returned by load_dataset
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    """Holds a fully prepared train/val/test split ready for any model."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_cols: List[str]
    train_index: pd.Index
    val_index: pd.Index
    test_index: pd.Index
    raw: pd.DataFrame  # original df (post-dropna) — useful for backtests
    target_col: str = "regime"

    def summary(self) -> str:
        def pct(s: pd.Series) -> str:
            counts = s.value_counts(normalize=True).sort_index() * 100
            return ", ".join(
                f"{REGIME_INT_TO_NAME[int(k)]}={v:.1f}%" for k, v in counts.items()
            )

        return (
            f"Dataset summary\n"
            f"---------------\n"
            f"features          : {len(self.feature_cols)}\n"
            f"train rows        : {len(self.X_train):,}  ({pct(self.y_train)})\n"
            f"val   rows        : {len(self.X_val):,}  ({pct(self.y_val)})\n"
            f"test  rows        : {len(self.X_test):,}  ({pct(self.y_test)})\n"
            f"train range       : {self.train_index.min()} -> {self.train_index.max()}\n"
            f"val   range       : {self.val_index.min()} -> {self.val_index.max()}\n"
            f"test  range       : {self.test_index.min()} -> {self.test_index.max()}\n"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the column names safe to feed into the classifier.

    Drops:
      - The regime columns themselves (the answer key).
      - Any rolling_return / realized_volatility column (they DEFINE the
        regime).
      - The raw `price` column (we keep it for backtesting, not modeling
        — its scale changes too much across years).
    Keeps everything else, which is dominated by the Stage 3 momentum
    encoder outputs (d1, d2, d1_fast, mstate, mstate_duration,
    pre_cross_warning, pre_trough_warning) plus the three raw indicator
    columns those were derived from.
    """
    drop = set(LEAKY_COLUMNS) | set(NON_FEATURE_COLUMNS)
    return [c for c in df.columns if c not in drop]


def load_dataset(
    parquet_path: str | Path = DEFAULT_INPUT_PARQUET,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
) -> Dataset:
    """Load Stage 3's parquet and produce a leak-free chronological split."""
    parquet_path = Path(parquet_path)
    df = pd.read_parquet(parquet_path).sort_index()

    # Drop warm-up rows where Stage 3 derivatives are still NaN. Stage 3
    # leaves NaN only in the first max(windows)=192 bars per feature.
    feature_cols = select_feature_columns(df)
    before = len(df)
    df = df.dropna(subset=feature_cols + ["regime"])
    after = len(df)
    if before != after:
        print(f"Dropped {before - after:,} warm-up rows with NaN features.")

    # Sanity check: target must be valid integer in [0, 3].
    df["regime"] = df["regime"].astype(int)
    assert df["regime"].between(0, NUM_REGIMES - 1).all()

    n = len(df)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_test = n - n_train - n_val

    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]

    feature_cols = select_feature_columns(df)
    bool_cols = [c for c in feature_cols if df[c].dtype == bool]

    def to_features(slc: pd.DataFrame) -> pd.DataFrame:
        X = slc[feature_cols].copy()
        for c in bool_cols:
            X[c] = X[c].astype(np.int8)
        return X

    return Dataset(
        X_train=to_features(train),
        y_train=train["regime"].astype(int),
        X_val=to_features(val),
        y_val=val["regime"].astype(int),
        X_test=to_features(test),
        y_test=test["regime"].astype(int),
        feature_cols=feature_cols,
        train_index=train.index,
        val_index=val.index,
        test_index=test.index,
        raw=df,
    )


def class_weights_inverse_freq(y: pd.Series) -> dict:
    """Class weights = inverse frequency, normalized so they average ~1.

    Used to keep the rare VOLATILE_EXPANSION class from being ignored.
    """
    counts = y.value_counts().sort_index()
    inv = 1.0 / counts
    inv = inv / inv.mean()  # normalize so weights average 1
    return {int(k): float(v) for k, v in inv.items()}


__all__ = [
    "Dataset",
    "DEFAULT_INPUT_PARQUET",
    "DEFAULT_TRAIN_FRAC",
    "DEFAULT_VAL_FRAC",
    "LEAKY_COLUMNS",
    "NON_FEATURE_COLUMNS",
    "NUM_REGIMES",
    "REGIME_INT_TO_NAME",
    "REGIME_NAME_TO_INT",
    "class_weights_inverse_freq",
    "load_dataset",
    "select_feature_columns",
]
