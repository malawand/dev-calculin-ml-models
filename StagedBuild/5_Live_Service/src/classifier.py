"""Load the trained Stage 4 LightGBM model and score encoded features."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from constants import REGIME_INT_TO_NAME

NUM_REGIMES = len(REGIME_INT_TO_NAME)


class RegimeClassifier:
    """Thin wrapper around the saved LightGBM booster."""

    def __init__(self, artifacts_dir: str | Path) -> None:
        artifacts_dir = Path(artifacts_dir)
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
