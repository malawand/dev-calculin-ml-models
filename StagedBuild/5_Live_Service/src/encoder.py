"""Load frozen Stage 3 encoder thresholds and transform live indicator data."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Prefer vendored copy (Docker / packaged deploy), fall back to Stage 3 source tree.
_LIB_DIR = Path(__file__).resolve().parents[1] / "lib"
_STAGE3_DIR = Path(__file__).resolve().parents[2] / "3_Momentum_State_Encoder"
for _p in (_LIB_DIR, _STAGE3_DIR):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from momentum_state import MomentumStateEncoder  # noqa: E402

from constants import DEFAULT_FEATURE_COLS, DEFAULT_WINDOWS  # noqa: E402


def encoder_artifacts_to_dict(enc: MomentumStateEncoder, *, source: str = "", row_count: int = 0) -> dict[str, Any]:
    """Serialize a fitted encoder to a JSON-safe dict."""
    return {
        "_meta": {
            "description": "Frozen Stage 3 encoder thresholds for live inference",
            "source_parquet": source,
            "row_count": row_count,
        },
        "feature_cols": enc.feature_cols,
        "windows": enc.windows,
        "threshold_pctl": enc.threshold_pctl,
        "epsilon_pctl": enc.epsilon_pctl,
        "train_fraction": enc.train_fraction,
        "thresholds": enc.thresholds,
        "epsilons": {feat: {str(w): v for w, v in wins.items()} for feat, wins in enc.epsilons.items()},
        "epsilons_fast": {
            feat: {str(w): v for w, v in wins.items()} for feat, wins in enc.epsilons_fast.items()
        },
    }


def save_encoder_artifacts(
    enc: MomentumStateEncoder,
    path: str | Path,
    *,
    source: str = "",
    row_count: int = 0,
) -> None:
    path = Path(path)
    path.write_text(json.dumps(
        encoder_artifacts_to_dict(enc, source=source, row_count=row_count),
        indent=2,
    ))


def load_encoder_from_artifacts(path: str | Path) -> MomentumStateEncoder:
    """
    Reconstruct a fitted MomentumStateEncoder from a JSON file produced by
    ``scripts/export_encoder_artifacts.py``.
    """
    data = json.loads(Path(path).read_text())

    enc = MomentumStateEncoder(
        feature_cols=data["feature_cols"],
        windows=[int(w) for w in data["windows"]],
        threshold_pctl=float(data.get("threshold_pctl", 30.0)),
        epsilon_pctl=float(data.get("epsilon_pctl", 10.0)),
        train_fraction=float(data.get("train_fraction", 0.70)),
    )
    enc.thresholds = {k: float(v) for k, v in data["thresholds"].items()}
    enc.epsilons = {
        feat: {int(w): float(v) for w, v in wins.items()}
        for feat, wins in data["epsilons"].items()
    }
    enc.epsilons_fast = {
        feat: {int(w): float(v) for w, v in wins.items()}
        for feat, wins in data["epsilons_fast"].items()
    }
    enc._fitted = True
    return enc


def fit_encoder_from_history(df: pd.DataFrame) -> MomentumStateEncoder:
    """Fit encoder thresholds on historical data (used by export script only)."""
    enc = MomentumStateEncoder(
        feature_cols=DEFAULT_FEATURE_COLS,
        windows=DEFAULT_WINDOWS,
    )
    enc.fit(df)
    return enc


def encode_live(df: pd.DataFrame, enc: MomentumStateEncoder) -> pd.DataFrame:
    """Apply the frozen encoder to a dataframe of raw indicator columns."""
    missing = [c for c in enc.feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Input missing indicator columns: {missing}")
    return enc.transform(df)
