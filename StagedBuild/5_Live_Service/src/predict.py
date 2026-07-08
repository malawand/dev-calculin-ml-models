"""Orchestrate one live regime prediction cycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from classifier import RegimeClassifier
from constants import MIN_BARS_FOR_FEATURES, RECOMMENDED_LOOKBACK_DAYS
from encoder import encode_live, load_encoder_from_artifacts
from prometheus_fetch import fetch_recent_bars


@dataclass
class LivePredictor:
    """Holds loaded model + encoder; runs fetch → encode → classify."""

    config: dict[str, Any]
    artifacts_dir: Path
    encoder_path: Path

    def __post_init__(self) -> None:
        self.encoder = load_encoder_from_artifacts(self.encoder_path)
        self.classifier = RegimeClassifier(self.artifacts_dir)

    @classmethod
    def from_config(cls, config: dict[str, Any], base_dir: Path | None = None) -> "LivePredictor":
        base_dir = base_dir or Path(__file__).resolve().parents[1]
        paths = config.get("paths", {})
        return cls(
            config=config,
            artifacts_dir=base_dir / paths.get("classifier_artifacts", "artifacts"),
            encoder_path=base_dir / paths.get("encoder_artifacts", "artifacts/encoder_artifacts.json"),
        )

    def predict_once(self) -> dict[str, Any]:
        prom_cfg = self.config.get("prometheus", {})
        url = prom_cfg["query_range_url"]
        lookback_days = float(prom_cfg.get("lookback_days", RECOMMENDED_LOOKBACK_DAYS))
        step = int(prom_cfg.get("step_seconds", 900))
        queries = prom_cfg.get("queries")

        raw = fetch_recent_bars(url, lookback_days=lookback_days, step=step, queries=queries)
        if len(raw) < MIN_BARS_FOR_FEATURES:
            raise RuntimeError(
                f"Need at least {MIN_BARS_FOR_FEATURES} bars (~96h at 15m) but got {len(raw)}. "
                f"Increase lookback_days (currently {lookback_days})."
            )

        encoded = encode_live(raw, self.encoder)

        feature_cols = self.classifier.feature_cols
        valid = encoded.dropna(subset=feature_cols)
        if valid.empty:
            raise RuntimeError(
                "No rows with complete features after encoding. "
                "Pull more history or check for NaN indicators in Prometheus."
            )

        latest_idx = valid.index[-1]
        latest_row = valid.iloc[[-1]]
        pred = self.classifier.predict(latest_row).iloc[0]

        price = None
        if "price" in raw.columns and latest_idx in raw.index:
            price = float(raw.loc[latest_idx, "price"])

        return {
            "timestamp": latest_idx.isoformat(),
            "price": price,
            "regime": pred["pred_name"],
            "confidence": float(pred["confidence"]),
            "prob_chop": float(pred["prob_CHOP"]),
            "prob_trending_up": float(pred["prob_TRENDING_UP"]),
            "prob_trending_down": float(pred["prob_TRENDING_DOWN"]),
            "prob_volatile_expansion": float(pred["prob_VOLATILE_EXPANSION"]),
            "bars_fetched": len(raw),
            "bars_usable": len(valid),
        }
