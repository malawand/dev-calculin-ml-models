"""Expose live regime predictions as Prometheus metrics for scrape → Cortex."""

from __future__ import annotations

import logging
import time
from typing import Any

from prometheus_client import Counter, Gauge, start_http_server

from constants import REGIME_INT_TO_NAME

logger = logging.getLogger(__name__)

REGIME_TO_INT = {v: k for k, v in REGIME_INT_TO_NAME.items()}


class RegimeTelemetry:
    """
    Prometheus exporter for the latest regime prediction.

    Prometheus scrapes GET :metrics_port/metrics on an interval (e.g. 60s).
    Each scrape reads the gauges last updated by the 15-minute prediction loop.
    Remote-write / federation into Cortex stores the time series long-term.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        telemetry_cfg = config.get("telemetry", {})
        self.enabled = telemetry_cfg.get("enabled", True)
        self.http_port = int(telemetry_cfg.get("metrics_port", 9109))
        self.symbol = telemetry_cfg.get("symbol", "BTCUSDT")
        labelnames = ["symbol"]

        self.last_timestamp = Gauge(
            "btc_regime_detector_last_timestamp",
            "Unix timestamp of the bar used for the last successful prediction",
            labelnames,
        )
        self.last_price = Gauge(
            "btc_regime_detector_last_price",
            "BTC price at last prediction",
            labelnames,
        )
        self.regime_value = Gauge(
            "btc_regime_detector_regime_value",
            "Predicted regime (CHOP=0, TRENDING_UP=1, TRENDING_DOWN=2, VOLATILE_EXPANSION=3)",
            labelnames,
        )
        self.confidence = Gauge(
            "btc_regime_detector_confidence",
            "Model confidence (max class probability, 0–1)",
            labelnames,
        )
        self.prob = Gauge(
            "btc_regime_detector_prob",
            "Per-regime probability from last prediction",
            labelnames + ["regime"],
        )
        self.last_run_success = Gauge(
            "btc_regime_detector_last_run_success",
            "1 if the most recent prediction cycle succeeded, 0 if it failed",
            labelnames,
        )
        self.last_run_unix = Gauge(
            "btc_regime_detector_last_run_unix",
            "Unix time when the prediction loop last completed (success or failure)",
            labelnames,
        )
        self.bars_fetched = Gauge(
            "btc_regime_detector_bars_fetched",
            "Prometheus bars pulled in the last successful cycle",
            labelnames,
        )
        self.bars_usable = Gauge(
            "btc_regime_detector_bars_usable",
            "Bars with complete features in the last successful cycle",
            labelnames,
        )
        self.prediction_runs = Counter(
            "btc_regime_detector_prediction_runs_total",
            "Prediction loop outcomes",
            labelnames + ["status"],
        )
        self._server_started = False

    def start(self) -> None:
        if not self.enabled or self._server_started:
            return
        start_http_server(self.http_port)
        self._server_started = True
        logger.info("Prometheus metrics listening on :%s/metrics", self.http_port)

    def emit(self, output: dict[str, Any]) -> None:
        if not self.enabled:
            return

        sym = self.symbol
        now = time.time()
        self.last_run_unix.labels(symbol=sym).set(now)
        self.last_run_success.labels(symbol=sym).set(1)
        self.prediction_runs.labels(symbol=sym, status="success").inc()

        ts = output.get("timestamp")
        if ts:
            import pandas as pd

            self.last_timestamp.labels(symbol=sym).set(pd.Timestamp(ts).timestamp())

        if output.get("price") is not None:
            self.last_price.labels(symbol=sym).set(output["price"])

        regime = output.get("regime", "CHOP")
        self.regime_value.labels(symbol=sym).set(REGIME_TO_INT.get(regime, 0))
        self.confidence.labels(symbol=sym).set(output.get("confidence", 0.0))

        self.prob.labels(symbol=sym, regime="CHOP").set(output.get("prob_chop", 0.0))
        self.prob.labels(symbol=sym, regime="TRENDING_UP").set(
            output.get("prob_trending_up", 0.0)
        )
        self.prob.labels(symbol=sym, regime="TRENDING_DOWN").set(
            output.get("prob_trending_down", 0.0)
        )
        self.prob.labels(symbol=sym, regime="VOLATILE_EXPANSION").set(
            output.get("prob_volatile_expansion", 0.0)
        )

        if output.get("bars_fetched") is not None:
            self.bars_fetched.labels(symbol=sym).set(output["bars_fetched"])
        if output.get("bars_usable") is not None:
            self.bars_usable.labels(symbol=sym).set(output["bars_usable"])

    def record_error(self) -> None:
        if not self.enabled:
            return
        sym = self.symbol
        self.last_run_unix.labels(symbol=sym).set(time.time())
        self.last_run_success.labels(symbol=sym).set(0)
        self.prediction_runs.labels(symbol=sym, status="failure").inc()
