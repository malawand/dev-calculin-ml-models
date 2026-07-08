"""Prometheus metrics for the bullish entry state machine."""

from __future__ import annotations

import logging
import time
from typing import Any

from prometheus_client import Counter, Gauge, start_http_server

from models import Action, SignalOutput, STATE_NUMERIC

logger = logging.getLogger(__name__)


class EntryStateMachineTelemetry:
    """Expose state machine observability on GET :port/metrics."""

    def __init__(self, config: dict[str, Any]) -> None:
        tel = config.get("telemetry", {})
        self.enabled = tel.get("enabled", True)
        self.http_port = int(tel.get("metrics_port", 9111))
        self.symbol = tel.get("symbol", "BTCUSDT")
        labelnames = ["symbol"]

        self.state_numeric = Gauge(
            "btc_entry_sm_state_numeric",
            "Current state machine state (0=NEUTRAL .. 6=COOLDOWN)",
            labelnames,
        )
        self.chop_dominance_ratio = Gauge(
            "btc_entry_sm_chop_dominance_ratio",
            "Fraction of chop lookback where CHOP was dominant",
            labelnames,
        )
        self.volatile_expansion_rise = Gauge(
            "btc_entry_sm_volatile_expansion_rise",
            "Current volatile expansion prob minus N bars ago",
            labelnames,
        )
        self.trend_spread = Gauge(
            "btc_entry_sm_trend_spread",
            "prob_trending_up minus prob_trending_down",
            labelnames,
        )
        self.recent_range_high = Gauge(
            "btc_entry_sm_recent_range_high",
            "Recent range high (prior bars only)",
            labelnames,
        )
        self.breakout_level = Gauge(
            "btc_entry_sm_breakout_level",
            "Breakout threshold above range high",
            labelnames,
        )
        self.price_breakout = Gauge(
            "btc_entry_sm_price_breakout",
            "1 if current price broke above breakout level",
            labelnames,
        )
        self.confidence_gap = Gauge(
            "btc_entry_sm_confidence_gap",
            "Top regime prob minus second highest",
            labelnames,
        )
        self.setup_age_bars = Gauge(
            "btc_entry_sm_setup_age_bars",
            "Bars since EXPANSION_ALERT within active setup",
            labelnames,
        )
        self.long_entry_signal = Gauge(
            "btc_entry_sm_long_entry_signal",
            "1 if ENTER_LONG was emitted this bar",
            labelnames,
        )
        self.last_run_success = Gauge(
            "btc_entry_sm_last_run_success",
            "1 if last poll cycle succeeded",
            labelnames,
        )
        self.last_run_unix = Gauge(
            "btc_entry_sm_last_run_unix",
            "Unix time of last poll cycle",
            labelnames,
        )
        self.enter_long_total = Counter(
            "btc_entry_sm_enter_long_total",
            "Total ENTER_LONG signals emitted",
            labelnames,
        )
        self._server_started = False

    def start(self) -> None:
        if not self.enabled or self._server_started:
            return
        start_http_server(self.http_port)
        self._server_started = True
        logger.info("Entry SM metrics listening on :%s/metrics", self.http_port)

    def emit(self, output: SignalOutput) -> None:
        if not self.enabled:
            return

        sym = self.symbol
        meta = output.metadata
        self.last_run_unix.labels(symbol=sym).set(time.time())
        self.last_run_success.labels(symbol=sym).set(1)
        self.state_numeric.labels(symbol=sym).set(STATE_NUMERIC.get(output.state, 0))
        self.chop_dominance_ratio.labels(symbol=sym).set(meta.chop_dominance_ratio)
        self.volatile_expansion_rise.labels(symbol=sym).set(meta.volatile_expansion_rise)
        self.trend_spread.labels(symbol=sym).set(meta.trend_spread)
        if meta.recent_range_high is not None:
            self.recent_range_high.labels(symbol=sym).set(meta.recent_range_high)
        if meta.breakout_level is not None:
            self.breakout_level.labels(symbol=sym).set(meta.breakout_level)
        self.price_breakout.labels(symbol=sym).set(1 if meta.price_breakout else 0)
        self.confidence_gap.labels(symbol=sym).set(meta.confidence_gap)
        self.setup_age_bars.labels(symbol=sym).set(meta.setup_age_bars)
        fired = output.action == Action.ENTER_LONG
        self.long_entry_signal.labels(symbol=sym).set(1 if fired else 0)
        if fired:
            self.enter_long_total.labels(symbol=sym).inc()

    def record_error(self) -> None:
        if not self.enabled:
            return
        sym = self.symbol
        self.last_run_unix.labels(symbol=sym).set(time.time())
        self.last_run_success.labels(symbol=sym).set(0)
