"""Configuration for the bullish entry state machine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StateMachineConfig:
    chop_lookback_bars: int = 12
    chop_dominance_threshold: float = 0.35
    required_chop_dominance_ratio: float = 0.65
    volatile_expansion_threshold: float = 0.25
    volatile_expansion_rise_threshold: float = 0.15
    volatile_expansion_lookback_bars: int = 4
    trend_up_threshold: float = 0.40
    trend_spread_threshold: float = 0.10
    require_trend_crossing: bool = False
    trend_crossing_lookback_bars: int = 4
    range_lookback_bars: int = 96
    breakout_buffer_pct: float = 0.001
    max_signal_age_bars: int = 8
    minimum_model_confidence: float = 0.55
    minimum_confidence_gap: float = 0.10
    entry_cooldown_bars: int = 96
    allow_reentry_while_in_position: bool = False

    @property
    def max_history_bars(self) -> int:
        return max(
            self.chop_lookback_bars,
            self.volatile_expansion_lookback_bars + 1,
            self.range_lookback_bars + 1,
            self.trend_crossing_lookback_bars + 1,
        )


@dataclass
class NowcasterPollConfig:
    url: str = "http://localhost:8080/prediction"
    poll_interval_seconds: int = 900


@dataclass
class ServiceConfig:
    api_port: int = 8082


@dataclass
class TelemetryConfig:
    enabled: bool = True
    metrics_port: int = 9111
    symbol: str = "BTCUSDT"


@dataclass
class AppConfig:
    state_machine: StateMachineConfig
    nowcaster: NowcasterPollConfig
    service: ServiceConfig
    telemetry: TelemetryConfig


def load_config(path: Path | str) -> AppConfig:
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    sm = raw.get("state_machine", {})
    nc = raw.get("nowcaster", {})
    svc = raw.get("service", {})
    tel = raw.get("telemetry", {})

    return AppConfig(
        state_machine=StateMachineConfig(
            chop_lookback_bars=int(sm.get("chop_lookback_bars", 12)),
            chop_dominance_threshold=float(sm.get("chop_dominance_threshold", 0.35)),
            required_chop_dominance_ratio=float(sm.get("required_chop_dominance_ratio", 0.65)),
            volatile_expansion_threshold=float(sm.get("volatile_expansion_threshold", 0.25)),
            volatile_expansion_rise_threshold=float(sm.get("volatile_expansion_rise_threshold", 0.15)),
            volatile_expansion_lookback_bars=int(sm.get("volatile_expansion_lookback_bars", 4)),
            trend_up_threshold=float(sm.get("trend_up_threshold", 0.40)),
            trend_spread_threshold=float(sm.get("trend_spread_threshold", 0.10)),
            require_trend_crossing=bool(sm.get("require_trend_crossing", False)),
            trend_crossing_lookback_bars=int(sm.get("trend_crossing_lookback_bars", 4)),
            range_lookback_bars=int(sm.get("range_lookback_bars", 96)),
            breakout_buffer_pct=float(sm.get("breakout_buffer_pct", 0.001)),
            max_signal_age_bars=int(sm.get("max_signal_age_bars", 8)),
            minimum_model_confidence=float(sm.get("minimum_model_confidence", 0.55)),
            minimum_confidence_gap=float(sm.get("minimum_confidence_gap", 0.10)),
            entry_cooldown_bars=int(sm.get("entry_cooldown_bars", 96)),
            allow_reentry_while_in_position=bool(sm.get("allow_reentry_while_in_position", False)),
        ),
        nowcaster=NowcasterPollConfig(
            url=str(nc.get("url", "http://localhost:8080/prediction")),
            poll_interval_seconds=int(nc.get("poll_interval_seconds", 900)),
        ),
        service=ServiceConfig(api_port=int(svc.get("api_port", 8082))),
        telemetry=TelemetryConfig(
            enabled=bool(tel.get("enabled", True)),
            metrics_port=int(tel.get("metrics_port", 9111)),
            symbol=str(tel.get("symbol", "BTCUSDT")),
        ),
    )
