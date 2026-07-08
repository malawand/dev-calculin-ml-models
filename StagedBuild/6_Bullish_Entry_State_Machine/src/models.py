"""Shared types for the bullish entry state machine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd


class TradingState(str, Enum):
    NEUTRAL = "NEUTRAL"
    CHOP_BASE = "CHOP_BASE"
    EXPANSION_ALERT = "EXPANSION_ALERT"
    BULLISH_CONFIRMATION = "BULLISH_CONFIRMATION"
    LONG_ENTRY = "LONG_ENTRY"
    IN_LONG = "IN_LONG"
    COOLDOWN = "COOLDOWN"


STATE_NUMERIC: dict[TradingState, int] = {
    TradingState.NEUTRAL: 0,
    TradingState.CHOP_BASE: 1,
    TradingState.EXPANSION_ALERT: 2,
    TradingState.BULLISH_CONFIRMATION: 3,
    TradingState.LONG_ENTRY: 4,
    TradingState.IN_LONG: 5,
    TradingState.COOLDOWN: 6,
}


class Action(str, Enum):
    NO_TRADE = "NO_TRADE"
    HOLD = "HOLD"
    ENTER_LONG = "ENTER_LONG"
    EXIT_LONG = "EXIT_LONG"


@dataclass(frozen=True)
class NowcasterBar:
    """One bar of nowcaster output — decoupled from Stage 5 imports."""

    timestamp: datetime
    price: float
    prob_chop: float
    prob_trending_up: float
    prob_trending_down: float
    prob_volatile_expansion: float
    confidence: float

    @classmethod
    def from_prediction_json(cls, data: dict[str, Any]) -> "NowcasterBar":
        ts = data.get("timestamp")
        if ts is None:
            raise ValueError("prediction JSON missing 'timestamp'")
        price = data.get("price")
        if price is None:
            raise ValueError("prediction JSON missing 'price'")
        return cls(
            timestamp=pd.Timestamp(ts).to_pydatetime(),
            price=float(price),
            prob_chop=float(data.get("prob_chop", 0.0)),
            prob_trending_up=float(data.get("prob_trending_up", 0.0)),
            prob_trending_down=float(data.get("prob_trending_down", 0.0)),
            prob_volatile_expansion=float(data.get("prob_volatile_expansion", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
        )


@dataclass
class SignalMetadata:
    prob_chop: float = 0.0
    prob_volatile_expansion: float = 0.0
    prob_trending_up: float = 0.0
    prob_trending_down: float = 0.0
    chop_dominance_ratio: float = 0.0
    volatile_expansion_rise: float = 0.0
    trend_spread: float = 0.0
    recent_range_high: float | None = None
    breakout_level: float | None = None
    price_breakout: bool = False
    confidence_gap: float = 0.0
    setup_age_bars: int = 0
    long_entry_signal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SignalOutput:
    timestamp: str
    state: TradingState
    state_numeric: int
    action: Action
    reason: str
    metadata: SignalMetadata = field(default_factory=SignalMetadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "state": self.state.value,
            "state_numeric": self.state_numeric,
            "action": self.action.value,
            "reason": self.reason,
            "metadata": self.metadata.to_dict(),
        }
