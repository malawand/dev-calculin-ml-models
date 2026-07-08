"""Synthetic bar builders for state machine tests."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import StateMachineConfig  # noqa: E402
from models import NowcasterBar  # noqa: E402
from state_machine import BullishTransitionStateMachine  # noqa: E402


def make_bar(
    i: int,
    *,
    price: float = 100.0,
    prob_chop: float = 0.5,
    prob_up: float = 0.2,
    prob_down: float = 0.15,
    prob_vol: float = 0.15,
    start: datetime | None = None,
) -> NowcasterBar:
    start = start or datetime(2026, 1, 1, tzinfo=timezone.utc)
    ts = start + timedelta(minutes=15 * i)
    probs = [prob_chop, prob_up, prob_down, prob_vol]
    return NowcasterBar(
        timestamp=ts,
        price=price,
        prob_chop=prob_chop,
        prob_trending_up=prob_up,
        prob_trending_down=prob_down,
        prob_volatile_expansion=prob_vol,
        confidence=max(probs),
    )


def chop_dominant_bar(i: int, price: float = 100.0) -> NowcasterBar:
    return make_bar(i, price=price, prob_chop=0.55, prob_up=0.15, prob_down=0.15, prob_vol=0.15)


def small_config() -> StateMachineConfig:
    return StateMachineConfig(
        chop_lookback_bars=4,
        chop_dominance_threshold=0.35,
        required_chop_dominance_ratio=0.75,
        volatile_expansion_threshold=0.25,
        volatile_expansion_rise_threshold=0.15,
        volatile_expansion_lookback_bars=2,
        trend_up_threshold=0.40,
        trend_spread_threshold=0.10,
        require_trend_crossing=False,
        trend_crossing_lookback_bars=2,
        range_lookback_bars=4,
        breakout_buffer_pct=0.001,
        max_signal_age_bars=3,
        minimum_model_confidence=0.50,
        minimum_confidence_gap=0.08,
        entry_cooldown_bars=2,
        allow_reentry_while_in_position=False,
    )


@pytest.fixture
def cfg() -> StateMachineConfig:
    return small_config()


@pytest.fixture
def sm(cfg: StateMachineConfig) -> BullishTransitionStateMachine:
    return BullishTransitionStateMachine(cfg)


def feed_bars(sm: BullishTransitionStateMachine, bars: list[NowcasterBar], in_position: bool = False):
    outputs = []
    for bar in bars:
        outputs.append(sm.process_bar(bar, in_position=in_position))
    return outputs


def warm_chop_base(sm: BullishTransitionStateMachine, cfg: StateMachineConfig, n: int | None = None) -> None:
    n = n or cfg.chop_lookback_bars
    for i in range(n):
        sm.process_bar(chop_dominant_bar(i))
