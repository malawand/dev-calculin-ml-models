"""Pure condition checks for the bullish entry state machine."""

from __future__ import annotations

from dataclasses import dataclass

from config import StateMachineConfig
from history import RollingBarHistory
from models import NowcasterBar


def bar_is_chop_dominant(bar: NowcasterBar, threshold: float) -> bool:
    return (
        bar.prob_chop >= threshold
        and bar.prob_chop > bar.prob_trending_up
        and bar.prob_chop > bar.prob_trending_down
        and bar.prob_chop > bar.prob_volatile_expansion
    )


def chop_dominance_ratio(history: RollingBarHistory, cfg: StateMachineConfig) -> float:
    window = history.lookback(cfg.chop_lookback_bars)
    if len(window) < cfg.chop_lookback_bars:
        return 0.0
    dominant = sum(1 for b in window if bar_is_chop_dominant(b, cfg.chop_dominance_threshold))
    return dominant / cfg.chop_lookback_bars


def chop_dominance_met(history: RollingBarHistory, cfg: StateMachineConfig) -> bool:
    return chop_dominance_ratio(history, cfg) >= cfg.required_chop_dominance_ratio


def volatile_expansion_rise(history: RollingBarHistory, cfg: StateMachineConfig) -> float:
    bars = history.bars
    if len(bars) < cfg.volatile_expansion_lookback_bars + 1:
        return 0.0
    current = bars[-1].prob_volatile_expansion
    past = bars[-1 - cfg.volatile_expansion_lookback_bars].prob_volatile_expansion
    return current - past


def expansion_alert_met(history: RollingBarHistory, cfg: StateMachineConfig) -> bool:
    bar = history.current()
    if bar is None:
        return False
    rise = volatile_expansion_rise(history, cfg)
    return (
        bar.prob_volatile_expansion >= cfg.volatile_expansion_threshold
        and rise >= cfg.volatile_expansion_rise_threshold
    )


def trend_spread(bar: NowcasterBar) -> float:
    return bar.prob_trending_up - bar.prob_trending_down


def trend_crossing_met(history: RollingBarHistory, cfg: StateMachineConfig) -> bool:
    """True if spread was below threshold recently and is now above."""
    bars = history.lookback(cfg.trend_crossing_lookback_bars + 1)
    if len(bars) < 2:
        return False
    current_spread = trend_spread(bars[-1])
    if current_spread < cfg.trend_spread_threshold:
        return False
    prior_spreads = [trend_spread(b) for b in bars[:-1]]
    return any(s <= cfg.trend_spread_threshold for s in prior_spreads)


def bullish_confirmation_met(history: RollingBarHistory, cfg: StateMachineConfig) -> bool:
    bar = history.current()
    if bar is None:
        return False
    spread = trend_spread(bar)
    base_ok = (
        bar.prob_trending_up >= cfg.trend_up_threshold
        and spread >= cfg.trend_spread_threshold
    )
    if not base_ok:
        return False
    if cfg.require_trend_crossing:
        return trend_crossing_met(history, cfg)
    return True


@dataclass
class BreakoutResult:
    recent_range_high: float | None
    breakout_level: float | None
    price_breakout: bool


def compute_breakout(history: RollingBarHistory, cfg: StateMachineConfig) -> BreakoutResult:
    bar = history.current()
    if bar is None:
        return BreakoutResult(None, None, False)

    prior_prices = history.price_window_prior(cfg.range_lookback_bars)
    if len(prior_prices) < cfg.range_lookback_bars:
        return BreakoutResult(None, None, False)

    recent_high = max(prior_prices)
    level = recent_high * (1.0 + cfg.breakout_buffer_pct)
    return BreakoutResult(
        recent_range_high=recent_high,
        breakout_level=level,
        price_breakout=bar.price > level,
    )


def confidence_gap(bar: NowcasterBar) -> float:
    probs = sorted(
        [
            bar.prob_chop,
            bar.prob_trending_up,
            bar.prob_trending_down,
            bar.prob_volatile_expansion,
        ],
        reverse=True,
    )
    return probs[0] - probs[1]


def confidence_filter_met(bar: NowcasterBar, cfg: StateMachineConfig) -> bool:
    probs = sorted(
        [
            bar.prob_chop,
            bar.prob_trending_up,
            bar.prob_trending_down,
            bar.prob_volatile_expansion,
        ],
        reverse=True,
    )
    gap = probs[0] - probs[1]
    return probs[0] >= cfg.minimum_model_confidence and gap >= cfg.minimum_confidence_gap


def long_entry_conditions_met(
    history: RollingBarHistory, cfg: StateMachineConfig
) -> tuple[bool, BreakoutResult, float]:
    bar = history.current()
    if bar is None:
        empty = BreakoutResult(None, None, False)
        return False, empty, 0.0
    breakout = compute_breakout(history, cfg)
    gap = confidence_gap(bar)
    ok = (
        bullish_confirmation_met(history, cfg)
        and breakout.price_breakout
        and confidence_filter_met(bar, cfg)
    )
    return ok, breakout, gap
