"""Bullish transition state machine — incremental bar processing."""

from __future__ import annotations

from config import StateMachineConfig
from conditions import (
    bullish_confirmation_met,
    chop_dominance_met,
    chop_dominance_ratio,
    compute_breakout,
    confidence_filter_met,
    confidence_gap,
    expansion_alert_met,
    long_entry_conditions_met,
    trend_spread,
    volatile_expansion_rise,
)
from history import RollingBarHistory
from models import (
    Action,
    NowcasterBar,
    SignalMetadata,
    SignalOutput,
    STATE_NUMERIC,
    TradingState,
)


class BullishTransitionStateMachine:
    """
    Detects CHOP_BASE → EXPANSION_ALERT → BULLISH_CONFIRMATION → LONG_ENTRY.

    Processes one nowcaster bar at a time for backtest and live use.
    """

    def __init__(self, config: StateMachineConfig) -> None:
        self.config = config
        self.history = RollingBarHistory(config.max_history_bars)
        self.state = TradingState.NEUTRAL
        self.setup_age_bars = 0
        self.cooldown_remaining = 0
        self._signal_fired_this_setup = False

    def reset(self) -> None:
        self.state = TradingState.NEUTRAL
        self.setup_age_bars = 0
        self.cooldown_remaining = 0
        self._signal_fired_this_setup = False
        self.history = RollingBarHistory(self.config.max_history_bars)

    def _reset_setup(self) -> None:
        self.setup_age_bars = 0
        self._signal_fired_this_setup = False

    def _build_metadata(self, bar: NowcasterBar) -> SignalMetadata:
        breakout = compute_breakout(self.history, self.config)
        return SignalMetadata(
            prob_chop=bar.prob_chop,
            prob_volatile_expansion=bar.prob_volatile_expansion,
            prob_trending_up=bar.prob_trending_up,
            prob_trending_down=bar.prob_trending_down,
            chop_dominance_ratio=chop_dominance_ratio(self.history, self.config),
            volatile_expansion_rise=volatile_expansion_rise(self.history, self.config),
            trend_spread=trend_spread(bar),
            recent_range_high=breakout.recent_range_high,
            breakout_level=breakout.breakout_level,
            price_breakout=breakout.price_breakout,
            confidence_gap=confidence_gap(bar),
            setup_age_bars=self.setup_age_bars,
            long_entry_signal=False,
        )

    def _output(
        self,
        bar: NowcasterBar,
        action: Action,
        reason: str,
        metadata: SignalMetadata | None = None,
    ) -> SignalOutput:
        meta = metadata or self._build_metadata(bar)
        return SignalOutput(
            timestamp=bar.timestamp.isoformat(),
            state=self.state,
            state_numeric=STATE_NUMERIC[self.state],
            action=action,
            reason=reason,
            metadata=meta,
        )

    def process_bar(self, bar: NowcasterBar, in_position: bool = False) -> SignalOutput:
        self.history.append(bar)

        if self.state == TradingState.COOLDOWN:
            self.cooldown_remaining -= 1
            if self.cooldown_remaining <= 0:
                self.state = TradingState.NEUTRAL
                self.cooldown_remaining = 0
                return self._output(bar, Action.NO_TRADE, "Cooldown complete; reset to NEUTRAL")
            return self._output(
                bar,
                Action.NO_TRADE,
                f"Cooldown active ({self.cooldown_remaining} bars remaining)",
            )

        if self.state == TradingState.IN_LONG and not in_position:
            self.state = TradingState.NEUTRAL
            self._reset_setup()
            return self._output(bar, Action.NO_TRADE, "Position closed; reset to NEUTRAL")

        if self.state in (TradingState.EXPANSION_ALERT, TradingState.BULLISH_CONFIRMATION):
            self.setup_age_bars += 1
            if self.setup_age_bars > self.config.max_signal_age_bars:
                self.state = TradingState.NEUTRAL
                self._reset_setup()
                return self._output(
                    bar,
                    Action.NO_TRADE,
                    f"Setup expired after {self.config.max_signal_age_bars} bars",
                )

        if self.state == TradingState.NEUTRAL:
            if chop_dominance_met(self.history, self.config):
                self.state = TradingState.CHOP_BASE
                return self._output(
                    bar,
                    Action.HOLD,
                    "Chop dominance confirmed; entering CHOP_BASE",
                )
            return self._output(bar, Action.NO_TRADE, "Waiting for chop dominance")

        if self.state == TradingState.CHOP_BASE:
            if not chop_dominance_met(self.history, self.config):
                self.state = TradingState.NEUTRAL
                return self._output(bar, Action.NO_TRADE, "Chop dominance lost; reset to NEUTRAL")
            if expansion_alert_met(self.history, self.config):
                self.state = TradingState.EXPANSION_ALERT
                self.setup_age_bars = 0
                self._signal_fired_this_setup = False
                return self._output(
                    bar,
                    Action.HOLD,
                    "Volatile expansion spike detected; entering EXPANSION_ALERT",
                )
            return self._output(bar, Action.HOLD, "In CHOP_BASE; waiting for expansion alert")

        if self.state == TradingState.EXPANSION_ALERT:
            if bullish_confirmation_met(self.history, self.config):
                self.state = TradingState.BULLISH_CONFIRMATION
                return self._output(
                    bar,
                    Action.HOLD,
                    "Trending up overtaking down; entering BULLISH_CONFIRMATION",
                )
            return self._output(
                bar,
                Action.HOLD,
                "In EXPANSION_ALERT; waiting for bullish trend confirmation",
            )

        if self.state == TradingState.BULLISH_CONFIRMATION:
            if in_position and not self.config.allow_reentry_while_in_position:
                return self._output(
                    bar,
                    Action.HOLD,
                    "Already in position; reentry disabled",
                )

            entry_ok, breakout, gap = long_entry_conditions_met(self.history, self.config)

            if entry_ok and not self._signal_fired_this_setup:
                meta = self._build_metadata(bar)
                meta.long_entry_signal = True
                meta.confidence_gap = gap
                meta.recent_range_high = breakout.recent_range_high
                meta.breakout_level = breakout.breakout_level
                meta.price_breakout = True

                self._signal_fired_this_setup = True
                self.state = TradingState.LONG_ENTRY
                long_out = self._output(
                    bar,
                    Action.ENTER_LONG,
                    "Price breakout above range high with confidence filter; ENTER_LONG",
                    metadata=meta,
                )

                if in_position:
                    self.state = TradingState.IN_LONG
                    self._reset_setup()
                else:
                    self.state = TradingState.COOLDOWN
                    self.cooldown_remaining = self.config.entry_cooldown_bars
                    self._reset_setup()

                return long_out

            if not entry_ok:
                if not bullish_confirmation_met(self.history, self.config):
                    return self._output(
                        bar,
                        Action.HOLD,
                        "Bullish trend confirmation lost; holding setup until expiry",
                    )
                if not breakout.price_breakout:
                    return self._output(
                        bar,
                        Action.HOLD,
                        "Awaiting price breakout above recent range high",
                    )
                if not confidence_filter_met(bar, self.config):
                    return self._output(
                        bar,
                        Action.HOLD,
                        "Breakout seen but model confidence filter not met",
                    )

            return self._output(
                bar,
                Action.HOLD,
                "In BULLISH_CONFIRMATION; awaiting entry conditions",
            )

        if self.state == TradingState.IN_LONG:
            return self._output(bar, Action.HOLD, "In long position")

        return self._output(bar, Action.NO_TRADE, "Unhandled state")
