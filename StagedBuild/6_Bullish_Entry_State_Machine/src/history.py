"""Rolling bar history for incremental state machine evaluation."""

from __future__ import annotations

from collections import deque

from models import NowcasterBar


class RollingBarHistory:
    """Bounded deque of nowcaster bars for lookback calculations."""

    def __init__(self, max_bars: int) -> None:
        self._bars: deque[NowcasterBar] = deque(maxlen=max_bars)

    def append(self, bar: NowcasterBar) -> None:
        self._bars.append(bar)

    def __len__(self) -> int:
        return len(self._bars)

    @property
    def bars(self) -> list[NowcasterBar]:
        return list(self._bars)

    def current(self) -> NowcasterBar | None:
        if not self._bars:
            return None
        return self._bars[-1]

    def lookback(self, n: int) -> list[NowcasterBar]:
        if n <= 0 or not self._bars:
            return []
        return list(self._bars)[-n:]

    def price_window_prior(self, n: int) -> list[float]:
        """Prices for the n bars immediately before the current bar (no lookahead)."""
        if len(self._bars) < 2:
            return []
        prior = list(self._bars)[:-1]
        if n <= 0:
            return prior
        return [b.price for b in prior[-n:]]

    def prob_series(self, attr: str, n: int | None = None) -> list[float]:
        bars = list(self._bars)
        if n is not None:
            bars = bars[-n:]
        return [getattr(b, attr) for b in bars]
